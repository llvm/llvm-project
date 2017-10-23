//===- FileAnalysis.h -------------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CFI_VERIFY_FILE_ANALYSIS_H
#define LLVM_CFI_VERIFY_FILE_ANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <set>
#include <string>
#include <unordered_map>

namespace llvm {
namespace cfi_verify {

// Disassembler and analysis tool for machine code files. Keeps track of non-
// sequential control flows, including indirect control flow instructions.
class FileAnalysis {
public:
  // A metadata struct for an instruction.
  struct Instr {
    uint64_t VMAddress;       // Virtual memory address of this instruction.
    MCInst Instruction;       // Instruction.
    uint64_t InstructionSize; // Size of this instruction.
    bool Valid; // Is this a valid instruction? If false, Instr::Instruction is
                // undefined.
  };

  // Construct a FileAnalysis from a file path.
  static Expected<FileAnalysis> Create(StringRef Filename);

  // Construct and take ownership of the supplied object. Do not use this
  // constructor, prefer to use FileAnalysis::Create instead.
  FileAnalysis(object::OwningBinary<object::Binary> Binary);
  FileAnalysis() = delete;
  FileAnalysis(const FileAnalysis &) = delete;
  FileAnalysis(FileAnalysis &&Other) = default;

  // Returns the instruction at the provided address. Returns nullptr if there
  // is no instruction at the provided address.
  const Instr *getInstruction(uint64_t Address) const;

  // Returns the instruction at the provided adress, dying if the instruction is
  // not found.
  const Instr &getInstructionOrDie(uint64_t Address) const;

  // Returns a pointer to the previous/next instruction in sequence,
  // respectively. Returns nullptr if the next/prev instruction doesn't exist,
  // or if the provided instruction doesn't exist.
  const Instr *getPrevInstructionSequential(const Instr &InstrMeta) const;
  const Instr *getNextInstructionSequential(const Instr &InstrMeta) const;

  // Returns whether this instruction is used by CFI to trap the program.
  bool isCFITrap(const Instr &InstrMeta) const;

  // Returns whether this function can fall through to the next instruction.
  // Undefined (and bad) instructions cannot fall through, and instruction that
  // modify the control flow can only fall through if they are conditional
  // branches or calls.
  bool canFallThrough(const Instr &InstrMeta) const;

  // Returns the definitive next instruction. This is different from the next
  // instruction sequentially as it will follow unconditional branches (assuming
  // they can be resolved at compile time, i.e. not indirect). This method
  // returns nullptr if the provided instruction does not transfer control flow
  // to exactly one instruction that is known deterministically at compile time.
  // Also returns nullptr if the deterministic target does not exist in this
  // file.
  const Instr *getDefiniteNextInstruction(const Instr &InstrMeta) const;

  // Get a list of deterministic control flows that lead to the provided
  // instruction. This list includes all static control flow cross-references as
  // well as the previous instruction if it can fall through.
  std::set<const Instr *>
  getDirectControlFlowXRefs(const Instr &InstrMeta) const;

  // Returns whether this instruction uses a register operand.
  bool usesRegisterOperand(const Instr &InstrMeta) const;

  // Returns the list of indirect instructions.
  const std::set<uint64_t> &getIndirectInstructions() const;

  const MCRegisterInfo *getRegisterInfo() const;
  const MCInstrInfo *getMCInstrInfo() const;
  const MCInstrAnalysis *getMCInstrAnalysis() const;

protected:
  // Construct a blank object with the provided triple and features. Used in
  // testing, where a sub class will dependency inject protected methods to
  // allow analysis of raw binary, without requiring a fully valid ELF file.
  FileAnalysis(const Triple &ObjectTriple, const SubtargetFeatures &Features);

  // Add an instruction to this object.
  void addInstruction(const Instr &Instruction);

  // Disassemble and parse the provided bytes into this object. Instruction
  // address calculation is done relative to the provided SectionAddress.
  void parseSectionContents(ArrayRef<uint8_t> SectionBytes,
                            uint64_t SectionAddress);

  // Constructs and initialises members required for disassembly.
  Error initialiseDisassemblyMembers();

  // Parses code sections from the internal object file. Saves them into the
  // internal members. Should only be called once by Create().
  Error parseCodeSections();

private:
  // Members that describe the input file.
  object::OwningBinary<object::Binary> Binary;
  const object::ObjectFile *Object = nullptr;
  Triple ObjectTriple;
  std::string ArchName;
  std::string MCPU;
  const Target *ObjectTarget = nullptr;
  SubtargetFeatures Features;

  // Members required for disassembly.
  std::unique_ptr<const MCRegisterInfo> RegisterInfo;
  std::unique_ptr<const MCAsmInfo> AsmInfo;
  std::unique_ptr<MCSubtargetInfo> SubtargetInfo;
  std::unique_ptr<const MCInstrInfo> MII;
  MCObjectFileInfo MOFI;
  std::unique_ptr<MCContext> Context;
  std::unique_ptr<const MCDisassembler> Disassembler;
  std::unique_ptr<const MCInstrAnalysis> MIA;
  std::unique_ptr<MCInstPrinter> Printer;

  // A mapping between the virtual memory address to the instruction metadata
  // struct.
  std::map<uint64_t, Instr> Instructions;

  // Contains a mapping between a specific address, and a list of instructions
  // that use this address as a branch target (including call instructions).
  DenseMap<uint64_t, std::vector<uint64_t>> StaticBranchTargetings;

  // A list of addresses of indirect control flow instructions.
  std::set<uint64_t> IndirectInstructions;
};

class UnsupportedDisassembly : public ErrorInfo<UnsupportedDisassembly> {
public:
  static char ID;
  std::string Text;

  UnsupportedDisassembly(StringRef Text);

  void log(raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;
};

} // namespace cfi_verify
} // namespace llvm

#endif // LLVM_CFI_VERIFY_FILE_ANALYSIS_H
