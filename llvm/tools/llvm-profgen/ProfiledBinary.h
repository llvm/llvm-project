//===-- ProfiledBinary.h - Binary decoder -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PROFGEN_PROFILEDBINARY_H
#define LLVM_TOOLS_LLVM_PROFGEN_PROFILEDBINARY_H

#include "CallContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Path.h"
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace llvm::object;

namespace llvm {
namespace sampleprof {

class ProfiledBinary;

struct InstructionPointer {
  ProfiledBinary *Binary;
  // Offset to the base address of the executable segment of the binary.
  uint64_t Offset;
  // Index to the sorted code address array of the binary.
  uint64_t Index;

  InstructionPointer(ProfiledBinary *Binary, uint64_t Offset)
      : Binary(Binary), Offset(Offset) {
    Index = 0;
  }
};

class ProfiledBinary {
  // Absolute path of the binary.
  std::string Path;
  // The target triple.
  Triple TheTriple;
  // The runtime base address that the executable sections are loaded at.
  mutable uint64_t BaseAddress = 0;
  // The preferred base address that the executable sections are loaded at.
  uint64_t PreferredBaseAddress = 0;
  // Mutiple MC component info
  std::unique_ptr<const MCRegisterInfo> MRI;
  std::unique_ptr<const MCAsmInfo> AsmInfo;
  std::unique_ptr<const MCSubtargetInfo> STI;
  std::unique_ptr<const MCInstrInfo> MII;
  std::unique_ptr<MCDisassembler> DisAsm;
  std::unique_ptr<const MCInstrAnalysis> MIA;
  std::unique_ptr<MCInstPrinter> IP;
  // A list of text sections sorted by start RVA and size. Used to check
  // if a given RVA is a valid code address.
  std::set<std::pair<uint64_t, uint64_t>> TextSections;
  // Function offset to name mapping.
  std::unordered_map<uint64_t, std::string> FuncStartAddrMap;
  // An array of offsets of all instructions sorted in increasing order. The
  // sorting is needed to fast advance to the next forward/backward instruction.
  std::vector<uint64_t> CodeAddrs;
  // A set of call instruction offsets. Used by virtual unwinding.
  std::unordered_set<uint64_t> CallAddrs;
  // A set of return instruction offsets. Used by virtual unwinding.
  std::unordered_set<uint64_t> RetAddrs;

  // The symbolizer used to get inline context for an instruction.
  std::unique_ptr<symbolize::LLVMSymbolizer> Symbolizer;

  void setPreferredBaseAddress(const ELFObjectFileBase *O);

  // Set up disassembler and related components.
  void setUpDisassembler(const ELFObjectFileBase *Obj);
  void setupSymbolizer();

  /// Dissassemble the text section and build various address maps.
  void disassemble(const ELFObjectFileBase *O);

  /// Helper function to dissassemble the symbol and extract info for unwinding
  bool dissassembleSymbol(std::size_t SI, ArrayRef<uint8_t> Bytes,
                          SectionSymbolsTy &Symbols, const SectionRef &Section);
  /// Symbolize a given instruction pointer and return a full call context.
  FrameLocationStack symbolize(const InstructionPointer &I);

  /// Decode the interesting parts of the binary and build internal data
  /// structures. On high level, the parts of interest are:
  ///   1. Text sections, including the main code section and the PLT
  ///   entries that will be used to handle cross-module call transitions.
  ///   2. The .debug_line section, used by Dwarf-based profile generation.
  ///   3. Pseudo probe related sections, used by probe-based profile
  ///   generation.
  void load();

public:
  ProfiledBinary(StringRef Path) : Path(Path) {
    setupSymbolizer();
    load();
  }

  const StringRef getPath() const { return Path; }
  const StringRef getName() const { return llvm::sys::path::filename(Path); }
  uint64_t getBaseAddress() const { return BaseAddress; }
  void setBaseAddress(uint64_t Address) { BaseAddress = Address; }
};

} // end namespace sampleprof
} // end namespace llvm

#endif
