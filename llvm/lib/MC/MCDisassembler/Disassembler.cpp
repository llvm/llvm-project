//===-- lib/MC/Disassembler.cpp - Disassembler Public C Interface ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Disassembler.h"
#include "llvm-c/Disassembler.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
class Target;
} // namespace llvm
using namespace llvm;

// LLVMCreateDisasm() creates a disassembler for the TripleName.  Symbolic
// disassembly is supported by passing a block of information in the DisInfo
// parameter and specifying the TagType and callback functions as described in
// the header llvm-c/Disassembler.h .  The pointer to the block and the 
// functions can all be passed as NULL.  If successful, this returns a
// disassembler context.  If not, it returns NULL.
//
LLVMDisasmContextRef LLVMCreateDisasm(const char *TripleName, void *DisInfo,
                                      int TagType, LLVMOpInfoCallback GetOpInfo,
                                      LLVMSymbolLookupCallback SymbolLookUp) {
  // Initialize targets and assembly printers/parsers.
  // FIXME: Clients are responsible for initializing the targets. And this
  // would be done by calling routines in "llvm-c/Target.h" which are static
  // line functions. But the current use of LLVMCreateDisasm() is to dynamically
  // load libLTO with dlopen() and then lookup the symbols using dlsym().
  // And since these initialize routines are static that does not work which
  // is why the call to them in this 'C' library API was added back.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  // Get the target.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  assert(TheTarget && "Unable to create target!");

  // Get the assembler info needed to setup the MCContext.
  const MCAsmInfo *MAI = TheTarget->createMCAsmInfo(TripleName);
  assert(MAI && "Unable to create target asm info!");

  const MCInstrInfo *MII = TheTarget->createMCInstrInfo();
  assert(MII && "Unable to create target instruction info!");

  const MCRegisterInfo *MRI = TheTarget->createMCRegInfo(TripleName);
  assert(MRI && "Unable to create target register info!");

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  std::string CPU;

  const MCSubtargetInfo *STI = TheTarget->createMCSubtargetInfo(TripleName, CPU,
                                                                FeaturesStr);
  assert(STI && "Unable to create subtarget info!");

  // Set up the MCContext for creating symbols and MCExpr's.
  MCContext *Ctx = new MCContext(*MAI, *MRI, 0);
  assert(Ctx && "Unable to create MCContext!");

  // Set up disassembler.
  MCDisassembler *DisAsm = TheTarget->createMCDisassembler(*STI);
  assert(DisAsm && "Unable to create disassembler!");
  DisAsm->setupForSymbolicDisassembly(GetOpInfo, SymbolLookUp, DisInfo, Ctx);

  // Set up the instruction printer.
  int AsmPrinterVariant = MAI->getAssemblerDialect();
  MCInstPrinter *IP = TheTarget->createMCInstPrinter(AsmPrinterVariant,
                                                     *MAI, *MII, *MRI, *STI);
  assert(IP && "Unable to create instruction printer!");

  LLVMDisasmContext *DC = new LLVMDisasmContext(TripleName, DisInfo, TagType,
                                                GetOpInfo, SymbolLookUp,
                                                TheTarget, MAI, MRI,
                                                STI, MII, Ctx, DisAsm, IP);
  assert(DC && "Allocation failure!");

  return DC;
}

//
// LLVMDisasmDispose() disposes of the disassembler specified by the context.
//
void LLVMDisasmDispose(LLVMDisasmContextRef DCR){
  LLVMDisasmContext *DC = (LLVMDisasmContext *)DCR;
  delete DC;
}

namespace {
//
// The memory object created by LLVMDisasmInstruction().
//
class DisasmMemoryObject : public MemoryObject {
  uint8_t *Bytes;
  uint64_t Size;
  uint64_t BasePC;
public:
  DisasmMemoryObject(uint8_t *bytes, uint64_t size, uint64_t basePC) :
                     Bytes(bytes), Size(size), BasePC(basePC) {}
 
  uint64_t getBase() const { return BasePC; }
  uint64_t getExtent() const { return Size; }

  int readByte(uint64_t Addr, uint8_t *Byte) const {
    if (Addr - BasePC >= Size)
      return -1;
    *Byte = Bytes[Addr - BasePC];
    return 0;
  }
};
} // end anonymous namespace

//
// LLVMDisasmInstruction() disassembles a single instruction using the
// disassembler context specified in the parameter DC.  The bytes of the
// instruction are specified in the parameter Bytes, and contains at least
// BytesSize number of bytes.  The instruction is at the address specified by
// the PC parameter.  If a valid instruction can be disassembled its string is
// returned indirectly in OutString which whos size is specified in the
// parameter OutStringSize.  This function returns the number of bytes in the
// instruction or zero if there was no valid instruction.  If this function
// returns zero the caller will have to pick how many bytes they want to step
// over by printing a .byte, .long etc. to continue.
//
size_t LLVMDisasmInstruction(LLVMDisasmContextRef DCR, uint8_t *Bytes,
                             uint64_t BytesSize, uint64_t PC, char *OutString,
                             size_t OutStringSize){
  LLVMDisasmContext *DC = (LLVMDisasmContext *)DCR;
  // Wrap the pointer to the Bytes, BytesSize and PC in a MemoryObject.
  DisasmMemoryObject MemoryObject(Bytes, BytesSize, PC);

  uint64_t Size;
  MCInst Inst;
  const MCDisassembler *DisAsm = DC->getDisAsm();
  MCInstPrinter *IP = DC->getIP();
  MCDisassembler::DecodeStatus S;
  S = DisAsm->getInstruction(Inst, Size, MemoryObject, PC,
                             /*REMOVE*/ nulls(), DC->CommentStream);
  switch (S) {
  case MCDisassembler::Fail:
  case MCDisassembler::SoftFail:
    // FIXME: Do something different for soft failure modes?
    return 0;

  case MCDisassembler::Success: {
    DC->CommentStream.flush();
    StringRef Comments = DC->CommentsToEmit.str();

    SmallVector<char, 64> InsnStr;
    raw_svector_ostream OS(InsnStr);
    IP->printInst(&Inst, OS, Comments);
    OS.flush();

    // Tell the comment stream that the vector changed underneath it.
    DC->CommentsToEmit.clear();
    DC->CommentStream.resync();

    assert(OutStringSize != 0 && "Output buffer cannot be zero size");
    size_t OutputSize = std::min(OutStringSize-1, InsnStr.size());
    std::memcpy(OutString, InsnStr.data(), OutputSize);
    OutString[OutputSize] = '\0'; // Terminate string.

    return Size;
  }
  }
  llvm_unreachable("Invalid DecodeStatus!");
}

//
// LLVMSetDisasmOptions() sets the disassembler's options.  It returns 1 if it
// can set all the Options and 0 otherwise.
//
int LLVMSetDisasmOptions(LLVMDisasmContextRef DCR, uint64_t Options){
  if (Options & LLVMDisassembler_Option_UseMarkup){
      LLVMDisasmContext *DC = (LLVMDisasmContext *)DCR;
      MCInstPrinter *IP = DC->getIP();
      IP->setUseMarkup(1);
      Options &= ~LLVMDisassembler_Option_UseMarkup;
  }
  return (Options == 0);
}
