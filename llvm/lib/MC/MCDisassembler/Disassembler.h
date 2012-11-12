//===------------- Disassembler.h - LLVM Disassembler -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the Disassembly library's disassembler 
// context.  The disassembler is responsible for producing strings for
// individual instructions according to a given architecture and disassembly
// syntax.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_DISASSEMBLER_H
#define LLVM_MC_DISASSEMBLER_H

#include "llvm-c/Disassembler.h"
#include <string>
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class MCContext;
class MCAsmInfo;
class MCDisassembler;
class MCInstPrinter; 
class MCInstrInfo;
class MCRegisterInfo;
class MCSubtargetInfo;
class Target;

//
// This is the disassembler context returned by LLVMCreateDisasm().
//
class LLVMDisasmContext {
private:
  //
  // The passed parameters when the disassembler context is created.
  //
  // The TripleName for this disassembler.
  std::string TripleName;
  // The pointer to the caller's block of symbolic information.
  void *DisInfo;
  // The Triple specific symbolic information type returned by GetOpInfo.
  int TagType;
  // The function to get the symbolic information for operands.
  LLVMOpInfoCallback GetOpInfo;
  // The function to look up a symbol name.
  LLVMSymbolLookupCallback SymbolLookUp;
  //
  // The objects created and saved by LLVMCreateDisasm() then used by
  // LLVMDisasmInstruction().
  //
  // The LLVM target corresponding to the disassembler.
  // FIXME: using llvm::OwningPtr<const llvm::Target> causes a malloc error
  //        when this LLVMDisasmContext is deleted.
  const Target *TheTarget;
  // The assembly information for the target architecture.
  llvm::OwningPtr<const llvm::MCAsmInfo> MAI;
  // The register information for the target architecture.
  llvm::OwningPtr<const llvm::MCRegisterInfo> MRI;
  // The subtarget information for the target architecture.
  llvm::OwningPtr<const llvm::MCSubtargetInfo> MSI;
  // The instruction information for the target architecture.
  llvm::OwningPtr<const llvm::MCInstrInfo> MII;
  // The assembly context for creating symbols and MCExprs.
  llvm::OwningPtr<const llvm::MCContext> Ctx;
  // The disassembler for the target architecture.
  llvm::OwningPtr<const llvm::MCDisassembler> DisAsm;
  // The instruction printer for the target architecture.
  llvm::OwningPtr<llvm::MCInstPrinter> IP;

public:
  // Comment stream and backing vector.
  SmallString<128> CommentsToEmit;
  raw_svector_ostream CommentStream;

  LLVMDisasmContext(std::string tripleName, void *disInfo, int tagType,
                    LLVMOpInfoCallback getOpInfo,
                    LLVMSymbolLookupCallback symbolLookUp,
                    const Target *theTarget, const MCAsmInfo *mAI,
                    const MCRegisterInfo *mRI,
                    const MCSubtargetInfo *mSI,
                    const MCInstrInfo *mII,
                    llvm::MCContext *ctx, const MCDisassembler *disAsm,
                    MCInstPrinter *iP) : TripleName(tripleName),
                    DisInfo(disInfo), TagType(tagType), GetOpInfo(getOpInfo),
                    SymbolLookUp(symbolLookUp), TheTarget(theTarget),
                    CommentStream(CommentsToEmit) {
    MAI.reset(mAI);
    MRI.reset(mRI);
    MSI.reset(mSI);
    MII.reset(mII);
    Ctx.reset(ctx);
    DisAsm.reset(disAsm);
    IP.reset(iP);
  }
  const std::string &getTripleName() const { return TripleName; }
  void *getDisInfo() const { return DisInfo; }
  int getTagType() const { return TagType; }
  LLVMOpInfoCallback getGetOpInfo() const { return GetOpInfo; }
  LLVMSymbolLookupCallback getSymbolLookupCallback() const {
    return SymbolLookUp;
  }
  const Target *getTarget() const { return TheTarget; }
  const MCDisassembler *getDisAsm() const { return DisAsm.get(); }
  const MCAsmInfo *getAsmInfo() const { return MAI.get(); }
  MCInstPrinter *getIP() { return IP.get(); }
};

} // namespace llvm

#endif
