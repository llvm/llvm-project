//===-- llvm/lib/CodeGen/AsmPrinter/CodeViewDebug.h ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Microsoft CodeView debug info.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_CODEVIEWDEBUG_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_CODEVIEWDEBUG_H

#include "AsmPrinterHandler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
/// \brief Collects and handles line tables information in a CodeView format.
class LLVM_LIBRARY_VISIBILITY CodeViewDebug : public AsmPrinterHandler {
  AsmPrinter *Asm;
  DebugLoc PrevInstLoc;

  // For each function, store a vector of labels to its instructions, as well as
  // to the end of the function.
  struct FunctionInfo {
    DebugLoc LastLoc;
    MCSymbol *End = nullptr;
    unsigned FuncId = 0;
    unsigned LastFileId;
    bool HaveLineInfo = false;
  };
  FunctionInfo *CurFn;

  unsigned NextFuncId = 0;

  /// Remember some debug info about each function. Keep it in a stable order to
  /// emit at the end of the TU.
  MapVector<const Function *, FunctionInfo> FnDebugInfo;

  /// Map from DIFile to .cv_file id.
  DenseMap<const DIFile *, unsigned> FileIdMap;

  typedef std::map<const DIFile *, std::string> FileToFilepathMapTy;
  FileToFilepathMapTy FileToFilepathMap;
  StringRef getFullFilepath(const DIFile *S);

  unsigned maybeRecordFile(const DIFile *F);

  void maybeRecordLocation(DebugLoc DL, const MachineFunction *MF);

  void clear() {
    assert(CurFn == nullptr);
    FileIdMap.clear();
    FnDebugInfo.clear();
    FileToFilepathMap.clear();
  }

  void emitDebugInfoForFunction(const Function *GV, FunctionInfo &FI);

public:
  CodeViewDebug(AsmPrinter *Asm);

  void setSymbolSize(const llvm::MCSymbol *, uint64_t) override {}

  /// \brief Emit the COFF section that holds the line table information.
  void endModule() override;

  /// \brief Gather pre-function debug information.
  void beginFunction(const MachineFunction *MF) override;

  /// \brief Gather post-function debug information.
  void endFunction(const MachineFunction *) override;

  /// \brief Process beginning of an instruction.
  void beginInstruction(const MachineInstr *MI) override;

  /// \brief Process end of an instruction.
  void endInstruction() override {}
};
} // End of namespace llvm

#endif
