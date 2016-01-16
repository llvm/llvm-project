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
    SmallVector<MCSymbol *, 10> Instrs;
    MCSymbol *End;
    FunctionInfo() : End(nullptr) {}
  };
  FunctionInfo *CurFn;

  typedef DenseMap<const Function *, FunctionInfo> FnDebugInfoTy;
  FnDebugInfoTy FnDebugInfo;
  // Store the functions we've visited in a vector so we can maintain a stable
  // order while emitting subsections.
  SmallVector<const Function *, 10> VisitedFunctions;

  DenseMap<MCSymbol *, DebugLoc> LabelsAndLocs;

  // FileNameRegistry - Manages filenames observed while generating debug info
  // by filtering out duplicates and bookkeeping the offsets in the string
  // table to be generated.
  struct FileNameRegistryTy {
    SmallVector<StringRef, 10> Filenames;
    struct PerFileInfo {
      size_t FilenameID, StartOffset;
    };
    StringMap<PerFileInfo> Infos;

    // The offset in the string table where we'll write the next unique
    // filename.
    size_t LastOffset;

    FileNameRegistryTy() {
      clear();
    }

    // Add Filename to the registry, if it was not observed before.
    size_t add(StringRef Filename) {
      size_t OldSize = Infos.size();
      bool Inserted;
      StringMap<PerFileInfo>::iterator It;
      std::tie(It, Inserted) = Infos.insert(
          std::make_pair(Filename, PerFileInfo{OldSize, LastOffset}));
      if (Inserted) {
        LastOffset += Filename.size() + 1;
        Filenames.push_back(Filename);
      }
      return It->second.FilenameID;
    }

    void clear() {
      LastOffset = 1;
      Infos.clear();
      Filenames.clear();
    }
  } FileNameRegistry;

  typedef std::map<const DIFile *, std::string> FileToFilepathMapTy;
  FileToFilepathMapTy FileToFilepathMap;
  StringRef getFullFilepath(const DIFile *S);

  void maybeRecordLocation(DebugLoc DL, const MachineFunction *MF);

  void clear() {
    assert(CurFn == nullptr);
    FileNameRegistry.clear();
    LabelsAndLocs.clear();
  }

  void emitDebugInfoForFunction(const Function *GV);

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
