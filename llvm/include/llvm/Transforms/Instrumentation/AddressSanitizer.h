//===--------- Definition of the AddressSanitizer class ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the AddressSanitizer class which is a port of the legacy
// AddressSanitizer pass to use the new PassManager infrastructure.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZER_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"

namespace llvm {

/// Frontend-provided metadata for source location.
struct LocationMetadata {
  StringRef Filename;
  int LineNo = 0;
  int ColumnNo = 0;

  LocationMetadata() = default;

  bool empty() const { return Filename.empty(); }
  void parse(MDNode *MDN);
};

/// Frontend-provided metadata for global variables.
class GlobalsMetadata {
public:
  struct Entry {
    LocationMetadata SourceLoc;
    StringRef Name;
    bool IsDynInit = false;
    bool IsExcluded = false;

    Entry() = default;
  };

  /// Create a default uninitialized GlobalsMetadata instance.
  GlobalsMetadata() = default;

  /// Create an initialized GlobalsMetadata instance.
  GlobalsMetadata(Module &M);

  /// Returns metadata entry for a given global.
  Entry get(GlobalVariable *G) const {
    auto Pos = Entries.find(G);
    return (Pos != Entries.end()) ? Pos->second : Entry();
  }

  /// Handle invalidation from the pass manager.
  /// These results are never invalidated.
  bool invalidate(Module &, const PreservedAnalyses &,
                  ModuleAnalysisManager::Invalidator &) {
    return false;
  }
  bool invalidate(Function &, const PreservedAnalyses &,
                  FunctionAnalysisManager::Invalidator &) {
    return false;
  }

private:
  DenseMap<GlobalVariable *, Entry> Entries;
};

/// The ASanGlobalsMetadataAnalysis initializes and returns a GlobalsMetadata
/// object. More specifically, ASan requires looking at all globals registered
/// in 'llvm.asan.globals' before running, which only depends on reading module
/// level metadata. This analysis is required to run before running the
/// AddressSanitizerPass since it collects that metadata.
/// The legacy pass manager equivalent of this is ASanGlobalsMetadataLegacyPass.
class ASanGlobalsMetadataAnalysis
    : public AnalysisInfoMixin<ASanGlobalsMetadataAnalysis> {
public:
  using Result = GlobalsMetadata;

  Result run(Module &, ModuleAnalysisManager &);

private:
  friend AnalysisInfoMixin<ASanGlobalsMetadataAnalysis>;
  static AnalysisKey Key;
};

/// Public interface to the address sanitizer pass for instrumenting code to
/// check for various memory errors at runtime.
///
/// The sanitizer itself is a function pass that works by inserting various
/// calls to the ASan runtime library functions. The runtime library essentially
/// replaces malloc() and free() with custom implementations that allow regions
/// surrounding requested memory to be checked for invalid accesses.
class AddressSanitizerPass : public PassInfoMixin<AddressSanitizerPass> {
public:
  explicit AddressSanitizerPass(
      bool CompileKernel = false, bool Recover = false,
      bool UseAfterScope = false,
      AsanDetectStackUseAfterReturnMode UseAfterReturn =
          AsanDetectStackUseAfterReturnMode::Runtime);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  bool CompileKernel;
  bool Recover;
  bool UseAfterScope;
  AsanDetectStackUseAfterReturnMode UseAfterReturn;
};

/// Public interface to the address sanitizer module pass for instrumenting code
/// to check for various memory errors.
///
/// This adds 'asan.module_ctor' to 'llvm.global_ctors'. This pass may also
/// run intependently of the function address sanitizer.
class ModuleAddressSanitizerPass
    : public PassInfoMixin<ModuleAddressSanitizerPass> {
public:
  explicit ModuleAddressSanitizerPass(
      bool CompileKernel = false, bool Recover = false, bool UseGlobalGC = true,
      bool UseOdrIndicator = false,
      AsanDtorKind DestructorKind = AsanDtorKind::Global);
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  bool CompileKernel;
  bool Recover;
  bool UseGlobalGC;
  bool UseOdrIndicator;
  AsanDtorKind DestructorKind;
};

// Insert AddressSanitizer (address sanity checking) instrumentation
FunctionPass *createAddressSanitizerFunctionPass(
    bool CompileKernel = false, bool Recover = false,
    bool UseAfterScope = false,
    AsanDetectStackUseAfterReturnMode UseAfterReturn =
        AsanDetectStackUseAfterReturnMode::Runtime);
ModulePass *createModuleAddressSanitizerLegacyPassPass(
    bool CompileKernel = false, bool Recover = false, bool UseGlobalsGC = true,
    bool UseOdrIndicator = true,
    AsanDtorKind DestructorKind = AsanDtorKind::Global);

} // namespace llvm

#endif
