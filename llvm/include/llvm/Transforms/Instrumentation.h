//===- Transforms/Instrumentation.h - Instrumentation passes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines constructor functions for instrumentation passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include <vector>

#if defined(__GNUC__) && defined(__linux__) && !defined(ANDROID)
inline void *getDFSanArgTLSPtrForJIT() {
  extern __thread __attribute__((tls_model("initial-exec")))
    void *__dfsan_arg_tls;
  return (void *)&__dfsan_arg_tls;
}

inline void *getDFSanRetValTLSPtrForJIT() {
  extern __thread __attribute__((tls_model("initial-exec")))
    void *__dfsan_retval_tls;
  return (void *)&__dfsan_retval_tls;
}
#endif

namespace llvm {

class TargetMachine;

/// Instrumentation passes often insert conditional checks into entry blocks.
/// Call this function before splitting the entry block to move instructions
/// that must remain in the entry block up before the split point. Static
/// allocas and llvm.localescape calls, for example, must remain in the entry
/// block.
BasicBlock::iterator PrepareToSplitEntryBlock(BasicBlock &BB,
                                              BasicBlock::iterator IP);

class ModulePass;
class FunctionPass;

// Insert GCOV profiling instrumentation
struct GCOVOptions {
  static GCOVOptions getDefault();

  // Specify whether to emit .gcno files.
  bool EmitNotes;

  // Specify whether to modify the program to emit .gcda files when run.
  bool EmitData;

  // A four-byte version string. The meaning of a version string is described in
  // gcc's gcov-io.h
  char Version[4];

  // Emit a "cfg checksum" that follows the "line number checksum" of a
  // function. This affects both .gcno and .gcda files.
  bool UseCfgChecksum;

  // Add the 'noredzone' attribute to added runtime library calls.
  bool NoRedZone;

  // Emit the name of the function in the .gcda files. This is redundant, as
  // the function identifier can be used to find the name from the .gcno file.
  bool FunctionNamesInData;

  // Emit the exit block immediately after the start block, rather than after
  // all of the function body's blocks.
  bool ExitBlockBeforeBody;
};
ModulePass *createGCOVProfilerPass(const GCOVOptions &Options =
                                   GCOVOptions::getDefault());

// PGO Instrumention
ModulePass *createPGOInstrumentationGenLegacyPass();
ModulePass *
createPGOInstrumentationUseLegacyPass(StringRef Filename = StringRef(""));
ModulePass *createPGOIndirectCallPromotionLegacyPass(bool InLTO = false);

/// Options for the frontend instrumentation based profiling pass.
struct InstrProfOptions {
  InstrProfOptions() : NoRedZone(false) {}

  // Add the 'noredzone' attribute to added runtime library calls.
  bool NoRedZone;

  // Name of the profile file to use as output
  std::string InstrProfileOutput;
};

/// Insert frontend instrumentation based profiling.
ModulePass *createInstrProfilingLegacyPass(
    const InstrProfOptions &Options = InstrProfOptions());

// Insert AddressSanitizer (address sanity checking) instrumentation
FunctionPass *createAddressSanitizerFunctionPass(bool CompileKernel = false,
                                                 bool Recover = false,
                                                 bool UseAfterScope = false);
ModulePass *createAddressSanitizerModulePass(bool CompileKernel = false,
                                             bool Recover = false);

// Insert MemorySanitizer instrumentation (detection of uninitialized reads)
FunctionPass *createMemorySanitizerPass(int TrackOrigins = 0);

// Insert ThreadSanitizer (race detection) instrumentation
FunctionPass *createThreadSanitizerPass();

// Insert DataFlowSanitizer (dynamic data flow analysis) instrumentation
ModulePass *createDataFlowSanitizerPass(
    const std::vector<std::string> &ABIListFiles = std::vector<std::string>(),
    void *(*getArgTLS)() = nullptr, void *(*getRetValTLS)() = nullptr);

// Options for EfficiencySanitizer sub-tools.
struct EfficiencySanitizerOptions {
  EfficiencySanitizerOptions() : ToolType(ESAN_None) {}
  enum Type {
    ESAN_None = 0,
    ESAN_CacheFrag,
    ESAN_WorkingSet,
  } ToolType;
};

// Insert EfficiencySanitizer instrumentation.
ModulePass *createEfficiencySanitizerPass(
    const EfficiencySanitizerOptions &Options = EfficiencySanitizerOptions());

// Options for sanitizer coverage instrumentation.
struct SanitizerCoverageOptions {
  SanitizerCoverageOptions()
      : CoverageType(SCK_None), IndirectCalls(false), TraceBB(false),
        TraceCmp(false), TraceDiv(false), TraceGep(false),
        Use8bitCounters(false), TracePC(false), TracePCGuard(false) {}

  enum Type {
    SCK_None = 0,
    SCK_Function,
    SCK_BB,
    SCK_Edge
  } CoverageType;
  bool IndirectCalls;
  bool TraceBB;
  bool TraceCmp;
  bool TraceDiv;
  bool TraceGep;
  bool Use8bitCounters;
  bool TracePC;
  bool TracePCGuard;
};

// Insert SanitizerCoverage instrumentation.
ModulePass *createSanitizerCoverageModulePass(
    const SanitizerCoverageOptions &Options = SanitizerCoverageOptions());

#if defined(__GNUC__) && defined(__linux__) && !defined(ANDROID)
inline ModulePass *createDataFlowSanitizerPassForJIT(
    const std::vector<std::string> &ABIListFiles = std::vector<std::string>()) {
  return createDataFlowSanitizerPass(ABIListFiles, getDFSanArgTLSPtrForJIT,
                                     getDFSanRetValTLSPtrForJIT);
}
#endif

// BoundsChecking - This pass instruments the code to perform run-time bounds
// checking on loads, stores, and other memory intrinsics.
FunctionPass *createBoundsCheckingPass();

/// \brief Calculate what to divide by to scale counts.
///
/// Given the maximum count, calculate a divisor that will scale all the
/// weights to strictly less than UINT32_MAX.
static inline uint64_t calculateCountScale(uint64_t MaxCount) {
  return MaxCount < UINT32_MAX ? 1 : MaxCount / UINT32_MAX + 1;
}

/// \brief Scale an individual branch count.
///
/// Scale a 64-bit weight down to 32-bits using \c Scale.
///
static inline uint32_t scaleBranchCount(uint64_t Count, uint64_t Scale) {
  uint64_t Scaled = Count / Scale;
  assert(Scaled <= UINT32_MAX && "overflow 32-bits");
  return Scaled;
}

} // End llvm namespace

#endif
