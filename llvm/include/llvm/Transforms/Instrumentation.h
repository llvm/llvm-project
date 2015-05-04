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
};
ModulePass *createGCOVProfilerPass(const GCOVOptions &Options =
                                   GCOVOptions::getDefault());

/// Options for the frontend instrumentation based profiling pass.
struct InstrProfOptions {
  InstrProfOptions() : NoRedZone(false) {}

  // Add the 'noredzone' attribute to added runtime library calls.
  bool NoRedZone;
};

/// Insert frontend instrumentation based profiling.
ModulePass *createInstrProfilingPass(
    const InstrProfOptions &Options = InstrProfOptions());

// Insert AddressSanitizer (address sanity checking) instrumentation
FunctionPass *createAddressSanitizerFunctionPass();
ModulePass *createAddressSanitizerModulePass();

// Insert MemorySanitizer instrumentation (detection of uninitialized reads)
FunctionPass *createMemorySanitizerPass(int TrackOrigins = 0);

// Insert ThreadSanitizer (race detection) instrumentation
FunctionPass *createThreadSanitizerPass();

// Insert DataFlowSanitizer (dynamic data flow analysis) instrumentation
ModulePass *createDataFlowSanitizerPass(StringRef ABIListFile = StringRef(),
                                        void *(*getArgTLS)() = nullptr,
                                        void *(*getRetValTLS)() = nullptr);

// Insert SanitizerCoverage instrumentation.
ModulePass *createSanitizerCoverageModulePass(int CoverageLevel);

#if defined(__GNUC__) && defined(__linux__) && !defined(ANDROID)
inline ModulePass *createDataFlowSanitizerPassForJIT(StringRef ABIListFile =
                                                         StringRef()) {
  return createDataFlowSanitizerPass(ABIListFile, getDFSanArgTLSPtrForJIT,
                                     getDFSanRetValTLSPtrForJIT);
}
#endif

// BoundsChecking - This pass instruments the code to perform run-time bounds
// checking on loads, stores, and other memory intrinsics.
FunctionPass *createBoundsCheckingPass();

} // End llvm namespace

#endif
