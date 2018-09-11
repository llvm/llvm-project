//===--- PerfMonitor.h --- Monitor time spent in scops --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PERF_MONITOR_H
#define PERF_MONITOR_H

#include "polly/CodeGen/IRBuilder.h"

namespace llvm {
class Function;
class Module;
class Value;
class Instruction;
} // namespace llvm

namespace polly {

class PerfMonitor {
public:
  /// Create a new performance monitor.
  ///
  /// @param S The scop for which to generate fine-grained performance
  ///          monitoring information.
  /// @param M The module for which to generate the performance monitor.
  PerfMonitor(const Scop &S, llvm::Module *M);

  /// Initialize the performance monitor.
  ///
  /// Ensure that all global variables, functions, and callbacks needed to
  /// manage the performance monitor are initialized and registered.
  void initialize();

  /// Mark the beginning of a timing region.
  ///
  /// @param InsertBefore The instruction before which the timing region starts.
  void insertRegionStart(llvm::Instruction *InsertBefore);

  /// Mark the end of a timing region.
  ///
  /// @param InsertBefore The instruction before which the timing region starts.
  void insertRegionEnd(llvm::Instruction *InsertBefore);

private:
  llvm::Module *M;
  PollyIRBuilder Builder;

  // The scop to profile against.
  const Scop &S;

  /// Indicates if performance profiling is supported on this architecture.
  bool Supported;

  /// The cycle counter at the beginning of the program execution.
  llvm::Value *CyclesTotalStartPtr;

  /// The total number of cycles spent in the current scop S.
  llvm::Value *CyclesInCurrentScopPtr;

  /// The total number of times the current scop S is executed.
  llvm::Value *TripCountForCurrentScopPtr;

  /// The total number of cycles spent within scops.
  llvm::Value *CyclesInScopsPtr;

  /// The value of the cycle counter at the beginning of the last scop.
  llvm::Value *CyclesInScopStartPtr;

  /// A global variable, that keeps track if the performance monitor
  /// initialization has already been run.
  llvm::Value *AlreadyInitializedPtr;

  llvm::Function *insertInitFunction(llvm::Function *FinalReporting);

  /// Add Function @p to list of global constructors
  ///
  /// If no global constructors are available in this current module, insert
  /// a new list of global constructors containing @p Fn as only global
  /// constructor. Otherwise, append @p Fn to the list of global constructors.
  ///
  /// All functions listed as global constructors are executed before the
  /// main() function is called.
  ///
  /// @param Fn Function to add to global constructors
  void addToGlobalConstructors(llvm::Function *Fn);

  /// Add global variables to module.
  ///
  /// Insert a set of global variables that are used to track performance,
  /// into the module (or obtain references to them if they already exist).
  void addGlobalVariables();

  /// Add per-scop tracking to module.
  ///
  /// Insert the global variable which is used to track the number of cycles
  /// this scop runs.
  void addScopCounter();

  /// Get a reference to the intrinsic "{ i64, i32 } @llvm.x86.rdtscp()".
  ///
  /// The rdtscp function returns the current value of the processor's
  /// time-stamp counter as well as the current CPU identifier. On modern x86
  /// systems, the returned value is independent of the dynamic clock frequency
  /// and consistent across multiple cores. It can consequently be used to get
  /// accurate and low-overhead timing information. Even though the counter is
  /// wrapping, it can be reliably used even for measuring longer time
  /// intervals, as on a 1 GHz processor the counter only wraps every 545 years.
  ///
  /// The RDTSCP instruction is "pseudo" serializing:
  ///
  /// "“The RDTSCP instruction waits until all previous instructions have been
  /// executed before reading the counter. However, subsequent instructions may
  /// begin execution before the read operation is performed.”
  ///
  /// To ensure that no later instructions are scheduled before the RDTSCP
  /// instruction it is often recommended to schedule a cpuid call after the
  /// RDTSCP instruction. We do not do this yet, trading some imprecision in
  /// our timing for a reduced overhead in our timing.
  ///
  /// @returns A reference to the declaration of @llvm.x86.rdtscp.
  llvm::Function *getRDTSCP();

  /// Get a reference to "int atexit(void (*function)(void))" function.
  ///
  /// This function allows to register function pointers that must be executed
  /// when the program is terminated.
  ///
  /// @returns A reference to @atexit().
  llvm::Function *getAtExit();

  /// Create function "__polly_perf_final_reporting".
  ///
  /// This function finalizes the performance measurements and prints the
  /// results to stdout. It is expected to be registered with 'atexit()'.
  llvm::Function *insertFinalReporting();

  /// Append Scop reporting data to "__polly_perf_final_reporting".
  ///
  /// This function appends the current scop (S)'s information to the final
  /// printing function.
  void AppendScopReporting();
};
} // namespace polly

#endif
