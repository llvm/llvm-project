//===--- ExecutorBase.h - Non-visitor methods of InstExecutor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares non-visitor methods of InstExecutor for code reuse.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLUBI_EXECUTORBASE_H
#define LLVM_TOOLS_LLUBI_EXECUTORBASE_H

#include "Context.h"
#include "Value.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <utility>

namespace llvm::ubi {

enum class FrameState {
  // It is about to enter the function.
  // Valid transition:
  //   -> Running
  Entry,
  // It is executing instructions inside the function.
  // Valid transitions:
  //   -> Pending (on call)
  //   -> Exit (on return)
  Running,
  // It is about to enter a callee or handle return value from the callee.
  // Valid transitions:
  //   -> Running (after returning from callee)
  Pending,
  // It is about to return the control to the caller.
  Exit,
};

/// Context for a function call.
/// This struct maintains the state during the execution of a function,
/// including the control flow, values of executed instructions, and stack
/// objects.
struct Frame {
  Function &Func;
  Frame *LastFrame;
  CallBase *CallSite;
  ArrayRef<AnyValue> Args;
  AnyValue &RetVal;

  TargetLibraryInfo TLI;
  BasicBlock *BB;
  BasicBlock::iterator PC;
  FrameState State = FrameState::Entry;
  // Stack objects allocated in this frame. They will be automatically freed
  // when the function returns.
  SmallVector<IntrusiveRefCntPtr<MemoryObject>> Allocas;
  // Values of arguments and executed instructions in this function.
  DenseMap<Value *, AnyValue> ValueMap;

  // Reserved for in-flight subroutines.
  Function *ResolvedCallee = nullptr;
  SmallVector<AnyValue> CalleeArgs;
  AnyValue CalleeRetVal;

  Frame(Function &F, CallBase *CallSite, Frame *LastFrame,
        ArrayRef<AnyValue> Args, AnyValue &RetVal,
        const TargetLibraryInfoImpl &TLIImpl);
};

enum class DiagnosticKind {
  ImmediateUB,
  Error,
};

class DiagnosticReporter;

class ExecutorBase {
  friend class DiagnosticReporter;

protected:
  Context &Ctx;
  EventHandler &Handler;
  Frame *CurrentFrame = nullptr;
  std::optional<ProgramExitInfo> ExitInfo;

  ExecutorBase(Context &C, EventHandler &H)
      : Ctx(C), Handler(H), ExitInfo(std::nullopt) {}
  ~ExecutorBase() = default;

private:
  void reportImmediateUBString(StringRef Msg);
  void reportErrorString(StringRef Msg);

public:
  DiagnosticReporter reportImmediateUB();
  DiagnosticReporter reportError();

  /// Check if the upcoming memory access is valid. Returns the offset relative
  /// to the underlying object if it is valid.
  std::optional<uint64_t> verifyMemAccess(const MemoryObject &MO,
                                          const APInt &Address,
                                          uint64_t AccessSize, Align Alignment,
                                          bool IsStore);

  AnyValue load(const AnyValue &Ptr, Align Alignment, Type *ValTy,
                bool NoUndef);
  void store(const AnyValue &Ptr, Align Alignment, const AnyValue &Val,
             Type *ValTy);

  void requestProgramExit(ProgramExitInfo::ProgramExitKind Kind,
                          uint64_t ExitCode = 0);
  void setFailed();

  bool hasProgramExited() const;
  std::optional<ProgramExitInfo> getExitInfo() const;

  unsigned getIntSize() const;

  void dumpStackTrace() const;
};

class DiagnosticReporter {
  ExecutorBase &Executor;
  std::string Buf;
  raw_string_ostream OS;
  DiagnosticKind Kind;

public:
  DiagnosticReporter(ExecutorBase &E, DiagnosticKind K)
      : Executor(E), OS(Buf), Kind(K) {}

  DiagnosticReporter(const DiagnosticReporter &) = delete;
  DiagnosticReporter(DiagnosticReporter &&) noexcept = delete;

  DiagnosticReporter &operator=(const DiagnosticReporter &) = delete;
  DiagnosticReporter &operator=(DiagnosticReporter &&) noexcept = delete;

  ~DiagnosticReporter() {
    switch (Kind) {
    case DiagnosticKind::ImmediateUB:
      Executor.reportImmediateUBString(Buf);
      break;
    case DiagnosticKind::Error:
      Executor.reportErrorString(Buf);
      break;
    }
  }

  template <typename T> DiagnosticReporter &operator<<(const T &Val) {
    OS << Val;
    return *this;
  }
};

} // namespace llvm::ubi

#endif // LLVM_TOOLS_LLUBI_EXECUTORBASE_H
