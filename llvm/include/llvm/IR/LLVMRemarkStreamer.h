//===- llvm/IR/LLVMRemarkStreamer.h - Streamer for LLVM remarks--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the conversion between IR Diagnostics and
// serializable remarks::Remark objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_LLVMREMARKSTREAMER_H
#define LLVM_IR_LLVMREMARKSTREAMER_H

#include "llvm/Remarks/Remark.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ToolOutputFile.h"
#include <memory>
#include <optional>
#include <string>

namespace llvm {

class DiagnosticInfoOptimizationBase;
class LLVMContext;
class ToolOutputFile;
namespace remarks {
class RemarkStreamer;
}

/// Streamer for LLVM remarks which has logic for dealing with DiagnosticInfo
/// objects.
class LLVMRemarkStreamer {
  remarks::RemarkStreamer &RS;
  /// Convert diagnostics into remark objects.
  /// The lifetime of the members of the result is bound to the lifetime of
  /// the LLVM diagnostics.
  remarks::Remark toRemark(const DiagnosticInfoOptimizationBase &Diag) const;

public:
  LLVMRemarkStreamer(remarks::RemarkStreamer &RS) : RS(RS) {}
  /// Emit a diagnostic through the streamer.
  LLVM_ABI void emit(const DiagnosticInfoOptimizationBase &Diag);
};

template <typename ThisError>
struct LLVMRemarkSetupErrorInfo : public ErrorInfo<ThisError> {
  std::string Msg;
  std::error_code EC;

  LLVMRemarkSetupErrorInfo(Error E) {
    handleAllErrors(std::move(E), [&](const ErrorInfoBase &EIB) {
      Msg = EIB.message();
      EC = EIB.convertToErrorCode();
    });
  }

  void log(raw_ostream &OS) const override { OS << Msg; }
  std::error_code convertToErrorCode() const override { return EC; }
};

struct LLVMRemarkSetupFileError
    : LLVMRemarkSetupErrorInfo<LLVMRemarkSetupFileError> {
  LLVM_ABI static char ID;
  using LLVMRemarkSetupErrorInfo<
      LLVMRemarkSetupFileError>::LLVMRemarkSetupErrorInfo;
};

struct LLVMRemarkSetupPatternError
    : LLVMRemarkSetupErrorInfo<LLVMRemarkSetupPatternError> {
  LLVM_ABI static char ID;
  using LLVMRemarkSetupErrorInfo<
      LLVMRemarkSetupPatternError>::LLVMRemarkSetupErrorInfo;
};

struct LLVMRemarkSetupFormatError
    : LLVMRemarkSetupErrorInfo<LLVMRemarkSetupFormatError> {
  LLVM_ABI static char ID;
  using LLVMRemarkSetupErrorInfo<
      LLVMRemarkSetupFormatError>::LLVMRemarkSetupErrorInfo;
};

/// RAII handle that manages the lifetime of the ToolOutputFile used to output
/// remarks. On destruction (or when calling releaseFile()), this handle ensures
/// that the optimization remarks are finalized and the RemarkStreamer is
/// correctly deregistered from the LLVMContext.
class LLVMRemarkFileHandle final {
  struct Finalizer {
    LLVMContext *Context;

    Finalizer(LLVMContext *Ctx) : Context(Ctx) {}

    Finalizer(const Finalizer &) = delete;
    Finalizer &operator=(const Finalizer &) = delete;

    Finalizer(Finalizer &&Other) : Context(Other.Context) {
      Other.Context = nullptr;
    }

    Finalizer &operator=(Finalizer &&Other) {
      std::swap(Context, Other.Context);
      return *this;
    }

    ~Finalizer() { finalize(); }

    LLVM_ABI void finalize();
  };

  std::unique_ptr<ToolOutputFile> OutputFile;
  Finalizer Finalize;

public:
  LLVMRemarkFileHandle() : OutputFile(nullptr), Finalize(nullptr) {}

  LLVMRemarkFileHandle(std::unique_ptr<ToolOutputFile> OutputFile,
                       LLVMContext &Ctx)
      : OutputFile(std::move(OutputFile)), Finalize(&Ctx) {}

  ToolOutputFile *get() { return OutputFile.get(); }
  explicit operator bool() { return bool(OutputFile); }

  /// Finalize remark emission and release the underlying ToolOutputFile.
  std::unique_ptr<ToolOutputFile> releaseFile() {
    finalize();
    return std::move(OutputFile);
  }

  void finalize() { Finalize.finalize(); }

  ToolOutputFile &operator*() { return *OutputFile; }
  ToolOutputFile *operator->() { return &*OutputFile; }
};

/// Set up optimization remarks that output to a file. The LLVMRemarkFileHandle
/// manages the lifetime of the underlying ToolOutputFile to ensure \ref
/// finalizeLLVMOptimizationRemarks() is called before the file is destroyed or
/// released from the handle. The handle must be kept alive until all remarks
/// were emitted through the remark streamer.
LLVM_ABI Expected<LLVMRemarkFileHandle> setupLLVMOptimizationRemarks(
    LLVMContext &Context, StringRef RemarksFilename, StringRef RemarksPasses,
    StringRef RemarksFormat, bool RemarksWithHotness,
    std::optional<uint64_t> RemarksHotnessThreshold = 0);

/// Set up optimization remarks that output directly to a raw_ostream.
/// \p OS is managed by the caller and must be open for writing until
/// \ref finalizeLLVMOptimizationRemarks() is called.
LLVM_ABI Error setupLLVMOptimizationRemarks(
    LLVMContext &Context, raw_ostream &OS, StringRef RemarksPasses,
    StringRef RemarksFormat, bool RemarksWithHotness,
    std::optional<uint64_t> RemarksHotnessThreshold = 0);

/// Finalize optimization remarks and deregister the RemarkStreamer from the \p
/// Context. This must be called before closing the (file) stream that was used
/// to set up the remarks.
LLVM_ABI void finalizeLLVMOptimizationRemarks(LLVMContext &Context);

} // end namespace llvm

#endif // LLVM_IR_LLVMREMARKSTREAMER_H
