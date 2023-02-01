//===- InteractiveModelRunner.cpp - noop ML model runner   ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A runner that communicates with an external agent via 2 file descriptors.
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/InteractiveModelRunner.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define _IMR_CL_VALS(T, N) clEnumValN(TensorType::N, #T, #T),

static cl::opt<TensorType> DebugReply(
    "interactive-model-runner-echo-type", cl::init(TensorType::Invalid),
    cl::Hidden,
    cl::desc("The InteractiveModelRunner will echo back to stderr "
             "the data received "
             "from the host as the specified type (for debugging purposes)."),
    cl::values(SUPPORTED_TENSOR_TYPES(_IMR_CL_VALS)
                   clEnumValN(TensorType::Invalid, "disable", "Don't echo")));

#undef _IMR_CL_VALS

InteractiveModelRunner::InteractiveModelRunner(
    LLVMContext &Ctx, const std::vector<TensorSpec> &Inputs,
    const TensorSpec &Advice, StringRef OutboundName, StringRef InboundName)
    : MLModelRunner(Ctx, MLModelRunner::Kind::Interactive, Inputs.size()),
      InputSpecs(Inputs), OutputSpec(Advice), Inbound(InboundName, InEC),
      OutputBuffer(OutputSpec.getTotalTensorBufferSize()),
      Log(std::make_unique<raw_fd_ostream>(OutboundName, OutEC), InputSpecs,
          Advice, /*IncludeReward=*/false, Advice) {
  if (InEC) {
    Ctx.emitError("Cannot open inbound file: " + InEC.message());
    return;
  }
  if (OutEC) {
    Ctx.emitError("Cannot open outbound file: " + OutEC.message());
    return;
  }
  // Just like in the no inference case, this will allocate an appropriately
  // sized buffer.
  for (size_t I = 0; I < InputSpecs.size(); ++I)
    setUpBufferForTensor(I, InputSpecs[I], nullptr);
  Log.flush();
}

void *InteractiveModelRunner::evaluateUntyped() {
  Log.startObservation();
  for (size_t I = 0; I < InputSpecs.size(); ++I)
    Log.logTensorValue(I, reinterpret_cast<const char *>(getTensorUntyped(I)));
  Log.endObservation();
  Log.flush();

  size_t InsPoint = 0;
  char *Buff = OutputBuffer.data();
  const size_t Limit = OutputBuffer.size();
  while (InsPoint < Limit) {
    auto Read = Inbound.read(Buff + InsPoint, OutputBuffer.size() - InsPoint);
    if (Read < 0) {
      Ctx.emitError("Failed reading from inbound file");
      break;
    }
    InsPoint += Read;
  }
  if (DebugReply != TensorType::Invalid)
    dbgs() << tensorValueToString(OutputBuffer.data(), OutputSpec);
  return OutputBuffer.data();
}