//===-- Executor.cpp - Executor implementation ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of Executor class internals.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/Executor.h"

#include <cassert>

#include "streamexecutor/PlatformInterfaces.h"
#include "streamexecutor/Stream.h"

#include "llvm/ADT/STLExtras.h"

namespace streamexecutor {

Executor::Executor(PlatformExecutor *PExecutor) : PExecutor(PExecutor) {}

Executor::~Executor() = default;

Expected<std::unique_ptr<Stream>> Executor::createStream() {
  Expected<std::unique_ptr<PlatformStreamHandle>> MaybePlatformStream =
      PExecutor->createStream();
  if (!MaybePlatformStream) {
    return MaybePlatformStream.takeError();
  }
  assert((*MaybePlatformStream)->getExecutor() == PExecutor &&
         "an executor created a stream with a different stored executor");
  return llvm::make_unique<Stream>(std::move(*MaybePlatformStream));
}

} // namespace streamexecutor
