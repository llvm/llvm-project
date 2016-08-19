//===-- Executor.h - The Executor class -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The Executor class which represents a single device of a specific platform.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_EXECUTOR_H
#define STREAMEXECUTOR_EXECUTOR_H

#include "streamexecutor/KernelSpec.h"
#include "streamexecutor/Utils/Error.h"

namespace streamexecutor {

class KernelInterface;
class PlatformExecutor;
class Stream;

class Executor {
public:
  explicit Executor(PlatformExecutor *PExecutor);
  virtual ~Executor();

  /// Gets the kernel implementation for the underlying platform.
  virtual Expected<std::unique_ptr<KernelInterface>>
  getKernelImplementation(const MultiKernelLoaderSpec &Spec) {
    // TODO(jhen): Implement this.
    return nullptr;
  }

  Expected<std::unique_ptr<Stream>> createStream();

private:
  PlatformExecutor *PExecutor;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_EXECUTOR_H
