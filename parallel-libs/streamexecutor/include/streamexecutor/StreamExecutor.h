//===-- StreamExecutor.h - The StreamExecutor class -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The StreamExecutor class which represents a single device of a specific
/// platform.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_STREAMEXECUTOR_H
#define STREAMEXECUTOR_STREAMEXECUTOR_H

#include "streamexecutor/Utils/Error.h"

namespace streamexecutor {

class KernelInterface;

class StreamExecutor {
public:
  /// Gets the kernel implementation for the underlying platform.
  virtual Expected<std::unique_ptr<KernelInterface>>
  getKernelImplementation(const MultiKernelLoaderSpec &Spec) {
    // TODO(jhen): Implement this.
    return nullptr;
  }

  // TODO(jhen): Add other methods.
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_STREAMEXECUTOR_H
