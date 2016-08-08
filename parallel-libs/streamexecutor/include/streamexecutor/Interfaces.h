//===-- Interfaces.h - Interfaces to platform-specific impls ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Interfaces to platform-specific StreamExecutor type implementations.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_INTERFACES_H
#define STREAMEXECUTOR_INTERFACES_H

namespace streamexecutor {

/// Methods supported by device kernel function objects on all platforms.
class KernelInterface {
  // TODO(jhen): Add methods.
};

// TODO(jhen): Add other interfaces such as Stream.

} // namespace streamexecutor

#endif // STREAMEXECUTOR_INTERFACES_H
