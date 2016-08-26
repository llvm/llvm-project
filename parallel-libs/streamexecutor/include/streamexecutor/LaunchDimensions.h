//===-- LaunchDimensions.h - Kernel block and grid sizes --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Structures to hold sizes for blocks and grids which are used as parameters
/// for kernel launches.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_LAUNCHDIMENSIONS_H
#define STREAMEXECUTOR_LAUNCHDIMENSIONS_H

namespace streamexecutor {

/// The dimensions of a device block of execution.
///
/// A block is made up of an array of X by Y by Z threads.
struct BlockDimensions {
  BlockDimensions(unsigned X = 1, unsigned Y = 1, unsigned Z = 1)
      : X(X), Y(Y), Z(Z) {}

  unsigned X;
  unsigned Y;
  unsigned Z;
};

/// The dimensions of a device grid of execution.
///
/// A grid is made up of an array of X by Y by Z blocks.
struct GridDimensions {
  GridDimensions(unsigned X = 1, unsigned Y = 1, unsigned Z = 1)
      : X(X), Y(Y), Z(Z) {}

  unsigned X;
  unsigned Y;
  unsigned Z;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_LAUNCHDIMENSIONS_H
