//===- SubsystemRAII.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_TESTINGSUPPORT_SUBSYSTEMRAII_H
#define LLDB_UNITTESTS_TESTINGSUPPORT_SUBSYSTEMRAII_H

#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <type_traits>

namespace lldb_private {

namespace detail {
/// Initializes and deinitializes a single subsystem.
/// @see SubsystemRAII
template <typename T> struct SubsystemRAIICase {

  /// Calls ::Initialize if it has a void return type.
  template <typename U = T>
  typename std::enable_if<
      std::is_same<decltype(U::Initialize()), void>::value>::type
  CallInitialize() {
    T::Initialize();
  }

  /// Calls ::Initialize if it has a llvm::Error return type and checks
  /// the Error instance for success.
  template <typename U = T>
  typename std::enable_if<
      std::is_same<decltype(U::Initialize()), llvm::Error>::value>::type
  CallInitialize() {
    ASSERT_THAT_ERROR(T::Initialize(), llvm::Succeeded());
  }

  SubsystemRAIICase() { CallInitialize(); }
  ~SubsystemRAIICase() { T::Terminate(); }
};
} // namespace detail

template <typename... T> class SubsystemRAII {};

/// RAII for initializing and deinitializing LLDB subsystems.
///
/// This RAII takes care of calling the Initialize and Terminate functions for
/// the subsystems specified by its template arguments. The ::Initialize
/// functions are called on construction for each subsystem template parameter
/// in the order in which they are passed as template parameters.
/// The ::Terminate functions are called in the reverse order at destruction
/// time.
///
/// If the ::Initialize function returns an llvm::Error this function handles
/// the Error instance (by checking that there is no error).
///
/// Constructing this RAII in a scope like this:
///
///   @code{.cpp}
///   {
///     SubsystemRAII<FileSystem, HostInfo, Socket> Subsystems;
///     DoingTestWork();
///   }
///   @endcode
///
/// is equivalent to the following code:
///
///   @code{.cpp}
///   {
///     FileSystem::Initialize();
///     HostInfo::Initialize();
///     ASSERT_THAT_ERROR(Socket::Initialize(), llvm::Succeeded());
///
///     DoingTestWork();
///
///     Socket::Terminate();
///     FileSystem::Terminate();
///     HostInfo::Terminate();
///   }
///   @endcode
template <typename T, typename... Ts> class SubsystemRAII<T, Ts...> {
  detail::SubsystemRAIICase<T> CurrentSubsystem;
  SubsystemRAII<Ts...> RemainingSubsystems;
};
} // namespace lldb_private

#endif // LLDB_UNITTESTS_TESTINGSUPPORT_SUBSYSTEMRAII_H
