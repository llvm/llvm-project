//===-- Platform.h - liboffload platform wrapper ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper around a liboffload platform handle caching its properties.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_PLATFORM_H
#define OMPTARGET_PLATFORM_H

#include "OffloadAPI.h"

#include "llvm/ADT/StringRef.h"

#include <string>

namespace llvm::omp::target::plugin {
struct GenericPluginTy;
}

/// A liboffload platform together with its cached properties.
struct PlatformTy {
  /// Query the properties of \p Handle and create the platform. Properties
  /// that cannot be queried are set to "Unknown".
  static PlatformTy create(ol_platform_handle_t Handle);

  ol_platform_handle_t getHandle() const { return Handle; }

  /// The platform name, without any trailing null terminator.
  llvm::StringRef getName() const { return Name; }

private:
  PlatformTy(ol_platform_handle_t Handle, std::string Name)
      : Handle(Handle), Name(std::move(Name)) {}

  ol_platform_handle_t Handle;
  std::string Name;
};

#endif // OMPTARGET_PLATFORM_H
