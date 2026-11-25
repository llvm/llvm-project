//===- DeviceOptions.h - Per-device plugin options --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGIN_COMMON_DEVICEOPTIONS_H
#define OFFLOAD_PLUGIN_COMMON_DEVICEOPTIONS_H

#include "Shared/EnvironmentVar.h"

#include "llvm/ADT/StringRef.h"

namespace llvm::offload::plugin {

/// A runtime option of a device. It holds the device default until the user
/// overrides it through the environment variable \p EnvName.
template <typename T> struct DeviceOptionTy {
  DeviceOptionTy(const char *EnvName, T Value)
      : EnvName(EnvName), Value(Value) {}

  operator T() const { return Value; }

  DeviceOptionTy &operator=(T NewValue) {
    Value = NewValue;
    return *this;
  }

  void readFromEnv() {
    Envar<T> Env(EnvName);
    if (!Env.isPresent())
      return;
    Value = Env.get();
  }

  const char *EnvName;
  T Value;
};

} // namespace llvm::offload::plugin

#endif // OFFLOAD_PLUGIN_COMMON_DEVICEOPTIONS_H
