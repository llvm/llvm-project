//===--- LTOUtils.h - LLVM Advisor ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm::advisor {

/// Resolve the path to a bitcode or combined index file for this unit.
/// Prefers the explicitly-set IR path (.bc), then derives one from the
/// object path.
inline std::string resolveLTOInputPath(const CapabilityContext &Ctx) {
  if (!Ctx.IRPath.empty() && sys::fs::exists(Ctx.IRPath))
    return Ctx.IRPath;
  if (!Ctx.ObjectPath.empty()) {
    SmallString<256> BC(Ctx.ObjectPath);
    sys::path::replace_extension(BC, "bc");
    if (sys::fs::exists(BC))
      return BC.str().str();
  }
  return {};
}

} // namespace llvm::advisor
