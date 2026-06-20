//===------------------- XRayDiffIngestor.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of XRayDiffIngestor in Runtime
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {
class XRayDiffIngestor {
public:
  Expected<json::Value> load(StringRef Before, StringRef After);
};
} // namespace llvm::advisor
