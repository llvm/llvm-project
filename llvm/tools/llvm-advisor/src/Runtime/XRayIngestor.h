//===------------------- XRayIngestor.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of XRayIngestor in Runtime
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {
class XRayIngestor {
public:
  Expected<json::Value> load(StringRef Path);
  Error ingest(StringRef Path);
};
} // namespace llvm::advisor
