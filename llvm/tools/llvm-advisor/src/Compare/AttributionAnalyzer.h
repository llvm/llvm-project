//===------------------- AttributionAnalyzer.h - LLVM Advisor
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of AttributionAnalyzer in Compare
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {
class AttributionAnalyzer {
public:
  json::Value empty() const {
    return json::Object{{"attributions", json::Array{}}};
  }
};
} // namespace llvm::advisor
