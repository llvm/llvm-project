//===-- DarwinLogEvent.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DarwinLogEvent_h
#define DarwinLogEvent_h

#include <memory>
#include <vector>

#include "JSONGenerator.h"

// =============================================================================
/// Each discrete unit of information is described as an event, such as
/// the emission of a single log message.
// =============================================================================

using DarwinLogEvent = JSONGenerator::Dictionary;
using DarwinLogEventSP = std::shared_ptr<DarwinLogEvent>;
using DarwinLogEventVector = std::vector<DarwinLogEventSP>;

#endif
