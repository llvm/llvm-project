//===-- DarwinLogEvent.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
