//===-- SurgicalInstrumentationConfig.h -- Surgical CSI ------*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is part of CSI, a framework that provides comprehensive static
// instrumentation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_SURGICALINSTRUMENTATIONCONFIG_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_SURGICALINSTRUMENTATIONCONFIG_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
enum InstrumentationConfigMode { WHITELIST = 0, BLACKLIST = 1 };

enum InstrumentationPoint : int {
  INSTR_INVALID_POINT = 0x0,
  INSTR_FUNCTION_ENTRY = 0x1,
  INSTR_FUNCTION_EXIT = 0x1 << 1,
  INSTR_BEFORE_CALL = 0x1 << 2,
  INSTR_AFTER_CALL = 0x1 << 3,
  INSTR_TAPIR_DETACH = 0x1 << 4,
  INSTR_TAPIR_SYNC = 0x1 << 5,
};

#define INSTR_ALL_POINTS InstrumentationPoint::INSTR_INVALID_POINT

inline InstrumentationPoint operator|(const InstrumentationPoint &a,
                                      const InstrumentationPoint &b) {
  return static_cast<InstrumentationPoint>(static_cast<int>(a) |
                                           static_cast<int>(b));
}

inline InstrumentationPoint operator&(const InstrumentationPoint &a,
                                      const InstrumentationPoint &b) {
  return static_cast<InstrumentationPoint>(static_cast<int>(a) &
                                           static_cast<int>(b));
}

inline bool operator==(InstrumentationPoint a, InstrumentationPoint b) {
  return static_cast<int>(a) == static_cast<int>(b);
}

inline InstrumentationPoint &operator|=(InstrumentationPoint &a,
                                        InstrumentationPoint b) {
  return a = a | b;
}

static StringMap<InstrumentationPoint> SurgicalInstrumentationPoints = {
    {"FunctionEntry", INSTR_FUNCTION_ENTRY},
    {
        "FunctionExit",
        INSTR_FUNCTION_EXIT,
    },
    {
        "BeforeCall",
        INSTR_BEFORE_CALL,
    },
    {
        "AfterCall",
        INSTR_AFTER_CALL,
    },
    {
        "TapirDetach",
        INSTR_TAPIR_DETACH,
    },
    {
        "TapirSync",
        INSTR_TAPIR_SYNC,
    },
};

InstrumentationPoint
ParseInstrumentationPoint(const StringRef &instrPointString);

class InstrumentationConfig {
public:
  virtual ~InstrumentationConfig() {}

  void SetConfigMode(InstrumentationConfigMode mode) { this->mode = mode; }

  static std::unique_ptr<InstrumentationConfig> GetDefault();

  static std::unique_ptr<InstrumentationConfig>
  ReadFromConfigurationFile(const std::string &filename);

  virtual bool DoesFunctionRequireInterposition(const StringRef &functionName) {
    return interposedFunctions.find(functionName) != interposedFunctions.end();
  }

  virtual bool DoesAnyFunctionRequireInterposition() {
    return interposedFunctions.size() > 0;
  }

  virtual bool DoesFunctionRequireInstrumentationForPoint(
      const StringRef &functionName, const InstrumentationPoint &point) {
    if (targetFunctions.size() == 0)
      return true;

    bool found = targetFunctions.find(functionName) != targetFunctions.end();

    if (found) // The function is in the configuration. Does it specify this
               // instrumentation point?
    {
      InstrumentationPoint &functionPoints = targetFunctions[functionName];

      if (functionPoints != INSTR_ALL_POINTS) {
        if ((targetFunctions[functionName] & point) != point)
          found = false;
      }
    }

    return mode == InstrumentationConfigMode::WHITELIST ? found : !found;
  }

protected:
  InstrumentationConfig(){};
  InstrumentationConfig(const StringMap<InstrumentationPoint> &targetFunctions,
                        const StringSet<> &interposedFunctions)
      : targetFunctions(targetFunctions),
        interposedFunctions(interposedFunctions) {}

  StringMap<InstrumentationPoint> targetFunctions;

  StringSet<> interposedFunctions;

  InstrumentationConfigMode mode = InstrumentationConfigMode::WHITELIST;
};

class DefaultInstrumentationConfig : public InstrumentationConfig {
public:
  virtual bool DoesFunctionRequireInstrumentationForPoint(
      const StringRef &functionName, const InstrumentationPoint &point) {
    return true;
  }

  virtual bool DoesAnyFunctionRequireInterposition() { return false; }

  virtual bool DoesFunctionRequireInterposition(const StringRef &functionName) {
    return false;
  }
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_SURGICALINSTRUMENTATIONCONFIG_H
