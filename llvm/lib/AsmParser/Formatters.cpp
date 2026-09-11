//===- Formatters.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement the facility of custom representations for constants.
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Formatters.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;

ParsedValue::~ParsedValue() { clear(); }

void ParsedValue::clear() {
  if (isInt())
    IntVal.~APSInt();
  ValKind = None;
}

void ParsedValue::setInt(const APSInt &V) {
  clear();
  new (&IntVal) APSInt(V);
  ValKind = Int;
}

static bool parseFPClass(StringRef Str, ParsedValue &Val,
                         std::string &ErrorMsg) {
  Val.clear();
  ErrorMsg.clear();
  FPClassTest Result = fcNone;

  SmallVector<StringRef, 4> Classes;
  Str.split(Classes, ' ');

  for (StringRef ClassSpec : Classes) {
    ClassSpec = ClassSpec.trim();

    if (ClassSpec.empty()) {
      ErrorMsg = "Empty class specification";
      return false;
    }

    FPClassTest ClassValue = StringSwitch<FPClassTest>(ClassSpec)
                                 .Case("nan", fcNan)
                                 .Case("snan", fcSNan)
                                 .Case("qnan", fcQNan)
                                 .Case("inf", fcInf)
                                 .Case("pinf", fcPosInf)
                                 .Case("ninf", fcNegInf)
                                 .Case("norm", fcNormal)
                                 .Case("pnorm", fcPosNormal)
                                 .Case("nnorm", fcNegNormal)
                                 .Case("sub", fcSubnormal)
                                 .Case("psub", fcPosSubnormal)
                                 .Case("nsub", fcNegSubnormal)
                                 .Case("zero", fcZero)
                                 .Case("pzero", fcPosZero)
                                 .Case("nzero", fcNegZero)
                                 .Case("finite", fcFinite)
                                 .Case("pfinite", fcPosFinite)
                                 .Case("nfinite", fcNegFinite)
                                 .Case("pos", fcPositive)
                                 .Case("neg", fcNegative)
                                 .Case("number", fcNumber)
                                 .Default(fcNone);

    if (ClassValue == fcNone) {
      ErrorMsg = "unknown floating-point class: " + ClassSpec.str();
      return false;
    }

    if (ClassValue & Result) {
      ErrorMsg = "class specifications must be disjoint: " + ClassSpec.str();
      return false;
    }

    Result |= ClassValue;
  }

  Val.setInt(APSInt(APInt(32, Result), true));
  return true;
}

bool llvm::parseFormattedValue(StringRef Formatter, StringRef Str,
                               ParsedValue &Val, std::string &ErrorMsg) {
  bool Success = false;
  Val.clear();
  ErrorMsg.clear();
  if (Formatter == "fc") {
    Success = parseFPClass(Str, Val, ErrorMsg);
  } else {
    ErrorMsg = "unknown formatter: " + Formatter.str();
  }
  return Success;
}
