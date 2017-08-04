//===-- SwiftSILManipulator.h -----------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftSILManipulator_h
#define liblldb_SwiftSILManipulator_h

#include "SwiftExpressionParser.h"

#include "lldb/lldb-types.h"

#include "swift/SIL/SILLocation.h"
#include "swift/SIL/SILValue.h"

namespace swift {
class SILBuilder;
};

namespace lldb_private {

class CompilerType;
class Log;

class SwiftSILManipulator {
public:
  SwiftSILManipulator(swift::SILBuilder &builder);

  swift::SILValue
  emitLValueForVariable(swift::VarDecl *var,
                        SwiftExpressionParser::SILVariableInfo &info);

protected:
  swift::SILBuilder &m_builder;
  Log *m_log;
};
}

#endif
