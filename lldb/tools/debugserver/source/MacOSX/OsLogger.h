//===-- OsLogger.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef OsLogger_h
#define OsLogger_h

#include "DNBDefs.h"

class OsLogger {
public:
  static DNBCallbackLog GetLogFunction();
};

#endif /* OsLogger_h */
