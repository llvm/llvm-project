//===-- OsLogger.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
