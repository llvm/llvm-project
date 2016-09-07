//===-- SysSignal.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/18/07.
//
//===----------------------------------------------------------------------===//

#ifndef __SysSignal_h__
#define __SysSignal_h__

class SysSignal {
public:
  static const char *Name(int signal);
};

#endif
