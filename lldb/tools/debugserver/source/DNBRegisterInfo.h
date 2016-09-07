//===-- DNBRegisterInfo.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 8/3/07.
//
//===----------------------------------------------------------------------===//

#ifndef __DNBRegisterInfo_h__
#define __DNBRegisterInfo_h__

#include "DNBDefs.h"
#include <stdint.h>
#include <stdio.h>

struct DNBRegisterValueClass : public DNBRegisterValue {
#ifdef __cplusplus
  DNBRegisterValueClass(const DNBRegisterInfo *regInfo = NULL);
  void Clear();
  void Dump(const char *pre, const char *post) const;
  bool IsValid() const;
#endif
};

#endif
