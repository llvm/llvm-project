//===-- CFData.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 1/16/08.
//
//===----------------------------------------------------------------------===//

#ifndef __CFData_h__
#define __CFData_h__

#include "CFUtils.h"

class CFData : public CFReleaser<CFDataRef> {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  CFData(CFDataRef data = NULL);
  CFData(const CFData &rhs);
  CFData &operator=(const CFData &rhs);
  virtual ~CFData();

  CFDataRef Serialize(CFPropertyListRef plist, CFPropertyListFormat format);
  const uint8_t *GetBytePtr() const;
  CFIndex GetLength() const;

protected:
  //------------------------------------------------------------------
  // Classes that inherit from CFData can see and modify these
  //------------------------------------------------------------------
};

#endif // #ifndef __CFData_h__
