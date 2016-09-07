//===-- CFCData.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CoreFoundationCPP_CFData_h_
#define CoreFoundationCPP_CFData_h_

#include "CFCReleaser.h"

class CFCData : public CFCReleaser<CFDataRef> {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  CFCData(CFDataRef data = NULL);
  CFCData(const CFCData &rhs);
  CFCData &operator=(const CFCData &rhs);
  virtual ~CFCData();

  CFDataRef Serialize(CFPropertyListRef plist, CFPropertyListFormat format);
  const uint8_t *GetBytePtr() const;
  CFIndex GetLength() const;

protected:
  //------------------------------------------------------------------
  // Classes that inherit from CFCData can see and modify these
  //------------------------------------------------------------------
};

#endif // #ifndef CoreFoundationCPP_CFData_h_
