//===-- CFCBundle.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CoreFoundationCPP_CFBundle_h_
#define CoreFoundationCPP_CFBundle_h_

#include "CFCReleaser.h"

class CFCBundle : public CFCReleaser<CFBundleRef> {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  CFCBundle(const char *path = NULL);
  CFCBundle(CFURLRef url);

  virtual ~CFCBundle();

  CFURLRef CopyExecutableURL() const;

  CFStringRef GetIdentifier() const;

  CFTypeRef GetValueForInfoDictionaryKey(CFStringRef key) const;

  bool GetPath(char *dst, size_t dst_len);

  bool SetPath(const char *path);

private:
  // Disallow copy and assignment constructors
  CFCBundle(const CFCBundle &);

  const CFCBundle &operator=(const CFCBundle &);
};

#endif // #ifndef CoreFoundationCPP_CFBundle_h_
