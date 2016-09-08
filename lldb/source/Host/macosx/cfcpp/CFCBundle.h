//===-- CFCBundle.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
