//===-- CFCMutableArray.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CoreFoundationCPP_CFMutableArray_h_
#define CoreFoundationCPP_CFMutableArray_h_

#include "CFCReleaser.h"

class CFCMutableArray : public CFCReleaser<CFMutableArrayRef> {
public:
  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  CFCMutableArray(CFMutableArrayRef array = NULL);
  CFCMutableArray(const CFCMutableArray &rhs); // This will copy the array
                                               // contents into a new array
  CFCMutableArray &operator=(const CFCMutableArray &rhs); // This will re-use
                                                          // the same array and
                                                          // just bump the ref
                                                          // count
  virtual ~CFCMutableArray();

  CFIndex GetCount() const;
  CFIndex GetCountOfValue(const void *value) const;
  CFIndex GetCountOfValue(CFRange range, const void *value) const;
  const void *GetValueAtIndex(CFIndex idx) const;
  bool SetValueAtIndex(CFIndex idx, const void *value);
  bool AppendValue(const void *value,
                   bool can_create = true); // Appends value and optionally
                                            // creates a CFCMutableArray if this
                                            // class doesn't contain one
  bool
  AppendCStringAsCFString(const char *cstr,
                          CFStringEncoding encoding = kCFStringEncodingUTF8,
                          bool can_create = true);
  bool AppendFileSystemRepresentationAsCFString(const char *s,
                                                bool can_create = true);
};

#endif // #ifndef CoreFoundationCPP_CFMutableArray_h_
