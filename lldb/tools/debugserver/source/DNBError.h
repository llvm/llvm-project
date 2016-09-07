//===-- DNBError.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/26/07.
//
//===----------------------------------------------------------------------===//

#ifndef __DNBError_h__
#define __DNBError_h__

#include <errno.h>
#include <mach/mach.h>
#include <stdio.h>
#include <string>

class DNBError {
public:
  typedef uint32_t ValueType;
  typedef enum {
    Generic = 0,
    MachKernel = 1,
    POSIX = 2
#ifdef WITH_SPRINGBOARD
    ,
    SpringBoard = 3
#endif
#ifdef WITH_BKS
    ,
    BackBoard = 4
#endif
#ifdef WITH_FBS
    ,
    FrontBoard = 5
#endif
  } FlavorType;

  explicit DNBError(ValueType err = 0, FlavorType flavor = Generic)
      : m_err(err), m_flavor(flavor) {}

  const char *AsString() const;
  void Clear() {
    m_err = 0;
    m_flavor = Generic;
    m_str.clear();
  }
  ValueType Error() const { return m_err; }
  FlavorType Flavor() const { return m_flavor; }

  ValueType operator=(kern_return_t err) {
    m_err = err;
    m_flavor = MachKernel;
    m_str.clear();
    return m_err;
  }

  void SetError(kern_return_t err) {
    m_err = err;
    m_flavor = MachKernel;
    m_str.clear();
  }

  void SetErrorToErrno() {
    m_err = errno;
    m_flavor = POSIX;
    m_str.clear();
  }

  void SetError(ValueType err, FlavorType flavor) {
    m_err = err;
    m_flavor = flavor;
    m_str.clear();
  }

  // Generic errors can set their own string values
  void SetErrorString(const char *err_str) {
    if (err_str && err_str[0])
      m_str = err_str;
    else
      m_str.clear();
  }
  bool Success() const { return m_err == 0; }
  bool Fail() const { return m_err != 0; }
  void LogThreadedIfError(const char *format, ...) const;
  void LogThreaded(const char *format, ...) const;

protected:
  ValueType m_err;
  FlavorType m_flavor;
  mutable std::string m_str;
};

#endif // #ifndef __DNBError_h__
