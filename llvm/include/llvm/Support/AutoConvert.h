/*===- AutoConvert.h - Auto conversion between ASCII/EBCDIC -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions used for auto conversion between
// ASCII/EBCDIC codepages specific to z/OS.
//
//===----------------------------------------------------------------------===*/

#ifndef LLVM_SUPPORT_AUTOCONVERT_H
#define LLVM_SUPPORT_AUTOCONVERT_H

#ifdef __MVS__
#include <_Ccsid.h>
#endif
#ifdef __cplusplus
#include "llvm/Support/Error.h"
#include <system_error>
#endif /* __cplusplus */

#define CCSID_IBM_1047 1047
#define CCSID_UTF_8 1208
#define CCSID_ISO8859_1 819

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int enablezOSAutoConversion(int FD);
int disablezOSAutoConversion(int FD);
int restorezOSStdHandleAutoConversion(int FD);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#ifdef __cplusplus
namespace llvm {

#ifdef __MVS__

/** \brief Set the tag information for a file descriptor. */
std::error_code setzOSFileTag(int FD, int CCSID, bool Text);

/** \brief Get the the tag ccsid for a file name or a file descriptor. */
ErrorOr<__ccsid_t> getzOSFileTag(const char *FileName, const int FD = -1);

/** \brief Query the file tag to determine if it needs conversion to UTF-8
 *  codepage.
 */
ErrorOr<bool> needzOSConversion(const char *FileName, const int FD = -1);

#endif /* __MVS__*/

inline std::error_code disableAutoConversion(int FD) {
#ifdef __MVS__
  if (::disablezOSAutoConversion(FD) == -1)
    return errnoAsErrorCode();
#endif
  return std::error_code();
}

inline std::error_code enableAutoConversion(int FD) {
#ifdef __MVS__
  if (::enablezOSAutoConversion(FD) == -1)
    return errnoAsErrorCode();
#endif
  return std::error_code();
}

inline std::error_code restoreStdHandleAutoConversion(int FD) {
#ifdef __MVS__
  if (::restorezOSStdHandleAutoConversion(FD) == -1)
    return errnoAsErrorCode();
#endif
  return std::error_code();
}

inline std::error_code setFileTag(int FD, int CCSID, bool Text) {
#ifdef __MVS__
  return setzOSFileTag(FD, CCSID, Text);
#endif
  return std::error_code();
}

inline ErrorOr<bool> needConversion(const char *FileName, const int FD = -1) {
#ifdef __MVS__
  return needzOSConversion(FileName, FD);
#endif
  return false;
}

} /* namespace llvm */
#endif /* __cplusplus */

#endif /* LLVM_SUPPORT_AUTOCONVERT_H */
