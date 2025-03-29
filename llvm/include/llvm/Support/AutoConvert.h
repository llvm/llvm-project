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
#ifdef __cplusplus
#include "llvm/Support/ErrorOr.h"
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

/** \brief Disable the z/OS enhanced ASCII auto-conversion for the file
 * descriptor.
 */
std::error_code disablezOSAutoConversion(int FD);

/** \brief Query the z/OS enhanced ASCII auto-conversion status of a file
 * descriptor and force the conversion if the file is not tagged with a
 * codepage.
 */
std::error_code enablezOSAutoConversion(int FD);

/** Restore the z/OS enhanced ASCII auto-conversion for the std handle. */
std::error_code restorezOSStdHandleAutoConversion(int FD);

/** \brief Set the tag information for a file descriptor. */
std::error_code setzOSFileTag(int FD, int CCSID, bool Text);

// Get the the tag ccsid for a file name or a file descriptor.
ErrorOr<__ccsid_t> getzOSFileTag(const char *FileName, const int FD = -1);

// Query the file tag to determine if it needs conversion to UTF-8 codepage.
ErrorOr<bool> needzOSConversion(const char *FileName, const int FD = -1);

} // namespace llvm
#endif // __cplusplus

#endif /* __MVS__ */

#endif /* LLVM_SUPPORT_AUTOCONVERT_H */
