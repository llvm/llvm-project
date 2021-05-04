//===- AutoConvert.h - Auto conversion between ASCII/EBCDIC -----*- C++ -*-===//
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
//===----------------------------------------------------------------------===//i

#ifndef LLVM_SUPPORT_AUTOCONVERT_H
#define LLVM_SUPPORT_AUTOCONVERT_H

#ifdef __MVS__
#define CCSID_IBM_1047 1047
#define CCSID_UTF_8 1208
#include <system_error>

namespace llvm {

/// \brief Disable the z/OS enhanced ASCII auto-conversion for the file
/// descriptor.
std::error_code disableAutoConversion(int FD);

/// \brief Query the z/OS enhanced ASCII auto-conversion status of a file
/// descriptor and force the conversion if the file is not tagged with a
/// codepage.
std::error_code enableAutoConversion(int FD);

/// \brief Set the tag information for a file descriptor.
std::error_code setFileTag(int FD, int CCSID, bool Text);

} // namespace llvm

#endif // __MVS__

#endif // LLVM_SUPPORT_AUTOCONVERT_H
