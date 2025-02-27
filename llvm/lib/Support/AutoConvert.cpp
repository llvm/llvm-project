//===- AutoConvert.cpp - Auto conversion between ASCII/EBCDIC -------------===//
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
//===----------------------------------------------------------------------===//

#ifdef __MVS__

#include "llvm/Support/AutoConvert.h"
#include "llvm/Support/Error.h"
#include <cassert>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace llvm;

static int savedStdHandleAutoConversionMode[3] = {-1, -1, -1};

int disablezOSAutoConversion(int FD) {
  static const struct f_cnvrt Convert = {
      SETCVTOFF, // cvtcmd
      0,         // pccsid
      0,         // fccsid
  };

  return fcntl(FD, F_CONTROL_CVT, &Convert);
}

int restorezOSStdHandleAutoConversion(int FD) {
  assert(FD == STDIN_FILENO || FD == STDOUT_FILENO || FD == STDERR_FILENO);
  if (savedStdHandleAutoConversionMode[FD] == -1)
    return 0;
  struct f_cnvrt Cvt = {
      savedStdHandleAutoConversionMode[FD], // cvtcmd
      0,                                    // pccsid
      0,                                    // fccsid
  };
  return (fcntl(FD, F_CONTROL_CVT, &Cvt));
}

int enablezOSAutoConversion(int FD) {
  struct f_cnvrt Query = {
      QUERYCVT, // cvtcmd
      0,        // pccsid
      0,        // fccsid
  };

  if (fcntl(FD, F_CONTROL_CVT, &Query) == -1)
    return -1;

  // We don't need conversion for UTF-8 tagged files.
  // TODO: Remove the assumption of ISO8859-1 = UTF-8 here when we fully resolve
  // problems related to UTF-8 tagged source files.
  // When the pccsid is not ISO8859-1, autoconversion is still needed.
  if (Query.pccsid == CCSID_ISO8859_1 &&
      (Query.fccsid == CCSID_UTF_8 || Query.fccsid == CCSID_ISO8859_1))
    return 0;

  // Save the state of std handles before we make changes to it.
  if ((FD == STDIN_FILENO || FD == STDOUT_FILENO || FD == STDERR_FILENO) &&
      savedStdHandleAutoConversionMode[FD] == -1)
    savedStdHandleAutoConversionMode[FD] = Query.cvtcmd;

  if (FD == STDOUT_FILENO || FD == STDERR_FILENO)
    Query.cvtcmd = SETCVTON;
  else
    Query.cvtcmd = SETCVTALL;

  Query.pccsid =
      (FD == STDIN_FILENO || FD == STDOUT_FILENO || FD == STDERR_FILENO)
          ? 0
          : CCSID_UTF_8;
  // Assume untagged files to be IBM-1047 encoded.
  Query.fccsid = (Query.fccsid == FT_UNTAGGED) ? CCSID_IBM_1047 : Query.fccsid;
  return fcntl(FD, F_CONTROL_CVT, &Query);
}

std::error_code llvm::disablezOSAutoConversion(int FD) {
  if (::disablezOSAutoConversion(FD) == -1)
    return errnoAsErrorCode();

  return std::error_code();
}

std::error_code llvm::enablezOSAutoConversion(int FD) {
  if (::enablezOSAutoConversion(FD) == -1)
    return errnoAsErrorCode();

  return std::error_code();
}

std::error_code llvm::restorezOSStdHandleAutoConversion(int FD) {
  if (::restorezOSStdHandleAutoConversion(FD) == -1)
    return errnoAsErrorCode();

  return std::error_code();
}

std::error_code llvm::setzOSFileTag(int FD, int CCSID, bool Text) {
  assert((!Text || (CCSID != FT_UNTAGGED && CCSID != FT_BINARY)) &&
         "FT_UNTAGGED and FT_BINARY are not allowed for text files");
  struct file_tag Tag;
  Tag.ft_ccsid = CCSID;
  Tag.ft_txtflag = Text;
  Tag.ft_deferred = 0;
  Tag.ft_rsvflags = 0;

  if (fcntl(FD, F_SETTAG, &Tag) == -1)
    return errnoAsErrorCode();
  return std::error_code();
}

ErrorOr<__ccsid_t> llvm::getzOSFileTag(const char *FileName, const int FD) {
  // If we have a file descriptor, use it to find out file tagging. Otherwise we
  // need to use stat() with the file path.
  if (FD != -1) {
    struct f_cnvrt Query = {
        QUERYCVT, // cvtcmd
        0,        // pccsid
        0,        // fccsid
    };
    if (fcntl(FD, F_CONTROL_CVT, &Query) == -1)
      return std::error_code(errno, std::generic_category());
    return Query.fccsid;
  }
  struct stat Attr;
  if (stat(FileName, &Attr) == -1)
    return std::error_code(errno, std::generic_category());
  return Attr.st_tag.ft_ccsid;
}

ErrorOr<bool> llvm::needzOSConversion(const char *FileName, const int FD) {
  ErrorOr<__ccsid_t> Ccsid = getzOSFileTag(FileName, FD);
  if (std::error_code EC = Ccsid.getError())
    return EC;
  // We don't need conversion for UTF-8 tagged files or binary files.
  // TODO: Remove the assumption of ISO8859-1 = UTF-8 here when we fully resolve
  // problems related to UTF-8 tagged source files.
  switch (*Ccsid) {
  case CCSID_UTF_8:
  case CCSID_ISO8859_1:
  case FT_BINARY:
    return false;
  default:
    return true;
  }
}

#endif //__MVS__
