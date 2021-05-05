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
#include <fcntl.h>
#include <sys/stat.h>

std::error_code llvm::disableAutoConversion(int FD) {
  static const struct f_cnvrt Convert = {
      SETCVTOFF,        // cvtcmd
      0,                // pccsid
      (short)FT_BINARY, // fccsid
  };
  if (fcntl(FD, F_CONTROL_CVT, &Convert) == -1)
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

std::error_code llvm::enableAutoConversion(int FD) {
  struct f_cnvrt Query = {
      QUERYCVT, // cvtcmd
      0,        // pccsid
      0,        // fccsid
  };

  if (fcntl(FD, F_CONTROL_CVT, &Query) == -1)
    return std::error_code(errno, std::generic_category());

  Query.cvtcmd = SETCVTALL;
  Query.pccsid =
      (FD == STDIN_FILENO || FD == STDOUT_FILENO || FD == STDERR_FILENO)
          ? 0
          : CCSID_UTF_8;
  // Assume untagged files to be IBM-1047 encoded.
  Query.fccsid = (Query.fccsid == FT_UNTAGGED) ? CCSID_IBM_1047 : Query.fccsid;
  if (fcntl(FD, F_CONTROL_CVT, &Query) == -1)
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

std::error_code llvm::setFileTag(int FD, int CCSID, bool Text) {
  assert((!Text || (CCSID != FT_UNTAGGED && CCSID != FT_BINARY)) &&
         "FT_UNTAGGED and FT_BINARY are not allowed for text files");
  struct file_tag Tag;
  Tag.ft_ccsid = CCSID;
  Tag.ft_txtflag = Text;
  Tag.ft_deferred = 0;
  Tag.ft_rsvflags = 0;

  if (fcntl(FD, F_SETTAG, &Tag) == -1)
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

#endif // __MVS__
