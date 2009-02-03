//===- llvm/System/Host.h - Host machine characteristics --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Methods for querying the nature of the host machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_HOST_H
#define LLVM_SYSTEM_HOST_H

#include <string>

namespace llvm {
namespace sys {

  inline bool isLittleEndianHost() {
    union {
      int i;
      char c;
    };
    i = 1;
    return c;
  }

  inline bool isBigEndianHost() {
    return !isLittleEndianHost();
  }

  /// getOSName() - Return the name of the host operating system or "" if
  /// unknown.
  std::string getOSName();

  /// getOSVersion() - Return the operating system version as a string or
  /// "" if unknown.
  std::string getOSVersion();
}
}

#endif
