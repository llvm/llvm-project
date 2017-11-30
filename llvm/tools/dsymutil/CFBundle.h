//===- tools/dsymutil/CFBundle.h - CFBundle helper --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm {
namespace dsymutil {

struct CFBundleInfo {
  std::string VersionStr = "1";
  std::string ShortVersionStr = "1.0";
  std::string IDStr;
  bool OmitShortVersion() const { return ShortVersionStr.empty(); }
};

CFBundleInfo getBundleInfo(llvm::StringRef ExePath);

} // end namespace dsymutil
} // end namespace llvm
