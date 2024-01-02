//===- Utils.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements utility functions for TextAPI Darwin operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/TextAPI/Utils.h"
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#endif

using namespace llvm;
using namespace llvm::MachO;

void llvm::MachO::replace_extension(SmallVectorImpl<char> &Path,
                                    const Twine &Extension) {
  StringRef P(Path.begin(), Path.size());
  auto ParentPath = sys::path::parent_path(P);
  auto Filename = sys::path::filename(P);

  if (!ParentPath.ends_with(Filename.str() + ".framework")) {
    sys::path::replace_extension(Path, Extension);
    return;
  }
  // Framework dylibs do not have a file extension, in those cases the new
  // extension is appended. e.g. given Path: "Foo.framework/Foo" and Extension:
  // "tbd", the result is "Foo.framework/Foo.tbd".
  SmallString<8> Storage;
  StringRef Ext = Extension.toStringRef(Storage);

  // Append '.' if needed.
  if (!Ext.empty() && Ext[0] != '.')
    Path.push_back('.');

  // Append extension.
  Path.append(Ext.begin(), Ext.end());
}

std::error_code llvm::MachO::shouldSkipSymLink(const Twine &Path,
                                               bool &Result) {
  Result = false;
  SmallString<PATH_MAX> Storage;
  auto P = Path.toNullTerminatedStringRef(Storage);
  sys::fs::file_status stat1;
  auto ec = sys::fs::status(P.data(), stat1);
  if (ec == std::errc::too_many_symbolic_link_levels) {
    Result = true;
    return {};
  }

  if (ec)
    return ec;

  StringRef Parent = sys::path::parent_path(P);
  while (!Parent.empty()) {
    sys::fs::file_status stat2;
    if (auto ec = sys::fs::status(Parent, stat2))
      return ec;

    if (sys::fs::equivalent(stat1, stat2)) {
      Result = true;
      return {};
    }

    Parent = sys::path::parent_path(Parent);
  }
  return {};
}

std::error_code
llvm::MachO::make_relative(StringRef From, StringRef To,
                           SmallVectorImpl<char> &RelativePath) {
  SmallString<PATH_MAX> Src = From;
  SmallString<PATH_MAX> Dst = To;
  if (auto ec = sys::fs::make_absolute(Src))
    return ec;

  if (auto ec = sys::fs::make_absolute(Dst))
    return ec;

  SmallString<PATH_MAX> Result;
  Src = sys::path::parent_path(From);
  auto it1 = sys::path::begin(Src), it2 = sys::path::begin(Dst),
       ie1 = sys::path::end(Src), ie2 = sys::path::end(Dst);
  // Ignore the common part.
  for (; it1 != ie1 && it2 != ie2; ++it1, ++it2) {
    if (*it1 != *it2)
      break;
  }

  for (; it1 != ie1; ++it1)
    sys::path::append(Result, "../");

  for (; it2 != ie2; ++it2)
    sys::path::append(Result, *it2);

  if (Result.empty())
    Result = ".";

  RelativePath.swap(Result);

  return {};
}

bool llvm::MachO::isPrivateLibrary(StringRef Path, bool IsSymLink) {
  // Remove the iOSSupport and DriverKit prefix to identify public locations.
  Path.consume_front(MACCATALYST_PREFIX_PATH);
  Path.consume_front(DRIVERKIT_PREFIX_PATH);
  // Also /Library/Apple prefix for ROSP.
  Path.consume_front("/Library/Apple");

  if (Path.starts_with("/usr/local/lib"))
    return true;

  if (Path.starts_with("/System/Library/PrivateFrameworks"))
    return true;

  // Everything in /usr/lib/swift (including sub-directories) are considered
  // public.
  if (Path.consume_front("/usr/lib/swift/"))
    return false;

  // Only libraries directly in /usr/lib are public. All other libraries in
  // sub-directories are private.
  if (Path.consume_front("/usr/lib/"))
    return Path.contains('/');

  // "/System/Library/Frameworks/" is a public location.
  if (Path.starts_with("/System/Library/Frameworks/")) {
    StringRef Name, Rest;
    std::tie(Name, Rest) =
        Path.drop_front(sizeof("/System/Library/Frameworks")).split('.');

    // Allow symlinks to top-level frameworks.
    if (IsSymLink && Rest == "framework")
      return false;

    // Only top level framework are public.
    // /System/Library/Frameworks/Foo.framework/Foo ==> true
    // /System/Library/Frameworks/Foo.framework/Versions/A/Foo ==> true
    // /System/Library/Frameworks/Foo.framework/Resources/libBar.dylib ==> false
    // /System/Library/Frameworks/Foo.framework/Frameworks/Bar.framework/Bar
    // ==> false
    // /System/Library/Frameworks/Foo.framework/Frameworks/Xfoo.framework/XFoo
    // ==> false
    return !(Rest.starts_with("framework/") &&
             (Rest.ends_with(Name) || Rest.ends_with((Name + ".tbd").str()) ||
              (IsSymLink && Rest.ends_with("Current"))));
  }
  return false;
}
