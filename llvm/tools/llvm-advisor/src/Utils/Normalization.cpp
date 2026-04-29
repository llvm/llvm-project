//===------------------- Normalization.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command line normalization and path canonicalization.
// Normalizes paths and commands for consistent comparison.
//
//===----------------------------------------------------------------------===//
#include "Utils/Normalization.h"

using namespace llvm;
using namespace llvm::advisor;

std::string llvm::advisor::normalizePath(StringRef Path, StringRef Base) {
  SmallString<256> Result;
  if (!sys::path::is_absolute(Path) && !Base.empty())
    Result = Base;
  sys::path::append(Result, Path);
  sys::path::remove_dots(Result, true);
  sys::path::native(Result);
  return Result.str().str();
}

static bool isInside(StringRef Path, StringRef Root) {
  if (Path == Root)
    return true;
  if (!Path.starts_with(Root))
    return false;
  return sys::path::is_separator(Path.drop_front(Root.size()).front());
}

Expected<std::string>
llvm::advisor::canonicalizePath(StringRef Path,
                                ArrayRef<StringRef> AllowedRoots) {
  SmallString<256> RealPath;
  if (std::error_code EC = sys::fs::real_path(Path, RealPath))
    return createStringError(EC, "path does not exist: %s", Path.str().c_str());
  sys::path::remove_dots(RealPath, true);

  for (StringRef Root : AllowedRoots) {
    SmallString<256> RealRoot;
    if (std::error_code EC = sys::fs::real_path(Root, RealRoot))
      return createStringError(EC, "path root does not exist: %s",
                               Root.str().c_str());
    sys::path::remove_dots(RealRoot, true);
    if (isInside(RealPath, RealRoot))
      return RealPath.str().str();
  }

  return createStringError(inconvertibleErrorCode(),
                           "path escapes allowed roots: %s",
                           Path.str().c_str());
}

SmallVector<std::string, 16>
llvm::advisor::normalizeCommand(ArrayRef<std::string> Arguments) {
  SmallVector<std::string, 16> Out;
  for (const std::string &Arg : Arguments) {
    StringRef Ref(Arg);
    if (Ref == "-ftime-trace-granularity=0")
      continue;
    if (Ref.starts_with("-fdebug-compilation-dir="))
      continue;
    Out.push_back(Arg);
  }
  return Out;
}

std::string llvm::advisor::inferLanguage(StringRef Path) {
  StringRef Ext = sys::path::extension(Path);
  if (Ext == ".c")
    return "c";
  if (Ext == ".m")
    return "objective-c";
  if (Ext == ".mm")
    return "objective-c++";
  if (Ext == ".s" || Ext == ".S")
    return "assembly";
  return "c++";
}

std::string llvm::advisor::inferTargetTriple(ArrayRef<std::string> Arguments) {
  constexpr StringRef LongTargetPrefix = "--target=";
  constexpr StringRef ShortTargetPrefix = "-target=";

  for (size_t I = 0, E = Arguments.size(); I != E; ++I) {
    StringRef Arg(Arguments[I]);
    if (Arg == "-target" && I + 1 != E)
      return Arguments[I + 1];
    if (Arg.starts_with(LongTargetPrefix))
      return Arg.drop_front(LongTargetPrefix.size()).str();
    if (Arg.starts_with(ShortTargetPrefix))
      return Arg.drop_front(ShortTargetPrefix.size()).str();
  }
  return {};
}
