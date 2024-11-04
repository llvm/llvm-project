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
