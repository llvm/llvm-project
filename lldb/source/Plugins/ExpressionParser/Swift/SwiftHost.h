//===-- SwiftHost.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_EXPRESSIONPARSER_SWIFT_SWIFTHOST_H
#define LLDB_PLUGINS_EXPRESSIONPARSER_SWIFT_SWIFTHOST_H

namespace lldb_private {

class FileSpec;

bool ComputeSwiftResourceDirectory(FileSpec &lldb_shlib_spec,
                                   FileSpec &file_spec, bool verify);

FileSpec GetSwiftResourceDir();

} // namespace lldb_private

#endif
