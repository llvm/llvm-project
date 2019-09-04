//===-- SwiftHost.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
