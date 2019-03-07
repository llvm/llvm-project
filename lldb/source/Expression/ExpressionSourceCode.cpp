//===-- ExpressionSourceCode.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ExpressionSourceCode.h"

#include <algorithm>

#include "llvm/ADT/StringRef.h"
#include "clang/Basic/CharInfo.h"

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"

using namespace lldb_private;

bool ExpressionSourceCode::SaveExpressionTextToTempFile(
    llvm::StringRef text, const EvaluateExpressionOptions &options,
    std::string &expr_source_path) {
  bool success = false;

  const uint32_t expr_number = options.GetExpressionNumber();

  const bool playground = options.GetPlaygroundTransformEnabled();
  const bool repl = options.GetREPLEnabled();

  llvm::StringRef file_prefix;
  if (playground)
    file_prefix = "playground";
  else if (repl)
    file_prefix = "repl";
  else
    file_prefix = "expr";

  llvm::Twine prefix = llvm::Twine(file_prefix).concat(llvm::Twine(expr_number));

  llvm::StringRef suffix;
  switch (options.GetLanguage()) {
  default:
    suffix = ".cpp";
    break;

  case lldb::eLanguageTypeSwift:
    suffix = ".swift";
    break;
  }

  int temp_fd;
  llvm::SmallString<128> buffer;
  std::error_code err =
      llvm::sys::fs::createTemporaryFile(prefix, suffix, temp_fd, buffer);
  if (!err) {
    lldb_private::File file(temp_fd, true);
    const size_t text_len = text.size();
    size_t bytes_written = text_len;
    if (file.Write(text.data(), bytes_written).Success()) {
      if (bytes_written == text_len) {
        // Make sure we have a newline in the file at the end
        bytes_written = 1;
        file.Write("\n", bytes_written);
        if (bytes_written == 1)
          success = true;
      }
    }
    if (!success)
      llvm::sys::fs::remove(expr_source_path);
  }
  if (!success)
    expr_source_path.clear();
  else
    expr_source_path = buffer.str().str();

  return success;
}
