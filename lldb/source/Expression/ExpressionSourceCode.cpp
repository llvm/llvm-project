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
#include "swift/AST/PlatformKind.h"
#include "swift/Basic/LangOptions.h"

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"

using namespace lldb_private;

const char *ExpressionSourceCode::g_expression_prefix = R"(
#ifndef NULL
#define NULL (__null)
#endif
#ifndef Nil
#define Nil (__null)
#endif
#ifndef nil
#define nil (__null)
#endif
#ifndef YES
#define YES ((BOOL)1)
#endif
#ifndef NO
#define NO ((BOOL)0)
#endif
typedef __INT8_TYPE__ int8_t;
typedef __UINT8_TYPE__ uint8_t;
typedef __INT16_TYPE__ int16_t;
typedef __UINT16_TYPE__ uint16_t;
typedef __INT32_TYPE__ int32_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __INT64_TYPE__ int64_t;
typedef __UINT64_TYPE__ uint64_t;
typedef __INTPTR_TYPE__ intptr_t;
typedef __UINTPTR_TYPE__ uintptr_t;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef unsigned short unichar;
extern "C"
{
    int printf(const char * __restrict, ...);
}
)";

uint32_t ExpressionSourceCode::GetNumBodyLines() {
  if (m_num_body_lines == 0)
    // 2 = <one for zero indexing> + <one for the body start marker>
    m_num_body_lines = 2 + std::count(m_body.begin(), m_body.end(), '\n');
  return m_num_body_lines;
}

static const char *c_start_marker = "    /*LLDB_BODY_START*/\n    ";
static const char *c_end_marker = ";\n    /*LLDB_BODY_END*/\n";

namespace {

class AddMacroState {
  enum State {
    CURRENT_FILE_NOT_YET_PUSHED,
    CURRENT_FILE_PUSHED,
    CURRENT_FILE_POPPED
  };

public:
  AddMacroState(const FileSpec &current_file, const uint32_t current_file_line)
      : m_state(CURRENT_FILE_NOT_YET_PUSHED), m_current_file(current_file),
        m_current_file_line(current_file_line) {}

  void StartFile(const FileSpec &file) {
    m_file_stack.push_back(file);
    if (file == m_current_file)
      m_state = CURRENT_FILE_PUSHED;
  }

  void EndFile() {
    if (m_file_stack.size() == 0)
      return;

    FileSpec old_top = m_file_stack.back();
    m_file_stack.pop_back();
    if (old_top == m_current_file)
      m_state = CURRENT_FILE_POPPED;
  }

  // An entry is valid if it occurs before the current line in the current
  // file.
  bool IsValidEntry(uint32_t line) {
    switch (m_state) {
    case CURRENT_FILE_NOT_YET_PUSHED:
      return true;
    case CURRENT_FILE_PUSHED:
      // If we are in file included in the current file, the entry should be
      // added.
      if (m_file_stack.back() != m_current_file)
        return true;

      return line < m_current_file_line;
    default:
      return false;
    }
  }

private:
  std::vector<FileSpec> m_file_stack;
  State m_state;
  FileSpec m_current_file;
  uint32_t m_current_file_line;
};

} // anonymous namespace

static void AddMacros(const DebugMacros *dm, CompileUnit *comp_unit,
                      AddMacroState &state, StreamString &stream) {
  if (dm == nullptr)
    return;

  for (size_t i = 0; i < dm->GetNumMacroEntries(); i++) {
    const DebugMacroEntry &entry = dm->GetMacroEntryAtIndex(i);
    uint32_t line;

    switch (entry.GetType()) {
    case DebugMacroEntry::DEFINE:
      if (state.IsValidEntry(entry.GetLineNumber()))
        stream.Printf("#define %s\n", entry.GetMacroString().AsCString());
      else
        return;
      break;
    case DebugMacroEntry::UNDEF:
      if (state.IsValidEntry(entry.GetLineNumber()))
        stream.Printf("#undef %s\n", entry.GetMacroString().AsCString());
      else
        return;
      break;
    case DebugMacroEntry::START_FILE:
      line = entry.GetLineNumber();
      if (state.IsValidEntry(line))
        state.StartFile(entry.GetFileSpec(comp_unit));
      else
        return;
      break;
    case DebugMacroEntry::END_FILE:
      state.EndFile();
      break;
    case DebugMacroEntry::INDIRECT:
      AddMacros(entry.GetIndirectDebugMacros(), comp_unit, state, stream);
      break;
    default:
      // This is an unknown/invalid entry. Ignore.
      break;
    }
  }
}

static bool ExprBodyContainsVar(llvm::StringRef var, llvm::StringRef body) {
  size_t from = 0;
  while ((from = body.find(var, from)) != llvm::StringRef::npos) {
    if ((from != 0 && clang::isIdentifierBody(body[from-1])) ||
        (from + var.size() != body.size() &&
         clang::isIdentifierBody(body[from+var.size()]))) {
      ++from;
      continue;
    }
    return true;
  }
  return false;
}

static void AddLocalVariableDecls(const lldb::VariableListSP &var_list_sp,
                                  StreamString &stream, const std::string &expr,
                                  lldb::LanguageType wrapping_language) {
  for (size_t i = 0; i < var_list_sp->GetSize(); i++) {
    lldb::VariableSP var_sp = var_list_sp->GetVariableAtIndex(i);

    ConstString var_name = var_sp->GetName();

    // We can check for .block_descriptor w/o checking for langauge since this
    // is not a valid identifier in either C or C++.
    if (!var_name || var_name == ConstString(".block_descriptor"))
      continue;

    if (!expr.empty() && !ExprBodyContainsVar(var_name.GetStringRef(), expr))
      continue;

    if ((var_name == ConstString("self") || var_name == ConstString("_cmd")) &&
        (wrapping_language == lldb::eLanguageTypeObjC ||
         wrapping_language == lldb::eLanguageTypeObjC_plus_plus))
      continue;

    if (var_name == ConstString("this") &&
        wrapping_language == lldb::eLanguageTypeC_plus_plus)
      continue;

    stream.Printf("using $__lldb_local_vars::%s;\n", var_name.AsCString());
  }
}

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
