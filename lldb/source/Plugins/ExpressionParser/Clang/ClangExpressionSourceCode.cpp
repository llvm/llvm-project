//===-- ClangExpressionSourceCode.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangExpressionSourceCode.h"

#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/StringRef.h"

#include "Plugins/ExpressionParser/Clang/ClangModulesDeclVendor.h"
#include "Plugins/ExpressionParser/Clang/ClangPersistentVariables.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/DebugMacros.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb_private;

const char *ClangExpressionSourceCode::g_expression_prefix = R"(
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

/// Checks if the expression body contains the given variable as a token.
/// \param body The expression body.
/// \param var The variable token we are looking for.
/// \return True iff the expression body containes the variable as a token.
static bool ExprBodyContainsVar(llvm::StringRef body, llvm::StringRef var) {
  assert(var.find_if([](char c) { return !clang::isIdentifierBody(c); }) ==
             llvm::StringRef::npos &&
         "variable contains non-identifier chars?");

  size_t start = 0;
  // Iterate over all occurences of the variable string in our expression.
  while ((start = body.find(var, start)) != llvm::StringRef::npos) {
    // We found our variable name in the expression. Check that the token
    // that contains our needle is equal to our variable and not just contains
    // the character sequence by accident.
    // Prevents situations where we for example inlcude the variable 'FOO' in an
    // expression like 'FOObar + 1'.
    bool has_characters_before =
        start != 0 && clang::isIdentifierBody(body[start - 1]);
    bool has_characters_after =
        start + var.size() < body.size() &&
        clang::isIdentifierBody(body[start + var.size()]);

    // Our token just contained the variable name as a substring. Continue
    // searching the rest of the expression.
    if (has_characters_before || has_characters_after) {
      ++start;
      continue;
    }
    return true;
  }
  return false;
}

static void AddLocalVariableDecls(const lldb::VariableListSP &var_list_sp,
                                  StreamString &stream,
                                  const std::string &expr,
                                  lldb::LanguageType wrapping_language) {
  for (size_t i = 0; i < var_list_sp->GetSize(); i++) {
    lldb::VariableSP var_sp = var_list_sp->GetVariableAtIndex(i);

    ConstString var_name = var_sp->GetName();


    // We can check for .block_descriptor w/o checking for langauge since this
    // is not a valid identifier in either C or C++.
    if (!var_name || var_name == ".block_descriptor")
      continue;

    if (!expr.empty() && !ExprBodyContainsVar(expr, var_name.GetStringRef()))
      continue;

    if ((var_name == "self" || var_name == "_cmd") &&
        (wrapping_language == lldb::eLanguageTypeObjC ||
         wrapping_language == lldb::eLanguageTypeObjC_plus_plus))
      continue;

    if (var_name == "this" &&
        wrapping_language == lldb::eLanguageTypeC_plus_plus)
      continue;

    stream.Printf("using $__lldb_local_vars::%s;\n", var_name.AsCString());
  }
}

bool ClangExpressionSourceCode::GetText(
    std::string &text, lldb::LanguageType wrapping_language, bool static_method,
    ExecutionContext &exe_ctx, bool add_locals, bool force_add_all_locals,
    llvm::ArrayRef<std::string> modules) const {
  const char *target_specific_defines = "typedef signed char BOOL;\n";
  std::string module_macros;

  Target *target = exe_ctx.GetTargetPtr();
  if (target) {
    if (target->GetArchitecture().GetMachine() == llvm::Triple::aarch64) {
      target_specific_defines = "typedef bool BOOL;\n";
    }
    if (target->GetArchitecture().GetMachine() == llvm::Triple::x86_64) {
      if (lldb::PlatformSP platform_sp = target->GetPlatform()) {
        static ConstString g_platform_ios_simulator("ios-simulator");
        if (platform_sp->GetPluginName() == g_platform_ios_simulator) {
          target_specific_defines = "typedef bool BOOL;\n";
        }
      }
    }

    if (ClangModulesDeclVendor *decl_vendor =
            target->GetClangModulesDeclVendor()) {
      ClangPersistentVariables *persistent_vars =
          llvm::cast<ClangPersistentVariables>(
              target->GetPersistentExpressionStateForLanguage(
                  lldb::eLanguageTypeC));
      const ClangModulesDeclVendor::ModuleVector &hand_imported_modules =
          persistent_vars->GetHandLoadedClangModules();
      ClangModulesDeclVendor::ModuleVector modules_for_macros;

      for (ClangModulesDeclVendor::ModuleID module : hand_imported_modules) {
        modules_for_macros.push_back(module);
      }

      if (target->GetEnableAutoImportClangModules()) {
        if (StackFrame *frame = exe_ctx.GetFramePtr()) {
          if (Block *block = frame->GetFrameBlock()) {
            SymbolContext sc;

            block->CalculateSymbolContext(&sc);

            if (sc.comp_unit) {
              StreamString error_stream;

              decl_vendor->AddModulesForCompileUnit(
                  *sc.comp_unit, modules_for_macros, error_stream);
            }
          }
        }
      }

      decl_vendor->ForEachMacro(
          modules_for_macros,
          [&module_macros](const std::string &expansion) -> bool {
            module_macros.append(expansion);
            module_macros.append("\n");
            return false;
          });
    }
  }

  StreamString debug_macros_stream;
  StreamString lldb_local_var_decls;
  if (StackFrame *frame = exe_ctx.GetFramePtr()) {
    const SymbolContext &sc = frame->GetSymbolContext(
        lldb::eSymbolContextCompUnit | lldb::eSymbolContextLineEntry);

    if (sc.comp_unit && sc.line_entry.IsValid()) {
      DebugMacros *dm = sc.comp_unit->GetDebugMacros();
      if (dm) {
        AddMacroState state(sc.line_entry.file, sc.line_entry.line);
        AddMacros(dm, sc.comp_unit, state, debug_macros_stream);
      }
    }

    if (add_locals)
      if (target->GetInjectLocalVariables(&exe_ctx)) {
        lldb::VariableListSP var_list_sp =
            frame->GetInScopeVariableList(false, true);
        AddLocalVariableDecls(var_list_sp, lldb_local_var_decls,
                              force_add_all_locals ? "" : m_body,
                              wrapping_language);
      }
  }

  if (m_wrap) {
    switch (wrapping_language) {
    default:
      return false;
    case lldb::eLanguageTypeC:
    case lldb::eLanguageTypeC_plus_plus:
    case lldb::eLanguageTypeObjC:
      break;
    }

    // Generate a list of @import statements that will import the specified
    // module into our expression.
    std::string module_imports;
    for (const std::string &module : modules) {
      module_imports.append("@import ");
      module_imports.append(module);
      module_imports.append(";\n");
    }

    StreamString wrap_stream;

    wrap_stream.Printf("%s\n%s\n%s\n%s\n%s\n", module_macros.c_str(),
                       debug_macros_stream.GetData(), g_expression_prefix,
                       target_specific_defines, m_prefix.c_str());

    // First construct a tagged form of the user expression so we can find it
    // later:
    std::string tagged_body;
    switch (wrapping_language) {
    default:
      tagged_body = m_body;
      break;
    case lldb::eLanguageTypeC:
    case lldb::eLanguageTypeC_plus_plus:
    case lldb::eLanguageTypeObjC:
      tagged_body.append(c_start_marker);
      tagged_body.append(m_body);
      tagged_body.append(c_end_marker);
      break;
    }
    switch (wrapping_language) {
    default:
      break;
    case lldb::eLanguageTypeC:
      wrap_stream.Printf("%s"
                         "void                           \n"
                         "%s(void *$__lldb_arg)          \n"
                         "{                              \n"
                         "    %s;                        \n"
                         "%s"
                         "}                              \n",
                         module_imports.c_str(), m_name.c_str(),
                         lldb_local_var_decls.GetData(), tagged_body.c_str());
      break;
    case lldb::eLanguageTypeC_plus_plus:
      wrap_stream.Printf("%s"
                         "void                                   \n"
                         "$__lldb_class::%s(void *$__lldb_arg)   \n"
                         "{                                      \n"
                         "    %s;                                \n"
                         "%s"
                         "}                                      \n",
                         module_imports.c_str(), m_name.c_str(),
                         lldb_local_var_decls.GetData(), tagged_body.c_str());
      break;
    case lldb::eLanguageTypeObjC:
      if (static_method) {
        wrap_stream.Printf(
            "%s"
            "@interface $__lldb_objc_class ($__lldb_category)        \n"
            "+(void)%s:(void *)$__lldb_arg;                          \n"
            "@end                                                    \n"
            "@implementation $__lldb_objc_class ($__lldb_category)   \n"
            "+(void)%s:(void *)$__lldb_arg                           \n"
            "{                                                       \n"
            "    %s;                                                 \n"
            "%s"
            "}                                                       \n"
            "@end                                                    \n",
            module_imports.c_str(), m_name.c_str(), m_name.c_str(),
            lldb_local_var_decls.GetData(), tagged_body.c_str());
      } else {
        wrap_stream.Printf(
            "%s"
            "@interface $__lldb_objc_class ($__lldb_category)       \n"
            "-(void)%s:(void *)$__lldb_arg;                         \n"
            "@end                                                   \n"
            "@implementation $__lldb_objc_class ($__lldb_category)  \n"
            "-(void)%s:(void *)$__lldb_arg                          \n"
            "{                                                      \n"
            "    %s;                                                \n"
            "%s"
            "}                                                      \n"
            "@end                                                   \n",
            module_imports.c_str(), m_name.c_str(), m_name.c_str(),
            lldb_local_var_decls.GetData(), tagged_body.c_str());
      }
      break;
    }

    text = wrap_stream.GetString();
  } else {
    text.append(m_body);
  }

  return true;
}

bool ClangExpressionSourceCode::GetOriginalBodyBounds(
    std::string transformed_text, lldb::LanguageType wrapping_language,
    size_t &start_loc, size_t &end_loc) {
  const char *start_marker;
  const char *end_marker;

  switch (wrapping_language) {
  default:
    return false;
  case lldb::eLanguageTypeC:
  case lldb::eLanguageTypeC_plus_plus:
  case lldb::eLanguageTypeObjC:
    start_marker = c_start_marker;
    end_marker = c_end_marker;
    break;
  }

  start_loc = transformed_text.find(start_marker);
  if (start_loc == std::string::npos)
    return false;
  start_loc += strlen(start_marker);
  end_loc = transformed_text.find(end_marker);
  return end_loc != std::string::npos;
}
