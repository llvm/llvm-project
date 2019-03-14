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

#include "Plugins/ExpressionParser/Clang/ClangModulesDeclVendor.h"
#include "Plugins/ExpressionParser/Clang/ClangPersistentVariables.h"
#include "Plugins/ExpressionParser/Clang/ClangUserExpression.h"
#include "Plugins/ExpressionParser/Swift/SwiftASTManipulator.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
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
                                  StreamString &stream, const std::string &expr) {
  for (size_t i = 0; i < var_list_sp->GetSize(); i++) {
    lldb::VariableSP var_sp = var_list_sp->GetVariableAtIndex(i);

    ConstString var_name = var_sp->GetName();
    if (!var_name || var_name == ConstString("this") ||
        var_name == ConstString(".block_descriptor") ||
        !ExprBodyContainsVar(var_name.AsCString(), expr))
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

/// Format the OS name the way that Swift availability attributes do.
static llvm::StringRef getAvailabilityName(llvm::Triple::OSType os) {
  switch (os) {
  case llvm::Triple::MacOSX: return "macOS";
  case llvm::Triple::IOS: return "iOS";
  case llvm::Triple::TvOS: return "tvOS";
  case llvm::Triple::WatchOS: return "watchOS";
  default:
    return llvm::Triple::getOSTypeName(os);
  }
}

bool ExpressionSourceCode::GetText(
    std::string &text, lldb::LanguageType wrapping_language,
    uint32_t language_flags, const EvaluateExpressionOptions &options,
    ExecutionContext &exe_ctx, uint32_t &first_body_line) const {
  first_body_line = 0;

  const char *target_specific_defines = "typedef signed char BOOL;\n";
  std::string module_macros;

  Target *target = exe_ctx.GetTargetPtr();
  if (ClangModulesDeclVendor::LanguageSupportsClangModules(wrapping_language)) {
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

      ClangPersistentVariables *persistent_vars =
          llvm::dyn_cast_or_null<ClangPersistentVariables>(
              target->GetPersistentExpressionStateForLanguage(
                  lldb::eLanguageTypeC));
      ClangModulesDeclVendor *decl_vendor = target->GetClangModulesDeclVendor();

      if (persistent_vars && decl_vendor) {
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

    ConstString object_name;
    if (Language::LanguageIsCPlusPlus(frame->GetLanguage())) {
      if (target->GetInjectLocalVariables(&exe_ctx)) {
        lldb::VariableListSP var_list_sp =
            frame->GetInScopeVariableList(false, true);
        AddLocalVariableDecls(var_list_sp, lldb_local_var_decls, m_body);
      }
    }
  }

  if (m_wrap) {
    const char *body = m_body.c_str();
    const char *pound_file = options.GetPoundLineFilePath();
    const uint32_t pound_line = options.GetPoundLineLine();
    StreamString pound_body;
    if (pound_file && pound_line) {
      if (wrapping_language == lldb::eLanguageTypeSwift) {
        pound_body.Printf("#sourceLocation(file: \"%s\", line: %u)\n%s",
                          pound_file, pound_line, body);
      } else {
        pound_body.Printf("#line %u \"%s\"\n%s", pound_line, pound_file, body);
      }
      body = pound_body.GetString().data();
    }

    switch (wrapping_language) {
    default:
      return false;
    case lldb::eLanguageTypeC:
    case lldb::eLanguageTypeC_plus_plus:
    case lldb::eLanguageTypeObjC:
    case lldb::eLanguageTypeSwift:
      break;
    }

    StreamString wrap_stream;

    if (ClangModulesDeclVendor::LanguageSupportsClangModules(
            wrapping_language)) {
      wrap_stream.Printf("%s\n%s\n%s\n%s\n%s\n", module_macros.c_str(),
                         debug_macros_stream.GetData(), g_expression_prefix,
                         target_specific_defines, m_prefix.c_str());
    }

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
      wrap_stream.Printf("void                           \n"
                         "%s(void *$__lldb_arg)          \n"
                         "{                              \n"
                         "    %s;                        \n"
                         "%s"
                         "}                              \n",
                         m_name.c_str(), lldb_local_var_decls.GetData(),
                         tagged_body.c_str());
      break;
    case lldb::eLanguageTypeC_plus_plus:
      wrap_stream.Printf("void                                   \n"
                         "$__lldb_class::%s(void *$__lldb_arg)   \n"
                         "{                                      \n"
                         "    %s;                                \n"
                         "%s"
                         "}                                      \n",
                         m_name.c_str(), lldb_local_var_decls.GetData(),
                         tagged_body.c_str());
      break;
    case lldb::eLanguageTypeObjC:
      if (language_flags & ClangUserExpression::eLanguageFlagInStaticMethod) {
        wrap_stream.Printf(
            "@interface $__lldb_objc_class ($__lldb_category)        \n"
            "+(void)%s:(void *)$__lldb_arg;                          \n"
            "@end                                                    \n"
            "@implementation $__lldb_objc_class ($__lldb_category)   \n"
            "+(void)%s:(void *)$__lldb_arg                           \n"
            "{                                                       \n"
            "%s"
            "}                                                       \n"
            "@end                                                    \n",
            m_name.c_str(), m_name.c_str(), tagged_body.c_str());
      } else {
        wrap_stream.Printf(
            "@interface $__lldb_objc_class ($__lldb_category)       \n"
            "-(void)%s:(void *)$__lldb_arg;                         \n"
            "@end                                                   \n"
            "@implementation $__lldb_objc_class ($__lldb_category)  \n"
            "-(void)%s:(void *)$__lldb_arg                          \n"
            "{                                                      \n"
            "%s"
            "}                                                      \n"
            "@end                                                   \n",
            m_name.c_str(), m_name.c_str(), tagged_body.c_str());
      }
      break;
    case lldb::eLanguageTypeSwift: {
      llvm::SmallString<16> buffer;
      llvm::raw_svector_ostream os_vers(buffer);

      auto arch_spec = target->GetArchitecture();
      auto triple = arch_spec.GetTriple();
      if (triple.isOSDarwin()) {
        if (auto process_sp = exe_ctx.GetProcessSP()) {
          os_vers << getAvailabilityName(triple.getOS()) << " ";
          auto platform = target->GetPlatform();
          bool is_simulator =
              platform->GetPluginName().GetStringRef().endswith("-simulator");
          if (is_simulator) {
            // The simulators look like the host OS to Process, but Platform
            // can the version out of an environment variable.
            os_vers << platform->GetOSVersion(process_sp.get()).getAsString();
          } else {
	    llvm::VersionTuple version = 
	      process_sp->GetHostOSVersion();
	    os_vers << version.getAsString();
          }
        }
      }
      SwiftPersistentExpressionState *persistent_state =
        llvm::cast<SwiftPersistentExpressionState>(target->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeSwift));
      std::vector<swift::ValueDecl *> persistent_results;
      // Check if we have already declared the playground stub debug functions
      persistent_state->GetSwiftPersistentDecls(ConstString("__builtin_log_with_id"), {},
                                                persistent_results);

      size_t num_persistent_results = persistent_results.size();
      bool need_to_declare_log_functions = num_persistent_results == 0;
      EvaluateExpressionOptions localOptions(options);

      localOptions.SetPreparePlaygroundStubFunctions(need_to_declare_log_functions);

      SwiftASTManipulator::WrapExpression(wrap_stream, m_body.c_str(),
                                          language_flags, localOptions,
                                          os_vers.str(), first_body_line);
    }
    }

    text = wrap_stream.GetString();
  } else {
    text.append(m_body);
  }

  return true;
}

bool ExpressionSourceCode::GetOriginalBodyBounds(
    std::string transformed_text, lldb::LanguageType wrapping_language,
    size_t &start_loc, size_t &end_loc) {
  const char *start_marker;
  const char *end_marker;

  switch (wrapping_language) {
  default:
    return false;
  case lldb::eLanguageTypeSwift:
    start_marker = SwiftASTManipulator::GetUserCodeStartMarker();
    end_marker = SwiftASTManipulator::GetUserCodeEndMarker();
    break;
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
