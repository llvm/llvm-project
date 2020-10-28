//===-- ClangUtilityFunction.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#include "ClangUtilityFunction.h"
#include "ClangExpressionDeclMap.h"
#include "ClangExpressionParser.h"
#include "ClangExpressionSourceCode.h"
#include "ClangPersistentVariables.h"

#include <stdio.h>
#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif


#include "lldb/Core/Module.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"

using namespace lldb_private;

char ClangUtilityFunction::ID;

/// Constructor
///
/// \param[in] text
///     The text of the function.  Must be a full translation unit.
///
/// \param[in] name
///     The name of the function, as used in the text.
ClangUtilityFunction::ClangUtilityFunction(ExecutionContextScope &exe_scope,
                                           std::string text, std::string name)
    : UtilityFunction(
          exe_scope,
          std::string(ClangExpressionSourceCode::g_expression_prefix) + text,
          std::move(name)) {}

ClangUtilityFunction::~ClangUtilityFunction() {}

/// Install the utility function into a process
///
/// \param[in] diagnostic_manager
///     A diagnostic manager to report errors and warnings to.
///
/// \param[in] exe_ctx
///     The execution context to install the utility function to.
///
/// \return
///     True on success (no errors); false otherwise.
bool ClangUtilityFunction::Install(DiagnosticManager &diagnostic_manager,
                                   ExecutionContext &exe_ctx) {
  if (m_jit_start_addr != LLDB_INVALID_ADDRESS) {
    diagnostic_manager.PutString(eDiagnosticSeverityWarning,
                                 "already installed");
    return false;
  }

  ////////////////////////////////////
  // Set up the target and compiler
  //

  Target *target = exe_ctx.GetTargetPtr();

  if (!target) {
    diagnostic_manager.PutString(eDiagnosticSeverityError, "invalid target");
    return false;
  }

  Process *process = exe_ctx.GetProcessPtr();

  if (!process) {
    diagnostic_manager.PutString(eDiagnosticSeverityError, "invalid process");
    return false;
  }

  //////////////////////////
  // Parse the expression
  //

  bool keep_result_in_memory = false;

  ResetDeclMap(exe_ctx, keep_result_in_memory);

  if (!DeclMap()->WillParse(exe_ctx, nullptr)) {
    diagnostic_manager.PutString(
        eDiagnosticSeverityError,
        "current process state is unsuitable for expression parsing");
    return false;
  }

  const bool generate_debug_info = true;
  ClangExpressionParser parser(exe_ctx.GetBestExecutionContextScope(), *this,
                               generate_debug_info);

  unsigned num_errors = parser.Parse(diagnostic_manager);

  if (num_errors) {
    ResetDeclMap();

    return false;
  }

  //////////////////////////////////
  // JIT the output of the parser
  //

  bool can_interpret = false; // should stay that way

  Status jit_error = parser.PrepareForExecution(
      m_jit_start_addr, m_jit_end_addr, m_execution_unit_sp, exe_ctx,
      can_interpret, eExecutionPolicyAlways);

  if (m_jit_start_addr != LLDB_INVALID_ADDRESS) {
    m_jit_process_wp = process->shared_from_this();
    if (parser.GetGenerateDebugInfo()) {
      lldb::ModuleSP jit_module_sp(m_execution_unit_sp->GetJITModule());

      if (jit_module_sp) {
        ConstString const_func_name(FunctionName());
        FileSpec jit_file;
        jit_file.GetFilename() = const_func_name;
        jit_module_sp->SetFileSpecAndObjectName(jit_file, ConstString());
        m_jit_module_wp = jit_module_sp;
        target->GetImages().Append(jit_module_sp);
      }
    }
  }

  DeclMap()->DidParse();

  ResetDeclMap();

  if (jit_error.Success()) {
    return true;
  } else {
    const char *error_cstr = jit_error.AsCString();
    if (error_cstr && error_cstr[0]) {
      diagnostic_manager.Printf(eDiagnosticSeverityError, "%s", error_cstr);
    } else {
      diagnostic_manager.PutString(eDiagnosticSeverityError,
                                   "expression can't be interpreted or run");
    }
    return false;
  }
}

void ClangUtilityFunction::ClangUtilityFunctionHelper::ResetDeclMap(
    ExecutionContext &exe_ctx, bool keep_result_in_memory) {
  std::shared_ptr<ClangASTImporter> ast_importer;
  auto *state = exe_ctx.GetTargetSP()->GetPersistentExpressionStateForLanguage(
      lldb::eLanguageTypeC);
  if (state) {
    auto *persistent_vars = llvm::cast<ClangPersistentVariables>(state);
    ast_importer = persistent_vars->GetClangASTImporter();
  }
  m_expr_decl_map_up = std::make_unique<ClangExpressionDeclMap>(
      keep_result_in_memory, nullptr, exe_ctx.GetTargetSP(), ast_importer,
      nullptr);
}
