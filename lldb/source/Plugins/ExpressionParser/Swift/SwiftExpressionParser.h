//===-- SwiftExpressionParser.h ---------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftExpressionParser_h_
#define liblldb_SwiftExpressionParser_h_

#include "Plugins/ExpressionParser/Clang/IRForTarget.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Error.h"
#include "lldb/Expression/ExpressionParser.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Target.h"
#include "lldb/lldb-public.h"

#include <string>
#include <vector>

namespace lldb_private {

class IRExecutionUnit;

//----------------------------------------------------------------------
/// @class SwiftExpressionParser SwiftExpressionParser.h
/// "lldb/Expression/SwiftExpressionParser.h"
/// @brief Encapsulates an instance of Swift that can parse expressions.
///
/// SwiftExpressionParser is responsible for preparing an instance of
/// ClangExpression for execution.  SwiftExpressionParser uses ClangExpression
/// as a glorified parameter list, performing the required parsing and
/// conversion to formats (DWARF bytecode, or JIT compiled machine code)
/// that can be executed.
//----------------------------------------------------------------------
class SwiftExpressionParser : public ExpressionParser {
public:
  //------------------------------------------------------------------
  /// Constructor
  ///
  /// Initializes class variabes.
  ///
  /// @param[in] exe_scope,
  ///     If non-NULL, an execution context scope that can help to
  ///     correctly create an expression with a valid process for
  ///     optional tuning Objective-C runtime support. Can be NULL.
  ///
  /// @param[in] expr
  ///     The expression to be parsed.
  ///
  /// @param[in] options
  ///     Additional options for the parser.
  //------------------------------------------------------------------
  SwiftExpressionParser(ExecutionContextScope *exe_scope, Expression &expr,
                        const EvaluateExpressionOptions &options);

  //------------------------------------------------------------------
  /// Parse a single expression and convert it to IR using Swift.  Don't
  /// wrap the expression in anything at all.
  ///
  /// @param[in] diagnostic_manager
  ///     The diagnostic manager to report errors to.
  ///
  /// @return
  ///     The number of errors encountered during parsing.  0 means
  ///     success.
  //------------------------------------------------------------------
  unsigned Parse(DiagnosticManager &diagnostic_manager, uint32_t first_line = 0,
                 uint32_t last_line = UINT32_MAX,
                 uint32_t line_offset = 0) override;

  //------------------------------------------------------------------
  /// Ready an already-parsed expression for execution, possibly
  /// evaluating it statically.
  ///
  /// @param[out] func_addr
  ///     The address to which the function has been written.
  ///
  /// @param[out] func_end
  ///     The end of the function's allocated memory region.  (func_addr
  ///     and func_end do not delimit an allocated region; the allocated
  ///     region may begin before func_addr.)
  ///
  /// @param[in] execution_unit_ap
  ///     After parsing, ownership of the execution unit for
  ///     for the expression is handed to this unique pointer.
  ///
  /// @param[in] exe_ctx
  ///     The execution context to write the function into.
  ///
  /// @param[out] evaluated_statically
  ///     Set to true if the expression could be interpreted statically;
  ///     untouched otherwise.
  ///
  /// @param[out] const_result
  ///     If the result of the expression is constant, and the
  ///     expression has no side effects, this is set to the result of the
  ///     expression.
  ///
  /// @param[in] execution_policy
  ///     Determines whether the expression must be JIT-compiled, must be
  ///     evaluated statically, or whether this decision may be made
  ///     opportunistically.
  ///
  /// @return
  ///     An error code indicating the success or failure of the operation.
  ///     Test with Success().
  //------------------------------------------------------------------
  Error
  PrepareForExecution(lldb::addr_t &func_addr, lldb::addr_t &func_end,
                      lldb::IRExecutionUnitSP &execution_unit_ap,
                      ExecutionContext &exe_ctx, bool &can_interpret,
                      lldb_private::ExecutionPolicy execution_policy) override;

  const EvaluateExpressionOptions &GetOptions() const { return m_options; }

  bool RewriteExpression(DiagnosticManager &diagnostic_manager) override;

  //------------------------------------------------------------------
  /// Information about each variable provided to the expression, so
  /// that we can generate proper accesses in the SIL.
  //------------------------------------------------------------------
  struct SILVariableInfo {
    CompilerType type;
    uint64_t offset;
    bool needs_init;

    SILVariableInfo(CompilerType t, uint64_t o, bool ni)
        : type(t), offset(o), needs_init(ni) {}

    SILVariableInfo() : type(), offset(0), needs_init(false) {}
  };

  //------------------------------------------------------------------
  /// A container for variables, created during a parse and discarded
  /// when done.
  //------------------------------------------------------------------
  typedef std::map<const char *, SILVariableInfo> SILVariableMap;

private:
  bool PerformAutoImport(swift::SourceFile &source_file, bool user_imports,
                         Error &error);

  Expression &m_expr;   ///< The expression to be parsed
  std::string m_triple; ///< The triple to use when compiling
  std::unique_ptr<llvm::LLVMContext>
      m_llvm_context; ///< The context to use for IR generation
  std::unique_ptr<llvm::Module> m_module;      ///< The module to build IR into
  lldb::IRExecutionUnitSP m_execution_unit_sp; ///< The container for the IR, to
                                               ///be JIT-compiled or interpreted
  SwiftASTContext
      *m_swift_ast_context; ///< The AST context to build the expression into
  SymbolContext m_sc;       ///< The symbol context to use when parsing
  lldb::StackFrameWP m_stack_frame_wp; ///< The stack frame to use (if possible)
                                       ///when determining dynamic types.
  EvaluateExpressionOptions m_options; ///< If true, we are running in REPL mode
};
}

#endif // liblldb_SwiftExpressionParser_h_
