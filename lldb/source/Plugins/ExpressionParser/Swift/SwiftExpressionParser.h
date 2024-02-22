//===-- SwiftExpressionParser.h ---------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftExpressionParser_h_
#define liblldb_SwiftExpressionParser_h_

#include "SwiftASTManipulator.h"

#include "Plugins/ExpressionParser/Clang/IRForTarget.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/Status.h"
#include "lldb/Expression/ExpressionParser.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Target.h"
#include "lldb/lldb-public.h"

#include <string>
#include <vector>

namespace lldb_private {

class IRExecutionUnit;
class SwiftLanguageRuntime;
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
  enum class ParseResult {
    success,
    retry_fresh_context, 
    retry_no_bind_generic_params,
    unrecoverable_error
  };
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
  /// @param[in] local_variables
  ///     The local variables that are in scope.
  ///
  /// @param[in] options
  ///     Additional options for the parser.
  //------------------------------------------------------------------
  SwiftExpressionParser(
       ExecutionContextScope *exe_scope,
       SwiftASTContextForExpressions &swift_ast_ctx, Expression &expr,
       llvm::SmallVector<SwiftASTManipulator::VariableInfo> &&local_variables,
       const EvaluateExpressionOptions &options);
  //------------------------------------------------------------------
  /// Attempts to find possible command line completions for the given
  /// expression.
  ///
  /// Currently unimplemented for Swift.
  //------------------------------------------------------------------
  bool Complete(CompletionRequest &request, unsigned line, unsigned pos,
                unsigned typed_pos) override;

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
  ParseResult Parse(DiagnosticManager &diagnostic_manager,
                    uint32_t first_line = 0, uint32_t last_line = UINT32_MAX);

  /// Returns true if the call to parse of this type is cacheable.
  bool IsParseCacheable() const {
    return m_is_cacheable;
  }

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
  Status
  PrepareForExecution(lldb::addr_t &func_addr, lldb::addr_t &func_end,
                      lldb::IRExecutionUnitSP &execution_unit_ap,
                      ExecutionContext &exe_ctx, bool &can_interpret,
                      lldb_private::ExecutionPolicy execution_policy) override;

  const EvaluateExpressionOptions &GetOptions() const { return m_options; }

  bool RewriteExpression(DiagnosticManager &diagnostic_manager) override;

  static CompilerType ResolveVariable(
      lldb::VariableSP variable_sp, lldb::StackFrameSP &stack_frame_sp,
      SwiftLanguageRuntime *runtime, lldb::DynamicValueType use_dynamic,
      lldb::BindGenericTypes bind_generic_types);

  static lldb::VariableSP FindSelfVariable(Block *block);

  //------------------------------------------------------------------
  /// Information about each variable provided to the expression, so
  /// that we can generate proper accesses in the SIL.
  //------------------------------------------------------------------
  struct SILVariableInfo {
    CompilerType type;
    uint64_t offset = 0;
    bool needs_init = false;
    bool is_unowned_self = false;

    SILVariableInfo() = default;
    SILVariableInfo(CompilerType t, uint64_t o, bool ni, bool s)
      : type(t), offset(o), needs_init(ni), is_unowned_self(s) {}
  };

  //------------------------------------------------------------------
  /// A container for variables, created during a parse and discarded
  /// when done.
  //------------------------------------------------------------------
  typedef std::map<const char *, SILVariableInfo> SILVariableMap;

private:
  /// The expression to be parsed.
  Expression &m_expr;
  /// The context to use for IR generation.
  std::unique_ptr<llvm::LLVMContext> m_llvm_context;
  /// The module to build IR into.
  std::unique_ptr<llvm::Module> m_module;
  /// The container for the IR, to be JIT-compiled or interpreted.
  lldb::IRExecutionUnitSP m_execution_unit_sp;
  /// The AST context to build the expression into.
  SwiftASTContextForExpressions &m_swift_ast_ctx;
  /// Used to manage the memory of a potential on-off context.
  //lldb::TypeSystemSP m_typesystem_sp;
  /// The symbol context to use when parsing.
  SymbolContext m_sc;
  // The execution context scope of the expression.
  ExecutionContextScope *m_exe_scope;
  /// The stack frame to use (if possible) when determining dynamic
  /// types.
  lldb::StackFrameWP m_stack_frame_wp;

  /// The variables in scope.
  llvm::SmallVector<SwiftASTManipulator::VariableInfo> m_local_variables;

  /// If true, we are running in REPL mode
  EvaluateExpressionOptions m_options;

  /// Indicates whether the call to Parse of this type is cacheable.
  bool m_is_cacheable;
};
} // namespace lldb_private

#endif // liblldb_SwiftExpressionParser_h_
