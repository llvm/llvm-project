//===-- SwiftASTManipulator.cpp ---------------------------------*- C++ -*-===//
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

#include "SwiftASTManipulator.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Expression/ExpressionParser.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/Target/Target.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/ASTWalker.h"
#include "swift/AST/ArchetypeBuilder.h"
#include "swift/AST/Decl.h"
#include "swift/AST/DiagnosticEngine.h"
#include "swift/AST/DiagnosticsFrontend.h"
#include "swift/AST/Expr.h"
#include "swift/AST/Initializer.h"
#include "swift/AST/Module.h"
#include "swift/AST/NameLookup.h"
#include "swift/AST/ParameterList.h"
#include "swift/AST/Pattern.h"
#include "swift/AST/Stmt.h"
#include "swift/AST/TypeRepr.h"
#include "swift/AST/Types.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "SwiftUserExpression.h"

using namespace lldb_private;

static void DumpGenericNames(
    lldb_private::Stream &wrapped_stream,
    llvm::ArrayRef<Expression::SwiftGenericInfo::Binding> generic_bindings) {
  if (generic_bindings.empty())
    return;

  wrapped_stream.PutChar('<');

  bool comma = false;

  for (const Expression::SwiftGenericInfo::Binding &binding :
       generic_bindings) {
    if (comma)
      wrapped_stream.PutCString(", ");
    comma = true;

    wrapped_stream.PutCString(binding.name);
  }

  wrapped_stream.PutChar('>');
}

static void DumpPlaceholderArguments(
    lldb_private::Stream &wrapped_stream,
    llvm::ArrayRef<Expression::SwiftGenericInfo::Binding> generic_bindings) {
  if (generic_bindings.empty())
    return;

  for (const Expression::SwiftGenericInfo::Binding &binding :
       generic_bindings) {
    const char *name = binding.name;

    wrapped_stream.Printf(", _ __lldb_placeholder_%s : UnsafePointer<%s>!",
                          name, name);
  }
}

static void DumpPlaceholdersIntoCall(
    lldb_private::Stream &wrapped_stream,
    llvm::ArrayRef<Expression::SwiftGenericInfo::Binding> generic_bindings) {
  if (generic_bindings.empty())
    return;

  for (const Expression::SwiftGenericInfo::Binding &binding :
       generic_bindings) {
    wrapped_stream.Printf(
        ",\n"
        "      (nil as UnsafePointer<$__lldb_typeof_generic_%s>?)",
        binding.name);
  }
}

bool SwiftASTManipulator::VariableInfo::GetIsLet() const {
  if (m_decl)
    return m_decl->isLet();
  else
    return m_is_let;
}

void SwiftASTManipulator::WrapExpression(
    lldb_private::Stream &wrapped_stream, const char *orig_text,
    uint32_t language_flags, const EvaluateExpressionOptions &options,
    const Expression::SwiftGenericInfo &generic_info,
    uint32_t &first_body_line) {
  first_body_line = 0; // set to invalid
  // TODO make the extension private so we're not polluting the class
  static unsigned int counter = 0;
  unsigned int current_counter = counter++;

  const bool playground = options.GetPlaygroundTransformEnabled();
  const bool repl = options.GetREPLEnabled();
  const bool generate_debug_info = options.GetGenerateDebugInfo();
  const char *pound_file = options.GetPoundLineFilePath();
  const uint32_t pound_line = options.GetPoundLineLine();
  const char *text = orig_text;
  StreamString fixed_text;

  if (playground) {
    const char *playground_prefix = R"(
@_silgen_name ("playground_logger_initialize") func $builtin_logger_initialize ()
@_silgen_name ("playground_log_hidden") func $builtin_log_with_id<T> (_ object : T, _ name : String, _ id : Int, _ sl : Int, _ el : Int, _ sc : Int, _ ec: Int) -> AnyObject
@_silgen_name ("playground_log_scope_entry") func $builtin_log_scope_entry (_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int) -> AnyObject
@_silgen_name ("playground_log_scope_exit") func $builtin_log_scope_exit (_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int) -> AnyObject
@_silgen_name ("playground_log_postprint") func $builtin_postPrint (_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int) -> AnyObject
@_silgen_name ("DVTSendPlaygroundLogData") func $builtin_send_data (_ :  AnyObject!)
$builtin_logger_initialize()
)";
    if (pound_file && pound_line) {
      wrapped_stream.Printf("%s#sourceLocation(file: \"%s\", line: %u)\n%s\n",
                            playground_prefix, pound_file, pound_line,
                            orig_text);
      first_body_line = 1;
    } else {
      wrapped_stream.Printf("%s%s", playground_prefix, orig_text);
      first_body_line = 7;
    }
    return;
  } else if (repl) {
    if (pound_file && pound_line) {
      wrapped_stream.Printf("#sourceLocation(file: \"%s\", line:  %u)\n%s\n",
                            llvm::sys::path::filename(pound_file).data(),
                            pound_line, orig_text);
    } else {
      wrapped_stream.Printf("%s", orig_text);
    }
    first_body_line = 1;
    return;
  }

  std::string expr_source_path;

  if (pound_file && pound_line) {
    fixed_text.Printf("#sourceLocation(file: \"%s\", line: %u)\n%s\n",
                      pound_file, pound_line, orig_text);
    text = fixed_text.GetString().c_str();
  } else if (generate_debug_info) {
    if (ExpressionSourceCode::SaveExpressionTextToTempFile(orig_text, options,
                                                           expr_source_path)) {
      fixed_text.Printf("#sourceLocation(file: \"%s\", line: 1)\n%s\n",
                        expr_source_path.c_str(), orig_text);
      text = fixed_text.GetString().c_str();
    }
  }

  // Note: All the wrapper functions we make are marked with the
  // @LLDBDebuggerFunction macro so that the compiler
  // can do whatever special treatment it need to do on them.  If you add new
  // variants be sure to mark them this way.
  // Also, any function that might end up being in an extension of swift class
  // needs to be marked final, since otherwise
  // the compiler might try to dispatch them dynamically, which it can't do
  // correctly for these functions.

  StreamString wrapped_expr_text;
  wrapped_expr_text.Printf("do\n"
                           "{\n"
                           "%s%s%s\n" // Don't indent the code so error columns
                                      // match up with errors from compiler
                           "}\n"
                           "catch (let __lldb_tmp_error)\n"
                           "{\n"
                           "    var %s = __lldb_tmp_error\n"
                           "}\n",
                           GetUserCodeStartMarker(), text,
                           GetUserCodeEndMarker(), GetErrorName());

  if (Flags(language_flags)
          .AnySet(SwiftUserExpression::eLanguageFlagNeedsObjectPointer |
                  SwiftUserExpression::eLanguageFlagInStaticMethod)) {
    const char *func_decorator = "";
    if (language_flags & SwiftUserExpression::eLanguageFlagInStaticMethod) {
      if (language_flags & SwiftUserExpression::eLanguageFlagIsClass)
        func_decorator = "final class";
      else
        func_decorator = "static";
    } else if (language_flags & SwiftUserExpression::eLanguageFlagIsClass &&
               !(language_flags &
                 SwiftUserExpression::eLanguageFlagIsWeakSelf)) {
      func_decorator = "final";
    } else {
      func_decorator = "mutating";
    }

    const char *optional_extension =
        (language_flags & SwiftUserExpression::eLanguageFlagIsWeakSelf)
            ? "Optional where Wrapped: "
            : "";

    if (generic_info.class_bindings.size()) {
      if (generic_info.function_bindings.size()) {
        wrapped_stream.Printf(
            "extension %s$__lldb_context {\n"
            "  @LLDBDebuggerFunction                                 \n"
            "  %s func $__lldb_wrapped_expr_%u",
            optional_extension, func_decorator, current_counter);
        DumpGenericNames(wrapped_stream, generic_info.function_bindings);
        wrapped_stream.Printf("(_ $__lldb_arg : UnsafeMutablePointer<Any>");
        DumpPlaceholderArguments(wrapped_stream,
                                 generic_info.function_bindings);
        wrapped_stream.Printf(
            ") {\n"
            "%s" // This is the expression text.  It has all the newlines it
                 // needs.
            "  }                                                    \n"
            "}                                                      \n"
            "func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {    "
            "  \n"
            "  do {                                          \n"
            "    $__lldb_injected_self.$__lldb_wrapped_expr_%u(     \n"
            "      $__lldb_arg                                        ",
            wrapped_expr_text.GetData(), current_counter);
        DumpPlaceholdersIntoCall(wrapped_stream,
                                 generic_info.function_bindings);
        wrapped_stream.Printf(
            "\n"
            "    )                                                  \n"
            "  }                                                    \n"
            "}                                                      \n");
        first_body_line = 5;

      } else {
        wrapped_stream.Printf(
            "extension %s$__lldb_context {                            \n"
            "  @LLDBDebuggerFunction                                \n"
            "  %s func $__lldb_wrapped_expr_%u(_ $__lldb_arg : "
            "UnsafeMutablePointer<Any>) {\n"
            "%s" // This is the expression text.  It has all the newlines it
                 // needs.
            "  }                                                    \n"
            "}                                                      \n"
            "func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {    "
            "  \n"
            "  do {                                          \n"
            "    $__lldb_injected_self.$__lldb_wrapped_expr_%u(     \n"
            "      $__lldb_arg                                      \n"
            "    )                                                  \n"
            "  }                                                    \n"
            "}                                                      \n",
            optional_extension, func_decorator, current_counter,
            wrapped_expr_text.GetData(), current_counter);

        first_body_line = 5;
      }
    } else {
      if (generic_info.function_bindings.size()) {
        wrapped_stream.Printf(
            "extension %s$__lldb_context {                            \n"
            "  @LLDBDebuggerFunction                                \n"
            "  %s func $__lldb_wrapped_expr_%u                        ",
            optional_extension, func_decorator, current_counter);
        DumpGenericNames(wrapped_stream, generic_info.function_bindings);
        wrapped_stream.Printf("(_ $__lldb_arg : UnsafeMutablePointer<Any>");
        DumpPlaceholderArguments(wrapped_stream,
                                 generic_info.function_bindings);
        wrapped_stream.Printf(
            ") {\n"
            "%s" // This is the expression text.  It has all the newlines it
                 // needs.
            "  }                                                    \n"
            "}                                                      \n"
            "func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {    "
            "  \n"
            "  do {                                          \n"
            "    $__lldb_injected_self.$__lldb_wrapped_expr_%u(     \n"
            "      $__lldb_arg                                        ",
            wrapped_expr_text.GetData(), current_counter);
        DumpPlaceholdersIntoCall(wrapped_stream,
                                 generic_info.function_bindings);
        wrapped_stream.Printf(
            "\n"
            "    )                                                  \n"
            "  }                                                    \n"
            "}                                                      \n");

        first_body_line = 5;

      } else {
        wrapped_stream.Printf(
            "extension %s$__lldb_context {                            \n"
            "@LLDBDebuggerFunction                                  \n"
            "  %s func $__lldb_wrapped_expr_%u(_ $__lldb_arg : "
            "UnsafeMutablePointer<Any>) {\n"
            "%s" // This is the expression text.  It has all the newlines it
                 // needs.
            "  }                                                    \n"
            "}                                                      \n"
            "func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {    "
            "  \n"
            "  do {                                          \n"
            "    $__lldb_injected_self.$__lldb_wrapped_expr_%u(     \n"
            "      $__lldb_arg                                      \n"
            "    )                                                  \n"
            "  }                                                    \n"
            "}                                                      \n",
            optional_extension, func_decorator, current_counter,
            wrapped_expr_text.GetData(), current_counter);

        first_body_line = 5;
      }
    }
  } else {
    if (generic_info.function_bindings.size()) {
      wrapped_stream.Printf(
          "@LLDBDebuggerFunction                                  \n"
          "func $__lldb_wrapped_expr_%u",
          current_counter);
      DumpGenericNames(wrapped_stream, generic_info.function_bindings);
      wrapped_stream.Printf("(_ $__lldb_arg : UnsafeMutablePointer<Any>");
      DumpPlaceholderArguments(wrapped_stream, generic_info.function_bindings);
      wrapped_stream.Printf(
          ") { \n"
          "%s" // This is the expression text.  It has all the newlines it
               // needs.
          "}                                                      \n"
          "func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {      "
          "\n"
          "  do {                                          \n"
          "    $__lldb_wrapped_expr_%u(                           \n"
          "      $__lldb_arg",
          wrapped_expr_text.GetData(), current_counter);
      DumpPlaceholdersIntoCall(wrapped_stream, generic_info.function_bindings);
      wrapped_stream.Printf(
          "\n"
          "    )                                                  \n"
          "  }                                                    \n"
          "}                                                      \n");
      first_body_line = 4;
    } else {
      wrapped_stream.Printf(
          "@LLDBDebuggerFunction                                  \n"
          "func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {      "
          "\n"
          "%s" // This is the expression text.  It has all the newlines it
               // needs.
          "}                                                      \n",
          wrapped_expr_text.GetData());
      first_body_line = 4;
    }
  }

  // The first source line will be 1 if we used the #sourceLocation directive
  if (!expr_source_path.empty() || (pound_file && pound_line))
    first_body_line = 1;
}

SwiftASTManipulatorBase::VariableMetadataResult::~VariableMetadataResult() {}

SwiftASTManipulatorBase::VariableMetadataError::~VariableMetadataError() {}

void SwiftASTManipulatorBase::VariableInfo::Print(
    lldb_private::Stream &stream) const {
  stream.Printf("[name=%s, type = ", m_name.str().str().c_str());

  if (m_type.IsValid())
    stream.PutCString(m_type.GetTypeName().AsCString("<no type name>"));
  else
    stream.PutCString("<no type>");

  if (MetadataIs<VariableMetadataResult>())
    stream.Printf(", is_result");

  if (MetadataIs<VariableMetadataError>())
    stream.Printf(", is_error");

  stream.PutChar(']');
}

void SwiftASTManipulatorBase::DoInitialization() {
  if (m_repl)
    return;

  static llvm::StringRef s_wrapped_func_prefix_str("$__lldb_wrapped_expr");
  static llvm::StringRef s_func_prefix_str("$__lldb_expr");

  // First pass: find whether we're dealing with a wrapped function or not

  class FuncAndExtensionFinder : public swift::ASTWalker {
  public:
    swift::FuncDecl *m_function_decl = nullptr; // This is the function in which
                                                // the expression code is
                                                // inserted.
    // It is always marked with the DebuggerFunction attribute.
    swift::ExtensionDecl *m_extension_decl =
        nullptr; // This is an optional extension holding the function
    swift::FuncDecl *m_wrapper_decl = nullptr; // This is an optional wrapper
                                               // function that calls
                                               // m_function_decl.
    llvm::StringRef m_wrapper_func_prefix; // This is the prefix name for the
                                           // wrapper function.  One tricky bit
    // is that in the case where there is no wrapper, the m_function_decl
    // has this name.  That's why we check first for the debugger attribute.

    FuncAndExtensionFinder(llvm::StringRef &wrapped_func_prefix)
        : m_wrapper_func_prefix(wrapped_func_prefix) {}

    virtual bool walkToDeclPre(swift::Decl *D) {
      swift::FuncDecl *func_decl = llvm::dyn_cast<swift::FuncDecl>(D);

      if (func_decl) {
        if (func_decl->getAttrs()
                .hasAttribute<swift::LLDBDebuggerFunctionAttr>()) {
          m_function_decl = func_decl;

          // Now walk back up the containing DeclContexts, and if we find an
          // extension Decl, that's our extension:
          swift::DeclContext *cur_ctx = m_function_decl->getDeclContext();
          while (cur_ctx) {
            swift::ExtensionDecl *extension_decl =
                llvm::dyn_cast<swift::ExtensionDecl>(cur_ctx);
            if (extension_decl) {
              m_extension_decl = extension_decl;
              break;
            }
            cur_ctx = cur_ctx->getParent();
          }
        } else if (func_decl->hasName() &&
                   func_decl->getName().str().startswith(m_wrapper_func_prefix))
          m_wrapper_decl = func_decl;

        // There's nothing buried in a function that we need to find in this
        // search.
        return false;
      }
      return true;
    }
  };

  FuncAndExtensionFinder func_finder(s_func_prefix_str);
  m_source_file.walk(func_finder);

  m_function_decl = func_finder.m_function_decl;
  m_wrapper_decl = func_finder.m_wrapper_decl;
  m_extension_decl = func_finder.m_extension_decl;

  assert(m_function_decl);

  // Find the body in the function

  if (m_function_decl) {
    swift::BraceStmt *function_body = m_function_decl->getBody();

    swift::DoCatchStmt *do_stmt = nullptr;

    for (swift::ASTNode &element : function_body->getElements()) {
      if (swift::Stmt *stmt = element.dyn_cast<swift::Stmt *>())
        if ((do_stmt = llvm::dyn_cast<swift::DoCatchStmt>(stmt)))
          break;
    }

    m_do_stmt = do_stmt;
    if (do_stmt) {
      // There should only be one catch:
      assert(m_do_stmt->getCatches().size() == 1);
      swift::CatchStmt *our_catch = m_do_stmt->getCatches().front();
      if (our_catch)
        m_catch_stmt = our_catch;
    }
  }
}

swift::BraceStmt *SwiftASTManipulatorBase::GetUserBody() {
  if (!IsValid())
    return nullptr;

  swift::Stmt *body_stmt = m_do_stmt->getBody();

  swift::BraceStmt *do_body = llvm::dyn_cast<swift::BraceStmt>(body_stmt);

  return do_body;
}

SwiftASTManipulator::SwiftASTManipulator(swift::SourceFile &source_file,
                                         bool repl)
    : SwiftASTManipulatorBase(source_file, repl) {}

void SwiftASTManipulator::FindSpecialNames(
    llvm::SmallVectorImpl<swift::Identifier> &names, llvm::StringRef prefix) {
  names.clear();

  class SpecialNameFinder : public swift::ASTWalker {
  public:
    typedef llvm::SmallVectorImpl<swift::Identifier> NameVector;

    SpecialNameFinder(NameVector &names, llvm::StringRef &prefix)
        : m_names(names), m_prefix(prefix) {}

    virtual std::pair<bool, swift::Expr *> walkToExprPre(swift::Expr *expr) {
      if (swift::UnresolvedDeclRefExpr *decl_ref_expr =
              llvm::dyn_cast<swift::UnresolvedDeclRefExpr>(expr)) {
        swift::Identifier name = decl_ref_expr->getName().getBaseName();

        if (m_prefix.empty() || name.str().startswith(m_prefix))
          m_names.push_back(name);
      }

      return {true, expr};
    }

  private:
    NameVector &m_names;
    llvm::StringRef m_prefix;
  };

  SpecialNameFinder special_name_finder(names, prefix);

  if (m_function_decl)
    m_function_decl->walkContext(special_name_finder);
}

// This call replaces:
//
//      <EXPR>
//
//  with:
//
//       do {
//           var __lldb_tmp_ret_<N> = <EXPR>
//       } while (false)
//
// and adds a "return" in the do-while if in_return is true.
// It records what it has done in a ResultLocationInfo, which gets pushed to the
// back of the ResultLocationInfo stack
// maintained by the SwiftASTManipulator, and returns the statement which
// effects the change.
//
// May return NULL if we can't make an appropriate variable assignment (e.g. for
// a bare "nil".)

swift::Stmt *SwiftASTManipulator::ConvertExpressionToTmpReturnVarAccess(
    swift::Expr *expr, const swift::SourceLoc &source_loc, bool in_return,
    swift::DeclContext *decl_context) {
  // swift doesn't know how to infer the type of a variable by assignment to
  // "nil".  So if the
  // expression is "nil" then we just drop it on the floor.
  if (swift::dyn_cast<swift::NilLiteralExpr>(expr))
    return nullptr;

  swift::ASTContext &ast_context = m_source_file.getASTContext();
  char name_buffer[64];
  snprintf(name_buffer, 64, "__lldb_tmp_ret_%d", m_tmpname_idx++);
  swift::Identifier name = ast_context.getIdentifier(name_buffer);
  swift::Identifier equalequal_name = ast_context.getIdentifier("==");

  ResultLocationInfo result_loc_info(source_loc);
  result_loc_info.orig_expr = expr;

  swift::DeclContext *new_decl_context = m_function_decl;

  if (m_repl) {
    new_decl_context = decl_context;
  }

  llvm::SmallVector<swift::ASTNode, 3> body;
  llvm::SmallVector<swift::Expr *, 3> false_body;
  const bool is_static = false;
  const bool is_let = false;
  result_loc_info.tmp_var_decl = new (ast_context) swift::VarDecl(
      is_static, is_let, source_loc, name, swift::Type(), new_decl_context);
  result_loc_info.tmp_var_decl->setImplicit();
  result_loc_info.tmp_var_decl->setAccessibility(
      swift::Accessibility::Internal);
  result_loc_info.tmp_var_decl->setSetterAccessibility(
      swift::Accessibility::Internal);

  swift::NamedPattern *var_pattern =
      new (ast_context) swift::NamedPattern(result_loc_info.tmp_var_decl, true);

  const swift::StaticSpellingKind static_spelling_kind =
      swift::StaticSpellingKind::KeywordStatic;
  result_loc_info.binding_decl = swift::PatternBindingDecl::create(
      ast_context, source_loc, static_spelling_kind, source_loc, var_pattern,
      expr, new_decl_context);
  result_loc_info.binding_decl->setImplicit();
  result_loc_info.binding_decl->setStatic(false);

  body.push_back(result_loc_info.binding_decl);
  body.push_back(result_loc_info.tmp_var_decl);

  if (in_return) {
    result_loc_info.return_stmt =
        new (ast_context) swift::ReturnStmt(source_loc, nullptr);
    body.push_back(result_loc_info.return_stmt);
  }
  swift::IntegerLiteralExpr *one_expr = new (ast_context)
      swift::IntegerLiteralExpr(swift::StringRef("1"), source_loc, true);
  false_body.push_back(one_expr);
  swift::UnresolvedDeclRefExpr *equalequal_expr = new (ast_context)
      swift::UnresolvedDeclRefExpr(equalequal_name,
                                   swift::DeclRefKind::BinaryOperator,
                                   swift::DeclNameLoc(source_loc));
  false_body.push_back(equalequal_expr);
  swift::IntegerLiteralExpr *zero_expr = new (ast_context)
      swift::IntegerLiteralExpr(swift::StringRef("0"), source_loc, true);
  false_body.push_back(zero_expr);
  swift::SequenceExpr *zero_equals_one_expr = swift::SequenceExpr::create(
      ast_context, llvm::ArrayRef<swift::Expr *>(false_body));

  zero_equals_one_expr->setImplicit();
  swift::BraceStmt *body_stmt = swift::BraceStmt::create(
      ast_context, source_loc, llvm::ArrayRef<swift::ASTNode>(body), source_loc,
      true);

  // Default construct a label info that contains nothing for the while
  // statement
  swift::LabeledStmtInfo label_info;

  swift::RepeatWhileStmt *assign_stmt = new (ast_context)
      swift::RepeatWhileStmt(label_info, source_loc, zero_equals_one_expr,
                             source_loc, body_stmt, true);
  result_loc_info.wrapper_stmt = assign_stmt;

  m_result_info.push_back(result_loc_info);
  return assign_stmt;
}

bool SwiftASTManipulator::RewriteResult() {
  class ReturnFinder : public swift::ASTWalker {
  public:
    ReturnFinder(SwiftASTManipulator &manipulator)
        : m_manipulator(manipulator) {}

    void SetDeclContext(swift::DeclContext *decl_context) {
      m_decl_context = decl_context;
    }

    virtual bool walkToDeclPre(swift::Decl *decl) {
      // Don't step into function declarations, they may have returns, but we
      // don't want
      // to instrument them.
      swift::DeclKind kind = decl->getKind();
      switch (kind) {
      case swift::DeclKind::Func:
      case swift::DeclKind::Class:
      case swift::DeclKind::Struct:
        return false;
      default:
        return true;
      }
    }

    virtual std::pair<bool, swift::Expr *> walkToExprPre(swift::Expr *expr) {
      // Don't step into closure definitions, they may have returns, but we
      // don't want
      // to instrument them either.
      swift::ExprKind kind = expr->getKind();
      if (kind == swift::ExprKind::Closure)
        return {false, expr};
      else
        return {true, expr};
    }

    virtual swift::Stmt *walkToStmtPost(swift::Stmt *stmt) {
      swift::ReturnStmt *possible_return =
          swift::dyn_cast<swift::ReturnStmt>(stmt);
      if (possible_return && possible_return->hasResult()) {
        swift::Expr *return_expr = possible_return->getResult();
        if (return_expr) {
          const bool add_return = true;
          swift::Stmt *return_stmt;

          return_stmt = m_manipulator.ConvertExpressionToTmpReturnVarAccess(
              return_expr, possible_return->getStartLoc(), add_return,
              m_decl_context);
          if (return_stmt)
            stmt = return_stmt;
        }
      }
      return stmt;
    }

  private:
    SwiftASTManipulator &m_manipulator;
    swift::DeclContext *m_decl_context = nullptr;
  };

  if (!IsValid())
    return false;

  if (m_repl) {
    ReturnFinder return_finder(*this);

    // First step, walk the function body converting returns to assignments to
    // temp variables + return:

    for (swift::Decl *decl : m_source_file.Decls) {
      if (auto top_level_code_decl =
              llvm::dyn_cast<swift::TopLevelCodeDecl>(decl)) {
        return_finder.SetDeclContext(top_level_code_decl);
        top_level_code_decl->getBody()->walk(return_finder);
      }
    }

    // Second step, fetch the last expression, and if it is non-null, set it to
    // a temp result as well:

    if (!m_source_file.Decls.empty()) {
      swift::Decl *last_decl = *(m_source_file.Decls.end() - 1);

      if (auto last_top_level_code_decl =
              llvm::dyn_cast<swift::TopLevelCodeDecl>(last_decl)) {
        llvm::MutableArrayRef<swift::ASTNode>::iterator back_iterator;

        back_iterator =
            last_top_level_code_decl->getBody()->getElements().end() - 1;
        swift::ASTNode last_element = *back_iterator;

        swift::Expr *last_expr = last_element.dyn_cast<swift::Expr *>();

        if (last_expr) {
          swift::Stmt *temp_result_decl = ConvertExpressionToTmpReturnVarAccess(
              last_expr, last_expr->getStartLoc(), false,
              last_top_level_code_decl);
          if (temp_result_decl)
            *back_iterator = temp_result_decl;
        }
      }
    }
  } else {
    swift::BraceStmt *user_body = GetUserBody();

    llvm::MutableArrayRef<swift::ASTNode> orig_elements =
        user_body->getElements();
    llvm::SmallVector<swift::Expr *, 1> return_values;

    // The function body is wrapped in an "if (true)" when constructed, so the
    // function body can not be empty
    // or it was one we didn't make (or the optimizer is getting smart on us
    // when it has no business doing that.)
    if (orig_elements.size() == 0) {
      // This is an empty expression, nothing to do here...
      return true;
    }

    // First step, walk the function body converting returns to assignments to
    // temp variables + return:
    ReturnFinder return_finder(*this);
    user_body->walk(return_finder);

    // Second step, fetch the last expression, and if it is non-null, set it to
    // a temp result as well:

    llvm::MutableArrayRef<swift::ASTNode>::iterator back_iterator;
    back_iterator = user_body->getElements().end() - 1;
    swift::ASTNode last_element = *back_iterator;

    swift::Expr *last_expr = last_element.dyn_cast<swift::Expr *>();

    if (last_expr) {
      swift::Stmt *temp_result_decl = ConvertExpressionToTmpReturnVarAccess(
          last_expr, last_expr->getStartLoc(), false, nullptr);
      if (temp_result_decl)
        *back_iterator = temp_result_decl;
    }
  }

  return true;
}

namespace {
class AssignmentMaker {
private:
  llvm::SmallSet<swift::VarDecl *, 1> &m_persistent_vars;
  swift::ASTContext &m_ast_context;
  llvm::SmallVector<swift::ASTNode, 3> &m_elements;
  llvm::SmallVectorImpl<swift::ASTNode>::iterator &m_ei;

public:
  void MakeOneAssignment(swift::VarDecl *var_decl, swift::Expr *initializer,
                         swift::SourceLoc location) {
    if (!m_persistent_vars.count(var_decl))
      return;

    swift::Type target_type = var_decl->getType();
    swift::LValueType *target_lvalue_type = swift::LValueType::get(target_type);

    const bool implicit = true;
    const swift::AccessSemantics uses_direct_property_access =
        swift::AccessSemantics::Ordinary;

    swift::DeclRefExpr *decl_ref = new (m_ast_context)
        swift::DeclRefExpr(var_decl, swift::DeclNameLoc(location), implicit,
                           uses_direct_property_access, target_lvalue_type);

    swift::AssignExpr *assignment = new (m_ast_context)
        swift::AssignExpr(decl_ref, location, initializer, implicit);

    assignment->setType(m_ast_context.TheEmptyTupleType);

    llvm::SmallVectorImpl<swift::ASTNode>::iterator next_iter = m_ei + 1;

    swift::ASTNode assignment_node((swift::Expr *)assignment);

    m_ei = m_elements.insert(next_iter, swift::ASTNode(assignment_node));
  }

  AssignmentMaker(llvm::SmallSet<swift::VarDecl *, 1> &persistent_vars,
                  swift::ASTContext &ast_context,
                  llvm::SmallVector<swift::ASTNode, 3> &elements,
                  llvm::SmallVectorImpl<swift::ASTNode>::iterator &ei)
      : m_persistent_vars(persistent_vars), m_ast_context(ast_context),
        m_elements(elements), m_ei(ei) {}
};
}

void SwiftASTManipulator::MakeDeclarationsPublic() {
  if (!IsValid())
    return;

  class Publicist : public swift::ASTWalker {
    virtual bool walkToDeclPre(swift::Decl *decl) {
      if (swift::ValueDecl *value_decl =
              llvm::dyn_cast<swift::ValueDecl>(decl)) {
        value_decl->overwriteAccessibility(swift::Accessibility::Public);
        if (swift::AbstractStorageDecl *var_decl =
                llvm::dyn_cast<swift::AbstractStorageDecl>(decl))
          var_decl->overwriteSetterAccessibility(swift::Accessibility::Public);
      }

      return true;
    }
  };

  Publicist p;

  for (swift::Decl *decl : m_source_file.Decls) {
    decl->walk(p);
  }
}

static bool hasInit(swift::PatternBindingDecl *pattern_binding) {
  for (unsigned i = 0, e = pattern_binding->getNumPatternEntries(); i != e; ++i)
    if (pattern_binding->getInit(i))
      return true;
  return false;
}

static swift::Expr *getFirstInit(swift::PatternBindingDecl *pattern_binding) {
  for (unsigned i = 0, e = pattern_binding->getNumPatternEntries(); i != e; ++i)
    if (pattern_binding->getInit(i))
      return pattern_binding->getInit(i);
  return nullptr;
}

bool SwiftASTManipulator::CheckPatternBindings() {
  for (swift::Decl *top_level_decl : m_source_file.Decls) {
    if (swift::TopLevelCodeDecl *top_level_code =
            llvm::dyn_cast<swift::TopLevelCodeDecl>(top_level_decl)) {
      for (swift::ASTNode &node : top_level_code->getBody()->getElements()) {
        if (swift::Decl *decl = node.dyn_cast<swift::Decl *>()) {
          if (swift::PatternBindingDecl *pattern_binding =
                  llvm::dyn_cast<swift::PatternBindingDecl>(decl)) {
            if (!(pattern_binding->isImplicit() || hasInit(pattern_binding))) {
              m_source_file.getASTContext().Diags.diagnose(
                  pattern_binding->getStartLoc(),
                  swift::diag::repl_must_be_initialized);

              return false;
            }
          }
        }
      }
    }
  }

  return true;
}
void SwiftASTManipulator::FindVariableDeclarations(
    llvm::SmallVectorImpl<size_t> &found_declarations, bool repl) {
  if (!IsValid())
    return;

  auto register_one_var = [this,
                           &found_declarations](swift::VarDecl *var_decl) {
    VariableInfo persistent_info;

    swift::Identifier name = var_decl->getName();

    size_t persistent_info_location = m_variables.size();

    persistent_info.m_name = name;
    persistent_info.m_type = CompilerType(&var_decl->getASTContext(),
                                          var_decl->getType().getPointer());
    persistent_info.m_decl = var_decl;

    m_variables.push_back(persistent_info);

    found_declarations.push_back(persistent_info_location);
  };

  if (m_repl) {
    for (swift::Decl *decl : m_source_file.Decls) {
      if (swift::VarDecl *var_decl = llvm::dyn_cast<swift::VarDecl>(decl)) {
        if (!var_decl->getName().str().startswith("$")) {
          register_one_var(var_decl);
        }
      }
    }
  } else {
    swift::BraceStmt *user_body = GetUserBody();

    llvm::ArrayRef<swift::ASTNode> body_elements = user_body->getElements();

    llvm::SmallVector<swift::ASTNode, 3> elements(body_elements.begin(),
                                                  body_elements.end());

    for (swift::ASTNode &element : elements) {
      if (swift::Decl *element_decl = element.dyn_cast<swift::Decl *>()) {
        if (swift::VarDecl *var_decl =
                llvm::dyn_cast<swift::VarDecl>(element_decl)) {
          if (!var_decl->isDebuggerVar()) // skip bona fide external variables
                                          // or variables we've already tagged
          {
            swift::Identifier name = var_decl->getName();

            if (name.str().startswith("$")) {
              var_decl->setDebuggerVar(true);
              register_one_var(var_decl);
            }
          }
        }
      }
    }
  }
}

void SwiftASTManipulator::FindNonVariableDeclarations(
    llvm::SmallVectorImpl<swift::ValueDecl *> &non_variables) {
  if (!IsValid())
    return;

  if (!m_repl)
    return; // we don't do this for non-REPL expressions... yet

  for (swift::Decl *decl : m_source_file.Decls) {
    if (swift::ValueDecl *value_decl = llvm::dyn_cast<swift::ValueDecl>(decl)) {
      if (!llvm::isa<swift::VarDecl>(value_decl) && value_decl->hasName()) {
        non_variables.push_back(value_decl);
      }
    }
  }
}

void SwiftASTManipulator::InsertResult(
    swift::VarDecl *result_var, swift::Type &result_type,
    SwiftASTManipulator::ResultLocationInfo &result_info) {
  swift::ASTContext &ast_context = m_source_file.getASTContext();

  CompilerType return_ast_type(&ast_context, result_type.getPointer());

  result_var->overwriteAccessibility(swift::Accessibility::Public);
  result_var->overwriteSetterAccessibility(swift::Accessibility::Public);

  // Finally, go reset the return expression to the new result variable for each
  // of the return expressions.

  // Make an LValueType of our result type for use in the assign expression.
  swift::LValueType *lvalue_result = swift::LValueType::get(result_type);

  // QUERY: Can I just make one of the LHS decl's and reuse it for all the
  // assigns?
  const swift::AccessSemantics uses_direct_property_access =
      swift::AccessSemantics::Ordinary;
  swift::DeclRefExpr *lhs_expr = new (ast_context)
      swift::DeclRefExpr(result_var, swift::DeclNameLoc(result_info.source_loc),
                         true, uses_direct_property_access, lvalue_result);

  swift::Expr *init_expr = getFirstInit(result_info.binding_decl);
  swift::AssignExpr *assign_expr = new (ast_context)
      swift::AssignExpr(lhs_expr, result_info.source_loc, init_expr, true);
  assign_expr->setType(ast_context.TheEmptyTupleType);

  llvm::SmallVector<swift::ASTNode, 2> new_body;
  new_body.push_back(assign_expr);
  if (result_info.return_stmt != nullptr)
    new_body.push_back(result_info.return_stmt);
  swift::BraceStmt *body_stmt = swift::BraceStmt::create(
      ast_context, result_info.source_loc,
      llvm::ArrayRef<swift::ASTNode>(new_body), result_info.source_loc, true);
  result_info.wrapper_stmt->setBody(body_stmt);
}

void SwiftASTManipulator::InsertError(swift::VarDecl *error_var,
                                      swift::Type &error_type) {
  if (!m_do_stmt)
    return;

  swift::ASTContext &ast_context = m_source_file.getASTContext();

  CompilerType error_ast_type(&ast_context, error_type.getPointer());

  error_var->overwriteAccessibility(swift::Accessibility::Public);
  error_var->overwriteSetterAccessibility(swift::Accessibility::Public);

  // Finally, go reset the return expression to the new result variable for each
  // of the return expressions.

  // Make an LValueType of our result type for use in the assign expression.
  swift::LValueType *lvalue_result = swift::LValueType::get(error_type);

  // QUERY: Can I just make one of the LHS decl's and reuse it for all the
  // assigns?
  swift::SourceLoc error_loc = m_do_stmt->getBody()->getStartLoc();

  const swift::AccessSemantics uses_direct_property_access =
      swift::AccessSemantics::Ordinary;
  swift::DeclRefExpr *lhs_expr = new (ast_context)
      swift::DeclRefExpr(error_var, swift::DeclNameLoc(error_loc), true,
                         uses_direct_property_access, lvalue_result);

  swift::BraceStmt *catch_body =
      llvm::dyn_cast<swift::BraceStmt>(m_catch_stmt->getBody());
  if (!catch_body) {
    // Fixme - log this error somehow.
    return;
  }
  llvm::ArrayRef<swift::ASTNode> body_elements = catch_body->getElements();

  llvm::SmallVector<swift::ASTNode, 3> elements(body_elements.begin(),
                                                body_elements.end());
  swift::PatternBindingDecl *binding_decl = nullptr;

  for (swift::ASTNode &element : elements) {
    if (swift::Decl *element_decl = element.dyn_cast<swift::Decl *>()) {
      binding_decl = llvm::dyn_cast<swift::PatternBindingDecl>(element_decl);
      if (binding_decl)
        break;
    }
  }

  swift::Expr *init_expr = getFirstInit(binding_decl);
  swift::AssignExpr *assign_expr =
      new (ast_context) swift::AssignExpr(lhs_expr, error_loc, init_expr, true);
  assign_expr->setType(ast_context.TheEmptyTupleType);

  llvm::SmallVector<swift::ASTNode, 2> new_body;
  new_body.push_back(assign_expr);

  swift::BraceStmt *body_stmt = swift::BraceStmt::create(
      ast_context, error_loc, llvm::ArrayRef<swift::ASTNode>(new_body),
      error_loc, true);
  m_catch_stmt->setBody(body_stmt);
}

bool SwiftASTManipulator::FixupResultAfterTypeChecking(Error &error) {
  if (!IsValid()) {
    error.SetErrorString("Operating on invalid SwiftASTManipulator");
    return false;
  }
  // Run through the result decls and figure out the return type.

  size_t num_results = m_result_info.size();
  if (num_results == 0)
    return true;

  swift::Type result_type;
  for (size_t i = 0; i < num_results; i++) {
    swift::VarDecl *the_decl = m_result_info[i].tmp_var_decl;
    if (the_decl->hasType()) {
      swift::Type its_type = the_decl->getType();
      if (result_type.isNull()) {
        result_type = its_type;
      } else if (!its_type.getPointer()->isEqual(result_type)) {
        std::string prev_type_name = result_type.getPointer()->getString();
        std::string cur_type_name = its_type.getPointer()->getString();

        error.SetErrorStringWithFormat(
            "Type for %zuth return value is inconsistent, previous type: %s, "
            "current type: %s.",
            i, prev_type_name.c_str(), cur_type_name.c_str());
        return false;
      }
    } else {
      error.SetErrorStringWithFormat(
          "Type of %zuth return value could not be determined.", i);
      return false;
    }
  }

  if (result_type.isNull()) {
    error.SetErrorString("Could not find the result type for this expression.");
    return false;
  } else if (result_type->is<swift::ErrorType>()) {
    error.SetErrorString("Result type is the error type.");
    return false;
  }

  swift::ASTContext &ast_context = m_source_file.getASTContext();

  CompilerType return_ast_type(&ast_context, result_type.getPointer());
  swift::Identifier result_var_name =
      ast_context.getIdentifier(GetResultName());
  SwiftASTManipulatorBase::VariableMetadataSP metadata_sp(
      new VariableMetadataResult());

  swift::VarDecl *result_var =
      AddExternalVariable(result_var_name, return_ast_type, metadata_sp);

  result_var->overwriteAccessibility(swift::Accessibility::Public);
  result_var->overwriteSetterAccessibility(swift::Accessibility::Public);

  // Finally, go reset the return expression to the new result variable for each
  // of the return expressions.

  for (SwiftASTManipulator::ResultLocationInfo &result_info : m_result_info) {
    InsertResult(result_var, result_type, result_info);
  }

  // Finally we have to do pretty much the same transformation on the error
  // object.
  // First we need to find it:
  if (m_catch_stmt) {
    // Search for the error variable, so we can read it and its type,
    // then call InsertError to replace it with an assignment to the error
    // variable.
    swift::BraceStmt *catch_body =
        llvm::dyn_cast<swift::BraceStmt>(m_catch_stmt->getBody());
    llvm::ArrayRef<swift::ASTNode> body_elements = catch_body->getElements();

    llvm::SmallVector<swift::ASTNode, 3> elements(body_elements.begin(),
                                                  body_elements.end());

    for (swift::ASTNode &element : elements) {
      if (swift::Decl *element_decl = element.dyn_cast<swift::Decl *>()) {
        if (swift::VarDecl *var_decl =
                llvm::dyn_cast<swift::VarDecl>(element_decl)) {
          if (var_decl->hasType()) {
            swift::Identifier error_var_name =
                ast_context.getIdentifier(GetErrorName());
            if (error_var_name != var_decl->getName())
              continue;

            swift::Type error_type = var_decl->getType();
            CompilerType error_ast_type(&ast_context, error_type.getPointer());
            SwiftASTManipulatorBase::VariableMetadataSP error_metadata_sp(
                new VariableMetadataError());

            swift::VarDecl *error_var = AddExternalVariable(
                error_var_name, error_ast_type, error_metadata_sp);

            error_var->overwriteAccessibility(swift::Accessibility::Public);
            error_var->overwriteSetterAccessibility(
                swift::Accessibility::Public);

            InsertError(error_var, error_type);
            break;
          }
        }
      }
    }
  }

  return true;
}

swift::VarDecl *
SwiftASTManipulator::AddExternalVariable(swift::Identifier name,
                                         CompilerType &type,
                                         VariableMetadataSP &metadata_sp) {
  if (!IsValid())
    return nullptr;

  VariableInfo variables[1];

  variables[0].m_name = name;
  variables[0].m_type = type;
  variables[0].m_metadata = metadata_sp;

  if (!AddExternalVariables(variables))
    return nullptr;

  return variables[0].m_decl;
}

static swift::PatternBindingDecl *
GetPatternBindingForVarDecl(swift::VarDecl *var_decl,
                            swift::DeclContext *containing_context) {
  swift::ASTContext &ast_context = var_decl->getASTContext();

  const bool is_implicit = true;

  swift::NamedPattern *named_pattern =
      new (ast_context) swift::NamedPattern(var_decl, is_implicit);

  swift::TypedPattern *typed_pattern = new (ast_context) swift::TypedPattern(
      named_pattern, swift::TypeLoc::withoutLoc(var_decl->getType()));

  swift::PatternBindingDecl *pattern_binding =
      swift::PatternBindingDecl::create(
          ast_context, swift::SourceLoc(), swift::StaticSpellingKind::None,
          var_decl->getLoc(), typed_pattern, nullptr, containing_context);
  pattern_binding->setImplicit(true);

  return pattern_binding;
}

static inline swift::Type GetSwiftType(CompilerType type) {
  return swift::Type(
      reinterpret_cast<swift::TypeBase *>(type.GetOpaqueQualType()));
}

bool SwiftASTManipulator::AddExternalVariables(
    llvm::MutableArrayRef<VariableInfo> variables) {
  if (!IsValid())
    return false;

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  swift::ASTContext &ast_context = m_source_file.getASTContext();

  if (m_repl) {
    // In the REPL, we're only adding the result variable.

    if (variables.empty()) {
      return true;
    }

    assert(variables.size() == 1);

    SwiftASTManipulator::VariableInfo &variable = variables[0];

    const bool is_static = false;
    bool is_let = variable.GetIsLet();
    swift::SourceLoc loc;
    swift::Identifier name = variable.m_name;
    swift::Type var_type = GetSwiftType(variable.m_type);

    // If the type is an inout or lvalue type (happens if this is an argument)
    // strip that part off:

    swift::VarDecl *redirected_var_decl = new (ast_context)
        swift::VarDecl(is_static, is_let, loc, name, var_type, &m_source_file);

    swift::TopLevelCodeDecl *top_level_code =
        new (ast_context) swift::TopLevelCodeDecl(&m_source_file);

    swift::PatternBindingDecl *pattern_binding =
        GetPatternBindingForVarDecl(redirected_var_decl, top_level_code);

    swift::ASTNode elements[] = {pattern_binding};

    swift::BraceStmt *brace_stmt =
        swift::BraceStmt::create(ast_context, loc, elements, loc, true);

    top_level_code->setBody(brace_stmt);

    redirected_var_decl->setImplicit(true);

    m_source_file.Decls.insert(m_source_file.Decls.begin(), top_level_code);
    m_source_file.Decls.insert(m_source_file.Decls.begin(),
                               redirected_var_decl);

    variable.m_decl = redirected_var_decl;

    if (log) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      variable.m_decl->dump(ss);
      ss.flush();

      log->Printf(
          "[SwiftASTManipulator::AddExternalVariables] Injected variable %s",
          s.c_str());
    }

    m_variables.push_back(variable);
  } else {
    swift::BraceStmt *body = m_function_decl->getBody();
    llvm::ArrayRef<swift::ASTNode> body_elements = body->getElements();

    llvm::SmallVector<swift::ASTNode, 3> elements(body_elements.begin(),
                                                  body_elements.end());
    llvm::SmallVectorImpl<swift::ASTNode>::iterator element_iterator =
        elements.begin();
    const bool is_static = false;

    for (SwiftASTManipulator::VariableInfo &variable : variables) {
      swift::SourceLoc loc = m_function_decl->getBody()->getLBraceLoc();
      swift::FuncDecl *containing_function = m_function_decl;
      swift::Identifier name = variable.m_name;
      bool is_let = variable.GetIsLet();

      bool is_self = !variable.m_name.str().compare("$__lldb_injected_self");

      if (is_self) {
        if (!m_wrapper_decl)
          continue;

        loc = m_wrapper_decl->getBody()->getLBraceLoc();
        containing_function = m_wrapper_decl;
      }

      // This might be a referenced type, which will confuse the type checker.
      // The access pattern for these types is the same as for the referent
      // type, so it is fine to
      // just strip it off.
      // FIXME: If this is a weak managed type, then it could ostensibly go away
      // out from under us,
      // but for now we aren't playing with reference counts to keep things
      // alive in the expression parser.
      SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(
          variable.m_type.GetTypeSystem());

      CompilerType referent_type;

      if (swift_ast_ctx)
        referent_type = swift_ast_ctx->GetReferentType(variable.m_type);

      // One tricky bit here is that this var may be an argument to the function
      // whose context we are
      // emulating, and that argument might be of "inout" type.  We need to
      // strip the inout off the type
      // or the initial parse will fail.  Fortunately, the variable access goes
      // the same regardless of whether
      // it is inout or not, so we don't have to do anything more to get this to
      // work.
      swift::Type var_type =
          GetSwiftType(referent_type)->getLValueOrInOutObjectType();
      if (is_self) {
        // Another tricky bit is that the Metatype types we get have the
        // "Representation" already attached (i.e.
        // "@thick", "@thin".)  But the representation is a SIL level thing, and
        // if it is attached to types that
        // we hand the parser, it throws a verifier error & aborts.  So we strip
        // it off here:
        swift::MetatypeType *metatype_type =
            llvm::dyn_cast<swift::MetatypeType>(var_type.getPointer());
        if (metatype_type) {
          var_type = swift::Type(
              swift::MetatypeType::get(metatype_type->getInstanceType()));
        }
      }

      swift::VarDecl *redirected_var_decl = new (ast_context) swift::VarDecl(
          is_static, is_let, loc, name, var_type, containing_function);
      redirected_var_decl->setDebuggerVar(true);
      redirected_var_decl->setImplicit(true);

      swift::PatternBindingDecl *pattern_binding =
          GetPatternBindingForVarDecl(redirected_var_decl, containing_function);

      if (var_type->getAs<swift::WeakStorageType>()) {
        redirected_var_decl->getAttrs().add(
            new (ast_context) swift::OwnershipAttr(swift::SourceRange(),
                                                   swift::Ownership::Weak));
      }

      if (is_self) {
        // we need to inject into the wrapper

        swift::BraceStmt *wrapper_body = m_wrapper_decl->getBody();
        llvm::ArrayRef<swift::ASTNode> wrapper_elements =
            wrapper_body->getElements();

        llvm::SmallVector<swift::ASTNode, 3> wrapper_elements_copy(
            wrapper_elements.begin(), wrapper_elements.end());
        llvm::SmallVectorImpl<swift::ASTNode>::iterator
            wrapper_element_iterator = wrapper_elements_copy.begin();

        wrapper_element_iterator = wrapper_elements_copy.insert(
            wrapper_element_iterator, swift::ASTNode(pattern_binding));
        wrapper_element_iterator = wrapper_elements_copy.insert(
            wrapper_element_iterator, swift::ASTNode(redirected_var_decl));

        m_wrapper_decl->setBody(swift::BraceStmt::create(
            ast_context, wrapper_body->getLBraceLoc(),
            ast_context.AllocateCopy(wrapper_elements_copy),
            wrapper_body->getRBraceLoc()));
      } else {
        element_iterator =
            elements.insert(element_iterator, swift::ASTNode(pattern_binding));
        element_iterator = elements.insert(element_iterator,
                                           swift::ASTNode(redirected_var_decl));
      }

      variable.m_decl = redirected_var_decl;

      if (log) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        variable.m_decl->dump(ss);
        ss.flush();

        log->Printf(
            "[SwiftASTManipulator::AddExternalVariables] Injected variable %s",
            s.c_str());
      }

      m_variables.push_back(variable);
    }

    m_function_decl->setBody(swift::BraceStmt::create(
        ast_context, body->getLBraceLoc(), ast_context.AllocateCopy(elements),
        body->getRBraceLoc()));
  }

  return true;
}

static void AppendToCaptures(swift::ASTContext &ast_context,
                             swift::FuncDecl *func_decl,
                             swift::VarDecl *capture_decl) {
  llvm::ArrayRef<swift::CapturedValue> old_captures =
      func_decl->getCaptureInfo().getCaptures();
  llvm::SmallVector<swift::CapturedValue, 2> captures(old_captures.begin(),
                                                      old_captures.end());

  captures.push_back(swift::CapturedValue(capture_decl, 0));

  func_decl->getCaptureInfo().setCaptures(ast_context.AllocateCopy(captures));
}

static swift::VarDecl *FindArgInFunction(swift::ASTContext &ast_context,
                                         swift::FuncDecl *func_decl) {
  auto name = ast_context.getIdentifier("$__lldb_arg");

  for (auto *paramList : func_decl->getParameterLists()) {
    for (auto param : *paramList)
      if (param->getName() == name)
        return param;
  }

  return nullptr;
}

bool SwiftASTManipulator::FixCaptures() {
  if (!IsValid())
    return false;

  swift::ASTContext &ast_context = m_source_file.getASTContext();

  swift::VarDecl *function_arg_decl =
      FindArgInFunction(ast_context, m_function_decl);
  swift::VarDecl *wrapper_arg_decl = nullptr;

  if (m_wrapper_decl)
    wrapper_arg_decl = FindArgInFunction(ast_context, m_wrapper_decl);

  if (!function_arg_decl)
    return false;

  if (m_wrapper_decl && (!wrapper_arg_decl))
    return false;

  for (VariableInfo &variable : m_variables) {
    if (!variable.m_decl)
      continue;

    if (variable.m_decl->getStorageKind() !=
        swift::AbstractStorageDecl::Computed)
      continue;

    swift::FuncDecl *getter_decl = variable.m_decl->getGetter();
    swift::FuncDecl *setter_decl = variable.m_decl->getSetter();

    swift::DeclContext *decl_context = variable.m_decl->getDeclContext();

    if (decl_context == (swift::DeclContext *)m_function_decl) {
      AppendToCaptures(ast_context, getter_decl, function_arg_decl);
      AppendToCaptures(ast_context, setter_decl, function_arg_decl);
    } else if (decl_context == (swift::DeclContext *)m_wrapper_decl) {
      AppendToCaptures(ast_context, getter_decl, wrapper_arg_decl);
      AppendToCaptures(ast_context, setter_decl, wrapper_arg_decl);
    } else {
      return false;
    }
  }

  return true;
}

swift::ValueDecl *SwiftASTManipulator::MakeGlobalTypealias(
    swift::Identifier name, CompilerType &type, bool make_private) {
  if (!IsValid())
    return nullptr;

  swift::SourceLoc source_loc;

  if (m_extension_decl)
    source_loc = m_extension_decl->getEndLoc();
  else
    source_loc = m_function_decl->getEndLoc();

  swift::ASTContext &ast_context = m_source_file.getASTContext();

  llvm::MutableArrayRef<swift::TypeLoc> inherited;
  swift::TypeAliasDecl *type_alias_decl = new (ast_context)
      swift::TypeAliasDecl(source_loc, name, source_loc,
                           swift::TypeLoc::withoutLoc(GetSwiftType(type)),
                           nullptr, &m_source_file);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));
  if (log) {

    std::string s;
    llvm::raw_string_ostream ss(s);
    type_alias_decl->dump(ss);
    ss.flush();

    log->Printf("Made global type alias for %s (%p) in context (%p):\n%s",
                name.get(), GetSwiftType(type).getPointer(), &ast_context,
                s.c_str());
  }

  if (type_alias_decl) {
    if (make_private) {
      type_alias_decl->overwriteAccessibility(swift::Accessibility::Private);
    }
    m_source_file.Decls.push_back(type_alias_decl);
  }

  return type_alias_decl;
}

SwiftASTManipulator::TypesForResultFixup
SwiftASTManipulator::GetTypesForResultFixup(uint32_t language_flags) {
  TypesForResultFixup ret;

  for (swift::Decl *decl : m_source_file.Decls) {
    if (auto extension_decl = llvm::dyn_cast<swift::ExtensionDecl>(decl)) {
      if (language_flags & SwiftUserExpression::eLanguageFlagIsWeakSelf) {
        if (extension_decl->getGenericParams() &&
            extension_decl->getGenericParams()->getParams().size() == 1) {
          swift::GenericTypeParamDecl *type_parameter =
              extension_decl->getGenericParams()->getParams()[0];
          swift::NameAliasType *name_alias_type =
              llvm::dyn_cast_or_null<swift::NameAliasType>(
                  type_parameter->getSuperclass().getPointer());

          if (name_alias_type) {
            // FIXME: What if the generic parameter is concrete?
            ret.Wrapper_archetype = swift::ArchetypeBuilder::mapTypeIntoContext(
                extension_decl, type_parameter->getDeclaredInterfaceType())
                    ->castTo<swift::ArchetypeType>();
            ret.context_alias = name_alias_type;
            ret.context_real = name_alias_type->getSinglyDesugaredType();
          }
        }
      } else if (!ret.context_alias) {
        swift::NameAliasType *name_alias_type =
            llvm::dyn_cast<swift::NameAliasType>(
                extension_decl->getExtendedType().getPointer());

        if (name_alias_type) {
          ret.context_alias = name_alias_type;
          ret.context_real = name_alias_type->getSinglyDesugaredType();
        }
      }
    }
  }

  return ret;
}

static swift::Type ReplaceInType(swift::Type orig, swift::TypeBase *from,
                                 swift::TypeBase *to) {
  std::function<swift::Type(swift::Type)> Replacer =
      [from, to](swift::Type orig_type) {
        if (orig_type.getPointer() == from) {
          return swift::Type(to);
        } else {
          return orig_type;
        }
      };

  return orig.transform(Replacer);
}

swift::Type SwiftASTManipulator::FixupResultType(swift::Type &result_type,
                                                 uint32_t language_flags) {
  TypesForResultFixup result_fixup_types =
      GetTypesForResultFixup(language_flags);

  if (result_fixup_types.Wrapper_archetype && result_fixup_types.context_real) {
    result_type =
        ReplaceInType(result_type, result_fixup_types.Wrapper_archetype,
                      result_fixup_types.context_real);
  }

  if (result_fixup_types.context_alias && result_fixup_types.context_real) {
    // This is what we ought to do, but the printing logic doesn't handle the
    // resulting types properly yet.
    // result_type = ReplaceInType(result_type,
    // result_fixup_types.context_alias, result_fixup_types.context_real);
    if (result_type.getPointer() == result_fixup_types.context_alias) {
      result_type = result_fixup_types.context_alias->getSinglyDesugaredType();
    }
  }

  return result_type;
}
