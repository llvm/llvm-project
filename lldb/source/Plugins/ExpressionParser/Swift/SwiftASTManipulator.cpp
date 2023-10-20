//===-- SwiftASTManipulator.cpp ---------------------------------*- C++ -*-===//
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

#include "SwiftASTManipulator.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "lldb/Expression/ExpressionParser.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/ASTWalker.h"
#include "swift/AST/Decl.h"
#include "swift/AST/DiagnosticEngine.h"
#include "swift/AST/DiagnosticsFrontend.h"
#include "swift/AST/Expr.h"
#include "swift/AST/Initializer.h"
#include "swift/AST/Module.h"
#include "swift/AST/NameLookup.h"
#include "swift/AST/ParameterList.h"
#include "swift/AST/Pattern.h"
#include "swift/AST/SourceFile.h"
#include "swift/AST/Stmt.h"
#include "swift/AST/TypeRepr.h"
#include "swift/AST/Types.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "SwiftUserExpression.h"

using namespace lldb_private;

swift::VarDecl::Introducer
SwiftASTManipulator::VariableInfo::GetVarIntroducer() const {
  if (m_decl)
    return m_decl->getIntroducer();
  else
    return m_var_introducer;
}

bool SwiftASTManipulator::VariableInfo::GetIsCaptureList() const {
  if (m_decl)
    return m_decl->isCaptureList();
  else
    return m_is_capture_list;
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

  if (llvm::isa<VariableMetadataResult>(m_metadata.get()))
    stream.Printf(", is_result");

  if (llvm::isa<VariableMetadataError>(m_metadata.get()))
    stream.Printf(", is_error");

  stream.PutChar(']');
}

void SwiftASTManipulatorBase::DoInitialization() {
  if (m_repl)
    return;

  // First pass: find whether we're dealing with a wrapped function or not.

  struct FuncAndExtensionFinder : public swift::ASTWalker {
    /// This is the toplevel entry function for the expression. It may
    /// call into \c ext_method_decl or hold the entire expression.
    swift::FuncDecl *entrypoint_decl = nullptr;
    /// This is optional.
    swift::FuncDecl *ext_method_decl = nullptr;
    /// When evaluating a generic expression this is the inner
    /// function containing the user expression.
    swift::FuncDecl *user_expr_decl = nullptr;
    /// When evaluating self as generic, this is the trampoline function that
    /// calls the ext_method_decl.
    swift::FuncDecl *trampoline_decl = nullptr;
    /// When evaluating self as generic, this is the sink function
    /// that the top level entry function calls in the AST level.
    swift::FuncDecl *sink_decl = nullptr;

    /// This is an optional extension holding method decl if not null.
    swift::ExtensionDecl *extension_decl = nullptr;

    PreWalkAction walkToDeclPre(swift::Decl *D) override {
      auto *FD = llvm::dyn_cast<swift::FuncDecl>(D);
      // Traverse into any non-function-decls.
      if (!FD)
        return Action::Continue();

      if (!FD->getAttrs().hasAttribute<swift::LLDBDebuggerFunctionAttr>())
        return Action::SkipChildren();

      // Walk up the DeclContext chain, searching for an extension.
      for (auto *DC = FD->getDeclContext(); DC; DC = DC->getParent()) {
        if (auto *extension = llvm::dyn_cast<swift::ExtensionDecl>(DC)) {
          extension_decl = extension;
          ext_method_decl = FD;
          return Action::SkipChildren();
        }
      }
      // Not in an extension.
      if (FD->getNameStr().equals("$__lldb_trampoline"))
        trampoline_decl = FD;
      else if (FD->getNameStr().equals("$__lldb_expr"))
        entrypoint_decl = FD;
      else if (FD->getNameStr().equals("$__lldb_sink"))
        sink_decl = FD;
      else if (FD->getNameStr().equals("$__lldb_user_expr"))
        user_expr_decl = FD;
      return Action::SkipChildren();
    }
  };

  FuncAndExtensionFinder func_finder;
  m_source_file.walk(func_finder);

  m_extension_decl = func_finder.extension_decl;
  if (func_finder.ext_method_decl) {
    m_function_decl = func_finder.ext_method_decl;
    m_entrypoint_decl = func_finder.entrypoint_decl;
  } else if (func_finder.user_expr_decl) {
    m_function_decl = func_finder.user_expr_decl;
    m_entrypoint_decl = func_finder.entrypoint_decl;
  } else
    m_function_decl = func_finder.entrypoint_decl;
  m_entrypoint_decl = func_finder.entrypoint_decl;
  m_trampoline_decl = func_finder.trampoline_decl;
  m_sink_decl = func_finder.sink_decl;
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
      swift::CaseStmt *our_catch = m_do_stmt->getCatches().front();
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

SwiftASTManipulator::SwiftASTManipulator(
    SwiftASTContextForExpressions &swift_ast_ctx,
    swift::SourceFile &source_file, bool repl,
    lldb::BindGenericTypes bind_generic_types)
    : SwiftASTManipulatorBase(source_file, repl, bind_generic_types),
      m_swift_ast_ctx(swift_ast_ctx) {}

void SwiftASTManipulator::FindSpecialNames(
    llvm::SmallVectorImpl<swift::Identifier> &names, llvm::StringRef prefix) {
  names.clear();

  class SpecialNameFinder : public swift::ASTWalker {
  public:
    typedef llvm::SmallVectorImpl<swift::Identifier> NameVector;

    SpecialNameFinder(NameVector &names, llvm::StringRef &prefix)
        : m_names(names), m_prefix(prefix) {}

    PreWalkResult<swift::Expr *> walkToExprPre(swift::Expr *E) override {
      if (auto *UDRE = llvm::dyn_cast<swift::UnresolvedDeclRefExpr>(E)) {
        swift::Identifier name = UDRE->getName().getBaseIdentifier();

        if (m_prefix.empty() || name.str().startswith(m_prefix))
          m_names.push_back(name);
      }

      return Action::Continue(E);
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
  std::string name_buffer;
  llvm::raw_string_ostream(name_buffer) << "__lldb_tmp_ret_" << m_tmpname_idx++;
  swift::Identifier name = ast_context.getIdentifier(name_buffer);

  ResultLocationInfo result_loc_info(source_loc);
  result_loc_info.orig_expr = expr;

  swift::DeclContext *new_decl_context = m_function_decl;
  if (m_repl)
    new_decl_context = decl_context;

  llvm::SmallVector<swift::ASTNode, 3> body;

  const bool is_static = false;
  const auto introducer = swift::VarDecl::Introducer::Let;
  result_loc_info.tmp_var_decl = new (ast_context) swift::VarDecl(
      is_static, introducer, source_loc, name,
      new_decl_context);
  result_loc_info.tmp_var_decl->setImplicit();
  const auto internal_access = swift::AccessLevel::Internal;
  result_loc_info.tmp_var_decl->setAccess(internal_access);
  result_loc_info.tmp_var_decl->setSetterAccess(internal_access);

  auto *var_pattern =
      swift::NamedPattern::createImplicit(ast_context,
                                          result_loc_info.tmp_var_decl);

  const auto static_spelling_kind = swift::StaticSpellingKind::KeywordStatic;
  result_loc_info.binding_decl = swift::PatternBindingDecl::createForDebugger(
      ast_context, static_spelling_kind, var_pattern, expr, new_decl_context);
  result_loc_info.binding_decl->setStatic(false);

  body.push_back(result_loc_info.binding_decl);
  body.push_back(result_loc_info.tmp_var_decl);

  if (in_return) {
    result_loc_info.return_stmt =
        new (ast_context) swift::ReturnStmt(source_loc, nullptr);
    body.push_back(result_loc_info.return_stmt);
  }

  swift::SourceLoc brace_end = result_loc_info.orig_expr->getEndLoc();
  if (brace_end.isInvalid()) {
    brace_end = source_loc;
  }
  auto *body_stmt = swift::BraceStmt::create(
      ast_context, source_loc, llvm::ArrayRef<swift::ASTNode>(body), brace_end,
      true);

  // Default construct a label info that contains nothing for the 'do'
  // statement
  swift::LabeledStmtInfo label_info;

  auto *assign_stmt = new (ast_context)
      swift::DoStmt(label_info, source_loc, body_stmt, true);
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

    PreWalkAction walkToDeclPre(swift::Decl *D) override {
      switch (D->getKind()) {
      default: return Action::Continue();

      // Don't step into function declarations, they may have returns, but we
      // don't want to instrument them.
      case swift::DeclKind::Accessor:
      case swift::DeclKind::Func:
      case swift::DeclKind::Class:
      case swift::DeclKind::Struct:
        return Action::SkipChildren();
      }
    }

    PreWalkResult<swift::Expr *> walkToExprPre(swift::Expr *expr) override {
      switch (expr->getKind()) {
      default: return Action::Continue(expr);

      // Don't step into closure definitions, they may have returns, but we
      // don't want to instrument them either.
      case swift::ExprKind::Closure:
        return Action::SkipChildren(expr);
      }
    }

    PostWalkResult<swift::Stmt *> walkToStmtPost(swift::Stmt *S) override {
      auto *RS = swift::dyn_cast<swift::ReturnStmt>(S);
      if (!RS || !RS->getResult())
        return Action::Continue(S);

      if (swift::Expr *RE = RS->getResult()) {
        if (swift::Stmt *S =
                m_manipulator.ConvertExpressionToTmpReturnVarAccess(
                    RE, RS->getStartLoc(), /*add_return=*/true, m_decl_context))
          return Action::Continue(S);
      }

      return Action::Continue(S);
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

    for (swift::Decl *decl : m_source_file.getTopLevelDecls()) {
      if (auto top_level_code_decl =
              llvm::dyn_cast<swift::TopLevelCodeDecl>(decl)) {
        return_finder.SetDeclContext(top_level_code_decl);
        top_level_code_decl->getBody()->walk(return_finder);
      }
    }

    // Second step, fetch the last expression, and if it is non-null, set it to
    // a temp result as well:

    if (!m_source_file.getTopLevelDecls().empty()) {
      swift::Decl *last_decl = *(m_source_file.getTopLevelDecls().end() - 1);

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

void SwiftASTManipulator::MakeDeclarationsPublic() {
  if (!IsValid())
    return;

  class Publicist : public swift::ASTWalker {
    static bool canMakePublic(swift::Decl *decl) {
      // Properties within structs that have attached property wrappers
      // shouldn't have their access reset; it impacts the implicit memberwise
      // initializer.
      if (llvm::isa<swift::StructDecl>(decl->getDeclContext())) {
        if (auto var = llvm::dyn_cast<swift::VarDecl>(decl)) {
          if (var->hasAttachedPropertyWrapper() ||
              var->getOriginalWrappedProperty())
            return false;

          return true;
        }

        if (auto accessor = llvm::dyn_cast<swift::AccessorDecl>(decl)) {
          return canMakePublic(accessor->getStorage());
        }
      }
      return true;
    }

    PreWalkAction walkToDeclPre(swift::Decl *D) override {
      if (!canMakePublic(D))
        return Action::Continue();

      if (auto *VD = llvm::dyn_cast<swift::ValueDecl>(D)) {
        auto access = swift::AccessLevel::Public;

        // We're making declarations 'public' so that they can be accessed from
        // later expressions, but in the case of classes we also want to be able
        // to subclass them, and override any overridable members. That is, we
        // should use 'open' when it is possible and correct to do so, rather
        // than just 'public'.
        if (llvm::isa<swift::ClassDecl>(VD) || VD->isSyntacticallyOverridable()) {
          if (!VD->isFinal())
            access = swift::AccessLevel::Open;
        }

        VD->overwriteAccess(access);
        if (auto *ASD = llvm::dyn_cast<swift::AbstractStorageDecl>(D))
          ASD->overwriteSetterAccess(access);
      }
      return Action::Continue();
    }
  };

  Publicist p;
  m_source_file.walk(p);
}

static swift::Expr *getFirstInit(swift::PatternBindingDecl *pattern_binding) {
  for (unsigned i = 0, e = pattern_binding->getNumPatternEntries(); i != e; ++i)
    if (pattern_binding->getInit(i))
      return pattern_binding->getInit(i);
  return nullptr;
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

    auto type = var_decl->getDeclContext()->mapTypeIntoContext(
        var_decl->getInterfaceType());
    persistent_info.m_name = name;
    persistent_info.m_type = ToCompilerType({type.getPointer()});
    persistent_info.m_decl = var_decl;

    m_variables.push_back(persistent_info);

    found_declarations.push_back(persistent_info_location);
  };

  if (m_repl) {
    for (swift::Decl *decl : m_source_file.getTopLevelDecls()) {
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

  for (swift::Decl *decl : m_source_file.getTopLevelDecls()) {
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

  result_var->overwriteAccess(swift::AccessLevel::Public);
  result_var->overwriteSetterAccess(swift::AccessLevel::Public);

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

  error_var->overwriteAccess(swift::AccessLevel::Public);
  error_var->overwriteSetterAccess(swift::AccessLevel::Public);

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

bool SwiftASTManipulator::FixupResultAfterTypeChecking(Status &error) {
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
    if (!the_decl->hasInterfaceType()) {
      error.SetErrorStringWithFormat(
          "Type of %zuth return value could not be determined.", i);
      return false;
    }
    swift::Type its_type = the_decl->getTypeInContext();
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
  }

  if (result_type.isNull()) {
    error.SetErrorString("Could not find the result type for this expression.");
    return false;
  } else if (result_type->is<swift::ErrorType>()) {
    error.SetErrorString("Result type is the error type.");
    return false;
  }

  swift::ASTContext &ast_context = m_source_file.getASTContext();
  CompilerType return_ast_type = ToCompilerType(result_type.getPointer());
  swift::Identifier result_var_name =
      ast_context.getIdentifier(GetResultName());
  SwiftASTManipulatorBase::VariableMetadataSP metadata_sp(
      new VariableMetadataResult());

  swift::VarDecl *result_var =
      AddExternalVariable(result_var_name, return_ast_type, metadata_sp);
  if (!result_var) {
    error.SetErrorString("Could not add external result variable.");
    return false;
  }

  result_var->overwriteAccess(swift::AccessLevel::Public);
  result_var->overwriteSetterAccess(swift::AccessLevel::Public);

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
          if (var_decl->hasInterfaceType()) {
            swift::Identifier error_var_name =
                ast_context.getIdentifier(GetErrorName());
            if (error_var_name != var_decl->getName())
              continue;

            swift::Type error_type = var_decl->getInterfaceType();
            CompilerType error_ast_type =
                ToCompilerType(error_type.getPointer());
            SwiftASTManipulatorBase::VariableMetadataSP error_metadata_sp(
                new VariableMetadataError());

            swift::VarDecl *error_var = AddExternalVariable(
                error_var_name, error_ast_type, error_metadata_sp);

            if (!error_var) {
              error.SetErrorString("Could not add external error variable.");
              return false;
            }

            error_var->overwriteAccess(swift::AccessLevel::Public);
            error_var->overwriteSetterAccess(
                swift::AccessLevel::Public);

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

  swift::NamedPattern *named_pattern =
      swift::NamedPattern::createImplicit(ast_context, var_decl);

  // Since lldb does not call bindExtensions, must ensure that any enclosing
  // extension is bound
  for (auto *context = containing_context; context;
       context = context->getParent()) {
    if (auto *d = context->getAsDecl()) {
      if (auto *ext = swift::dyn_cast<swift::ExtensionDecl>(d))
        ext->computeExtendedNominal();
    }
  }
  swift::Type type = containing_context->mapTypeIntoContext(
      var_decl->getInterfaceType());
  swift::TypedPattern *typed_pattern =
    swift::TypedPattern::createImplicit(ast_context, named_pattern, type);

  swift::PatternBindingDecl *pattern_binding =
      swift::PatternBindingDecl::createImplicit(
          ast_context, swift::StaticSpellingKind::None, typed_pattern, nullptr,
          containing_context, var_decl->getLoc());

  return pattern_binding;
}

swift::FuncDecl *SwiftASTManipulator::GetFunctionToInjectVariableInto(
    const SwiftASTManipulator::VariableInfo &variable) const {
  // We always inject self in the wrapper, so we can
  // call the function declared in the type extension.
  if (variable.IsSelf())
    return m_entrypoint_decl;

  // When not binding generic type parameters, we want to inject the metadata
  // pointers in the wrapper, so we can pass them as opaque pointers in the
  // trampoline function later on.
  if (m_bind_generic_types == lldb::eDontBind &&
      (variable.IsMetadataPointer() || variable.IsPackCount() ||
       variable.IsUnboundPack()))
    return m_entrypoint_decl;

  return m_function_decl;
}

llvm::Optional<swift::Type> SwiftASTManipulator::GetSwiftTypeForVariable(
    const SwiftASTManipulator::VariableInfo &variable) const {
  auto type_system_swift =
      variable.m_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();

  if (!type_system_swift)
    return {};

  // When injecting a value pack or pack count into the outer
  // lldb_expr function, treat it as an opaque raw pointer.
  if (m_bind_generic_types == lldb::eDontBind && variable.IsUnboundPack()) {
    auto swift_ast_ctx = type_system_swift->GetSwiftASTContext();
    if (swift_ast_ctx) {
      auto it = m_type_aliases.find("$__lldb_builtin_ptr_t");
      if (it == m_type_aliases.end())
        return {};
      return swift::Type(it->second);
    }
    return {};
  }

  // This might be a referenced type, which will confuse the type checker.
  // The access pattern for these types is the same as for the referent
  // type, so it is fine to just strip it off.
  CompilerType referent_type =
      type_system_swift->GetReferentType(variable.m_type.GetOpaqueQualType());

  swift::Type swift_type = m_swift_ast_ctx.GetSwiftType(referent_type);
  if (!swift_type)
    return {};

  // One tricky bit here is that this var may be an argument to the
  // function whose context we are emulating, and that argument might be
  // of "inout" type.  We need to strip the inout off the type or the
  // initial parse will fail.  Fortunately, the variable access goes the
  // same regardless of whether it is inout or not, so we don't have to do
  // anything more to get this to work.
  swift_type = swift_type->getWithoutSpecifierType();

  if (variable.IsSelf()) {
    // Another tricky bit is that the Metatype types we get have the
    // "Representation" already attached (i.e.
    // "@thick", "@thin".)  But the representation is a SIL level thing,
    // and if it is attached to types that we hand the parser, it throws a
    // verifier error & aborts.  So we strip it off here:
    swift::MetatypeType *metatype_type =
        llvm::dyn_cast<swift::MetatypeType>(swift_type.getPointer());
    if (metatype_type) {
      swift_type = swift::Type(
          swift::MetatypeType::get(metatype_type->getInstanceType()));
    }
  }

  if (swift_type->hasArchetype())
    swift_type = swift_type->mapTypeOutOfContext();

  return {swift_type};
}

swift::VarDecl *SwiftASTManipulator::GetVarDeclForVariableInFunction(
    const SwiftASTManipulator::VariableInfo &variable,
    swift::FuncDecl *containing_function) {
  const auto maybe_swift_type = GetSwiftTypeForVariable(variable);

  if (!maybe_swift_type)
    return {};

  const auto &swift_type = *maybe_swift_type;

  const swift::SourceLoc loc = containing_function->getBody()->getLBraceLoc();
  const swift::Identifier name = variable.GetName();
  // We may need to mutate the self variable later, so hardcode it to a var
  // in that case.
  const auto introducer = (variable.IsSelf() || variable.IsUnboundPack())
                              ? swift::VarDecl::Introducer::Var
                              : variable.GetVarIntroducer();

  const swift::ASTContext &ast_context = m_source_file.getASTContext();
  swift::VarDecl *redirected_var_decl = new (ast_context) swift::VarDecl(
      /*is_static*/ false, introducer, loc, name, containing_function);

  redirected_var_decl->setInterfaceType(swift_type);
  redirected_var_decl->setDebuggerVar(true);
  redirected_var_decl->setImplicit(true);
  redirected_var_decl->setImplInfo(swift::StorageImplInfo(
      swift::ReadImplKind::Stored, swift::WriteImplKind::Stored,
      swift::ReadWriteImplKind::Stored));

  // This avoids having local variables filtered out by
  // swift::namelookup::filterForDiscriminator().
  redirected_var_decl->overwriteAccess(swift::AccessLevel::Public);

  if (swift_type->getAs<swift::WeakStorageType>()) {
    redirected_var_decl->getAttrs().add(
        new (ast_context) swift::ReferenceOwnershipAttr(
            swift::SourceRange(), swift::ReferenceOwnership::Weak));
  }

  return redirected_var_decl;
}

/// Adds the new nodes to the beginning of the function.
static void AddNodesToBeginningFunction(
    swift::FuncDecl *function,
    const llvm::SmallVectorImpl<swift::ASTNode> &new_nodes,
    swift::ASTContext &ast_context) {
  swift::BraceStmt *body = function->getBody();
  llvm::ArrayRef<swift::ASTNode> current_elements = body->getElements();

  llvm::SmallVector<swift::ASTNode, 3> new_elements(current_elements.begin(),
                                                    current_elements.end());

  // Make sure to add the new nodes at the beginning of the function.
  new_elements.insert(new_elements.begin(), new_nodes.begin(), new_nodes.end());
  auto *new_function_body = swift::BraceStmt::create(
      ast_context, body->getLBraceLoc(), ast_context.AllocateCopy(new_elements),
      body->getRBraceLoc());

  function->setBody(new_function_body, function->getBodyKind());
  function->setHasSingleExpressionBody(false);
}

bool SwiftASTManipulator::AddExternalVariables(
    llvm::MutableArrayRef<VariableInfo> variables) {
  if (!IsValid())
    return false;

  Log *log = GetLog(LLDBLog::Expressions);

  swift::ASTContext &ast_context = m_source_file.getASTContext();

  if (m_repl) {
    // In the REPL, we're only adding the result variable.

    if (variables.empty()) {
      return true;
    }

    assert(variables.size() == 1);

    SwiftASTManipulator::VariableInfo &variable = variables[0];

    const bool is_static = false;
    auto introducer = variable.GetVarIntroducer();
    swift::SourceLoc loc;
    swift::Identifier name = variable.m_name;
    swift::Type var_type = m_swift_ast_ctx.GetSwiftType(variable.m_type);

    if (!var_type)
      return false;

    // If the type is an inout or lvalue type (happens if this is an argument)
    // strip that part off:

    swift::VarDecl *redirected_var_decl = new (ast_context)
        swift::VarDecl(is_static, introducer, loc, name,
                       &m_source_file);    
    redirected_var_decl->setInterfaceType(var_type);
    redirected_var_decl->setTopLevelGlobal(true);

    swift::TopLevelCodeDecl *top_level_code =
        new (ast_context) swift::TopLevelCodeDecl(&m_source_file);

    swift::PatternBindingDecl *pattern_binding =
        GetPatternBindingForVarDecl(redirected_var_decl, top_level_code);

    swift::ASTNode elements[] = {pattern_binding};

    swift::BraceStmt *brace_stmt =
        swift::BraceStmt::create(ast_context, loc, elements, loc, true);

    top_level_code->setBody(brace_stmt);

    redirected_var_decl->setImplicit(true);

    // FIXME: This should use SourceFile::addTopLevelDecl, but if these decls
    // are not inserted at the beginning of the source file then
    // SwiftREPL/FoundationTypes.test fails.
    //
    // See rdar://58355191
    m_source_file.prependTopLevelDecl(top_level_code);
    m_source_file.prependTopLevelDecl(redirected_var_decl);

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
    // The new nodes that should be added to each function.
    std::unordered_map<swift::FuncDecl *, llvm::SmallVector<swift::ASTNode, 3>>
        injected_nodes;

    for (SwiftASTManipulator::VariableInfo &variable : variables) {
      swift::FuncDecl *containing_function =
          GetFunctionToInjectVariableInto(variable);
      assert(containing_function && "No function to inject variable into!");
      if (!containing_function)
        continue;

      swift::VarDecl *redirected_var_decl =
          GetVarDeclForVariableInFunction(variable, containing_function);
      if (!redirected_var_decl) {
        LLDB_LOG(log, "No var decl.");
        continue;
      }
      swift::PatternBindingDecl *pattern_binding =
          GetPatternBindingForVarDecl(redirected_var_decl, containing_function);
      if (!pattern_binding) {
        LLDB_LOG(log, "No pattern binding.");
        continue;
      }

      // Push the var decl and pattern binding so we add them to the function
      // later.
      injected_nodes[containing_function].push_back(
          swift::ASTNode(pattern_binding));
      injected_nodes[containing_function].push_back(
          swift::ASTNode(redirected_var_decl));

      variable.m_decl = redirected_var_decl;

      if (log) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        variable.m_decl->dump(ss);
        ss.flush();

        LLDB_LOG(log,
                 "[SwiftASTManipulator::AddExternalVariables] Injected "
                 "variable {0} into {1}",
                 s, containing_function->getName().getBaseIdentifier().str());
      }

      m_variables.push_back(variable);
    }

    for (auto &pair : injected_nodes) {
      auto *containing_function = pair.first;
      auto &new_nodes = pair.second;
      AddNodesToBeginningFunction(containing_function, new_nodes, ast_context);
    }
  }

  return true;
}

static void AppendToCaptures(swift::ASTContext &ast_context,
                             swift::FuncDecl *func_decl,
                             swift::VarDecl *capture_decl) {
  const auto old_capture_info = func_decl->getCaptureInfo();
  const auto old_captures = old_capture_info.getCaptures();
  llvm::SmallVector<swift::CapturedValue, 2> captures(old_captures.begin(),
                                                      old_captures.end());

  captures.push_back(swift::CapturedValue(capture_decl, 0, swift::SourceLoc()));
  const auto new_capture_info =
      swift::CaptureInfo(ast_context, captures,
                         old_capture_info.getDynamicSelfType(),
                         old_capture_info.getOpaqueValue(),
                         old_capture_info.hasGenericParamCaptures());
  func_decl->setCaptureInfo(new_capture_info);
}

static swift::VarDecl *FindArgInFunction(swift::ASTContext &ast_context,
                                         swift::FuncDecl *func_decl) {
  auto name = ast_context.getIdentifier("$__lldb_arg");

  if (auto *self_decl = func_decl->getImplicitSelfDecl())
    if (self_decl->getName() == name)
      return self_decl;

  for (auto *param : *func_decl->getParameters()) {
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

  if (m_entrypoint_decl)
    wrapper_arg_decl = FindArgInFunction(ast_context, m_entrypoint_decl);

  if (!function_arg_decl)
    return false;

  if (m_entrypoint_decl && (!wrapper_arg_decl))
    return false;

  for (VariableInfo &variable : m_variables) {
    if (!variable.m_decl)
      continue;

    auto impl = variable.m_decl->getImplInfo();
    if (impl.getReadImpl() != swift::ReadImplKind::Get ||
        (impl.getWriteImpl() != swift::WriteImplKind::Set &&
         impl.getWriteImpl() != swift::WriteImplKind::Immutable))
      continue;

    // FIXME: What about _read and _modify?
    swift::FuncDecl *getter_decl = variable.m_decl->getAccessor(
        swift::AccessorKind::Get);
    swift::FuncDecl *setter_decl = variable.m_decl->getAccessor(
        swift::AccessorKind::Set);

    swift::DeclContext *decl_context = variable.m_decl->getDeclContext();

    if (decl_context == (swift::DeclContext *)m_function_decl) {
      AppendToCaptures(ast_context, getter_decl, function_arg_decl);
      AppendToCaptures(ast_context, setter_decl, function_arg_decl);
    } else if (decl_context == (swift::DeclContext *)m_entrypoint_decl) {
      AppendToCaptures(ast_context, getter_decl, wrapper_arg_decl);
      AppendToCaptures(ast_context, setter_decl, wrapper_arg_decl);
    } else {
      return false;
    }
  }

  return true;
}

swift::TypeAliasDecl *
SwiftASTManipulator::MakeTypealias(swift::Identifier name, CompilerType &type,
                                   bool make_private,
                                   swift::DeclContext *decl_ctx) {
  if (!IsValid())
    return nullptr;

  // If no DeclContext was passed in make this a global typealias.
  if (!decl_ctx)
    decl_ctx = &m_source_file;

  swift::ASTContext &ast_context = m_source_file.getASTContext();

  swift::TypeAliasDecl *type_alias_decl = new (ast_context)
      swift::TypeAliasDecl(swift::SourceLoc(), swift::SourceLoc(), name,
                           swift::SourceLoc(), nullptr, decl_ctx);
  swift::Type underlying_type = m_swift_ast_ctx.GetSwiftType(type);
  if (!underlying_type)
    return nullptr;

  type_alias_decl->setUnderlyingType(underlying_type);
  type_alias_decl->markAsDebuggerAlias(true);
  type_alias_decl->setImplicit(true);

  Log *log = GetLog(LLDBLog::Expressions);

  if (!type_alias_decl) {
    LLDB_LOGF(log,
              "Could not make typealias from %s to %s in decl context (%p) and "
              "context (%p)",
              name.get(), type.GetMangledTypeName().GetCString(),
              static_cast<void *>(decl_ctx), static_cast<void *>(&ast_context));
    return nullptr;
  }
  if (log) {

    std::string s;
    llvm::raw_string_ostream ss(s);
    type_alias_decl->dump(ss);
    ss.flush();

    log->Printf(
        "Made type alias for %s (%p) in decl context (%p) and context "
        "(%p):\n%s",
        name.get(),
        static_cast<void *>(m_swift_ast_ctx.GetSwiftType(type).getPointer()),
        static_cast<void *>(decl_ctx), static_cast<void *>(&ast_context),
        s.c_str());
  }

  if (make_private)
    type_alias_decl->overwriteAccess(swift::AccessLevel::Private);

  if (auto *f = llvm::dyn_cast<swift::FuncDecl>(decl_ctx)) {
    llvm::SmallVector<swift::ASTNode, 1> node;
    node.push_back(type_alias_decl);
    AddNodesToBeginningFunction(f, node, ast_context);
  } else {
    m_source_file.addTopLevelDecl(type_alias_decl);
  }

  m_type_aliases.insert(
      {name.str(), type_alias_decl->getStructuralType().getPointer()});
  return type_alias_decl;
}

bool SwiftASTManipulator::SaveExpressionTextToTempFile(
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
    lldb_private::NativeFile file(temp_fd,
                                  /*options*/ File::eOpenOptionWriteOnly,
                                  /*transfer_ownership*/ true);
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
