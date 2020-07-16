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

#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "lldb/Expression/ExpressionParser.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
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

void SwiftASTManipulator::WrapExpression(
    lldb_private::Stream &wrapped_stream, const char *orig_text,
    bool needs_object_ptr,
    bool static_method,
    bool is_class,
    bool weak_self,
    const EvaluateExpressionOptions &options,
    llvm::StringRef os_version,
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
    const char *playground_logger_declarations = R"(
@_silgen_name ("playground_logger_initialize") func __builtin_logger_initialize ()
@_silgen_name ("playground_log_hidden") func __builtin_log_with_id<T> (_ object : T, _ name : String, _ id : Int, _ sl : Int, _ el : Int, _ sc : Int, _ ec: Int, _ moduleID: Int, _ fileID: Int) -> AnyObject
@_silgen_name ("playground_log_scope_entry") func __builtin_log_scope_entry (_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int, _ moduleID: Int, _ fileID: Int) -> AnyObject
@_silgen_name ("playground_log_scope_exit") func __builtin_log_scope_exit (_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int, _ moduleID: Int, _ fileID: Int) -> AnyObject
@_silgen_name ("playground_log_postprint") func __builtin_postPrint (_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int, _ moduleID: Int, _ fileID: Int) -> AnyObject
@_silgen_name ("DVTSendPlaygroundLogData") func __builtin_send_data (_ :  AnyObject!)
__builtin_logger_initialize()
)";

    // The debug function declarations need only be declared once per session -
    // on the first REPL call.  This code assumes that the first call is the
    // first REPL call; don't call playground once then playground || repl
    // again
    bool first_expression = options.GetPreparePlaygroundStubFunctions();

    const char *playground_prefix =
        first_expression ? playground_logger_declarations : "";

    if (pound_file && pound_line) {
      wrapped_stream.Printf("%s#sourceLocation(file: \"%s\", line: %u)\n%s\n",
                            playground_prefix, pound_file, pound_line,
                            orig_text);
    } else {
      // In 2017+, xcode playgrounds send orig_text that starts with a module
      // loading prefix (not the above prefix), then a sourceLocation specifier
      // that indicates the page name, and then the page body text.  The
      // first_body_line mechanism in this function cannot be used to
      // compensate for the playground_prefix added here, since it incorrectly
      // continues to apply even after sourceLocation directives are read from
      // the orig_text.  To make sure playgrounds work correctly whether or not
      // they supply their own sourceLocation, create a dummy sourceLocation
      // here with a fake filename that starts counting the first line of
      // orig_text as line 1.
      wrapped_stream.Printf("%s#sourceLocation(file: \"%s\", line: %u)\n%s\n",
                            playground_prefix, "Playground.swift", 1,
                            orig_text);
    }
    first_body_line = 1;
    return;
  }

  assert(!playground && "Playground mode not expected");

  if (repl) {
    if (pound_file && pound_line) {
      wrapped_stream.Printf("#sourceLocation(file: \"%s\", line:  %u)\n%s\n",
                            llvm::sys::path::filename(pound_file).str().c_str(),
                            pound_line, orig_text);
    } else {
      wrapped_stream.Printf("%s", orig_text);
    }
    first_body_line = 1;
    return;
  }

  assert(!playground && !repl && "Playground/REPL mode not expected");

  if (pound_file && pound_line) {
    fixed_text.Printf("#sourceLocation(file: \"%s\", line: %u)\n%s\n",
                      pound_file, pound_line, orig_text);
    text = fixed_text.GetString().data();
  } else if (generate_debug_info) {
    std::string expr_source_path;
    if (SwiftASTManipulator::SaveExpressionTextToTempFile(orig_text, options,
                                                          expr_source_path)) {
      fixed_text.Printf("#sourceLocation(file: \"%s\", line: 1)\n%s\n",
                        expr_source_path.c_str(), orig_text);
      text = fixed_text.GetString().data();
    }
  }

  // Note: All the wrapper functions we make are marked with the
  // @LLDBDebuggerFunction macro so that the compiler can do whatever special
  // treatment it need to do on them.  If you add new variants be sure to mark
  // them this way.  Also, any function that might end up being in an extension
  // of swift class needs to be marked final, since otherwise the compiler
  // might try to dispatch them dynamically, which it can't do correctly for
  // these functions.

  std::string availability = "";
  if (!os_version.empty())
    availability = (llvm::Twine("@available(") + os_version + ", *)").str();

  StreamString wrapped_expr_text;

  // Avoid indenting user code: this makes column information from compiler
  // errors match up with what the user typed.
  wrapped_expr_text.Printf(R"(
do {
%s%s%s
} catch (let __lldb_tmp_error) {
  var %s = __lldb_tmp_error
}
)",
                           GetUserCodeStartMarker(), text,
                           GetUserCodeEndMarker(), GetErrorName());

  if (needs_object_ptr || static_method) {
    const char *func_decorator = "";
    if (static_method) {
      if (is_class)
        func_decorator = "final class";
      else
        func_decorator = "static";
    } else if (is_class && !weak_self) {
      func_decorator = "final";
    } else {
      func_decorator = "mutating";
    }

    const char *optional_extension =
        weak_self ? "Swift.Optional where Wrapped == " : "";

    // The expression text is inserted into the body of $__lldb_wrapped_expr_%u.
    wrapped_stream.Printf(R"(
extension %s$__lldb_context {
  @LLDBDebuggerFunction %s
  %s func $__lldb_wrapped_expr_%u(_ $__lldb_arg : UnsafeMutablePointer<Any>) {
    %s
  }
}
%s
func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {
  do {
    $__lldb_injected_self.$__lldb_wrapped_expr_%u(
      $__lldb_arg
    )
  }
}
)",
                          optional_extension, availability.c_str(),
                          func_decorator, current_counter,
                          wrapped_expr_text.GetData(), availability.c_str(),
                          current_counter);

    first_body_line = 5;
  } else {
    wrapped_stream.Printf(
        "@LLDBDebuggerFunction %s\n"
        "func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {\n"
        "%s" // This is the expression text (with newlines).
        "}\n",
        availability.c_str(), wrapped_expr_text.GetData());
    first_body_line = 4;
  }
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

    bool walkToDeclPre(swift::Decl *D) override {
      auto *FD = llvm::dyn_cast<swift::FuncDecl>(D);
      if (!FD)
        return true;

      if (FD->getAttrs().hasAttribute<swift::LLDBDebuggerFunctionAttr>()) {
        m_function_decl = FD;

        // Now walk back up the containing DeclContexts, and if we find an
        // extension Decl, that's our extension:
        for (swift::DeclContext *DC = m_function_decl->getDeclContext(); DC;
             DC = DC->getParent()) {
          if (auto *extension_decl = llvm::dyn_cast<swift::ExtensionDecl>(DC)) {
            m_extension_decl = extension_decl;
            break;
          }
        }
      } else if (FD->hasName() && FD->getBaseIdentifier().str()
                                    .startswith(m_wrapper_func_prefix)) {
        m_wrapper_decl = FD;
      }

      // There's nothing buried in a function that we need to find in this
      // search.
      return false;
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

    std::pair<bool, swift::Expr *> walkToExprPre(swift::Expr *E) override {
      if (auto *UDRE = llvm::dyn_cast<swift::UnresolvedDeclRefExpr>(E)) {
        swift::Identifier name = UDRE->getName().getBaseIdentifier();

        if (m_prefix.empty() || name.str().startswith(m_prefix))
          m_names.push_back(name);
      }

      return {true, E};
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
  swift::Identifier equalequal_name = ast_context.getIdentifier("==");

  ResultLocationInfo result_loc_info(source_loc);
  result_loc_info.orig_expr = expr;

  swift::DeclContext *new_decl_context = m_function_decl;
  if (m_repl)
    new_decl_context = decl_context;

  llvm::SmallVector<swift::ASTNode, 3> body;
  llvm::SmallVector<swift::Expr *, 3> false_body;
  const bool is_static = false;
  const auto introducer = swift::VarDecl::Introducer::Let;
  const bool is_capture_list = false;
  result_loc_info.tmp_var_decl = new (ast_context) swift::VarDecl(
      is_static, introducer, is_capture_list, source_loc, name,
      new_decl_context);
  result_loc_info.tmp_var_decl->setImplicit();
  const auto internal_access = swift::AccessLevel::Internal;
  result_loc_info.tmp_var_decl->setAccess(internal_access);
  result_loc_info.tmp_var_decl->setSetterAccess(internal_access);

  auto *var_pattern =
      swift::NamedPattern::createImplicit(ast_context,
                                          result_loc_info.tmp_var_decl);

  const auto static_spelling_kind = swift::StaticSpellingKind::KeywordStatic;
  result_loc_info.binding_decl = swift::PatternBindingDecl::createImplicit(
      ast_context, static_spelling_kind, var_pattern, expr, new_decl_context);
  result_loc_info.binding_decl->setStatic(false);

  body.push_back(result_loc_info.binding_decl);
  body.push_back(result_loc_info.tmp_var_decl);

  if (in_return) {
    result_loc_info.return_stmt =
        new (ast_context) swift::ReturnStmt(source_loc, nullptr);
    body.push_back(result_loc_info.return_stmt);
  }
  auto *one_expr = new (ast_context)
      swift::IntegerLiteralExpr(swift::StringRef("1"), source_loc, true);
  false_body.push_back(one_expr);
  auto *equalequal_expr = new (ast_context) swift::UnresolvedDeclRefExpr(
      swift::DeclNameRef(equalequal_name), swift::DeclRefKind::BinaryOperator,
      swift::DeclNameLoc(source_loc));
  false_body.push_back(equalequal_expr);
  auto *zero_expr = new (ast_context)
      swift::IntegerLiteralExpr(swift::StringRef("0"), source_loc, true);
  false_body.push_back(zero_expr);
  auto *zero_equals_one_expr = swift::SequenceExpr::create(
      ast_context, llvm::ArrayRef<swift::Expr *>(false_body));

  zero_equals_one_expr->setImplicit();
  auto *body_stmt = swift::BraceStmt::create(
      ast_context, source_loc, llvm::ArrayRef<swift::ASTNode>(body), source_loc,
      true);

  // Default construct a label info that contains nothing for the while
  // statement
  swift::LabeledStmtInfo label_info;

  auto *assign_stmt = new (ast_context)
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

    bool walkToDeclPre(swift::Decl *D) override {
      switch (D->getKind()) {
      default: return true;

      // Don't step into function declarations, they may have returns, but we
      // don't want to instrument them.
      case swift::DeclKind::Accessor:
      case swift::DeclKind::Func:
      case swift::DeclKind::Class:
      case swift::DeclKind::Struct:
        return false;
      }
    }

    std::pair<bool, swift::Expr *> walkToExprPre(swift::Expr *expr) override {
      switch (expr->getKind()) {
      default: return {true, expr};

      // Don't step into closure definitions, they may have returns, but we
      // don't want to instrument them either.
      case swift::ExprKind::Closure:
        return {false, expr};
      }
    }

    swift::Stmt *walkToStmtPost(swift::Stmt *S) override {
      auto *RS = swift::dyn_cast<swift::ReturnStmt>(S);
      if (!RS || !RS->getResult())
        return S;

      if (swift::Expr *RE = RS->getResult()) {
        if (swift::Stmt *S =
                m_manipulator.ConvertExpressionToTmpReturnVarAccess(
                    RE, RS->getStartLoc(), /*add_return=*/true, m_decl_context))
          return S;
      }

      return S;
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

    bool walkToDeclPre(swift::Decl *D) override {
      if (!canMakePublic(D))
        return true;

      if (auto *VD = llvm::dyn_cast<swift::ValueDecl>(D)) {
        auto access = swift::AccessLevel::Public;

        // We're making declarations 'public' so that they can be accessed from
        // later expressions, but in the case of classes we also want to be able
        // to subclass them, and override any overridable members. That is, we
        // should use 'open' when it is possible and correct to do so, rather
        // than just 'public'.
        if (llvm::isa<swift::ClassDecl>(VD) || VD->isPotentiallyOverridable()) {
          if (!VD->isFinal())
            access = swift::AccessLevel::Open;
        }

        VD->overwriteAccess(access);
        if (auto *ASD = llvm::dyn_cast<swift::AbstractStorageDecl>(D))
          ASD->overwriteSetterAccess(access);
      }
      return true;
    }
  };

  Publicist p;
  m_source_file.walk(p);
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

  CompilerType return_ast_type = ToCompilerType(result_type.getPointer());

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

  CompilerType error_ast_type = ToCompilerType(error_type.getPointer());

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
    if (the_decl->hasInterfaceType()) {
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

  CompilerType return_ast_type = ToCompilerType(result_type.getPointer());
  swift::Identifier result_var_name =
      ast_context.getIdentifier(GetResultName());
  SwiftASTManipulatorBase::VariableMetadataSP metadata_sp(
      new VariableMetadataResult());

  swift::VarDecl *result_var =
      AddExternalVariable(result_var_name, return_ast_type, metadata_sp);

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
    auto introducer = variable.GetVarIntroducer();
    bool is_capture_list = variable.GetIsCaptureList();
    swift::SourceLoc loc;
    swift::Identifier name = variable.m_name;
    swift::Type var_type = GetSwiftType(variable.m_type);

    // If the type is an inout or lvalue type (happens if this is an argument)
    // strip that part off:

    swift::VarDecl *redirected_var_decl = new (ast_context)
        swift::VarDecl(is_static, introducer, is_capture_list, loc, name,
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
      auto introducer = variable.GetVarIntroducer();
      bool is_capture_list = variable.GetIsCaptureList();

      bool is_self = !variable.m_name.str().compare("$__lldb_injected_self");

      if (is_self) {
        if (!m_wrapper_decl)
          continue;

        // We need to mutate the $__lldb_wrapped_expr_%d member of self later.
        introducer = swift::VarDecl::Introducer::Var;
        loc = m_wrapper_decl->getBody()->getLBraceLoc();
        containing_function = m_wrapper_decl;
      }

      // This might be a referenced type, which will confuse the type checker.
      // The access pattern for these types is the same as for the referent
      // type, so it is fine to
      // just strip it off.
      auto *swift_ast_ctx = llvm::dyn_cast_or_null<TypeSystemSwift>(
          variable.m_type.GetTypeSystem());

      CompilerType referent_type;
      if (swift_ast_ctx)
        referent_type =
            swift_ast_ctx->GetReferentType(variable.m_type.GetOpaqueQualType());
      if (!referent_type)
        continue;

      // One tricky bit here is that this var may be an argument to the function
      // whose context we are
      // emulating, and that argument might be of "inout" type.  We need to
      // strip the inout off the type
      // or the initial parse will fail.  Fortunately, the variable access goes
      // the same regardless of whether
      // it is inout or not, so we don't have to do anything more to get this to
      // work.
      swift::Type var_type =
          GetSwiftType(referent_type)->getWithoutSpecifierType();
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
          is_static, introducer, is_capture_list, loc, name,
          containing_function);
      auto interface_type = var_type;
      if (interface_type->hasArchetype())
        interface_type = interface_type->mapTypeOutOfContext();
      redirected_var_decl->setInterfaceType(interface_type);
      redirected_var_decl->setDebuggerVar(true);
      redirected_var_decl->setImplicit(true);
      // This avoids having local variables filtered out by
      // swift::namelookup::filterForDiscriminator().
      redirected_var_decl->overwriteAccess(swift::AccessLevel::Public);

      swift::PatternBindingDecl *pattern_binding =
          GetPatternBindingForVarDecl(redirected_var_decl, containing_function);

      if (var_type->getAs<swift::WeakStorageType>()) {
        redirected_var_decl->getAttrs().add(
            new (ast_context) swift::ReferenceOwnershipAttr(
                swift::SourceRange(), swift::ReferenceOwnership::Weak));
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

  if (m_wrapper_decl)
    wrapper_arg_decl = FindArgInFunction(ast_context, m_wrapper_decl);

  if (!function_arg_decl)
    return false;

  if (m_wrapper_decl && (!wrapper_arg_decl))
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

  swift::TypeAliasDecl *type_alias_decl = new (ast_context)
      swift::TypeAliasDecl(source_loc, swift::SourceLoc(), name, source_loc,
                           nullptr, &m_source_file);
  swift::Type underlying_type = GetSwiftType(type);
  type_alias_decl->setUnderlyingType(underlying_type);
  type_alias_decl->markAsDebuggerAlias(true);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));
  if (log) {

    std::string s;
    llvm::raw_string_ostream ss(s);
    type_alias_decl->dump(ss);
    ss.flush();

    log->Printf("Made global type alias for %s (%p) in context (%p):\n%s",
                name.get(), static_cast<void *>(GetSwiftType(type).getPointer()),
                static_cast<void *>(&ast_context), s.c_str());
  }

  if (type_alias_decl) {
    if (make_private) {
      type_alias_decl->overwriteAccess(swift::AccessLevel::Private);
    }
    m_source_file.addTopLevelDecl(type_alias_decl);
  }

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
    lldb_private::NativeFile file(temp_fd, /*options*/ File::eOpenOptionWrite,
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
