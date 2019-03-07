//===-- SwiftASTManipulator.h -----------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftASTManipulator_h
#define liblldb_SwiftASTManipulator_h

#include "lldb/Core/ClangForward.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Expression/Expression.h"
#include "lldb/Symbol/CompilerType.h"

#include "swift/AST/Decl.h"
#include "swift/Basic/SourceLoc.h"
#include "llvm/ADT/SmallVector.h"

namespace swift {
class CatchStmt;
class DoCatchStmt;
class ExtensionDecl;
class FuncDecl;
class RepeatWhileStmt;
class ReturnStmt;
class VarDecl;
} // namespace swift

namespace lldb_private {

class SwiftASTManipulatorBase {
public:
  class VariableMetadata {
  public:
    VariableMetadata() {}
    virtual ~VariableMetadata() {}
    virtual unsigned GetType() = 0;
  };

  class VariableMetadataResult
      : public SwiftASTManipulatorBase::VariableMetadata {
  public:
    virtual ~VariableMetadataResult();
    constexpr static unsigned Type() { return 'Resu'; }
    virtual unsigned GetType() { return Type(); }
  };

  class VariableMetadataError
      : public SwiftASTManipulatorBase::VariableMetadata {
  public:
    virtual ~VariableMetadataError();
    constexpr static unsigned Type() { return 'Erro'; }
    virtual unsigned GetType() { return Type(); }
  };

  typedef std::shared_ptr<VariableMetadata> VariableMetadataSP;

  struct VariableInfo {
    CompilerType GetType() const { return m_type; }
    swift::Identifier GetName() const { return m_name; }
    swift::VarDecl *GetDecl() const { return m_decl; }
    swift::VarDecl::Specifier GetVarSpecifier() const;
    bool GetIsCaptureList() const;

    VariableInfo() : m_type(), m_name(), m_metadata() {}

    VariableInfo(CompilerType &type, swift::Identifier name,
                 VariableMetadataSP metadata, 
                 swift::VarDecl::Specifier specifier, 
                 bool is_capture_list = false)
        : m_type(type), m_name(name), m_var_specifier(specifier),
          m_is_capture_list(is_capture_list), m_metadata(metadata) {}

    template <class T> bool MetadataIs() const {
      return (m_metadata && m_metadata->GetType() == T::Type());
    }

    void Print(Stream &stream) const;

    void SetType(CompilerType new_type) { m_type = new_type; }

    friend class SwiftASTManipulator;

  protected:
    CompilerType m_type;
    swift::Identifier m_name;
    swift::VarDecl *m_decl = nullptr;
    swift::VarDecl::Specifier m_var_specifier = swift::VarDecl::Specifier::Var;
    bool m_is_capture_list = false;

  public:
    VariableMetadataSP m_metadata;
  };

  SwiftASTManipulatorBase(swift::SourceFile &source_file, bool repl)
      : m_source_file(source_file), m_variables(), m_repl(repl) {
    DoInitialization();
  }

  llvm::MutableArrayRef<VariableInfo> GetVariableInfo() { return m_variables; }

  bool IsValid() {
    return m_repl || (m_function_decl &&
                      (m_wrapper_decl || (!m_extension_decl)) && m_do_stmt);
  }

  swift::BraceStmt *GetUserBody();

private:
  void DoInitialization();

protected:
  swift::SourceFile &m_source_file;
  llvm::SmallVector<VariableInfo, 1> m_variables;

  bool m_repl = false;

  swift::FuncDecl *m_function_decl =
      nullptr; // the function containing the expression's code
  swift::FuncDecl *m_wrapper_decl =
      nullptr; // the wrapper that invokes the right generic function.
  swift::ExtensionDecl *m_extension_decl =
      nullptr; // the extension m_function_decl lives in, if it's a method.
  swift::DoCatchStmt *m_do_stmt =
      nullptr; // the do{}catch(){} statement whose body is the main body.
  swift::CatchStmt *m_catch_stmt =
      nullptr; // the body of the catch - we patch the assignment there to
               // capture any error thrown.
};

class SwiftASTManipulator : public SwiftASTManipulatorBase {
public:
  SwiftASTManipulator(swift::SourceFile &source_file, bool repl);

  static void WrapExpression(Stream &wrapped_stream, const char *text,
                             bool needs_object_ptr,
                             bool static_method,
                             bool is_class,
                             bool weak_self,
                             const EvaluateExpressionOptions &options,
                             const Expression::SwiftGenericInfo &generic_info,
                             llvm::StringRef os_version,
                             uint32_t &first_body_line);

  void FindSpecialNames(llvm::SmallVectorImpl<swift::Identifier> &names,
                        llvm::StringRef prefix);

  swift::VarDecl *AddExternalVariable(swift::Identifier name,
                                      CompilerType &type,
                                      VariableMetadataSP &metadata_sp);

  bool AddExternalVariables(llvm::MutableArrayRef<VariableInfo> variables);

  bool RewriteResult();

  void MakeDeclarationsPublic();

  void
  FindVariableDeclarations(llvm::SmallVectorImpl<size_t> &found_declarations,
                           bool repl);

  void FindNonVariableDeclarations(
      llvm::SmallVectorImpl<swift::ValueDecl *> &non_variables);

  bool FixCaptures();

  swift::ValueDecl *MakeGlobalTypealias(swift::Identifier name,
                                        CompilerType &type,
                                        bool make_private = true);

  bool FixupResultAfterTypeChecking(Status &error);

  static const char *GetArgumentName() { return "$__lldb_arg"; }
  static const char *GetResultName() { return "$__lldb_result"; }
  static const char *GetErrorName() { return "$__lldb_error_result"; }
  static const char *GetUserCodeStartMarker() {
    return "/*__LLDB_USER_START__*/\n";
  }
  static const char *GetUserCodeEndMarker() {
    return "\n/*__LLDB_USER_END__*/";
  }

private:
  uint32_t m_tmpname_idx = 0;

  typedef llvm::SmallVectorImpl<swift::ASTNode> Body;

  swift::Stmt *ConvertExpressionToTmpReturnVarAccess(
      swift::Expr *expr, const swift::SourceLoc &source_loc, bool in_return,
      swift::DeclContext *decl_context);

  struct ResultLocationInfo {
    swift::VarDecl
        *tmp_var_decl; // This points to the first stage tmp result decl
    swift::RepeatWhileStmt
        *wrapper_stmt; // This is the RepeatWhile statement that we make up.
    swift::PatternBindingDecl
        *binding_decl;      // This is the expression returned by this block
    swift::Expr *orig_expr; // This is the original expression that we resolved
                            // to this type
    swift::ReturnStmt *return_stmt; // If this block does a return, this is the
                                    // return statement
    const swift::SourceLoc source_loc; // This is the source location of this
                                       // return in the overall expression.

    ResultLocationInfo(const swift::SourceLoc &in_source_loc)
        : tmp_var_decl(nullptr), wrapper_stmt(nullptr), binding_decl(nullptr),
          orig_expr(nullptr), return_stmt(nullptr), source_loc(in_source_loc) {}

    ResultLocationInfo(const ResultLocationInfo &rhs)
        : tmp_var_decl(rhs.tmp_var_decl), wrapper_stmt(rhs.wrapper_stmt),
          binding_decl(rhs.binding_decl), orig_expr(rhs.orig_expr),
          return_stmt(rhs.return_stmt), source_loc(rhs.source_loc) {}
  };

  void InsertResult(swift::VarDecl *result_var, swift::Type &result_type,
                    ResultLocationInfo &result_info);

  void InsertError(swift::VarDecl *error_var, swift::Type &error_type);

  std::vector<ResultLocationInfo> m_result_info;
};
}

#endif
