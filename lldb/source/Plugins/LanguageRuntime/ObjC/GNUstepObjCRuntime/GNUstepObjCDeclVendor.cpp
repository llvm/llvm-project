//===-- GNUstepObjCDeclVendor.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GNUstepObjCDeclVendor.h"

#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include "lldb/Core/Module.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExternalASTSource.h"

#include "llvm/ADT/StringExtras.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {

/// Completes an interface the moment clang needs its members, rather than
/// when the class was first named. Mirrors AppleObjCExternalASTSource.
class GNUstepObjCExternalASTSource : public clang::ExternalASTSource {
public:
  explicit GNUstepObjCExternalASTSource(GNUstepObjCDeclVendor &decl_vendor)
      : m_decl_vendor(decl_vendor) {}

  bool FindExternalVisibleDeclsByName(
      const clang::DeclContext *decl_ctx, clang::DeclarationName name,
      const clang::DeclContext *original_dc) override {
    auto *interface_decl = llvm::dyn_cast<clang::ObjCInterfaceDecl>(decl_ctx);
    if (!interface_decl) {
      SetNoExternalVisibleDeclsForName(decl_ctx, name);
      return false;
    }
    if (!m_decl_vendor.FinishDecl(
            const_cast<clang::ObjCInterfaceDecl *>(interface_decl)))
      return false;
    return !interface_decl->lookup(name).empty();
  }

  void CompleteType(clang::TagDecl *tag_decl) override {}

  void CompleteType(clang::ObjCInterfaceDecl *interface_decl) override {
    m_decl_vendor.FinishDecl(interface_decl);
  }

  bool layoutRecordType(
      const clang::RecordDecl *Record, uint64_t &Size, uint64_t &Alignment,
      llvm::DenseMap<const clang::FieldDecl *, uint64_t> &FieldOffsets,
      llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>
          &BaseOffsets,
      llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>
          &VirtualBaseOffsets) override {
    return false;
  }

  void StartTranslationUnit(clang::ASTConsumer *Consumer) override {
    clang::TranslationUnitDecl *tu_decl =
        m_decl_vendor.m_ast_ctx_sp->getASTContext().getTranslationUnitDecl();
    tu_decl->setHasExternalVisibleStorage();
    tu_decl->setHasExternalLexicalStorage();
  }

private:
  GNUstepObjCDeclVendor &m_decl_vendor;
};

} // namespace lldb_private

GNUstepObjCDeclVendor::GNUstepObjCDeclVendor(ObjCLanguageRuntime &runtime)
    : DeclVendor(eGNUstepObjCDeclVendor), m_runtime(runtime) {
  m_ast_ctx_sp = std::make_shared<TypeSystemClang>(
      "GNUstepObjCDeclVendor AST",
      runtime.GetProcess()->GetTarget().GetArchitecture().GetTriple());
  auto external_source_owning_ptr =
      llvm::makeIntrusiveRefCnt<GNUstepObjCExternalASTSource>(*this);
  m_external_source = external_source_owning_ptr.get();
  m_ast_ctx_sp->getASTContext().setExternalSource(external_source_owning_ptr);
}

clang::ObjCInterfaceDecl *
GNUstepObjCDeclVendor::GetDeclForISA(ObjCLanguageRuntime::ObjCISA isa) {
  auto iter = m_isa_to_interface.find(isa);
  if (iter != m_isa_to_interface.end())
    return iter->second;

  ObjCLanguageRuntime::ClassDescriptorSP descriptor =
      m_runtime.GetClassDescriptorFromISA(isa);
  if (!descriptor)
    return nullptr;
  ConstString name(descriptor->GetClassName());
  if (!name)
    return nullptr;

  clang::ASTContext &ast_ctx = m_ast_ctx_sp->getASTContext();
  clang::IdentifierInfo &identifier_info =
      ast_ctx.Idents.get(name.GetStringRef());

  clang::ObjCInterfaceDecl *new_iface_decl = clang::ObjCInterfaceDecl::Create(
      ast_ctx, ast_ctx.getTranslationUnitDecl(), clang::SourceLocation(),
      &identifier_info, /*typeParamList=*/nullptr, /*PrevDecl=*/nullptr);

  // The ISA is how FinishDecl gets back to the runtime from a bare decl.
  ClangASTMetadata meta_data;
  meta_data.SetISAPtr(isa);
  m_ast_ctx_sp->SetMetadata(new_iface_decl, meta_data);

  new_iface_decl->setHasExternalVisibleStorage();
  new_iface_decl->setHasExternalLexicalStorage();
  ast_ctx.getTranslationUnitDecl()->addDecl(new_iface_decl);

  m_isa_to_interface[isa] = new_iface_decl;
  return new_iface_decl;
}

void MethodTypeSplitter::Parse(llvm::StringRef types) {
  std::string current;
  unsigned depth = 0;
  bool in_quotes = false;
  for (char c : types) {
    // A quoted string is a name, not structure: clang's extended encoding
    // spells an object parameter `@"ClassName"` and a qualified one
    // `@"<Protocol>"` (setEncodeClassNames, ASTContext.cpp), and a digit in
    // either is part of the name rather than an argument-frame offset.
    if (in_quotes) {
      current.push_back(c);
      if (c == '"')
        in_quotes = false;
      continue;
    }
    if (c == '"') {
      current.push_back(c);
      in_quotes = true;
      continue;
    }

    const bool is_digit = llvm::isDigit(c);
    if (depth == 0 && is_digit) {
      // An offset: it ends the type that preceded it.
      if (!current.empty()) {
        m_types.push_back(current);
        current.clear();
      }
      continue;
    }
    // Angle brackets bound a block's own signature, which the extended
    // encoding writes inline as `@?<...>` (setEncodeBlockParameters); like an
    // aggregate, everything inside belongs to the one type.
    if (c == '{' || c == '(' || c == '[' || c == '<')
      ++depth;
    else if (c == '}' || c == ')' || c == ']' || c == '>')
      if (depth > 0)
        --depth;
    current.push_back(c);
    // A complete aggregate ends the type too, since no offset follows it
    // until the next argument.
    if (depth == 0 && (c == '}' || c == ')' || c == ']')) {
      m_types.push_back(current);
      current.clear();
    }
  }
  if (!current.empty())
    m_types.push_back(current);
  // An unterminated quote means the encoding was truncated or corrupt. Say so
  // rather than handing back a name that runs to the end of the buffer.
  m_valid = !m_types.empty() && !in_quotes;
}

clang::ObjCMethodDecl *GNUstepObjCDeclVendor::BuildMethodDecl(
    clang::ObjCInterfaceDecl *interface_decl, llvm::StringRef name,
    llvm::StringRef types, bool is_instance_method) {
  MethodTypeSplitter splitter(types);
  if (!splitter.IsValid())
    return nullptr;

  ObjCLanguageRuntime::EncodingToTypeSP encoding_to_type_sp =
      m_runtime.GetEncodingToType();
  if (!encoding_to_type_sp)
    return nullptr;

  clang::ASTContext &ast_ctx = m_ast_ctx_sp->getASTContext();

  // Method types are realized for expressions: unlike an ivar, a return type
  // of @"NSString" is worth resolving, and recursing back into this vendor
  // for it is safe because the method is not part of any interface yet.
  CompilerType return_type = encoding_to_type_sp->RealizeType(
      *m_ast_ctx_sp, splitter.GetReturnType().str().c_str(),
      /*for_expression=*/true);
  if (!return_type)
    return nullptr;

  // "initWithFoo:bar:" has as many pieces as colons; a zero-argument
  // selector is a single identifier with none.
  llvm::SmallVector<llvm::StringRef, 4> pieces;
  name.split(pieces, ':');
  const bool has_colons = name.contains(':');
  if (has_colons && !pieces.empty() && pieces.back().empty())
    pieces.pop_back();
  if (pieces.empty())
    return nullptr;

  llvm::SmallVector<const clang::IdentifierInfo *, 4> selector_pieces;
  for (llvm::StringRef piece : pieces)
    selector_pieces.push_back(&ast_ctx.Idents.get(piece));

  const unsigned num_selector_args = has_colons ? selector_pieces.size() : 0;
  clang::Selector selector =
      ast_ctx.Selectors.getSelector(num_selector_args, selector_pieces.data());

  clang::ObjCMethodDecl *method_decl = clang::ObjCMethodDecl::Create(
      ast_ctx, clang::SourceLocation(), clang::SourceLocation(), selector,
      ClangUtil::GetQualType(return_type), /*ReturnTInfo=*/nullptr,
      interface_decl, is_instance_method, /*isVariadic=*/false,
      /*isPropertyAccessor=*/false, /*isSynthesizedAccessorStub=*/false,
      /*isImplicitlyDeclared=*/true, /*isDefined=*/false,
      clang::ObjCImplementationControl::None,
      /*HasRelatedResultType=*/false);
  if (!method_decl)
    return nullptr;

  llvm::SmallVector<clang::ParmVarDecl *, 4> params;
  for (size_t i = 0; i < splitter.GetNumArguments(); ++i) {
    CompilerType arg_type = encoding_to_type_sp->RealizeType(
        *m_ast_ctx_sp, splitter.GetArgumentType(i).str().c_str(),
        /*for_expression=*/true);
    if (!arg_type)
      return nullptr;
    params.push_back(clang::ParmVarDecl::Create(
        ast_ctx, method_decl, clang::SourceLocation(), clang::SourceLocation(),
        /*Id=*/nullptr, ClangUtil::GetQualType(arg_type), /*TInfo=*/nullptr,
        clang::SC_None, /*DefArg=*/nullptr));
  }
  method_decl->setMethodParams(ast_ctx, params, {});
  return method_decl;
}

bool GNUstepObjCDeclVendor::FinishDecl(
    clang::ObjCInterfaceDecl *interface_decl) {
  if (!interface_decl)
    return false;

  std::optional<ClangASTMetadata> meta_data =
      m_ast_ctx_sp->GetMetadata(interface_decl);
  if (!meta_data)
    return false;
  const ObjCLanguageRuntime::ObjCISA isa = meta_data->GetISAPtr();
  if (isa == 0)
    return false;

  // Already done. This also terminates the superclass recursion below: a
  // cycle in the isa graph re-enters here and stops.
  if (!interface_decl->hasExternalVisibleStorage())
    return true;

  // Ask the runtime before publishing anything. Starting the definition and
  // then failing would leave a members-less interface that the guard above
  // makes permanent, so a class that was momentarily unreadable would stay
  // empty for the rest of the session.
  ObjCLanguageRuntime::ClassDescriptorSP descriptor =
      m_runtime.GetClassDescriptorFromISA(isa);
  if (!descriptor)
    return false;

  // The definition has to be started, and the external-storage bits cleared,
  // *before* the callbacks run - otherwise a class that reaches itself
  // through its superclass chain recurses without bound.
  interface_decl->startDefinition();
  interface_decl->setHasExternalVisibleStorage(false);
  interface_decl->setHasExternalLexicalStorage(false);

  ObjCLanguageRuntime::EncodingToTypeSP encoding_to_type_sp =
      m_runtime.GetEncodingToType();
  Log *log = GetLog(LLDBLog::Types);

  auto superclass_func = [this,
                          interface_decl](ObjCLanguageRuntime::ObjCISA super) {
    clang::ObjCInterfaceDecl *superclass_decl = GetDeclForISA(super);
    if (!superclass_decl || superclass_decl == interface_decl)
      return;
    // A superclass has to be complete before it can be attached, so this is
    // eager where everything else here is lazy.
    FinishDecl(superclass_decl);

    // The recursion latch above stops FinishDecl re-entering, but it does not
    // stop the *edge* being written: with A's superclass B and B's superclass
    // A, both setSuperClass calls would still run and leave a cycle in the
    // AST that clang then walks without bound. Reject any edge that is
    // already reachable upwards from the proposed superclass.
    for (const clang::ObjCInterfaceDecl *ancestor = superclass_decl; ancestor;
         ancestor = ancestor->getSuperClass()) {
      if (ancestor == interface_decl) {
        LLDB_LOG(GetLog(LLDBLog::Types),
                 "GNUstep class {0} would close a superclass cycle; leaving it "
                 "without a superclass",
                 interface_decl->getName());
        return;
      }
    }

    clang::ASTContext &ast_ctx = m_ast_ctx_sp->getASTContext();
    interface_decl->setSuperClass(ast_ctx.getTrivialTypeSourceInfo(
        ast_ctx.getObjCInterfaceType(superclass_decl)));
  };

  auto ivar_func = [this, interface_decl, &encoding_to_type_sp,
                    log](const char *name, const char *type,
                         lldb::addr_t offset, uint64_t size) -> bool {
    if (!name || !type || !encoding_to_type_sp)
      return false;
    // Deliberately not for_expression: an ivar's precise Objective-C type
    // would send the parser back through this vendor while it is mid-
    // FinishDecl, and dynamic typing resolves it anyway.
    CompilerType ivar_type = encoding_to_type_sp->RealizeType(
        *m_ast_ctx_sp, type, /*for_expression=*/false);
    if (!ivar_type) {
      // The ivar still occupies space even when its encoding cannot be
      // realized. Dropping it would leave clang to lay the interface out from
      // the ivars that remain, silently shifting the offset of every ivar
      // after it - so the debugger would report wrong values rather than
      // decline to answer. Stand in an opaque block of the right size.
      LLDB_LOG(log,
               "GNUstep ivar {0} has an unrealizable type {1}; substituting "
               "an opaque {2}-byte placeholder",
               name, type, size);
      if (size == 0)
        return false;
      ivar_type = m_ast_ctx_sp->CreateArrayType(
          m_ast_ctx_sp->GetBasicType(lldb::eBasicTypeChar), size,
          /*is_vector=*/false);
      if (!ivar_type)
        return false;
    }
    clang::ASTContext &ast_ctx = m_ast_ctx_sp->getASTContext();
    clang::ObjCIvarDecl *ivar_decl = clang::ObjCIvarDecl::Create(
        ast_ctx, interface_decl, clang::SourceLocation(),
        clang::SourceLocation(), &ast_ctx.Idents.get(name),
        ClangUtil::GetQualType(ivar_type), /*TInfo=*/nullptr,
        clang::ObjCIvarDecl::Public, /*BW=*/nullptr, /*synthesized=*/false);
    if (ivar_decl)
      interface_decl->addDecl(ivar_decl);
    return false;
  };

  auto make_method_func = [this, interface_decl, log](bool is_instance_method) {
    return [this, interface_decl, log,
            is_instance_method](const char *name, const char *types) -> bool {
      if (!name || !types)
        return false;
      if (clang::ObjCMethodDecl *method_decl =
              BuildMethodDecl(interface_decl, name, types, is_instance_method))
        interface_decl->addDecl(method_decl);
      else
        LLDB_LOG(log, "GNUstep method {0} has an unrealizable signature {1}",
                 name, types);
      return false;
    };
  };

  if (!descriptor->Describe(
          superclass_func, make_method_func(/*is_instance_method=*/true),
          make_method_func(/*is_instance_method=*/false), ivar_func)) {
    LLDB_LOG(log, "GNUstep runtime could not describe class {0}",
             descriptor->GetClassName());
    return false;
  }
  return true;
}

uint32_t GNUstepObjCDeclVendor::FindDecls(ConstString name, bool append,
                                          uint32_t max_matches,
                                          std::vector<CompilerDecl> &decls) {
  if (!append)
    decls.clear();
  if (!name || max_matches == 0)
    return 0;

  clang::ASTContext &ast_ctx = m_ast_ctx_sp->getASTContext();
  clang::IdentifierInfo &identifier_info =
      ast_ctx.Idents.get(name.GetStringRef());
  clang::DeclarationName decl_name =
      ast_ctx.DeclarationNames.getIdentifier(&identifier_info);

  // Anything already vended stays vended, so repeated lookups of the same
  // name return the same decl rather than a fresh one.
  for (clang::NamedDecl *candidate :
       ast_ctx.getTranslationUnitDecl()->lookup(decl_name)) {
    if (auto *result_iface_decl =
            llvm::dyn_cast<clang::ObjCInterfaceDecl>(candidate)) {
      decls.push_back(m_ast_ctx_sp->GetCompilerDecl(result_iface_decl));
      return 1;
    }
  }

  const ObjCLanguageRuntime::ObjCISA isa = m_runtime.GetISA(name);
  if (isa == 0)
    return 0;

  clang::ObjCInterfaceDecl *iface_decl = GetDeclForISA(isa);
  if (!iface_decl)
    return 0;

  decls.push_back(m_ast_ctx_sp->GetCompilerDecl(iface_decl));
  return 1;
}
