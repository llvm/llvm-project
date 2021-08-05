//===-- ClangASTSource.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangASTSource.h"

#include "ClangDeclVendor.h"
#include "ClangModulesDeclVendor.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Symbol/CompilerDeclContext.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/TaggedASTType.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Log.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/SourceManager.h"

#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include <memory>
#include <vector>

using namespace clang;
using namespace lldb_private;

// Scoped class that will remove an active lexical decl from the set when it
// goes out of scope.
namespace {
class ScopedLexicalDeclEraser {
public:
  ScopedLexicalDeclEraser(std::set<const clang::Decl *> &decls,
                          const clang::Decl *decl)
      : m_active_lexical_decls(decls), m_decl(decl) {}

  ~ScopedLexicalDeclEraser() { m_active_lexical_decls.erase(m_decl); }

private:
  std::set<const clang::Decl *> &m_active_lexical_decls;
  const clang::Decl *m_decl;
};
}

ClangASTSource::ClangASTSource(
    const lldb::TargetSP &target,
    const std::shared_ptr<ClangASTImporter> &importer)
    : m_lookups_enabled(false), m_target(target), m_ast_context(nullptr),
      m_ast_importer_sp(importer), m_active_lexical_decls(),
      m_active_lookups() {
  assert(m_ast_importer_sp && "No ClangASTImporter passed to ClangASTSource?");
}

void ClangASTSource::InstallASTContext(TypeSystemClang &clang_ast_context) {
  m_ast_context = &clang_ast_context.getASTContext();
  m_clang_ast_context = &clang_ast_context;
  m_file_manager = &m_ast_context->getSourceManager().getFileManager();
  m_ast_importer_sp->InstallMapCompleter(m_ast_context, *this);
}

ClangASTSource::~ClangASTSource() {
  m_ast_importer_sp->ForgetDestination(m_ast_context);

  if (!m_target)
    return;

  // Unregister the current ASTContext as a source for all scratch
  // ASTContexts in the ClangASTImporter. Without this the scratch AST might
  // query the deleted ASTContext for additional type information.
  // We unregister from *all* scratch ASTContexts in case a type got exported
  // to a scratch AST that isn't the best fitting scratch ASTContext.
  TypeSystemClang *scratch_ast = ScratchTypeSystemClang::GetForTarget(
      *m_target, ScratchTypeSystemClang::DefaultAST, false);

  if (!scratch_ast)
    return;

  ScratchTypeSystemClang *default_scratch_ast =
      llvm::cast<ScratchTypeSystemClang>(scratch_ast);
  // Unregister from the default scratch AST (and all sub-ASTs).
  default_scratch_ast->ForgetSource(m_ast_context, *m_ast_importer_sp);
}

void ClangASTSource::StartTranslationUnit(ASTConsumer *Consumer) {
  if (!m_ast_context)
    return;

  m_ast_context->getTranslationUnitDecl()->setHasExternalVisibleStorage();
  m_ast_context->getTranslationUnitDecl()->setHasExternalLexicalStorage();
}

// The core lookup interface.
bool ClangASTSource::FindExternalVisibleDeclsByName(
    const DeclContext *decl_ctx, DeclarationName clang_decl_name) {
  if (!m_ast_context) {
    SetNoExternalVisibleDeclsForName(decl_ctx, clang_decl_name);
    return false;
  }

  std::string decl_name(clang_decl_name.getAsString());

  switch (clang_decl_name.getNameKind()) {
  // Normal identifiers.
  case DeclarationName::Identifier: {
    clang::IdentifierInfo *identifier_info =
        clang_decl_name.getAsIdentifierInfo();

    if (!identifier_info || identifier_info->getBuiltinID() != 0) {
      SetNoExternalVisibleDeclsForName(decl_ctx, clang_decl_name);
      return false;
    }
  } break;

  // Operator names.
  case DeclarationName::CXXOperatorName:
  case DeclarationName::CXXLiteralOperatorName:
    break;

  // Using directives found in this context.
  // Tell Sema we didn't find any or we'll end up getting asked a *lot*.
  case DeclarationName::CXXUsingDirective:
    SetNoExternalVisibleDeclsForName(decl_ctx, clang_decl_name);
    return false;

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector: {
    llvm::SmallVector<NamedDecl *, 1> method_decls;

    NameSearchContext method_search_context(*m_clang_ast_context, method_decls,
                                            clang_decl_name, decl_ctx);

    FindObjCMethodDecls(method_search_context);

    SetExternalVisibleDeclsForName(decl_ctx, clang_decl_name, method_decls);
    return (method_decls.size() > 0);
  }
  // These aren't possible in the global context.
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
  case DeclarationName::CXXDeductionGuideName:
    SetNoExternalVisibleDeclsForName(decl_ctx, clang_decl_name);
    return false;
  }

  if (!GetLookupsEnabled()) {
    // Wait until we see a '$' at the start of a name before we start doing any
    // lookups so we can avoid lookup up all of the builtin types.
    if (!decl_name.empty() && decl_name[0] == '$') {
      SetLookupsEnabled(true);
    } else {
      SetNoExternalVisibleDeclsForName(decl_ctx, clang_decl_name);
      return false;
    }
  }

  ConstString const_decl_name(decl_name.c_str());

  const char *uniqued_const_decl_name = const_decl_name.GetCString();
  if (m_active_lookups.find(uniqued_const_decl_name) !=
      m_active_lookups.end()) {
    // We are currently looking up this name...
    SetNoExternalVisibleDeclsForName(decl_ctx, clang_decl_name);
    return false;
  }
  m_active_lookups.insert(uniqued_const_decl_name);
  llvm::SmallVector<NamedDecl *, 4> name_decls;
  NameSearchContext name_search_context(*m_clang_ast_context, name_decls,
                                        clang_decl_name, decl_ctx);
  FindExternalVisibleDecls(name_search_context);
  SetExternalVisibleDeclsForName(decl_ctx, clang_decl_name, name_decls);
  m_active_lookups.erase(uniqued_const_decl_name);
  return (name_decls.size() != 0);
}

TagDecl *ClangASTSource::FindCompleteType(const TagDecl *decl) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  if (const NamespaceDecl *namespace_context =
          dyn_cast<NamespaceDecl>(decl->getDeclContext())) {
    ClangASTImporter::NamespaceMapSP namespace_map =
        m_ast_importer_sp->GetNamespaceMap(namespace_context);

    LLDB_LOGV(log, "      CTD Inspecting namespace map{0} ({1} entries)",
              namespace_map.get(), namespace_map->size());

    if (!namespace_map)
      return nullptr;

    for (const ClangASTImporter::NamespaceMapItem &item : *namespace_map) {
      LLDB_LOG(log, "      CTD Searching namespace {0} in module {1}",
               item.second.GetName(), item.first->GetFileSpec().GetFilename());

      TypeList types;

      ConstString name(decl->getName());

      item.first->FindTypesInNamespace(name, item.second, UINT32_MAX, types);

      for (uint32_t ti = 0, te = types.GetSize(); ti != te; ++ti) {
        lldb::TypeSP type = types.GetTypeAtIndex(ti);

        if (!type)
          continue;

        CompilerType clang_type(type->GetFullCompilerType());

        if (!ClangUtil::IsClangType(clang_type))
          continue;

        const TagType *tag_type =
            ClangUtil::GetQualType(clang_type)->getAs<TagType>();

        if (!tag_type)
          continue;

        TagDecl *candidate_tag_decl =
            const_cast<TagDecl *>(tag_type->getDecl());

        if (TypeSystemClang::GetCompleteDecl(
                &candidate_tag_decl->getASTContext(), candidate_tag_decl))
          return candidate_tag_decl;
      }
    }
  } else {
    TypeList types;

    ConstString name(decl->getName());

    const ModuleList &module_list = m_target->GetImages();

    bool exact_match = false;
    llvm::DenseSet<SymbolFile *> searched_symbol_files;
    module_list.FindTypes(nullptr, name, exact_match, UINT32_MAX,
                          searched_symbol_files, types);

    for (uint32_t ti = 0, te = types.GetSize(); ti != te; ++ti) {
      lldb::TypeSP type = types.GetTypeAtIndex(ti);

      if (!type)
        continue;

      CompilerType clang_type(type->GetFullCompilerType());

      if (!ClangUtil::IsClangType(clang_type))
        continue;

      const TagType *tag_type =
          ClangUtil::GetQualType(clang_type)->getAs<TagType>();

      if (!tag_type)
        continue;

      TagDecl *candidate_tag_decl = const_cast<TagDecl *>(tag_type->getDecl());

      // We have found a type by basename and we need to make sure the decl
      // contexts are the same before we can try to complete this type with
      // another
      if (!TypeSystemClang::DeclsAreEquivalent(const_cast<TagDecl *>(decl),
                                               candidate_tag_decl))
        continue;

      if (TypeSystemClang::GetCompleteDecl(&candidate_tag_decl->getASTContext(),
                                           candidate_tag_decl))
        return candidate_tag_decl;
    }
  }
  return nullptr;
}

void ClangASTSource::CompleteType(TagDecl *tag_decl) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  if (log) {
    LLDB_LOG(log,
             "    CompleteTagDecl on (ASTContext*){0} Completing "
             "(TagDecl*){1} named {2}",
             m_clang_ast_context->getDisplayName(), tag_decl,
             tag_decl->getName());

    LLDB_LOG(log, "      CTD Before:\n{0}", ClangUtil::DumpDecl(tag_decl));
  }

  auto iter = m_active_lexical_decls.find(tag_decl);
  if (iter != m_active_lexical_decls.end())
    return;
  m_active_lexical_decls.insert(tag_decl);
  ScopedLexicalDeclEraser eraser(m_active_lexical_decls, tag_decl);

  if (!m_ast_importer_sp->CompleteTagDecl(tag_decl)) {
    // We couldn't complete the type.  Maybe there's a definition somewhere
    // else that can be completed.
    if (TagDecl *alternate = FindCompleteType(tag_decl))
      m_ast_importer_sp->CompleteTagDeclWithOrigin(tag_decl, alternate);
  }

  LLDB_LOG(log, "      [CTD] After:\n{0}", ClangUtil::DumpDecl(tag_decl));
}

void ClangASTSource::CompleteType(clang::ObjCInterfaceDecl *interface_decl) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  LLDB_LOG(log,
           "    [CompleteObjCInterfaceDecl] on (ASTContext*){0} '{1}' "
           "Completing an ObjCInterfaceDecl named {1}",
           m_ast_context, m_clang_ast_context->getDisplayName(),
           interface_decl->getName());
  LLDB_LOG(log, "      [COID] Before:\n{0}",
           ClangUtil::DumpDecl(interface_decl));

  ClangASTImporter::DeclOrigin original = m_ast_importer_sp->GetDeclOrigin(interface_decl);

  if (original.Valid()) {
    if (ObjCInterfaceDecl *original_iface_decl =
            dyn_cast<ObjCInterfaceDecl>(original.decl)) {
      ObjCInterfaceDecl *complete_iface_decl =
          GetCompleteObjCInterface(original_iface_decl);

      if (complete_iface_decl && (complete_iface_decl != original_iface_decl)) {
        m_ast_importer_sp->SetDeclOrigin(interface_decl, complete_iface_decl);
      }
    }
  }

  m_ast_importer_sp->CompleteObjCInterfaceDecl(interface_decl);

  if (interface_decl->getSuperClass() &&
      interface_decl->getSuperClass() != interface_decl)
    CompleteType(interface_decl->getSuperClass());

  LLDB_LOG(log, "      [COID] After:");
  LLDB_LOG(log, "      [COID] {0}", ClangUtil::DumpDecl(interface_decl));
}

clang::ObjCInterfaceDecl *ClangASTSource::GetCompleteObjCInterface(
    const clang::ObjCInterfaceDecl *interface_decl) {
  lldb::ProcessSP process(m_target->GetProcessSP());

  if (!process)
    return nullptr;

  ObjCLanguageRuntime *language_runtime(ObjCLanguageRuntime::Get(*process));

  if (!language_runtime)
    return nullptr;

  ConstString class_name(interface_decl->getNameAsString().c_str());

  lldb::TypeSP complete_type_sp(
      language_runtime->LookupInCompleteClassCache(class_name));

  if (!complete_type_sp)
    return nullptr;

  TypeFromUser complete_type =
      TypeFromUser(complete_type_sp->GetFullCompilerType());
  lldb::opaque_compiler_type_t complete_opaque_type =
      complete_type.GetOpaqueQualType();

  if (!complete_opaque_type)
    return nullptr;

  const clang::Type *complete_clang_type =
      QualType::getFromOpaquePtr(complete_opaque_type).getTypePtr();
  const ObjCInterfaceType *complete_interface_type =
      dyn_cast<ObjCInterfaceType>(complete_clang_type);

  if (!complete_interface_type)
    return nullptr;

  ObjCInterfaceDecl *complete_iface_decl(complete_interface_type->getDecl());

  return complete_iface_decl;
}

void ClangASTSource::FindExternalLexicalDecls(
    const DeclContext *decl_context,
    llvm::function_ref<bool(Decl::Kind)> predicate,
    llvm::SmallVectorImpl<Decl *> &decls) {

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  const Decl *context_decl = dyn_cast<Decl>(decl_context);

  if (!context_decl)
    return;

  auto iter = m_active_lexical_decls.find(context_decl);
  if (iter != m_active_lexical_decls.end())
    return;
  m_active_lexical_decls.insert(context_decl);
  ScopedLexicalDeclEraser eraser(m_active_lexical_decls, context_decl);

  if (log) {
    if (const NamedDecl *context_named_decl = dyn_cast<NamedDecl>(context_decl))
      LLDB_LOG(log,
               "FindExternalLexicalDecls on (ASTContext*){0} '{1}' in "
               "'{2}' (%sDecl*){3}",
               m_ast_context, m_clang_ast_context->getDisplayName(),
               context_named_decl->getNameAsString().c_str(),
               context_decl->getDeclKindName(),
               static_cast<const void *>(context_decl));
    else if (context_decl)
      LLDB_LOG(log,
               "FindExternalLexicalDecls on (ASTContext*){0} '{1}' in "
               "({2}Decl*){3}",
               m_ast_context, m_clang_ast_context->getDisplayName(),
               context_decl->getDeclKindName(),
               static_cast<const void *>(context_decl));
    else
      LLDB_LOG(log,
               "FindExternalLexicalDecls on (ASTContext*){0} '{1}' in a "
               "NULL context",
               m_ast_context, m_clang_ast_context->getDisplayName());
  }

  ClangASTImporter::DeclOrigin original = m_ast_importer_sp->GetDeclOrigin(context_decl);

  if (!original.Valid())
    return;

  LLDB_LOG(log, "  FELD Original decl {0} (Decl*){1:x}:\n{2}",
           static_cast<void *>(original.ctx),
           static_cast<void *>(original.decl),
           ClangUtil::DumpDecl(original.decl));

  if (ObjCInterfaceDecl *original_iface_decl =
          dyn_cast<ObjCInterfaceDecl>(original.decl)) {
    ObjCInterfaceDecl *complete_iface_decl =
        GetCompleteObjCInterface(original_iface_decl);

    if (complete_iface_decl && (complete_iface_decl != original_iface_decl)) {
      original.decl = complete_iface_decl;
      original.ctx = &complete_iface_decl->getASTContext();

      m_ast_importer_sp->SetDeclOrigin(context_decl, complete_iface_decl);
    }
  }

  if (TagDecl *original_tag_decl = dyn_cast<TagDecl>(original.decl)) {
    ExternalASTSource *external_source = original.ctx->getExternalSource();

    if (external_source)
      external_source->CompleteType(original_tag_decl);
  }

  const DeclContext *original_decl_context =
      dyn_cast<DeclContext>(original.decl);

  if (!original_decl_context)
    return;

  // Indicates whether we skipped any Decls of the original DeclContext.
  bool SkippedDecls = false;
  for (Decl *decl : original_decl_context->decls()) {
    // The predicate function returns true if the passed declaration kind is
    // the one we are looking for.
    // See clang::ExternalASTSource::FindExternalLexicalDecls()
    if (predicate(decl->getKind())) {
      if (log) {
        std::string ast_dump = ClangUtil::DumpDecl(decl);
        if (const NamedDecl *context_named_decl =
                dyn_cast<NamedDecl>(context_decl))
          LLDB_LOG(log, "  FELD Adding [to {0}Decl {1}] lexical {2}Decl {3}",
                   context_named_decl->getDeclKindName(),
                   context_named_decl->getName(), decl->getDeclKindName(),
                   ast_dump);
        else
          LLDB_LOG(log, "  FELD Adding lexical {0}Decl {1}",
                   decl->getDeclKindName(), ast_dump);
      }

      CopyDecl(decl);

      // FIXME: We should add the copied decl to the 'decls' list. This would
      // add the copied Decl into the DeclContext and make sure that we
      // correctly propagate that we added some Decls back to Clang.
      // By leaving 'decls' empty we incorrectly return false from
      // DeclContext::LoadLexicalDeclsFromExternalStorage which might cause
      // lookup issues later on.
      // We can't just add them for now as the ASTImporter already added the
      // decl into the DeclContext and this would add it twice.
    } else {
      SkippedDecls = true;
    }
  }

  // CopyDecl may build a lookup table which may set up ExternalLexicalStorage
  // to false.  However, since we skipped some of the external Decls we must
  // set it back!
  if (SkippedDecls) {
    decl_context->setHasExternalLexicalStorage(true);
    // This sets HasLazyExternalLexicalLookups to true.  By setting this bit we
    // ensure that the lookup table is rebuilt, which means the external source
    // is consulted again when a clang::DeclContext::lookup is called.
    const_cast<DeclContext *>(decl_context)->setMustBuildLookupTable();
  }

  return;
}

void ClangASTSource::FindExternalVisibleDecls(NameSearchContext &context) {
  assert(m_ast_context);

  const ConstString name(context.m_decl_name.getAsString().c_str());

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  if (log) {
    if (!context.m_decl_context)
      LLDB_LOG(log,
               "ClangASTSource::FindExternalVisibleDecls on "
               "(ASTContext*){0} '{1}' for '{2}' in a NULL DeclContext",
               m_ast_context, m_clang_ast_context->getDisplayName(), name);
    else if (const NamedDecl *context_named_decl =
                 dyn_cast<NamedDecl>(context.m_decl_context))
      LLDB_LOG(log,
               "ClangASTSource::FindExternalVisibleDecls on "
               "(ASTContext*){0} '{1}' for '{2}' in '{3}'",
               m_ast_context, m_clang_ast_context->getDisplayName(), name,
               context_named_decl->getName());
    else
      LLDB_LOG(log,
               "ClangASTSource::FindExternalVisibleDecls on "
               "(ASTContext*){0} '{1}' for '{2}' in a '{3}'",
               m_ast_context, m_clang_ast_context->getDisplayName(), name,
               context.m_decl_context->getDeclKindName());
  }

  if (isa<NamespaceDecl>(context.m_decl_context)) {
    LookupInNamespace(context);
  } else if (isa<ObjCInterfaceDecl>(context.m_decl_context)) {
    FindObjCPropertyAndIvarDecls(context);
  } else if (!isa<TranslationUnitDecl>(context.m_decl_context)) {
    // we shouldn't be getting FindExternalVisibleDecls calls for these
    return;
  } else {
    CompilerDeclContext namespace_decl;

    LLDB_LOG(log, "  CAS::FEVD Searching the root namespace");

    FindExternalVisibleDecls(context, lldb::ModuleSP(), namespace_decl);
  }

  if (!context.m_namespace_map->empty()) {
    if (log && log->GetVerbose())
      LLDB_LOG(log, "  CAS::FEVD Registering namespace map {0} ({1} entries)",
               context.m_namespace_map.get(), context.m_namespace_map->size());

    NamespaceDecl *clang_namespace_decl =
        AddNamespace(context, context.m_namespace_map);

    if (clang_namespace_decl)
      clang_namespace_decl->setHasExternalVisibleStorage();
  }
}

clang::Sema *ClangASTSource::getSema() {
  return m_clang_ast_context->getSema();
}

bool ClangASTSource::IgnoreName(const ConstString name,
                                bool ignore_all_dollar_names) {
  static const ConstString id_name("id");
  static const ConstString Class_name("Class");

  if (m_ast_context->getLangOpts().ObjC)
    if (name == id_name || name == Class_name)
      return true;

  StringRef name_string_ref = name.GetStringRef();

  // The ClangASTSource is not responsible for finding $-names.
  return name_string_ref.empty() ||
         (ignore_all_dollar_names && name_string_ref.startswith("$")) ||
         name_string_ref.startswith("_$");
}

void ClangASTSource::FindExternalVisibleDecls(
    NameSearchContext &context, lldb::ModuleSP module_sp,
    CompilerDeclContext &namespace_decl) {
  assert(m_ast_context);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  SymbolContextList sc_list;

  const ConstString name(context.m_decl_name.getAsString().c_str());
  if (IgnoreName(name, true))
    return;

  if (!m_target)
    return;

  FillNamespaceMap(context, module_sp, namespace_decl);

  if (context.m_found_type)
    return;

  TypeList types;
  const bool exact_match = true;
  llvm::DenseSet<lldb_private::SymbolFile *> searched_symbol_files;
  if (module_sp && namespace_decl)
    module_sp->FindTypesInNamespace(name, namespace_decl, 1, types);
  else {
    m_target->GetImages().FindTypes(module_sp.get(), name, exact_match, 1,
                                    searched_symbol_files, types);
  }

  if (size_t num_types = types.GetSize()) {
    for (size_t ti = 0; ti < num_types; ++ti) {
      lldb::TypeSP type_sp = types.GetTypeAtIndex(ti);

      if (log) {
        const char *name_string = type_sp->GetName().GetCString();

        LLDB_LOG(log, "  CAS::FEVD Matching type found for \"{0}\": {1}", name,
                 (name_string ? name_string : "<anonymous>"));
      }

      CompilerType full_type = type_sp->GetFullCompilerType();

      CompilerType copied_clang_type(GuardedCopyType(full_type));

      if (!copied_clang_type) {
        LLDB_LOG(log, "  CAS::FEVD - Couldn't export a type");

        continue;
      }

      context.AddTypeDecl(copied_clang_type);

      context.m_found_type = true;
      break;
    }
  }

  if (!context.m_found_type) {
    // Try the modules next.
    FindDeclInModules(context, name);
  }

  if (!context.m_found_type) {
    FindDeclInObjCRuntime(context, name);
  }
}

void ClangASTSource::FillNamespaceMap(
    NameSearchContext &context, lldb::ModuleSP module_sp,
    const CompilerDeclContext &namespace_decl) {
  const ConstString name(context.m_decl_name.getAsString().c_str());
  if (IgnoreName(name, true))
    return;

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  if (module_sp && namespace_decl) {
    CompilerDeclContext found_namespace_decl;

    if (SymbolFile *symbol_file = module_sp->GetSymbolFile()) {
      found_namespace_decl = symbol_file->FindNamespace(name, namespace_decl);

      if (found_namespace_decl) {
        context.m_namespace_map->push_back(
            std::pair<lldb::ModuleSP, CompilerDeclContext>(
                module_sp, found_namespace_decl));

        LLDB_LOG(log, "  CAS::FEVD Found namespace {0} in module {1}", name,
                 module_sp->GetFileSpec().GetFilename());
      }
    }
    return;
  }

  for (lldb::ModuleSP image : m_target->GetImages().Modules()) {
    if (!image)
      continue;

    CompilerDeclContext found_namespace_decl;

    SymbolFile *symbol_file = image->GetSymbolFile();

    if (!symbol_file)
      continue;

    found_namespace_decl = symbol_file->FindNamespace(name, namespace_decl);

    if (found_namespace_decl) {
      context.m_namespace_map->push_back(
          std::pair<lldb::ModuleSP, CompilerDeclContext>(image,
                                                         found_namespace_decl));

      LLDB_LOG(log, "  CAS::FEVD Found namespace {0} in module {1}", name,
               image->GetFileSpec().GetFilename());
    }
  }
}

template <class D> class TaggedASTDecl {
public:
  TaggedASTDecl() : decl(nullptr) {}
  TaggedASTDecl(D *_decl) : decl(_decl) {}
  bool IsValid() const { return (decl != nullptr); }
  bool IsInvalid() const { return !IsValid(); }
  D *operator->() const { return decl; }
  D *decl;
};

template <class D2, template <class D> class TD, class D1>
TD<D2> DynCast(TD<D1> source) {
  return TD<D2>(dyn_cast<D2>(source.decl));
}

template <class D = Decl> class DeclFromParser;
template <class D = Decl> class DeclFromUser;

template <class D> class DeclFromParser : public TaggedASTDecl<D> {
public:
  DeclFromParser() : TaggedASTDecl<D>() {}
  DeclFromParser(D *_decl) : TaggedASTDecl<D>(_decl) {}

  DeclFromUser<D> GetOrigin(ClangASTSource &source);
};

template <class D> class DeclFromUser : public TaggedASTDecl<D> {
public:
  DeclFromUser() : TaggedASTDecl<D>() {}
  DeclFromUser(D *_decl) : TaggedASTDecl<D>(_decl) {}

  DeclFromParser<D> Import(ClangASTSource &source);
};

template <class D>
DeclFromUser<D> DeclFromParser<D>::GetOrigin(ClangASTSource &source) {
  ClangASTImporter::DeclOrigin origin = source.GetDeclOrigin(this->decl);
  if (!origin.Valid())
    return DeclFromUser<D>();
  return DeclFromUser<D>(dyn_cast<D>(origin.decl));
}

template <class D>
DeclFromParser<D> DeclFromUser<D>::Import(ClangASTSource &source) {
  DeclFromParser<> parser_generic_decl(source.CopyDecl(this->decl));
  if (parser_generic_decl.IsInvalid())
    return DeclFromParser<D>();
  return DeclFromParser<D>(dyn_cast<D>(parser_generic_decl.decl));
}

bool ClangASTSource::FindObjCMethodDeclsWithOrigin(
    NameSearchContext &context, ObjCInterfaceDecl *original_interface_decl,
    const char *log_info) {
  const DeclarationName &decl_name(context.m_decl_name);
  clang::ASTContext *original_ctx = &original_interface_decl->getASTContext();

  Selector original_selector;

  if (decl_name.isObjCZeroArgSelector()) {
    IdentifierInfo *ident = &original_ctx->Idents.get(decl_name.getAsString());
    original_selector = original_ctx->Selectors.getSelector(0, &ident);
  } else if (decl_name.isObjCOneArgSelector()) {
    const std::string &decl_name_string = decl_name.getAsString();
    std::string decl_name_string_without_colon(decl_name_string.c_str(),
                                               decl_name_string.length() - 1);
    IdentifierInfo *ident =
        &original_ctx->Idents.get(decl_name_string_without_colon);
    original_selector = original_ctx->Selectors.getSelector(1, &ident);
  } else {
    SmallVector<IdentifierInfo *, 4> idents;

    clang::Selector sel = decl_name.getObjCSelector();

    unsigned num_args = sel.getNumArgs();

    for (unsigned i = 0; i != num_args; ++i) {
      idents.push_back(&original_ctx->Idents.get(sel.getNameForSlot(i)));
    }

    original_selector =
        original_ctx->Selectors.getSelector(num_args, idents.data());
  }

  DeclarationName original_decl_name(original_selector);

  llvm::SmallVector<NamedDecl *, 1> methods;

  TypeSystemClang::GetCompleteDecl(original_ctx, original_interface_decl);

  if (ObjCMethodDecl *instance_method_decl =
          original_interface_decl->lookupInstanceMethod(original_selector)) {
    methods.push_back(instance_method_decl);
  } else if (ObjCMethodDecl *class_method_decl =
                 original_interface_decl->lookupClassMethod(
                     original_selector)) {
    methods.push_back(class_method_decl);
  }

  if (methods.empty()) {
    return false;
  }

  for (NamedDecl *named_decl : methods) {
    if (!named_decl)
      continue;

    ObjCMethodDecl *result_method = dyn_cast<ObjCMethodDecl>(named_decl);

    if (!result_method)
      continue;

    Decl *copied_decl = CopyDecl(result_method);

    if (!copied_decl)
      continue;

    ObjCMethodDecl *copied_method_decl = dyn_cast<ObjCMethodDecl>(copied_decl);

    if (!copied_method_decl)
      continue;

    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

    LLDB_LOG(log, "  CAS::FOMD found ({0}) {1}", log_info,
             ClangUtil::DumpDecl(copied_method_decl));

    context.AddNamedDecl(copied_method_decl);
  }

  return true;
}

void ClangASTSource::FindDeclInModules(NameSearchContext &context,
                                       ConstString name) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  std::shared_ptr<ClangModulesDeclVendor> modules_decl_vendor =
      GetClangModulesDeclVendor();
  if (!modules_decl_vendor)
    return;

  bool append = false;
  uint32_t max_matches = 1;
  std::vector<clang::NamedDecl *> decls;

  if (!modules_decl_vendor->FindDecls(name, append, max_matches, decls))
    return;

  LLDB_LOG(log, "  CAS::FEVD Matching entity found for \"{0}\" in the modules",
           name);

  clang::NamedDecl *const decl_from_modules = decls[0];

  if (llvm::isa<clang::TypeDecl>(decl_from_modules) ||
      llvm::isa<clang::ObjCContainerDecl>(decl_from_modules) ||
      llvm::isa<clang::EnumConstantDecl>(decl_from_modules)) {
    clang::Decl *copied_decl = CopyDecl(decl_from_modules);
    clang::NamedDecl *copied_named_decl =
        copied_decl ? dyn_cast<clang::NamedDecl>(copied_decl) : nullptr;

    if (!copied_named_decl) {
      LLDB_LOG(log, "  CAS::FEVD - Couldn't export a type from the modules");

      return;
    }

    context.AddNamedDecl(copied_named_decl);

    context.m_found_type = true;
  }
}

void ClangASTSource::FindDeclInObjCRuntime(NameSearchContext &context,
                                           ConstString name) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  lldb::ProcessSP process(m_target->GetProcessSP());

  if (!process)
    return;

  ObjCLanguageRuntime *language_runtime(ObjCLanguageRuntime::Get(*process));

  if (!language_runtime)
    return;

  DeclVendor *decl_vendor = language_runtime->GetDeclVendor();

  if (!decl_vendor)
    return;

  bool append = false;
  uint32_t max_matches = 1;
  std::vector<clang::NamedDecl *> decls;

  auto *clang_decl_vendor = llvm::cast<ClangDeclVendor>(decl_vendor);
  if (!clang_decl_vendor->FindDecls(name, append, max_matches, decls))
    return;

  LLDB_LOG(log, "  CAS::FEVD Matching type found for \"{0}\" in the runtime",
           name);

  clang::Decl *copied_decl = CopyDecl(decls[0]);
  clang::NamedDecl *copied_named_decl =
      copied_decl ? dyn_cast<clang::NamedDecl>(copied_decl) : nullptr;

  if (!copied_named_decl) {
    LLDB_LOG(log, "  CAS::FEVD - Couldn't export a type from the runtime");

    return;
  }

  context.AddNamedDecl(copied_named_decl);
}

void ClangASTSource::FindObjCMethodDecls(NameSearchContext &context) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  const DeclarationName &decl_name(context.m_decl_name);
  const DeclContext *decl_ctx(context.m_decl_context);

  const ObjCInterfaceDecl *interface_decl =
      dyn_cast<ObjCInterfaceDecl>(decl_ctx);

  if (!interface_decl)
    return;

  do {
    ClangASTImporter::DeclOrigin original = m_ast_importer_sp->GetDeclOrigin(interface_decl);

    if (!original.Valid())
      break;

    ObjCInterfaceDecl *original_interface_decl =
        dyn_cast<ObjCInterfaceDecl>(original.decl);

    if (FindObjCMethodDeclsWithOrigin(context, original_interface_decl,
                                      "at origin"))
      return; // found it, no need to look any further
  } while (false);

  StreamString ss;

  if (decl_name.isObjCZeroArgSelector()) {
    ss.Printf("%s", decl_name.getAsString().c_str());
  } else if (decl_name.isObjCOneArgSelector()) {
    ss.Printf("%s", decl_name.getAsString().c_str());
  } else {
    clang::Selector sel = decl_name.getObjCSelector();

    for (unsigned i = 0, e = sel.getNumArgs(); i != e; ++i) {
      llvm::StringRef r = sel.getNameForSlot(i);
      ss.Printf("%s:", r.str().c_str());
    }
  }
  ss.Flush();

  if (ss.GetString().contains("$__lldb"))
    return; // we don't need any results

  ConstString selector_name(ss.GetString());

  LLDB_LOG(log,
           "ClangASTSource::FindObjCMethodDecls on (ASTContext*){0} '{1}' "
           "for selector [{2} {3}]",
           m_ast_context, m_clang_ast_context->getDisplayName(),
           interface_decl->getName(), selector_name);
  SymbolContextList sc_list;

  ModuleFunctionSearchOptions function_options;
  function_options.include_symbols = false;
  function_options.include_inlines = false;

  std::string interface_name = interface_decl->getNameAsString();

  do {
    StreamString ms;
    ms.Printf("-[%s %s]", interface_name.c_str(), selector_name.AsCString());
    ms.Flush();
    ConstString instance_method_name(ms.GetString());

    sc_list.Clear();
    m_target->GetImages().FindFunctions(instance_method_name,
                                        lldb::eFunctionNameTypeFull,
                                        function_options, sc_list);

    if (sc_list.GetSize())
      break;

    ms.Clear();
    ms.Printf("+[%s %s]", interface_name.c_str(), selector_name.AsCString());
    ms.Flush();
    ConstString class_method_name(ms.GetString());

    sc_list.Clear();
    m_target->GetImages().FindFunctions(class_method_name,
                                        lldb::eFunctionNameTypeFull,
                                        function_options, sc_list);

    if (sc_list.GetSize())
      break;

    // Fall back and check for methods in categories.  If we find methods this
    // way, we need to check that they're actually in categories on the desired
    // class.

    SymbolContextList candidate_sc_list;

    m_target->GetImages().FindFunctions(selector_name,
                                        lldb::eFunctionNameTypeSelector,
                                        function_options, candidate_sc_list);

    for (uint32_t ci = 0, ce = candidate_sc_list.GetSize(); ci != ce; ++ci) {
      SymbolContext candidate_sc;

      if (!candidate_sc_list.GetContextAtIndex(ci, candidate_sc))
        continue;

      if (!candidate_sc.function)
        continue;

      const char *candidate_name = candidate_sc.function->GetName().AsCString();

      const char *cursor = candidate_name;

      if (*cursor != '+' && *cursor != '-')
        continue;

      ++cursor;

      if (*cursor != '[')
        continue;

      ++cursor;

      size_t interface_len = interface_name.length();

      if (strncmp(cursor, interface_name.c_str(), interface_len))
        continue;

      cursor += interface_len;

      if (*cursor == ' ' || *cursor == '(')
        sc_list.Append(candidate_sc);
    }
  } while (false);

  if (sc_list.GetSize()) {
    // We found a good function symbol.  Use that.

    for (uint32_t i = 0, e = sc_list.GetSize(); i != e; ++i) {
      SymbolContext sc;

      if (!sc_list.GetContextAtIndex(i, sc))
        continue;

      if (!sc.function)
        continue;

      CompilerDeclContext function_decl_ctx = sc.function->GetDeclContext();
      if (!function_decl_ctx)
        continue;

      ObjCMethodDecl *method_decl =
          TypeSystemClang::DeclContextGetAsObjCMethodDecl(function_decl_ctx);

      if (!method_decl)
        continue;

      ObjCInterfaceDecl *found_interface_decl =
          method_decl->getClassInterface();

      if (!found_interface_decl)
        continue;

      if (found_interface_decl->getName() == interface_decl->getName()) {
        Decl *copied_decl = CopyDecl(method_decl);

        if (!copied_decl)
          continue;

        ObjCMethodDecl *copied_method_decl =
            dyn_cast<ObjCMethodDecl>(copied_decl);

        if (!copied_method_decl)
          continue;

        LLDB_LOG(log, "  CAS::FOMD found (in symbols)\n{0}",
                 ClangUtil::DumpDecl(copied_method_decl));

        context.AddNamedDecl(copied_method_decl);
      }
    }

    return;
  }

  // Try the debug information.

  do {
    ObjCInterfaceDecl *complete_interface_decl = GetCompleteObjCInterface(
        const_cast<ObjCInterfaceDecl *>(interface_decl));

    if (!complete_interface_decl)
      break;

    // We found the complete interface.  The runtime never needs to be queried
    // in this scenario.

    DeclFromUser<const ObjCInterfaceDecl> complete_iface_decl(
        complete_interface_decl);

    if (complete_interface_decl == interface_decl)
      break; // already checked this one

    LLDB_LOG(log,
             "CAS::FOPD trying origin "
             "(ObjCInterfaceDecl*){0}/(ASTContext*){1}...",
             complete_interface_decl, &complete_iface_decl->getASTContext());

    FindObjCMethodDeclsWithOrigin(context, complete_interface_decl,
                                  "in debug info");

    return;
  } while (false);

  do {
    // Check the modules only if the debug information didn't have a complete
    // interface.

    if (std::shared_ptr<ClangModulesDeclVendor> modules_decl_vendor =
            GetClangModulesDeclVendor()) {
      ConstString interface_name(interface_decl->getNameAsString().c_str());
      bool append = false;
      uint32_t max_matches = 1;
      std::vector<clang::NamedDecl *> decls;

      if (!modules_decl_vendor->FindDecls(interface_name, append, max_matches,
                                          decls))
        break;

      ObjCInterfaceDecl *interface_decl_from_modules =
          dyn_cast<ObjCInterfaceDecl>(decls[0]);

      if (!interface_decl_from_modules)
        break;

      if (FindObjCMethodDeclsWithOrigin(context, interface_decl_from_modules,
                                        "in modules"))
        return;
    }
  } while (false);

  do {
    // Check the runtime only if the debug information didn't have a complete
    // interface and the modules don't get us anywhere.

    lldb::ProcessSP process(m_target->GetProcessSP());

    if (!process)
      break;

    ObjCLanguageRuntime *language_runtime(ObjCLanguageRuntime::Get(*process));

    if (!language_runtime)
      break;

    DeclVendor *decl_vendor = language_runtime->GetDeclVendor();

    if (!decl_vendor)
      break;

    ConstString interface_name(interface_decl->getNameAsString().c_str());
    bool append = false;
    uint32_t max_matches = 1;
    std::vector<clang::NamedDecl *> decls;

    auto *clang_decl_vendor = llvm::cast<ClangDeclVendor>(decl_vendor);
    if (!clang_decl_vendor->FindDecls(interface_name, append, max_matches,
                                      decls))
      break;

    ObjCInterfaceDecl *runtime_interface_decl =
        dyn_cast<ObjCInterfaceDecl>(decls[0]);

    if (!runtime_interface_decl)
      break;

    FindObjCMethodDeclsWithOrigin(context, runtime_interface_decl,
                                  "in runtime");
  } while (false);
}

static bool FindObjCPropertyAndIvarDeclsWithOrigin(
    NameSearchContext &context, ClangASTSource &source,
    DeclFromUser<const ObjCInterfaceDecl> &origin_iface_decl) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  if (origin_iface_decl.IsInvalid())
    return false;

  std::string name_str = context.m_decl_name.getAsString();
  StringRef name(name_str);
  IdentifierInfo &name_identifier(
      origin_iface_decl->getASTContext().Idents.get(name));

  DeclFromUser<ObjCPropertyDecl> origin_property_decl(
      origin_iface_decl->FindPropertyDeclaration(
          &name_identifier, ObjCPropertyQueryKind::OBJC_PR_query_instance));

  bool found = false;

  if (origin_property_decl.IsValid()) {
    DeclFromParser<ObjCPropertyDecl> parser_property_decl(
        origin_property_decl.Import(source));
    if (parser_property_decl.IsValid()) {
      LLDB_LOG(log, "  CAS::FOPD found\n{0}",
               ClangUtil::DumpDecl(parser_property_decl.decl));

      context.AddNamedDecl(parser_property_decl.decl);
      found = true;
    }
  }

  DeclFromUser<ObjCIvarDecl> origin_ivar_decl(
      origin_iface_decl->getIvarDecl(&name_identifier));

  if (origin_ivar_decl.IsValid()) {
    DeclFromParser<ObjCIvarDecl> parser_ivar_decl(
        origin_ivar_decl.Import(source));
    if (parser_ivar_decl.IsValid()) {
      LLDB_LOG(log, "  CAS::FOPD found\n{0}",
               ClangUtil::DumpDecl(parser_ivar_decl.decl));

      context.AddNamedDecl(parser_ivar_decl.decl);
      found = true;
    }
  }

  return found;
}

void ClangASTSource::FindObjCPropertyAndIvarDecls(NameSearchContext &context) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  DeclFromParser<const ObjCInterfaceDecl> parser_iface_decl(
      cast<ObjCInterfaceDecl>(context.m_decl_context));
  DeclFromUser<const ObjCInterfaceDecl> origin_iface_decl(
      parser_iface_decl.GetOrigin(*this));

  ConstString class_name(parser_iface_decl->getNameAsString().c_str());

  LLDB_LOG(log,
           "ClangASTSource::FindObjCPropertyAndIvarDecls on "
           "(ASTContext*){0} '{1}' for '{2}.{3}'",
           m_ast_context, m_clang_ast_context->getDisplayName(),
           parser_iface_decl->getName(), context.m_decl_name.getAsString());

  if (FindObjCPropertyAndIvarDeclsWithOrigin(context, *this, origin_iface_decl))
    return;

  LLDB_LOG(log,
           "CAS::FOPD couldn't find the property on origin "
           "(ObjCInterfaceDecl*){0}/(ASTContext*){1}, searching "
           "elsewhere...",
           origin_iface_decl.decl, &origin_iface_decl->getASTContext());

  SymbolContext null_sc;
  TypeList type_list;

  do {
    ObjCInterfaceDecl *complete_interface_decl = GetCompleteObjCInterface(
        const_cast<ObjCInterfaceDecl *>(parser_iface_decl.decl));

    if (!complete_interface_decl)
      break;

    // We found the complete interface.  The runtime never needs to be queried
    // in this scenario.

    DeclFromUser<const ObjCInterfaceDecl> complete_iface_decl(
        complete_interface_decl);

    if (complete_iface_decl.decl == origin_iface_decl.decl)
      break; // already checked this one

    LLDB_LOG(log,
             "CAS::FOPD trying origin "
             "(ObjCInterfaceDecl*){0}/(ASTContext*){1}...",
             complete_iface_decl.decl, &complete_iface_decl->getASTContext());

    FindObjCPropertyAndIvarDeclsWithOrigin(context, *this, complete_iface_decl);

    return;
  } while (false);

  do {
    // Check the modules only if the debug information didn't have a complete
    // interface.

    std::shared_ptr<ClangModulesDeclVendor> modules_decl_vendor =
        GetClangModulesDeclVendor();

    if (!modules_decl_vendor)
      break;

    bool append = false;
    uint32_t max_matches = 1;
    std::vector<clang::NamedDecl *> decls;

    if (!modules_decl_vendor->FindDecls(class_name, append, max_matches, decls))
      break;

    DeclFromUser<const ObjCInterfaceDecl> interface_decl_from_modules(
        dyn_cast<ObjCInterfaceDecl>(decls[0]));

    if (!interface_decl_from_modules.IsValid())
      break;

    LLDB_LOG(log,
             "CAS::FOPD[{0}] trying module "
             "(ObjCInterfaceDecl*){0}/(ASTContext*){1}...",
             interface_decl_from_modules.decl,
             &interface_decl_from_modules->getASTContext());

    if (FindObjCPropertyAndIvarDeclsWithOrigin(context, *this,
                                               interface_decl_from_modules))
      return;
  } while (false);

  do {
    // Check the runtime only if the debug information didn't have a complete
    // interface and nothing was in the modules.

    lldb::ProcessSP process(m_target->GetProcessSP());

    if (!process)
      return;

    ObjCLanguageRuntime *language_runtime(ObjCLanguageRuntime::Get(*process));

    if (!language_runtime)
      return;

    DeclVendor *decl_vendor = language_runtime->GetDeclVendor();

    if (!decl_vendor)
      break;

    bool append = false;
    uint32_t max_matches = 1;
    std::vector<clang::NamedDecl *> decls;

    auto *clang_decl_vendor = llvm::cast<ClangDeclVendor>(decl_vendor);
    if (!clang_decl_vendor->FindDecls(class_name, append, max_matches, decls))
      break;

    DeclFromUser<const ObjCInterfaceDecl> interface_decl_from_runtime(
        dyn_cast<ObjCInterfaceDecl>(decls[0]));

    if (!interface_decl_from_runtime.IsValid())
      break;

    LLDB_LOG(log,
             "CAS::FOPD[{0}] trying runtime "
             "(ObjCInterfaceDecl*){0}/(ASTContext*){1}...",
             interface_decl_from_runtime.decl,
             &interface_decl_from_runtime->getASTContext());

    if (FindObjCPropertyAndIvarDeclsWithOrigin(context, *this,
                                               interface_decl_from_runtime))
      return;
  } while (false);
}

void ClangASTSource::LookupInNamespace(NameSearchContext &context) {
  const NamespaceDecl *namespace_context =
      dyn_cast<NamespaceDecl>(context.m_decl_context);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  ClangASTImporter::NamespaceMapSP namespace_map =
      m_ast_importer_sp->GetNamespaceMap(namespace_context);

  LLDB_LOGV(log, "  CAS::FEVD Inspecting namespace map {0} ({1} entries)",
            namespace_map.get(), namespace_map->size());

  if (!namespace_map)
    return;

  for (ClangASTImporter::NamespaceMap::iterator i = namespace_map->begin(),
                                                e = namespace_map->end();
       i != e; ++i) {
    LLDB_LOG(log, "  CAS::FEVD Searching namespace {0} in module {1}",
             i->second.GetName(), i->first->GetFileSpec().GetFilename());

    FindExternalVisibleDecls(context, i->first, i->second);
  }
}

typedef llvm::DenseMap<const FieldDecl *, uint64_t> FieldOffsetMap;
typedef llvm::DenseMap<const CXXRecordDecl *, CharUnits> BaseOffsetMap;

template <class D, class O>
static bool ImportOffsetMap(llvm::DenseMap<const D *, O> &destination_map,
                            llvm::DenseMap<const D *, O> &source_map,
                            ClangASTSource &source) {
  // When importing fields into a new record, clang has a hard requirement that
  // fields be imported in field offset order.  Since they are stored in a
  // DenseMap with a pointer as the key type, this means we cannot simply
  // iterate over the map, as the order will be non-deterministic.  Instead we
  // have to sort by the offset and then insert in sorted order.
  typedef llvm::DenseMap<const D *, O> MapType;
  typedef typename MapType::value_type PairType;
  std::vector<PairType> sorted_items;
  sorted_items.reserve(source_map.size());
  sorted_items.assign(source_map.begin(), source_map.end());
  llvm::sort(sorted_items.begin(), sorted_items.end(),
             [](const PairType &lhs, const PairType &rhs) {
               return lhs.second < rhs.second;
             });

  for (const auto &item : sorted_items) {
    DeclFromUser<D> user_decl(const_cast<D *>(item.first));
    DeclFromParser<D> parser_decl(user_decl.Import(source));
    if (parser_decl.IsInvalid())
      return false;
    destination_map.insert(
        std::pair<const D *, O>(parser_decl.decl, item.second));
  }

  return true;
}

template <bool IsVirtual>
bool ExtractBaseOffsets(const ASTRecordLayout &record_layout,
                        DeclFromUser<const CXXRecordDecl> &record,
                        BaseOffsetMap &base_offsets) {
  for (CXXRecordDecl::base_class_const_iterator
           bi = (IsVirtual ? record->vbases_begin() : record->bases_begin()),
           be = (IsVirtual ? record->vbases_end() : record->bases_end());
       bi != be; ++bi) {
    if (!IsVirtual && bi->isVirtual())
      continue;

    const clang::Type *origin_base_type = bi->getType().getTypePtr();
    const clang::RecordType *origin_base_record_type =
        origin_base_type->getAs<RecordType>();

    if (!origin_base_record_type)
      return false;

    DeclFromUser<RecordDecl> origin_base_record(
        origin_base_record_type->getDecl());

    if (origin_base_record.IsInvalid())
      return false;

    DeclFromUser<CXXRecordDecl> origin_base_cxx_record(
        DynCast<CXXRecordDecl>(origin_base_record));

    if (origin_base_cxx_record.IsInvalid())
      return false;

    CharUnits base_offset;

    if (IsVirtual)
      base_offset =
          record_layout.getVBaseClassOffset(origin_base_cxx_record.decl);
    else
      base_offset =
          record_layout.getBaseClassOffset(origin_base_cxx_record.decl);

    base_offsets.insert(std::pair<const CXXRecordDecl *, CharUnits>(
        origin_base_cxx_record.decl, base_offset));
  }

  return true;
}

bool ClangASTSource::layoutRecordType(const RecordDecl *record, uint64_t &size,
                                      uint64_t &alignment,
                                      FieldOffsetMap &field_offsets,
                                      BaseOffsetMap &base_offsets,
                                      BaseOffsetMap &virtual_base_offsets) {

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  LLDB_LOG(log,
           "LayoutRecordType on (ASTContext*){0} '{1}' for (RecordDecl*)"
           "{3} [name = '{4}']",
           m_ast_context, m_clang_ast_context->getDisplayName(), record,
           record->getName());

  DeclFromParser<const RecordDecl> parser_record(record);
  DeclFromUser<const RecordDecl> origin_record(
      parser_record.GetOrigin(*this));

  if (origin_record.IsInvalid())
    return false;

  FieldOffsetMap origin_field_offsets;
  BaseOffsetMap origin_base_offsets;
  BaseOffsetMap origin_virtual_base_offsets;

  TypeSystemClang::GetCompleteDecl(
      &origin_record->getASTContext(),
      const_cast<RecordDecl *>(origin_record.decl));

  clang::RecordDecl *definition = origin_record.decl->getDefinition();
  if (!definition || !definition->isCompleteDefinition())
    return false;

  const ASTRecordLayout &record_layout(
      origin_record->getASTContext().getASTRecordLayout(origin_record.decl));

  int field_idx = 0, field_count = record_layout.getFieldCount();

  for (RecordDecl::field_iterator fi = origin_record->field_begin(),
                                  fe = origin_record->field_end();
       fi != fe; ++fi) {
    if (field_idx >= field_count)
      return false; // Layout didn't go well.  Bail out.

    uint64_t field_offset = record_layout.getFieldOffset(field_idx);

    origin_field_offsets.insert(
        std::pair<const FieldDecl *, uint64_t>(*fi, field_offset));

    field_idx++;
  }

  lldbassert(&record->getASTContext() == m_ast_context);

  DeclFromUser<const CXXRecordDecl> origin_cxx_record(
      DynCast<const CXXRecordDecl>(origin_record));

  if (origin_cxx_record.IsValid()) {
    if (!ExtractBaseOffsets<false>(record_layout, origin_cxx_record,
                                   origin_base_offsets) ||
        !ExtractBaseOffsets<true>(record_layout, origin_cxx_record,
                                  origin_virtual_base_offsets))
      return false;
  }

  if (!ImportOffsetMap(field_offsets, origin_field_offsets, *this) ||
      !ImportOffsetMap(base_offsets, origin_base_offsets, *this) ||
      !ImportOffsetMap(virtual_base_offsets, origin_virtual_base_offsets,
                       *this))
    return false;

  size = record_layout.getSize().getQuantity() * m_ast_context->getCharWidth();
  alignment = record_layout.getAlignment().getQuantity() *
              m_ast_context->getCharWidth();

  if (log) {
    LLDB_LOG(log, "LRT returned:");
    LLDB_LOG(log, "LRT   Original = (RecordDecl*){0}",
             static_cast<const void *>(origin_record.decl));
    LLDB_LOG(log, "LRT   Size = {0}", size);
    LLDB_LOG(log, "LRT   Alignment = {0}", alignment);
    LLDB_LOG(log, "LRT   Fields:");
    for (RecordDecl::field_iterator fi = record->field_begin(),
                                    fe = record->field_end();
         fi != fe; ++fi) {
      LLDB_LOG(log,
               "LRT     (FieldDecl*){0}, Name = '{1}', Offset = {2} bits",
               *fi, fi->getName(), field_offsets[*fi]);
    }
    DeclFromParser<const CXXRecordDecl> parser_cxx_record =
        DynCast<const CXXRecordDecl>(parser_record);
    if (parser_cxx_record.IsValid()) {
      LLDB_LOG(log, "LRT   Bases:");
      for (CXXRecordDecl::base_class_const_iterator
               bi = parser_cxx_record->bases_begin(),
               be = parser_cxx_record->bases_end();
           bi != be; ++bi) {
        bool is_virtual = bi->isVirtual();

        QualType base_type = bi->getType();
        const RecordType *base_record_type = base_type->getAs<RecordType>();
        DeclFromParser<RecordDecl> base_record(base_record_type->getDecl());
        DeclFromParser<CXXRecordDecl> base_cxx_record =
            DynCast<CXXRecordDecl>(base_record);

        LLDB_LOG(log,
                 "LRT     {0}(CXXRecordDecl*){1}, Name = '{2}', Offset = "
                 "{3} chars",
                 (is_virtual ? "Virtual " : ""), base_cxx_record.decl,
                 base_cxx_record.decl->getName(),
                 (is_virtual
                      ? virtual_base_offsets[base_cxx_record.decl].getQuantity()
                      : base_offsets[base_cxx_record.decl].getQuantity()));
      }
    } else {
      LLDB_LOG(log, "LRD   Not a CXXRecord, so no bases");
    }
  }

  return true;
}

void ClangASTSource::CompleteNamespaceMap(
    ClangASTImporter::NamespaceMapSP &namespace_map, ConstString name,
    ClangASTImporter::NamespaceMapSP &parent_map) const {

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  if (log) {
    if (parent_map && parent_map->size())
      LLDB_LOG(log,
               "CompleteNamespaceMap on (ASTContext*){0} '{1}' Searching "
               "for namespace {2} in namespace {3}",
               m_ast_context, m_clang_ast_context->getDisplayName(), name,
               parent_map->begin()->second.GetName());
    else
      LLDB_LOG(log,
               "CompleteNamespaceMap on (ASTContext*){0} '{1}' Searching "
               "for namespace {2}",
               m_ast_context, m_clang_ast_context->getDisplayName(), name);
  }

  if (parent_map) {
    for (ClangASTImporter::NamespaceMap::iterator i = parent_map->begin(),
                                                  e = parent_map->end();
         i != e; ++i) {
      CompilerDeclContext found_namespace_decl;

      lldb::ModuleSP module_sp = i->first;
      CompilerDeclContext module_parent_namespace_decl = i->second;

      SymbolFile *symbol_file = module_sp->GetSymbolFile();

      if (!symbol_file)
        continue;

      found_namespace_decl =
          symbol_file->FindNamespace(name, module_parent_namespace_decl);

      if (!found_namespace_decl)
        continue;

      namespace_map->push_back(std::pair<lldb::ModuleSP, CompilerDeclContext>(
          module_sp, found_namespace_decl));

      LLDB_LOG(log, "  CMN Found namespace {0} in module {1}", name,
               module_sp->GetFileSpec().GetFilename());
    }
  } else {
    CompilerDeclContext null_namespace_decl;
    for (lldb::ModuleSP image : m_target->GetImages().Modules()) {
      if (!image)
        continue;

      CompilerDeclContext found_namespace_decl;

      SymbolFile *symbol_file = image->GetSymbolFile();

      if (!symbol_file)
        continue;

      found_namespace_decl =
          symbol_file->FindNamespace(name, null_namespace_decl);

      if (!found_namespace_decl)
        continue;

      namespace_map->push_back(std::pair<lldb::ModuleSP, CompilerDeclContext>(
          image, found_namespace_decl));

      LLDB_LOG(log, "  CMN[{0}] Found namespace {0} in module {1}", name,
               image->GetFileSpec().GetFilename());
    }
  }
}

NamespaceDecl *ClangASTSource::AddNamespace(
    NameSearchContext &context,
    ClangASTImporter::NamespaceMapSP &namespace_decls) {
  if (!namespace_decls)
    return nullptr;

  const CompilerDeclContext &namespace_decl = namespace_decls->begin()->second;

  clang::ASTContext *src_ast =
      TypeSystemClang::DeclContextGetTypeSystemClang(namespace_decl);
  if (!src_ast)
    return nullptr;
  clang::NamespaceDecl *src_namespace_decl =
      TypeSystemClang::DeclContextGetAsNamespaceDecl(namespace_decl);

  if (!src_namespace_decl)
    return nullptr;

  Decl *copied_decl = CopyDecl(src_namespace_decl);

  if (!copied_decl)
    return nullptr;

  NamespaceDecl *copied_namespace_decl = dyn_cast<NamespaceDecl>(copied_decl);

  if (!copied_namespace_decl)
    return nullptr;

  context.m_decls.push_back(copied_namespace_decl);

  m_ast_importer_sp->RegisterNamespaceMap(copied_namespace_decl,
                                          namespace_decls);

  return dyn_cast<NamespaceDecl>(copied_decl);
}

clang::Decl *ClangASTSource::CopyDecl(Decl *src_decl) {
  return m_ast_importer_sp->CopyDecl(m_ast_context, src_decl);
}

ClangASTImporter::DeclOrigin ClangASTSource::GetDeclOrigin(const clang::Decl *decl) {
  return m_ast_importer_sp->GetDeclOrigin(decl);
}

CompilerType ClangASTSource::GuardedCopyType(const CompilerType &src_type) {
  TypeSystemClang *src_ast =
      llvm::dyn_cast_or_null<TypeSystemClang>(src_type.GetTypeSystem());
  if (src_ast == nullptr)
    return CompilerType();

  QualType copied_qual_type = ClangUtil::GetQualType(
      m_ast_importer_sp->CopyType(*m_clang_ast_context, src_type));

  if (copied_qual_type.getAsOpaquePtr() &&
      copied_qual_type->getCanonicalTypeInternal().isNull())
    // this shouldn't happen, but we're hardening because the AST importer
    // seems to be generating bad types on occasion.
    return CompilerType();

  return m_clang_ast_context->GetType(copied_qual_type);
}

std::shared_ptr<ClangModulesDeclVendor>
ClangASTSource::GetClangModulesDeclVendor() {
  auto persistent_vars = llvm::cast<ClangPersistentVariables>(
      m_target->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeC));
  return persistent_vars->GetClangModulesDeclVendor();
}
