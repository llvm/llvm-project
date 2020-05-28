//===-- TypeSystemSwiftTypeRef.cpp ----------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SwiftASTContext.h"

#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Utility/Log.h"

#include "Plugins/ExpressionParser/Clang/ClangExternalASTSourceCallbacks.h"
#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include "swift/AST/ClangModuleLoader.h"
#include "swift/Basic/Version.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/Demangler.h"
#include "swift/Strings.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/APINotes/APINotesManager.h"
#include "clang/APINotes/APINotesReader.h"

using namespace lldb;
using namespace lldb_private;

char TypeSystemSwift::ID;
char TypeSystemSwiftTypeRef::ID;

TypeSystemSwift::TypeSystemSwift() : TypeSystem() {}

/// Create a mangled name for a type alias node.
static ConstString GetTypeAlias(swift::Demangle::Demangler &Dem,
                                swift::Demangle::NodePointer node) {
  using namespace swift::Demangle;
  auto global = Dem.createNode(Node::Kind::Global);
  auto type_mangling = Dem.createNode(Node::Kind::TypeMangling);
  global->addChild(type_mangling, Dem);
  type_mangling->addChild(node, Dem);
  return ConstString(mangleNode(global));
}

/// Find a Clang type by name in module \p M.
static TypeSP LookupClangType(Module &M, StringRef name) {
  llvm::SmallVector<CompilerContext, 2> decl_context;
  decl_context.push_back({CompilerContextKind::AnyModule, ConstString()});
  decl_context.push_back({CompilerContextKind::AnyType, ConstString(name)});
  llvm::DenseSet<SymbolFile *> searched_symbol_files;
  TypeMap clang_types;
  M.FindTypes(decl_context, TypeSystemClang::GetSupportedLanguagesForTypes(),
              searched_symbol_files, clang_types);
  if (clang_types.Empty())
    return {};
  return clang_types.GetTypeAtIndex(0);
}

/// Find a Clang type by name in module \p M.
static CompilerType LookupClangForwardType(Module &M, StringRef name) {
  if (TypeSP type = LookupClangType(M, name))
    return type->GetForwardCompilerType();
  return {};
}

/// Return a demangle tree leaf node representing \p clang_type.
static swift::Demangle::NodePointer
GetClangTypeNode(CompilerType clang_type, swift::Demangle::Demangler &Dem) {
  using namespace swift::Demangle;
  clang::QualType qual_type = ClangUtil::GetQualType(clang_type);
  NodePointer structure = Dem.createNode(
      qual_type->isClassType() ? Node::Kind::Class : Node::Kind::Structure);
  NodePointer module = Dem.createNodeWithAllocatedText(
      Node::Kind::Module, swift::MANGLING_MODULE_OBJC);
  structure->addChild(module, Dem);
  NodePointer identifier = Dem.createNodeWithAllocatedText(
      Node::Kind::Module, clang_type.GetTypeName().GetStringRef());
  structure->addChild(identifier, Dem);
  return structure;
}

/// Iteratively resolve all type aliases in \p node by looking up their
/// desugared types in the debug info of module \p M.
static swift::Demangle::NodePointer
GetCanonicalNode(lldb_private::Module *M, swift::Demangle::Demangler &Dem,
                 swift::Demangle::NodePointer node) {
  if (!node)
    return node;
  using namespace swift::Demangle;
  auto getCanonicalNode = [&](NodePointer node) -> NodePointer {
    return GetCanonicalNode(M, Dem, node);
  };

  NodePointer canonical = nullptr;
  auto kind = node->getKind();
  switch (kind) {
  case Node::Kind::SugaredOptional:
    // FIXME: Factor these three cases out.
    assert(node->getNumChildren() == 1);
    if (node->getNumChildren() != 1)
      return node;

    canonical = Dem.createNode(Node::Kind::BoundGenericEnum);
    {
      NodePointer type = Dem.createNode(Node::Kind::Type);
      NodePointer e = Dem.createNode(Node::Kind::Enum);
      NodePointer module = Dem.createNodeWithAllocatedText(Node::Kind::Module,
                                                           swift::STDLIB_NAME);
      e->addChild(module, Dem);
      NodePointer optional =
          Dem.createNodeWithAllocatedText(Node::Kind::Identifier, "Optional");
      e->addChild(optional, Dem);
      type->addChild(e, Dem);
      canonical->addChild(type, Dem);
    }
    {
      NodePointer typelist = Dem.createNode(Node::Kind::TypeList);
      NodePointer type = Dem.createNode(Node::Kind::Type);
      type->addChild(getCanonicalNode(node->getFirstChild()), Dem);
      typelist->addChild(type, Dem);
      canonical->addChild(typelist, Dem);
    }
    return canonical;
  case Node::Kind::SugaredArray: {
    assert(node->getNumChildren() == 1);
    if (node->getNumChildren() != 1)
      return node;

    canonical = Dem.createNode(Node::Kind::BoundGenericStructure);
    {
      NodePointer type = Dem.createNode(Node::Kind::Type);
      NodePointer structure = Dem.createNode(Node::Kind::Structure);
      NodePointer module = Dem.createNodeWithAllocatedText(Node::Kind::Module,
                                                           swift::STDLIB_NAME);
      structure->addChild(module, Dem);
      NodePointer array =
          Dem.createNodeWithAllocatedText(Node::Kind::Identifier, "Array");
      structure->addChild(array, Dem);
      type->addChild(structure, Dem);
      canonical->addChild(type, Dem);
    }
    {
      NodePointer typelist = Dem.createNode(Node::Kind::TypeList);
      NodePointer type = Dem.createNode(Node::Kind::Type);
      type->addChild(getCanonicalNode(node->getFirstChild()), Dem);
      typelist->addChild(type, Dem);
      canonical->addChild(typelist, Dem);
    }
    return canonical;
  }
  case Node::Kind::SugaredDictionary:
    // FIXME: This isnt covered by any test.
    assert(node->getNumChildren() == 2);
    if (node->getNumChildren() != 2)
      return node;

    canonical = Dem.createNode(Node::Kind::BoundGenericStructure);
    {
      NodePointer type = Dem.createNode(Node::Kind::Type);
      NodePointer structure = Dem.createNode(Node::Kind::Structure);
      NodePointer module = Dem.createNodeWithAllocatedText(Node::Kind::Module,
                                                           swift::STDLIB_NAME);
      structure->addChild(module, Dem);
      NodePointer dict =
          Dem.createNodeWithAllocatedText(Node::Kind::Identifier, "Dictionary");
      structure->addChild(dict, Dem);
      type->addChild(structure, Dem);
      canonical->addChild(type, Dem);
    }
    {
      NodePointer typelist = Dem.createNode(Node::Kind::TypeList);
      {
        NodePointer type = Dem.createNode(Node::Kind::Type);
        type->addChild(getCanonicalNode(node->getChild(0)), Dem);
        typelist->addChild(type, Dem);
      }
      {
        NodePointer type = Dem.createNode(Node::Kind::Type);
        type->addChild(getCanonicalNode(node->getChild(1)), Dem);
        typelist->addChild(type, Dem);
      }
      canonical->addChild(typelist, Dem);
    }
    return canonical;
  case Node::Kind::SugaredParen:
    assert(node->getNumChildren() == 1);
    if (node->getNumChildren() != 1)
      return node;
    return getCanonicalNode(node->getFirstChild());

  case Node::Kind::BoundGenericTypeAlias:
  case Node::Kind::TypeAlias: {
    // Try to look this up as a Swift type alias. For each *Swift*
    // type alias there is a debug info entry that has the mangled
    // name as name and the aliased type as a type.
    ConstString mangled = GetTypeAlias(Dem, node);
    if (!M) {
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "No module. Couldn't resolve type alias %s",
                mangled.AsCString());
      return node;
    }
    llvm::DenseSet<lldb_private::SymbolFile *> searched_symbol_files;
    TypeList types;
    M->FindTypes({mangled}, false, 1, searched_symbol_files, types);
    if (types.Empty()) {
      // No Swift type found -- this could be a Clang typdef.  This
      // check is not done earlier because a Clang typedef that points
      // to a builtin type, e.g., "typedef unsigned uint32_t", could
      // end up pointing to a *Swift* type!
      if (node->getNumChildren() == 2 && node->getChild(0)->hasText() &&
          node->getChild(0)->getText() == swift::MANGLING_MODULE_OBJC &&
          node->getChild(1)->hasText()) {
        // Resolve the typedef within the Clang debug info.
        auto clang_type = LookupClangForwardType(*M, node->getChild(1)->getText());
        if (!clang_type)
          return node;
        clang_type = clang_type.GetCanonicalType();
        return GetClangTypeNode(clang_type, Dem);
      }

      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "Couldn't resolve type alias %s", mangled.AsCString());
      return node;
    }
    auto type = types.GetTypeAtIndex(0);
    if (!type) {
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "Found empty type alias %s", mangled.AsCString());
      return node;
    }

    // DWARFASTParserSwift stashes the desugared mangled name of a
    // type alias into the Type's name field.
    ConstString desugared_name = type->GetName();
    if (!isMangledName(desugared_name.GetStringRef())) {
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "Found non-Swift type alias %s", mangled.AsCString());
      return node;
    }
    NodePointer n = Dem.demangleSymbol(desugared_name.GetStringRef());
    if (n && n->getKind() == Node::Kind::Global && n->hasChildren())
      n = n->getFirstChild();
    if (n && n->getKind() == Node::Kind::TypeMangling && n->hasChildren())
      n = n->getFirstChild();
    if (n && n->getKind() == Node::Kind::Type && n->hasChildren())
      n = n->getFirstChild();
    if (!n) {
      LLDB_LOG(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
               "Unrecognized demangling %s", desugared_name.AsCString());
      return node;
    }
    return getCanonicalNode(n);
  }
  default:
    break;
  }

  // Recurse through all children.
  // FIXME: don't create new nodes if children don't change!
  if (node->hasText())
    canonical = Dem.createNodeWithAllocatedText(kind, node->getText());
  else if (node->hasIndex())
    canonical = Dem.createNode(kind, node->getIndex());
  else
    canonical = Dem.createNode(kind);
  for (unsigned i = 0; i < node->getNumChildren(); ++i)
    canonical->addChild(getCanonicalNode(node->getChild(i)), Dem);
  return canonical;
}

/// Return the demangle tree representation of this type's canonical
/// (type aliases resolved) type.
static swift::Demangle::NodePointer
GetCanonicalDemangleTree(lldb_private::Module *Module,
                         swift::Demangle::Demangler &Dem,
                         const char *mangled_name) {
  NodePointer node = Dem.demangleSymbol(mangled_name);
  NodePointer canonical = GetCanonicalNode(Module, Dem, node);
  return canonical;
}

static clang::Decl *GetDeclForTypeAndKind(clang::QualType qual_type,
                                          swift::ClangTypeKind kind) {
  switch (kind) {
  case swift::ClangTypeKind::Typedef:
    /*=swift::ClangTypeKind::ObjCClass:*/
    if (auto *obj_type = qual_type->getAsObjCInterfaceType())
      return obj_type->getInterface();
    if (auto *typedef_type = qual_type->getAs<clang::TypedefType>())
      return typedef_type->getDecl();
    break;
  case swift::ClangTypeKind::Tag:
    return qual_type->getAsTagDecl();
  case swift::ClangTypeKind::ObjCProtocol:
    // Not implemented since Objective-C protocols aren't yet
    // described in DWARF.
    break;
  }
  return nullptr;
}

clang::api_notes::APINotesManager *TypeSystemSwiftTypeRef::GetAPINotesManager(
    ClangExternalASTSourceCallbacks *source, unsigned id) {
  if (!source)
    return nullptr;
  auto desc = source->getSourceDescriptor(id);
  if (!desc)
    return nullptr;
  clang::Module *module = desc->getModuleOrNull();
  if (!module)
    return nullptr;
  auto &apinotes_manager = m_apinotes_manager[module];
  if (apinotes_manager)
    return apinotes_manager.get();

  std::string path;
  for (clang::Module *parent = module; parent; parent = parent->Parent) {
    if (!parent->APINotesFile.empty()) {
      path = std::string(llvm::sys::path::parent_path(parent->APINotesFile));
      break;
    }
  }
  if (path.empty())
    return nullptr;

  apinotes_manager.reset(new clang::api_notes::APINotesManager(
      *source->GetTypeSystem().GetSourceMgr(),
      *source->GetTypeSystem().GetLangOpts()));
  // FIXME: get Swift version from the target instead of using the embedded
  // Swift version.
  auto swift_version = swift::version::getSwiftNumericVersion();
  apinotes_manager->setSwiftVersion(
      llvm::VersionTuple(swift_version.first, swift_version.second));
  apinotes_manager->loadCurrentModuleAPINotes(module, false, {path});
  return apinotes_manager.get();
}

/// Desugar a sugared type.
static swift::Demangle::NodePointer Desugar(swift::Demangle::Demangler &Dem,
                                            swift::Demangle::NodePointer node,
                                            Node::Kind bound_kind,
                                            Node::Kind kind, llvm::StringRef name) {
  using namespace swift::Demangle;
  NodePointer desugared = Dem.createNode(bound_kind);
  NodePointer type = Dem.createNode(Node::Kind::Type);
  {
    NodePointer concrete = Dem.createNode(kind);
    NodePointer swift =
        Dem.createNodeWithAllocatedText(Node::Kind::Module, swift::STDLIB_NAME);
    concrete->addChild(swift, Dem);
    NodePointer ident =
        Dem.createNodeWithAllocatedText(Node::Kind::Identifier, name);
    concrete->addChild(ident, Dem);
    type->addChild(concrete, Dem);
  }
  NodePointer type_list = Dem.createNode(Node::Kind::TypeList);
  {
    NodePointer type = Dem.createNode(Node::Kind::Type);
    type->addChild(node->getFirstChild(), Dem);
    type_list->addChild(type, Dem);
  }
  desugared->addChild(type, Dem);
  desugared->addChild(type_list, Dem);
  return desugared;
}

using GetAPINotesManagerFn = std::function<clang::api_notes::APINotesManager *(
    ClangExternalASTSourceCallbacks *source, unsigned id)>;
using GetClangImporterFn = std::function<swift::ClangImporter *()>;

/// Replace all "__C" module names with their actual Clang module names.
static swift::Demangle::NodePointer
GetNodeForPrinting(const std::string &m_description, lldb_private::Module &M,
                   GetAPINotesManagerFn get_apinotes_manager,
                   GetClangImporterFn get_clangimporter,
                   swift::Demangle::Demangler &Dem,
                   swift::Demangle::NodePointer node,
                   bool desugar = true) {
  if (!node)
    return node;
  using namespace swift::Demangle;
  auto getNodeForPrinting = [&](NodePointer node) -> NodePointer {
    return GetNodeForPrinting(m_description, M, get_apinotes_manager,
                              get_clangimporter, Dem, node, desugar);
  };

  NodePointer canonical = nullptr;
  auto kind = node->getKind();
  switch (kind) {
  case Node::Kind::Class:
  case Node::Kind::Structure:
  case Node::Kind::TypeAlias: {
    if (node->getNumChildren() != 2 || !node->getChild(0)->hasText() ||
        node->getChild(0)->getText() != swift::MANGLING_MODULE_OBJC ||
        !node->getChild(1)->hasText())
      break;

    // This is an imported Objective-C type; look it up in the debug info.
    StringRef ident = node->getChild(1)->getText();
    TypeSP clang_type = LookupClangType(M, ident);
    if (!clang_type)
      return node;

    // Extract the toplevel Clang module name from the debug info.
    llvm::SmallVector<CompilerContext, 4> DeclCtx;
    clang_type->GetDeclContext(DeclCtx);
    StringRef toplevel_module;
    for (auto &Context : DeclCtx)
      if (Context.kind == CompilerContextKind::Module) {
        toplevel_module = Context.name.GetStringRef();
        break;
      }
    if (toplevel_module.empty())
      break;

    // Create a new node with the Clang module instead of "__C".
    NodePointer renamed = Dem.createNode(kind);
    NodePointer module =
        Dem.createNode(Node::Kind::Module, toplevel_module);
    renamed->addChild(module, Dem);

    // This is unfortunate performance-wise, but only ClangImporter
    // knows how to translate a clang::Decl's name into a Swift name.

    // This order is significant, because of `typedef tag`.
    swift::ClangTypeKind kinds[] = {swift::ClangTypeKind::Typedef,
                                    swift::ClangTypeKind::Tag,
                                    swift::ClangTypeKind::ObjCProtocol};
    clang::NamedDecl *clang_decl = nullptr;
    CompilerType compiler_type = clang_type->GetForwardCompilerType();
    clang::QualType qual_type = ClangUtil::GetQualType(compiler_type);
    for (auto kind : kinds) {
      clang_decl = llvm::dyn_cast_or_null<clang::NamedDecl>(
          GetDeclForTypeAndKind(qual_type, kind));
      if (clang_decl)
        break;
    }
    if (!clang_decl)
      break;

    // Read the Swift name from the APINotes.
    std::string swift_name = std::string(ident);
    if (unsigned id = clang_decl->getOwningModuleID()) {
      auto *clang_typesystem = llvm::dyn_cast_or_null<TypeSystemClang>(
          compiler_type.GetTypeSystem());
      if (!clang_typesystem)
        break;
      auto *ast_source =
          llvm::dyn_cast_or_null<ClangExternalASTSourceCallbacks>(
              clang_typesystem->getASTContext().getExternalSource());
      if (auto *apinotes_manager = get_apinotes_manager(ast_source, id)) {
        for (auto reader : apinotes_manager->getCurrentModuleReaders()) {
          if (llvm::isa<clang::TypedefNameDecl>(clang_decl)) {
            auto info = reader->lookupTypedef(ident);
            if (auto version = info.getSelected()) {
              clang::api_notes::TypedefInfo typedef_info =
                  info[*version].second;
              if (!typedef_info.SwiftName.empty())
                swift_name = typedef_info.SwiftName;
              break;
            }
          }
          if (llvm::isa<clang::TagDecl>(clang_decl)) {
            auto info = reader->lookupTag(ident);
            if (auto version = info.getSelected()) {
              clang::api_notes::TagInfo tag_info = info[*version].second;
              if (!tag_info.SwiftName.empty())
                swift_name = tag_info.SwiftName;
              break;
            }
          }
          auto extract_context_info =
              [&](clang::api_notes::APINotesReader::VersionedInfo<
                  clang::api_notes::ObjCContextInfo>
                      info) -> bool {
            if (auto version = info.getSelected()) {
              clang::api_notes::ObjCContextInfo context_info =
                  info[*version].second;
              if (!context_info.SwiftName.empty())
                swift_name = context_info.SwiftName;
              return true;
            }
            return false;
          };
          if (llvm::isa<clang::ObjCInterfaceDecl>(clang_decl)) {
            auto info = reader->lookupObjCClassInfo(ident);
            if (extract_context_info(info))
              break;
          }
          if (llvm::isa<clang::ObjCProtocolDecl>(clang_decl)) {
            auto info = reader->lookupObjCProtocolInfo(ident);
            if (extract_context_info(info))
              break;
          }
        }
      }
    }
      
    auto clang_importer = get_clangimporter();
    if (!clang_importer)
      break;
    //swift::DeclName imported_name = clang_importer->importName(clang_decl, {});
    //imported_name.getBaseName().userFacingName()
    NodePointer identifier = Dem.createNode(Node::Kind::Identifier, swift_name);
    renamed->addChild(identifier, Dem);
    return renamed;
  }

  //
  // The remaining cases are all about bug-for-bug compatibility
  // with the type dumper and we don't need to carry them forward
  // necessarily.
  //

  // The type dumper doesn't print these.
#define REF_STORAGE(Name, ...)                                                 \
  case Node::Kind::Name:                                                       \
    return (node->getNumChildren() == 1)                                       \
               ? getNodeForPrinting(node->getChild(0))                         \
               : node;
#include "swift/AST/ReferenceStorage.def"

  case Node::Kind::ImplFunctionType: {
    // Rewrite ImplFunctionType nodes as FunctionType nodes.
    NodePointer fnty = Dem.createNode(Node::Kind::FunctionType);
    NodePointer args = Dem.createNode(Node::Kind::ArgumentTuple);
    NodePointer rett = Dem.createNode(Node::Kind::ReturnType);
    NodePointer args_ty = Dem.createNode(Node::Kind::Type);
    NodePointer args_tuple = Dem.createNode(Node::Kind::Tuple);
    for (NodePointer child : *node) {
      if (child->getKind() == Node::Kind::ImplParameter)
        for (NodePointer type : *node)
          if (type->getKind() == Node::Kind::Type &&
              type->getNumChildren() == 1)
            rett->addChild(type->getChild(0), Dem);
      else if (child->getKind() == Node::Kind::ImplResult)
        for (NodePointer type : *node)
          if (type->getKind() == Node::Kind::Type)
            rett->addChild(type, Dem);
    }
    args_ty->addChild(args_tuple, Dem);
    args->addChild(args_ty, Dem);
    fnty->addChild(args, Dem);
    if (rett->getNumChildren() != 1)
      rett->addChild(Dem.createNode(Node::Kind::Tuple), Dem);
    fnty->addChild(rett, Dem);
    return fnty;
  }

  case Node::Kind::SugaredOptional:
    // This is particularly silly. The outermost sugared Optional is desugared.
    // See SwiftASTContext::GetTypeName() and remove it there, too!
    if (desugar && node->getNumChildren() == 1) {
      desugar = false;
      return Desugar(Dem, node, Node::Kind::BoundGenericEnum, Node::Kind::Enum,
                     "Optional");
    }
    return node;
  case Node::Kind::SugaredArray:
    // See comment on SugaredOptional.
    if (desugar && node->getNumChildren() == 1) {
      desugar = false;
      return Desugar(Dem, node, Node::Kind::BoundGenericStructure,
                     Node::Kind::Structure, "Array");
    }
    return node;
  case Node::Kind::SugaredDictionary:
    // See comment on SugaredOptional.
    if (desugar && node->getNumChildren() == 1) {
      desugar = false;
      return Desugar(Dem, node, Node::Kind::BoundGenericStructure,
                     Node::Kind::Structure, "Dictionary");
    }
    return node;
  case Node::Kind::DependentAssociatedTypeRef:
    if (node->getNumChildren() == 2 &&
        node->getChild(0)->getKind() == Node::Kind::Identifier)
      return node->getChild(0);
    break;    
  default:
    break;
  }

  // Recurse through all children.
  // FIXME: don't create new nodes if children don't change!
  if (node->hasText())
    canonical = Dem.createNodeWithAllocatedText(kind, node->getText());
  else if (node->hasIndex())
    canonical = Dem.createNode(kind, node->getIndex());
  else
    canonical = Dem.createNode(kind);

  // Bug-for-bug compatibility. Remove this loop!
  // Strip out LocalDeclNames.
  for (unsigned i = 0; i < node->getNumChildren(); ++i) {
    NodePointer child = node->getChild(i);
    if (child->getKind() == Node::Kind::LocalDeclName)
      for (NodePointer identifier : *child)
        if (identifier->getKind() == Node::Kind::Identifier) {
          NodePointer module = nullptr;
          if (node->getChild(0)->getNumChildren() > 1)
            module = node->getChild(0)->getChild(0);
          if (module->getKind() != Node::Kind::Module)
            break;
            
          canonical->addChild(module, Dem);
          canonical->addChild(identifier, Dem);
          return canonical;
        }
  }

  for (unsigned i = 0; i < node->getNumChildren(); ++i)
    canonical->addChild(getNodeForPrinting(node->getChild(i)), Dem);
  return canonical;
}

/// Return the demangle tree representation with all "__C" module
/// names with their actual Clang module names.
static swift::Demangle::NodePointer GetDemangleTreeForPrinting(
    const std::string &m_description, lldb_private::Module *Module,
    GetAPINotesManagerFn get_apinotes_manager,
    GetClangImporterFn get_clangimporter, swift::Demangle::Demangler &Dem,
    const char *mangled_name) {
  NodePointer node = Dem.demangleSymbol(mangled_name);
  if (!Module)
    return node;
  NodePointer canonical =
      GetNodeForPrinting(m_description, *Module, get_apinotes_manager,
                         get_clangimporter, Dem, node);
  return canonical;
}

CompilerType TypeSystemSwift::GetInstanceType(CompilerType compiler_type) {
  auto *ts = compiler_type.GetTypeSystem();
  if (auto *tr = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(ts))
    return tr->GetInstanceType(compiler_type.GetOpaqueQualType());
  if (auto *ast = llvm::dyn_cast_or_null<SwiftASTContext>(ts))
    return ast->GetInstanceType(compiler_type.GetOpaqueQualType());
  return {};
}

TypeSystemSwiftTypeRef::TypeSystemSwiftTypeRef(
    SwiftASTContext *swift_ast_context)
    : m_swift_ast_context(swift_ast_context) {
  m_description = "TypeSystemSwiftTypeRef";
}

const char *TypeSystemSwiftTypeRef::AsMangledName(opaque_compiler_type_t type) {
  assert(type && *reinterpret_cast<const char *>(type) == '$' &&
         "wrong type system");
  return reinterpret_cast<const char *>(type);
}

ConstString
TypeSystemSwiftTypeRef::GetMangledTypeName(opaque_compiler_type_t type) {
  // FIXME: Suboptimal performance, because the ConstString is looked up again.
  return ConstString(AsMangledName(type));
}

void *TypeSystemSwiftTypeRef::ReconstructType(opaque_compiler_type_t type) {
  Status error;
  return m_swift_ast_context->ReconstructType(GetMangledTypeName(type), error);
}

CompilerType TypeSystemSwiftTypeRef::ReconstructType(CompilerType type) {
  return {m_swift_ast_context, ReconstructType(type.GetOpaqueQualType())};
}

CompilerType TypeSystemSwiftTypeRef::GetTypeFromMangledTypename(
    ConstString mangled_typename) {
  return {this, (opaque_compiler_type_t)mangled_typename.AsCString()};
}

lldb::TypeSP TypeSystemSwiftTypeRef::GetCachedType(ConstString mangled) {
  return m_swift_ast_context->GetCachedType(mangled);
}

void TypeSystemSwiftTypeRef::SetCachedType(ConstString mangled,
                                           const lldb::TypeSP &type_sp) {
  return m_swift_ast_context->SetCachedType(mangled, type_sp);
}

Module *TypeSystemSwiftTypeRef::GetModule() const {
  return m_swift_ast_context ? m_swift_ast_context->GetModule() : nullptr;
}

ConstString TypeSystemSwiftTypeRef::GetPluginName() {
  return ConstString("TypeSystemSwiftTypeRef");
}
uint32_t TypeSystemSwiftTypeRef::GetPluginVersion() { return 1; }

bool TypeSystemSwiftTypeRef::SupportsLanguage(lldb::LanguageType language) {
  return language == eLanguageTypeSwift;
}

Status TypeSystemSwiftTypeRef::IsCompatible() {
  return m_swift_ast_context->IsCompatible();
}

void TypeSystemSwiftTypeRef::DiagnoseWarnings(Process &process,
                                              Module &module) const {
  m_swift_ast_context->DiagnoseWarnings(process, module);
}
DWARFASTParser *TypeSystemSwiftTypeRef::GetDWARFParser() {
  return m_swift_ast_context->GetDWARFParser();
}

// Tests

#ifndef NDEBUG
bool TypeSystemSwiftTypeRef::Verify(opaque_compiler_type_t type) {
  if (!type)
    return true;

  const char *str = reinterpret_cast<const char *>(type);
  if (!SwiftLanguageRuntime::IsSwiftMangledName(str))
    return false;

  // Finally, check that the mangled name is canonical.
  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = dem.demangleSymbol(str);
  std::string remangled = mangleNode(node);
  return remangled == std::string(str);
}

#include <regex>
namespace {
template <typename T> bool Equivalent(T l, T r) { return l == r; }
/// Compare two swift types from different type systems by comparing their
/// (canonicalized) mangled name.
template <> bool Equivalent<CompilerType>(CompilerType l, CompilerType r) {
  return l.GetMangledTypeName() == r.GetMangledTypeName();
} // namespace
// This one is particularly taylored for GetName().
template <> bool Equivalent<ConstString>(ConstString l, ConstString r) {
  if (l != r) {
    // For some reason the Swift type dumper doesn't attach a module
    // name to the AnyObject protocol, and only that one.
    std::string l_prime = std::regex_replace(
        l.GetStringRef().str(), std::regex("Swift.AnyObject"), "AnyObject");
    if (llvm::StringRef(l_prime) == r.GetStringRef())
      return true;

    // If the new variant supports something the old one didn't, accept it.
    if (r.IsEmpty() || r.GetStringRef().equals("<invalid>") ||
        r.GetStringRef().contains("__ObjC.") || r.GetStringRef().contains(" -> ()"))
      return true;

    // Failure. Dump it for easier debugging.
    llvm::dbgs() << l.GetStringRef() << " != " << r.GetStringRef() << "\n";
  }
  return l == r;
}

/// A list of testcases for which the validation is disabled.
bool Whitelisted(SwiftASTContext &swift_ast_context) {
  auto *sym_file = swift_ast_context.GetSymbolFile();
  if (!sym_file)
    return false;
  auto *obj_file = sym_file->GetObjectFile();
  if (!obj_file)
    return false;
  auto &file_spec = obj_file->GetFileSpec();
  std::string path = file_spec.GetPath(false);
  llvm::StringRef p(path);
  return p.contains("lang/swift/clangimporter/include_conflict/"
                    "TestSwiftIncludeConflict") ||
         p.contains("lang/swift/clangimporter/headermap_conflict/"
                    "TestSwiftHeadermapConflict") ||
         p.contains("lang/swift/clangimporter/extra_clang_flags/"
                    "TestSwiftExtraClangFlags");
}
}
#endif

// This can be removed once the transition is complete.
#define VALIDATE_AND_RETURN(IMPL, EXPECTED)                                    \
  do {                                                                         \
    auto result = IMPL();                                                      \
    if (m_swift_ast_context)                                                   \
      assert((Equivalent(result, (EXPECTED)) ||                                \
              Whitelisted(*m_swift_ast_context)) &&                            \
             "TypeSystemSwiftTypeRef diverges from SwiftASTContext");          \
    return result;                                                             \
  } while (0)

CompilerType
TypeSystemSwiftTypeRef::RemangleAsType(swift::Demangle::Demangler &Dem,
                                       swift::Demangle::NodePointer node) {
  if (!node)
    return {};
  assert(node->getKind() == Node::Kind::Type && "expected type node");

  using namespace swift::Demangle;
  auto global = Dem.createNode(Node::Kind::Global);
  auto type_mangling = Dem.createNode(Node::Kind::TypeMangling);
  global->addChild(type_mangling, Dem);
  type_mangling->addChild(node, Dem);
  ConstString mangled_element(mangleNode(global));
  return GetTypeFromMangledTypename(mangled_element);
}

swift::Demangle::NodePointer TypeSystemSwiftTypeRef::DemangleCanonicalType(
    swift::Demangle::Demangler &Dem, opaque_compiler_type_t opaque_type) {
  using namespace swift::Demangle;
  if (!opaque_type)
    return nullptr;
  NodePointer node =
      GetCanonicalDemangleTree(GetModule(), Dem, AsMangledName(opaque_type));

  if (!node || node->getNumChildren() != 1 ||
      node->getKind() != Node::Kind::Global)
    return nullptr;
  node = node->getFirstChild();
  if (node->getNumChildren() != 1 ||
      node->getKind() != Node::Kind::TypeMangling)
    return nullptr;
  node = node->getFirstChild();
  if (node->getNumChildren() != 1 || node->getKind() != Node::Kind::Type)
    return nullptr;
  node = node->getFirstChild();
  return node;
}

bool TypeSystemSwiftTypeRef::IsArrayType(opaque_compiler_type_t type,
                                         CompilerType *element_type,
                                         uint64_t *size, bool *is_incomplete) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = DemangleCanonicalType(Dem, type);
    if (!node || node->getNumChildren() != 2 ||
        node->getKind() != Node::Kind::BoundGenericStructure)
      return false;
    auto elem_node = node->getChild(1);
    node = node->getFirstChild();
    if (node->getNumChildren() != 1 || node->getKind() != Node::Kind::Type)
      return false;
    node = node->getFirstChild();
    if (node->getNumChildren() != 2 ||
        node->getKind() != Node::Kind::Structure ||
        node->getChild(0)->getKind() != Node::Kind::Module ||
        !node->getChild(0)->hasText() ||
        node->getChild(0)->getText() != swift::STDLIB_NAME ||
        node->getChild(1)->getKind() != Node::Kind::Identifier ||
        !node->getChild(1)->hasText() ||
        (node->getChild(1)->getText() != "Array" &&
         node->getChild(1)->getText() != "NativeArray" &&
         node->getChild(1)->getText() != "ArraySlice"))
      return false;

    if (elem_node->getNumChildren() != 1 ||
        elem_node->getKind() != Node::Kind::TypeList)
      return false;
    elem_node = elem_node->getFirstChild();
    if (element_type)
      *element_type = RemangleAsType(Dem, elem_node);

    if (is_incomplete)
      *is_incomplete = true;
    if (size)
      *size = 0;

    return true;
  };
  VALIDATE_AND_RETURN(
      impl, m_swift_ast_context->IsArrayType(ReconstructType(type), nullptr,
                                             nullptr, nullptr));
}
bool TypeSystemSwiftTypeRef::IsAggregateType(opaque_compiler_type_t type) {
  return m_swift_ast_context->IsAggregateType(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsDefined(opaque_compiler_type_t type) {
  return m_swift_ast_context->IsDefined(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsFloatingPointType(opaque_compiler_type_t type,
                                                 uint32_t &count,
                                                 bool &is_complex) {
  return m_swift_ast_context->IsFloatingPointType(ReconstructType(type), count,
                                                  is_complex);
}

bool TypeSystemSwiftTypeRef::IsFunctionType(opaque_compiler_type_t type,
                                            bool *is_variadic_ptr) {
  auto impl = [&]() -> bool {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = DemangleCanonicalType(Dem, type);
    return node && (node->getKind() == Node::Kind::FunctionType ||
                    node->getKind() == Node::Kind::ImplFunctionType);
  };
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->IsFunctionType(
                                ReconstructType(type), nullptr));
}
size_t TypeSystemSwiftTypeRef::GetNumberOfFunctionArguments(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> size_t {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = DemangleCanonicalType(Dem, type);
    if (!node || (node->getKind() != Node::Kind::FunctionType &&
                  node->getKind() != Node::Kind::ImplFunctionType))
      return 0;
    unsigned num_args = 0;
    for (NodePointer child : *node) {
      if (child->getKind() == Node::Kind::ImplParameter)
        ++num_args;
      if (child->getKind() == Node::Kind::ArgumentTuple &&
          child->getNumChildren() == 1) {
        NodePointer node = child->getFirstChild();
        if (node->getNumChildren() != 1 || node->getKind() != Node::Kind::Type)
          break;
        node = node->getFirstChild();
        if (node->getKind() == Node::Kind::Tuple)
          return node->getNumChildren();
      }
    }
    return num_args;
  };
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->GetNumberOfFunctionArguments(
                                ReconstructType(type)));
}
CompilerType
TypeSystemSwiftTypeRef::GetFunctionArgumentAtIndex(opaque_compiler_type_t type,
                                                   const size_t index) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = DemangleCanonicalType(Dem, type);
    if (!node || (node->getKind() != Node::Kind::FunctionType &&
                  node->getKind() != Node::Kind::ImplFunctionType))
      return {};
    unsigned num_args = 0;
    for (NodePointer child : *node) {
      if (child->getKind() == Node::Kind::ImplParameter) {
        if (num_args == index)
          for (NodePointer type : *child)
            if (type->getKind() == Node::Kind::Type)
              return RemangleAsType(Dem, type);
        ++num_args;
      }
      if (child->getKind() == Node::Kind::ArgumentTuple &&
          child->getNumChildren() == 1) {
        NodePointer node = child->getFirstChild();
        if (node->getNumChildren() != 1 || node->getKind() != Node::Kind::Type)
          break;
        node = node->getFirstChild();
        if (node->getKind() == Node::Kind::Tuple)
          for (NodePointer child : *node) {
            if (child->getNumChildren() == 1 &&
                child->getKind() == Node::Kind::TupleElement) {
              NodePointer type = child->getFirstChild();
              if (num_args == index && type->getKind() == Node::Kind::Type)
                return RemangleAsType(Dem, type);
              ++num_args;
            }
          }
      }
    }
    return {};
  };
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->GetFunctionArgumentAtIndex(
                                ReconstructType(type), index));
}
bool TypeSystemSwiftTypeRef::IsFunctionPointerType(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> bool { return IsFunctionType(type, nullptr); };
  VALIDATE_AND_RETURN(
      impl, m_swift_ast_context->IsFunctionPointerType(ReconstructType(type)));
}
bool TypeSystemSwiftTypeRef::IsIntegerType(opaque_compiler_type_t type,
                                           bool &is_signed) {
  return m_swift_ast_context->IsIntegerType(ReconstructType(type), is_signed);
}
bool TypeSystemSwiftTypeRef::IsPossibleDynamicType(opaque_compiler_type_t type,
                                                   CompilerType *target_type,
                                                   bool check_cplusplus,
                                                   bool check_objc) {
  return m_swift_ast_context->IsPossibleDynamicType(
      ReconstructType(type), target_type, check_cplusplus, check_objc);
}
bool TypeSystemSwiftTypeRef::IsPointerType(opaque_compiler_type_t type,
                                           CompilerType *pointee_type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = DemangleCanonicalType(Dem, type);
    if (!node || node->getKind() != Node::Kind::BuiltinTypeName ||
        !node->hasText())
      return false;
    return ((node->getText() == swift::BUILTIN_TYPE_NAME_RAWPOINTER) ||
            (node->getText() == swift::BUILTIN_TYPE_NAME_UNSAFEVALUEBUFFER) ||
            (node->getText() == swift::BUILTIN_TYPE_NAME_NATIVEOBJECT) ||
            (node->getText() == swift::BUILTIN_TYPE_NAME_BRIDGEOBJECT));
  };
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->IsPointerType(
                                ReconstructType(type), pointee_type));
}
bool TypeSystemSwiftTypeRef::IsScalarType(opaque_compiler_type_t type) {
  return m_swift_ast_context->IsScalarType(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsVoidType(opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = DemangleCanonicalType(Dem, type);
    return node && node->getNumChildren() == 0 &&
           node->getKind() == Node::Kind::Tuple;
  };
  VALIDATE_AND_RETURN(impl,
                      m_swift_ast_context->IsVoidType(ReconstructType(type)));
}
// Type Completion
bool TypeSystemSwiftTypeRef::GetCompleteType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetCompleteType(ReconstructType(type));
}
// AST related queries
uint32_t TypeSystemSwiftTypeRef::GetPointerByteSize() {
  return m_swift_ast_context->GetPointerByteSize();
}
// Accessors
ConstString TypeSystemSwiftTypeRef::GetTypeName(opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer print_node = GetDemangleTreeForPrinting(
        m_description, GetModule(),
        [&](ClangExternalASTSourceCallbacks *source, unsigned id) {
          return GetAPINotesManager(source, id);
        },
        [&]() { return m_swift_ast_context->GetClangImporter(); }, Dem,
        AsMangledName(type));
    std::string remangled = mangleNode(print_node);
    bool simplified = false;
    return ConstString(
        SwiftLanguageRuntime::DemangleSymbolAsString(remangled, simplified));
  };
  VALIDATE_AND_RETURN(impl,
                      m_swift_ast_context->GetTypeName(ReconstructType(type)));
}
ConstString
TypeSystemSwiftTypeRef::GetDisplayTypeName(opaque_compiler_type_t type,
                                           const SymbolContext *sc) {
  return m_swift_ast_context->GetDisplayTypeName(ReconstructType(type), sc);
}
uint32_t TypeSystemSwiftTypeRef::GetTypeInfo(
    opaque_compiler_type_t type, CompilerType *pointee_or_element_clang_type) {
  return m_swift_ast_context->GetTypeInfo(ReconstructType(type),
                                          pointee_or_element_clang_type);
}
lldb::LanguageType
TypeSystemSwiftTypeRef::GetMinimumLanguage(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetMinimumLanguage(ReconstructType(type));
}
lldb::TypeClass
TypeSystemSwiftTypeRef::GetTypeClass(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetTypeClass(ReconstructType(type));
}

// Creating related types
CompilerType
TypeSystemSwiftTypeRef::GetArrayElementType(opaque_compiler_type_t type,
                                            uint64_t *stride) {
  auto impl = [&]() {
    CompilerType element_type;
    IsArrayType(type, &element_type, nullptr, nullptr);
    return element_type;
  };
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->GetArrayElementType(
                                ReconstructType(type), nullptr));
}
CompilerType
TypeSystemSwiftTypeRef::GetCanonicalType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetCanonicalType(ReconstructType(type));
}
int TypeSystemSwiftTypeRef::GetFunctionArgumentCount(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> int { return GetNumberOfFunctionArguments(type); };
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->GetFunctionArgumentCount(
                                ReconstructType(type)));
}
CompilerType TypeSystemSwiftTypeRef::GetFunctionArgumentTypeAtIndex(
    opaque_compiler_type_t type, size_t idx) {
  auto impl = [&] { return GetFunctionArgumentAtIndex(type, idx); };
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->GetFunctionArgumentTypeAtIndex(
                                ReconstructType(type), idx));
}
CompilerType
TypeSystemSwiftTypeRef::GetFunctionReturnType(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = DemangleCanonicalType(Dem, type);
    if (!node || (node->getKind() != Node::Kind::FunctionType &&
                  node->getKind() != Node::Kind::ImplFunctionType))
      return {};
    unsigned num_args = 0;
    for (NodePointer child : *node) {
      if (child->getKind() == Node::Kind::ImplResult) {
        for (NodePointer type : *child)
          if (type->getKind() == Node::Kind::Type)
            return RemangleAsType(Dem, type);
      }
      if (child->getKind() == Node::Kind::ReturnType &&
          child->getNumChildren() == 1) {
        NodePointer type = child->getFirstChild();
        if (type->getKind() == Node::Kind::Type)
          return RemangleAsType(Dem, type);
      }
    }
    // Else this is a void / "()" type.
    NodePointer type = Dem.createNode(Node::Kind::Type);
    NodePointer tuple = Dem.createNode(Node::Kind::Tuple);
    type->addChild(tuple, Dem);
    return RemangleAsType(Dem, type);
  };
  VALIDATE_AND_RETURN(
      impl, m_swift_ast_context->GetFunctionReturnType(ReconstructType(type)));
}
size_t
TypeSystemSwiftTypeRef::GetNumMemberFunctions(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetNumMemberFunctions(ReconstructType(type));
}
TypeMemberFunctionImpl
TypeSystemSwiftTypeRef::GetMemberFunctionAtIndex(opaque_compiler_type_t type,
                                                 size_t idx) {
  return m_swift_ast_context->GetMemberFunctionAtIndex(ReconstructType(type),
                                                       idx);
}
CompilerType
TypeSystemSwiftTypeRef::GetPointeeType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetPointeeType(ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetPointerType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetPointerType(ReconstructType(type));
}

// Exploring the type
llvm::Optional<uint64_t>
TypeSystemSwiftTypeRef::GetBitSize(opaque_compiler_type_t type,
                                   ExecutionContextScope *exe_scope) {
  return m_swift_ast_context->GetBitSize(ReconstructType(type), exe_scope);
}
llvm::Optional<uint64_t>
TypeSystemSwiftTypeRef::GetByteStride(opaque_compiler_type_t type,
                                      ExecutionContextScope *exe_scope) {
  return m_swift_ast_context->GetByteStride(ReconstructType(type), exe_scope);
}
lldb::Encoding TypeSystemSwiftTypeRef::GetEncoding(opaque_compiler_type_t type,
                                                   uint64_t &count) {
  return m_swift_ast_context->GetEncoding(ReconstructType(type), count);
}
lldb::Format TypeSystemSwiftTypeRef::GetFormat(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetFormat(ReconstructType(type));
}
uint32_t
TypeSystemSwiftTypeRef::GetNumChildren(opaque_compiler_type_t type,
                                       bool omit_empty_base_classes,
                                       const ExecutionContext *exe_ctx) {
  return m_swift_ast_context->GetNumChildren(ReconstructType(type),
                                             omit_empty_base_classes, exe_ctx);
}
uint32_t TypeSystemSwiftTypeRef::GetNumFields(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetNumFields(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetFieldAtIndex(
    opaque_compiler_type_t type, size_t idx, std::string &name,
    uint64_t *bit_offset_ptr, uint32_t *bitfield_bit_size_ptr,
    bool *is_bitfield_ptr) {
  return m_swift_ast_context->GetFieldAtIndex(
      ReconstructType(type), idx, name, bit_offset_ptr, bitfield_bit_size_ptr,
      is_bitfield_ptr);
}
CompilerType TypeSystemSwiftTypeRef::GetChildCompilerTypeAtIndex(
    opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
    bool transparent_pointers, bool omit_empty_base_classes,
    bool ignore_array_bounds, std::string &child_name,
    uint32_t &child_byte_size, int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class, bool &child_is_deref_of_parent,
    ValueObject *valobj, uint64_t &language_flags) {
  return m_swift_ast_context->GetChildCompilerTypeAtIndex(
      ReconstructType(type), exe_ctx, idx, transparent_pointers,
      omit_empty_base_classes, ignore_array_bounds, child_name, child_byte_size,
      child_byte_offset, child_bitfield_bit_size, child_bitfield_bit_offset,
      child_is_base_class, child_is_deref_of_parent, valobj, language_flags);
}
uint32_t
TypeSystemSwiftTypeRef::GetIndexOfChildWithName(opaque_compiler_type_t type,
                                                const char *name,
                                                bool omit_empty_base_classes) {
  return m_swift_ast_context->GetIndexOfChildWithName(
      ReconstructType(type), name, omit_empty_base_classes);
}
size_t TypeSystemSwiftTypeRef::GetIndexOfChildMemberWithName(
    opaque_compiler_type_t type, const char *name, bool omit_empty_base_classes,
    std::vector<uint32_t> &child_indexes) {
  return m_swift_ast_context->GetIndexOfChildMemberWithName(
      ReconstructType(type), name, omit_empty_base_classes, child_indexes);
}
size_t
TypeSystemSwiftTypeRef::GetNumTemplateArguments(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetNumTemplateArguments(ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetTypeForFormatters(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetTypeForFormatters(ReconstructType(type));
}
LazyBool
TypeSystemSwiftTypeRef::ShouldPrintAsOneLiner(opaque_compiler_type_t type,
                                              ValueObject *valobj) {
  return m_swift_ast_context->ShouldPrintAsOneLiner(ReconstructType(type),
                                                    valobj);
}
bool TypeSystemSwiftTypeRef::IsMeaninglessWithoutDynamicResolution(
    opaque_compiler_type_t type) {
  return m_swift_ast_context->IsMeaninglessWithoutDynamicResolution(
      ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsImportedType(opaque_compiler_type_t type,
                                            CompilerType *original_type) {
  return m_swift_ast_context->IsImportedType(ReconstructType(type),
                                             original_type);
}
CompilerType TypeSystemSwiftTypeRef::GetErrorType() {
  return m_swift_ast_context->GetErrorType();
}

CompilerType
TypeSystemSwiftTypeRef::GetReferentType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetReferentType(ReconstructType(type));
}

CompilerType
TypeSystemSwiftTypeRef::GetInstanceType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetInstanceType(ReconstructType(type));
}
TypeSystemSwift::TypeAllocationStrategy
TypeSystemSwiftTypeRef::GetAllocationStrategy(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetAllocationStrategy(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::CreateTupleType(
    const std::vector<TupleElement> &elements) {
  return m_swift_ast_context->CreateTupleType(elements);
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(
    opaque_compiler_type_t type, bool print_help_if_available,
    bool print_extensions_if_available, lldb::DescriptionLevel level) {
  return m_swift_ast_context->DumpTypeDescription(
      ReconstructType(type), print_help_if_available, print_help_if_available,
      level);
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(
    opaque_compiler_type_t type, Stream *s, bool print_help_if_available,
    bool print_extensions_if_available, lldb::DescriptionLevel level) {
  return m_swift_ast_context->DumpTypeDescription(
      ReconstructType(type), s, print_help_if_available,
      print_extensions_if_available, level);
}

// Dumping types
#ifndef NDEBUG
/// Convenience LLVM-style dump method for use in the debugger only.
LLVM_DUMP_METHOD void
TypeSystemSwiftTypeRef::dump(opaque_compiler_type_t type) const {
  llvm::dbgs() << reinterpret_cast<const char *>(type) << "\n";
}
#endif

void TypeSystemSwiftTypeRef::DumpValue(
    opaque_compiler_type_t type, ExecutionContext *exe_ctx, Stream *s,
    lldb::Format format, const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, bool show_types, bool show_summary,
    bool verbose, uint32_t depth) {
  return m_swift_ast_context->DumpValue(
      ReconstructType(type), exe_ctx, s, format, data, data_offset,
      data_byte_size, bitfield_bit_size, bitfield_bit_offset, show_types,
      show_summary, verbose, depth);
}

bool TypeSystemSwiftTypeRef::DumpTypeValue(
    opaque_compiler_type_t type, Stream *s, lldb::Format format,
    const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, ExecutionContextScope *exe_scope,
    bool is_base_class) {
  return m_swift_ast_context->DumpTypeValue(
      ReconstructType(type), s, format, data, data_offset, data_byte_size,
      bitfield_bit_size, bitfield_bit_offset, exe_scope, is_base_class);
}

void TypeSystemSwiftTypeRef::DumpTypeDescription(opaque_compiler_type_t type,
                                                 lldb::DescriptionLevel level) {
  return m_swift_ast_context->DumpTypeDescription(ReconstructType(type), level);
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(opaque_compiler_type_t type,
                                                 Stream *s,
                                                 lldb::DescriptionLevel level) {
  return m_swift_ast_context->DumpTypeDescription(ReconstructType(type), s,
                                                  level);
}
void TypeSystemSwiftTypeRef::DumpSummary(opaque_compiler_type_t type,
                                         ExecutionContext *exe_ctx, Stream *s,
                                         const DataExtractor &data,
                                         lldb::offset_t data_offset,
                                         size_t data_byte_size) {
  return m_swift_ast_context->DumpSummary(ReconstructType(type), exe_ctx, s,
                                          data, data_offset, data_byte_size);
}
bool TypeSystemSwiftTypeRef::IsPointerOrReferenceType(
    opaque_compiler_type_t type, CompilerType *pointee_type) {
  return m_swift_ast_context->IsPointerOrReferenceType(ReconstructType(type),
                                                       pointee_type);
}
llvm::Optional<size_t>
TypeSystemSwiftTypeRef::GetTypeBitAlign(opaque_compiler_type_t type,
                                        ExecutionContextScope *exe_scope) {
  return m_swift_ast_context->GetTypeBitAlign(ReconstructType(type), exe_scope);
}
bool TypeSystemSwiftTypeRef::IsTypedefType(opaque_compiler_type_t type) {
  return m_swift_ast_context->IsTypedefType(ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetTypedefedType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetTypedefedType(ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetFullyUnqualifiedType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetFullyUnqualifiedType(ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetNonReferenceType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetNonReferenceType(ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetLValueReferenceType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetLValueReferenceType(ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetRValueReferenceType(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetRValueReferenceType(ReconstructType(type));
}
uint32_t
TypeSystemSwiftTypeRef::GetNumDirectBaseClasses(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetNumDirectBaseClasses(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetDirectBaseClassAtIndex(
    opaque_compiler_type_t type, size_t idx, uint32_t *bit_offset_ptr) {
  return m_swift_ast_context->GetDirectBaseClassAtIndex(ReconstructType(type),
                                                        idx, bit_offset_ptr);
}
bool TypeSystemSwiftTypeRef::IsReferenceType(opaque_compiler_type_t type,
                                             CompilerType *pointee_type,
                                             bool *is_rvalue) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = DemangleCanonicalType(Dem, type);
    if (!node || node->getNumChildren() != 1 ||
        node->getKind() != Node::Kind::InOut)
      return false;

    if (pointee_type) {
      NodePointer referenced = node->getFirstChild();
      auto type = Dem.createNode(Node::Kind::Type);
      type->addChild(referenced, Dem);
      *pointee_type = RemangleAsType(Dem, type);
    }

    if (is_rvalue)
      *is_rvalue = false;

    return true;
  };

  VALIDATE_AND_RETURN(
      impl, m_swift_ast_context->IsReferenceType(ReconstructType(type),
                                                 pointee_type, is_rvalue));
}
bool TypeSystemSwiftTypeRef::ShouldTreatScalarValueAsAddress(
    opaque_compiler_type_t type) {
  return m_swift_ast_context->ShouldTreatScalarValueAsAddress(
      ReconstructType(type));
}
