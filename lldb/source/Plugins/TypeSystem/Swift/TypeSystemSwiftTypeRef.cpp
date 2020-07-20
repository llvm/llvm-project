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

#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"

#include "lldb/Symbol/CompileUnit.h"
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

#include "clang/APINotes/APINotesManager.h"
#include "clang/APINotes/APINotesReader.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"

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

/// Resolve a type alias node and return a demangle tree for the
/// resolved type. If the type alias resolves to a Clang type, return
/// a Clang CompilerType.
static std::pair<swift::Demangle::NodePointer, CompilerType>
ResolveTypeAlias(lldb_private::Module *M, swift::Demangle::Demangler &Dem,
                 swift::Demangle::NodePointer node) {
  // Try to look this up as a Swift type alias. For each *Swift*
  // type alias there is a debug info entry that has the mangled
  // name as name and the aliased type as a type.
  ConstString mangled = GetTypeAlias(Dem, node);
  if (!M) {
    LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
              "No module. Couldn't resolve type alias %s", mangled.AsCString());
    return {{}, {}};
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
      auto clang_type =
          LookupClangForwardType(*M, node->getChild(1)->getText());
      if (!clang_type)
        return {{}, {}};
      return {{}, clang_type};
    }

    LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
              "Couldn't resolve type alias %s", mangled.AsCString());
    return {{}, {}};
  }
  auto type = types.GetTypeAtIndex(0);
  if (!type) {
    LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
              "Found empty type alias %s", mangled.AsCString());
    return {{}, {}};
  }

  // DWARFASTParserSwift stashes the desugared mangled name of a
  // type alias into the Type's name field.
  ConstString desugared_name = type->GetName();
  if (!isMangledName(desugared_name.GetStringRef())) {
    LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
              "Found non-Swift type alias %s", mangled.AsCString());
    return {{}, {}};
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
    return {{}, {}};
  }
  return {n, {}};
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
    auto node_clangtype = ResolveTypeAlias(M, Dem, node);
    if (CompilerType clang_type = node_clangtype.second)
      return getCanonicalNode(
          GetClangTypeNode(clang_type.GetCanonicalType(), Dem));
    if (node_clangtype.first)
      return getCanonicalNode(node_clangtype.first);
    return node;
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
      path = llvm::sys::path::parent_path(parent->APINotesFile);
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
static swift::Demangle::NodePointer
Desugar(swift::Demangle::Demangler &Dem, swift::Demangle::NodePointer node,
        Node::Kind bound_kind, Node::Kind kind, llvm::StringRef name) {
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
                   swift::Demangle::NodePointer node, bool resolve_objc_module,
                   bool desugar = true) {
  if (!node)
    return node;
  using namespace swift::Demangle;
  auto getNodeForPrinting = [&](NodePointer node) -> NodePointer {
    return GetNodeForPrinting(m_description, M, get_apinotes_manager,
                              get_clangimporter, Dem, node, resolve_objc_module,
                              desugar);
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
    if (resolve_objc_module) {
      for (auto &Context : DeclCtx)
        if (Context.kind == CompilerContextKind::Module) {
          toplevel_module = Context.name.GetStringRef();
          break;
        }
      if (toplevel_module.empty())
        break;
    } else {
      toplevel_module = swift::MANGLING_MODULE_OBJC;
    }

    // Create a new node with the Clang module instead of "__C".
    NodePointer renamed = Dem.createNode(kind);
    NodePointer module = Dem.createNode(Node::Kind::Module, toplevel_module);
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
    std::string swift_name = ident;
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
    // swift::DeclName imported_name = clang_importer->importName(clang_decl,
    // {}); imported_name.getBaseName().userFacingName()
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
    const char *mangled_name, bool resolve_objc_module) {
  NodePointer node = Dem.demangleSymbol(mangled_name);
  if (!Module)
    return node;
  NodePointer canonical =
      GetNodeForPrinting(m_description, *Module, get_apinotes_manager,
                         get_clangimporter, Dem, node, resolve_objc_module);
  return canonical;
}

/// Collect TypeInfo flags from a demangle tree. For most attributes
/// this can stop scanning at the outmost type, however in order to
/// determine whether a node is generic or not, it needs to visit all
/// nodes. The \p generic_walk argument specifies that the primary
/// attributes have been collected and that we only look for generics.
static uint32_t collectTypeInfo(Module *M, swift::Demangle::Demangler &Dem,
                                swift::Demangle::NodePointer node,
                                bool generic_walk = false) {
  if (!node)
    return 0;
  uint32_t swift_flags = eTypeIsSwift;

  /// Collect type info from a clang-imported type.
  auto collect_clang_type = [&](CompilerType clang_type) {
    auto type_class = clang_type.GetTypeClass();
    // Classes.
    if ((type_class & eTypeClassClass) ||
        (type_class & eTypeClassObjCObjectPointer)) {
      swift_flags &= ~eTypeIsStructUnion;
      swift_flags |= eTypeIsClass | eTypeHasChildren | eTypeHasValue |
                     eTypeInstanceIsPointer;
      return;
    }
    // Structs.
    if ((type_class & eTypeClassStruct)) {
      swift_flags |= eTypeIsStructUnion | eTypeHasChildren;
      return;
    }
    // Enums.
    if ((type_class & eTypeClassEnumeration)) {
      swift_flags &= ~eTypeIsStructUnion;
      swift_flags |= eTypeIsEnumeration | eTypeHasChildren | eTypeHasValue;
      return;
    }
  };

  using namespace swift::Demangle;
  if (generic_walk)
    switch (node->getKind()) {
    // Bug-for-bug-compatibility.
    // FIXME: There should be more cases here.
    case Node::Kind::DynamicSelf:
      swift_flags |= eTypeIsGeneric;
      break;
    default:
      break;
    }
  else
    switch (node->getKind()) {
    case Node::Kind::SugaredOptional:
      swift_flags |= eTypeIsGeneric | eTypeIsBound | eTypeHasChildren |
                     eTypeHasValue | eTypeIsEnumeration;
      break;
    case Node::Kind::SugaredArray:
    case Node::Kind::SugaredDictionary:
      swift_flags |=
          eTypeIsGeneric | eTypeIsBound | eTypeHasChildren | eTypeIsStructUnion;
      break;

    case Node::Kind::DependentGenericParamType:
      swift_flags |= eTypeHasValue | eTypeIsPointer | eTypeIsScalar |
                     eTypeIsGenericTypeParam;
      break;
    case Node::Kind::DependentGenericType:
    case Node::Kind::DependentMemberType:
      swift_flags |= eTypeHasValue | eTypeIsPointer | eTypeIsScalar |
                     eTypeIsGenericTypeParam;
      break;

    case Node::Kind::DynamicSelf:
      swift_flags |= eTypeHasValue | eTypeIsGeneric | eTypeIsBound;
      break;

    case Node::Kind::ImplFunctionType:
      // Bug-for-bug-compatibility. Not sure if this is correct.
      swift_flags |= eTypeIsPointer | eTypeHasValue;
      return swift_flags;
      LLVM_FALLTHROUGH;
    case Node::Kind::BoundGenericFunction:
      swift_flags |= eTypeIsGeneric | eTypeIsBound;
      LLVM_FALLTHROUGH;
    case Node::Kind::FunctionType:
      swift_flags |= eTypeIsPointer | eTypeHasValue;
      break;
    case Node::Kind::BuiltinTypeName:
      swift_flags |= eTypeIsBuiltIn | eTypeHasValue;
      if (node->hasText()) {
        // TODO (performance): It may be safe to switch over the pointers here.
        if (node->getText() == swift::BUILTIN_TYPE_NAME_RAWPOINTER)
          swift_flags |= eTypeHasChildren | eTypeIsPointer | eTypeIsScalar;
        else if (node->getText() == swift::BUILTIN_TYPE_NAME_UNSAFEVALUEBUFFER)
          swift_flags |= eTypeIsPointer | eTypeIsScalar;
        else if (node->getText() == swift::BUILTIN_TYPE_NAME_NATIVEOBJECT)
          swift_flags |= eTypeHasChildren | eTypeIsPointer | eTypeIsScalar;
        else if (node->getText() == swift::BUILTIN_TYPE_NAME_BRIDGEOBJECT)
          swift_flags |=
              eTypeHasChildren | eTypeIsPointer | eTypeIsScalar | eTypeIsObjC;
        else if (node->getText().startswith(swift::BUILTIN_TYPE_NAME_VEC))
          swift_flags |= eTypeHasChildren | eTypeIsVector;
      }
      break;
    case Node::Kind::Tuple:
      swift_flags |= eTypeHasChildren | eTypeIsTuple;
      break;
    case Node::Kind::BoundGenericEnum:
      swift_flags |= eTypeIsGeneric | eTypeIsBound;
      LLVM_FALLTHROUGH;
    case Node::Kind::Enum: {
      // FIXME: do C-style enums have children?
      // The AST implementation is getting eTypeHasChildren out of the Decl.
      swift_flags |= eTypeIsEnumeration;
      if (node->getNumChildren() != 2)
        break;
      // Bug-for-bug compatibility.
      if (!(collectTypeInfo(M, Dem, node->getChild(1)) &
            eTypeIsGenericTypeParam))
        swift_flags |= eTypeHasValue | eTypeHasChildren;
      auto module = node->getChild(0);
      auto ident = node->getChild(1);
      if (module->hasText() &&
          module->getText() == swift::MANGLING_MODULE_OBJC) {
        swift_flags |= eTypeHasValue /*| eTypeIsObjC*/;
      }
      break;
    }
    case Node::Kind::BoundGenericStructure:
      swift_flags |= eTypeIsGeneric | eTypeIsBound;
      LLVM_FALLTHROUGH;
    case Node::Kind::Structure: {
      swift_flags |= eTypeHasChildren | eTypeIsStructUnion;
      if (node->getNumChildren() != 2)
        break;
      auto module = node->getChild(0);
      auto ident = node->getChild(1);
      // Builtin types.
      if (module->hasText() && module->getText() == swift::STDLIB_NAME) {
        if (ident->hasText() &&
            ident->getText().startswith(swift::BUILTIN_TYPE_NAME_INT))
          swift_flags |= eTypeIsScalar | eTypeIsInteger;
        else if (ident->hasText() &&
                 ident->getText().startswith(swift::BUILTIN_TYPE_NAME_FLOAT))
          swift_flags |= eTypeIsScalar | eTypeIsFloat;
      }

      // Clang-imported types.
      if (module->hasText() &&
          module->getText() == swift::MANGLING_MODULE_OBJC) {
        if (ident->getKind() != Node::Kind::Identifier || !ident->hasText())
          break;

        if (!M)
          break;
        // Look up the Clang type in DWARF.
        CompilerType clang_type = LookupClangForwardType(*M, ident->getText());
        collect_clang_type(clang_type.GetCanonicalType());
        return swift_flags;
      }
      break;
    }
    case Node::Kind::BoundGenericClass:
      swift_flags |= eTypeIsGeneric | eTypeIsBound;
      LLVM_FALLTHROUGH;
    case Node::Kind::Class:
      swift_flags |= eTypeHasChildren | eTypeIsClass | eTypeHasValue |
                     eTypeInstanceIsPointer;
      break;

    case Node::Kind::BoundGenericOtherNominalType:
      swift_flags |= eTypeIsGeneric | eTypeIsBound;
      swift_flags |= eTypeHasValue;
      break;

    case Node::Kind::BoundGenericProtocol:
      swift_flags |= eTypeIsGeneric | eTypeIsBound;
      LLVM_FALLTHROUGH;
    case Node::Kind::Protocol:
      swift_flags |= eTypeHasChildren | eTypeIsStructUnion | eTypeIsProtocol;
      break;
    case Node::Kind::ProtocolList:
    case Node::Kind::ProtocolListWithClass:
    case Node::Kind::ProtocolListWithAnyObject:
      swift_flags |= eTypeIsProtocol;
      // Bug-for-bug-compatibility.
      swift_flags |= eTypeHasChildren | eTypeIsStructUnion;
      break;

    case Node::Kind::ExistentialMetatype:
    case Node::Kind::Metatype:
      swift_flags |= eTypeIsMetatype | eTypeHasValue;
      break;

    case Node::Kind::InOut:
      swift_flags |= eTypeHasChildren | eTypeIsReference | eTypeHasValue;
      break;

    case Node::Kind::BoundGenericTypeAlias:
      // Bug-for-bug compatibility.
      // swift_flags |= eTypeIsGeneric | eTypeIsBound;
      LLVM_FALLTHROUGH;
    case Node::Kind::TypeAlias: {
      // Bug-for-bug compatibility.
      // swift_flags |= eTypeIsTypedef;
      auto node_clangtype = ResolveTypeAlias(M, Dem, node);
      if (CompilerType clang_type = node_clangtype.second) {
        collect_clang_type(clang_type.GetCanonicalType());
        return swift_flags;
      }
      swift_flags |=
          collectTypeInfo(M, Dem, node_clangtype.first, generic_walk);
      return swift_flags;
    }
    default:
      break;
    }

  // If swift_flags were collected we're done here except for
  // determining whether the type is generic.
  generic_walk |= (swift_flags != eTypeIsSwift);

  // Visit the child nodes.
  for (unsigned i = 0; i < node->getNumChildren(); ++i)
    swift_flags |= collectTypeInfo(M, Dem, node->getChild(i), generic_walk);

  return swift_flags;
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

/// Specialization for GetTypeInfo().
template <> bool Equivalent<uint32_t>(uint32_t l, uint32_t r) {
  if (l != r) {
    // Failure. Dump it for easier debugging.
    llvm::dbgs() << "TypeSystemSwiftTypeRef diverges from SwiftASTContext:\n";
#define HANDLE_ENUM_CASE(VAL, CASE)                                            \
  if (VAL & CASE)                                                              \
  llvm::dbgs() << " | " << #CASE

    llvm::dbgs() << "l = " << l;
    HANDLE_ENUM_CASE(l, eTypeHasChildren);
    HANDLE_ENUM_CASE(l, eTypeHasValue);
    HANDLE_ENUM_CASE(l, eTypeIsArray);
    HANDLE_ENUM_CASE(l, eTypeIsBlock);
    HANDLE_ENUM_CASE(l, eTypeIsBuiltIn);
    HANDLE_ENUM_CASE(l, eTypeIsClass);
    HANDLE_ENUM_CASE(l, eTypeIsCPlusPlus);
    HANDLE_ENUM_CASE(l, eTypeIsEnumeration);
    HANDLE_ENUM_CASE(l, eTypeIsFuncPrototype);
    HANDLE_ENUM_CASE(l, eTypeIsMember);
    HANDLE_ENUM_CASE(l, eTypeIsObjC);
    HANDLE_ENUM_CASE(l, eTypeIsPointer);
    HANDLE_ENUM_CASE(l, eTypeIsReference);
    HANDLE_ENUM_CASE(l, eTypeIsStructUnion);
    HANDLE_ENUM_CASE(l, eTypeIsTemplate);
    HANDLE_ENUM_CASE(l, eTypeIsTypedef);
    HANDLE_ENUM_CASE(l, eTypeIsVector);
    HANDLE_ENUM_CASE(l, eTypeIsScalar);
    HANDLE_ENUM_CASE(l, eTypeIsInteger);
    HANDLE_ENUM_CASE(l, eTypeIsFloat);
    HANDLE_ENUM_CASE(l, eTypeIsComplex);
    HANDLE_ENUM_CASE(l, eTypeIsSigned);
    HANDLE_ENUM_CASE(l, eTypeInstanceIsPointer);
    HANDLE_ENUM_CASE(l, eTypeIsSwift);
    HANDLE_ENUM_CASE(l, eTypeIsGenericTypeParam);
    HANDLE_ENUM_CASE(l, eTypeIsProtocol);
    HANDLE_ENUM_CASE(l, eTypeIsTuple);
    HANDLE_ENUM_CASE(l, eTypeIsMetatype);
    HANDLE_ENUM_CASE(l, eTypeIsGeneric);
    HANDLE_ENUM_CASE(l, eTypeIsBound);
    llvm::dbgs() << "\nr = " << r;

    HANDLE_ENUM_CASE(r, eTypeHasChildren);
    HANDLE_ENUM_CASE(r, eTypeHasValue);
    HANDLE_ENUM_CASE(r, eTypeIsArray);
    HANDLE_ENUM_CASE(r, eTypeIsBlock);
    HANDLE_ENUM_CASE(r, eTypeIsBuiltIn);
    HANDLE_ENUM_CASE(r, eTypeIsClass);
    HANDLE_ENUM_CASE(r, eTypeIsCPlusPlus);
    HANDLE_ENUM_CASE(r, eTypeIsEnumeration);
    HANDLE_ENUM_CASE(r, eTypeIsFuncPrototype);
    HANDLE_ENUM_CASE(r, eTypeIsMember);
    HANDLE_ENUM_CASE(r, eTypeIsObjC);
    HANDLE_ENUM_CASE(r, eTypeIsPointer);
    HANDLE_ENUM_CASE(r, eTypeIsReference);
    HANDLE_ENUM_CASE(r, eTypeIsStructUnion);
    HANDLE_ENUM_CASE(r, eTypeIsTemplate);
    HANDLE_ENUM_CASE(r, eTypeIsTypedef);
    HANDLE_ENUM_CASE(r, eTypeIsVector);
    HANDLE_ENUM_CASE(r, eTypeIsScalar);
    HANDLE_ENUM_CASE(r, eTypeIsInteger);
    HANDLE_ENUM_CASE(r, eTypeIsFloat);
    HANDLE_ENUM_CASE(r, eTypeIsComplex);
    HANDLE_ENUM_CASE(r, eTypeIsSigned);
    HANDLE_ENUM_CASE(r, eTypeInstanceIsPointer);
    HANDLE_ENUM_CASE(r, eTypeIsSwift);
    HANDLE_ENUM_CASE(r, eTypeIsGenericTypeParam);
    HANDLE_ENUM_CASE(r, eTypeIsProtocol);
    HANDLE_ENUM_CASE(r, eTypeIsTuple);
    HANDLE_ENUM_CASE(r, eTypeIsMetatype);
    HANDLE_ENUM_CASE(r, eTypeIsGeneric);
    HANDLE_ENUM_CASE(r, eTypeIsBound);
    llvm::dbgs() << "\n";
  }
  return l == r;
}

/// Compare two swift types from different type systems by comparing their
/// (canonicalized) mangled name.
template <> bool Equivalent<CompilerType>(CompilerType l, CompilerType r) {
  return l.GetMangledTypeName() == r.GetMangledTypeName();
} // namespace
/// This one is particularly taylored for GetTypeName() and
/// GetDisplayTypeName().
///
/// String divergences are mostly cosmetic in nature and usually
/// TypeSystemSwiftTypeRef is returning more accurate results. They only really
/// matter for GetTypeName() and there only if there is a data formatter
/// matching that name.
template <> bool Equivalent<ConstString>(ConstString l, ConstString r) {
  if (l != r) {
    // Failure. Dump it for easier debugging.
    llvm::dbgs() << "TypeSystemSwiftTypeRef diverges from SwiftASTContext: "
                 << l.GetStringRef() << " != " << r.GetStringRef() << "\n";

    // For some reason the Swift type dumper doesn't attach a module
    // name to the AnyObject protocol, and only that one.
    std::string l_prime = std::regex_replace(
        l.GetStringRef().str(), std::regex("Swift.AnyObject"), "AnyObject");
    if (llvm::StringRef(l_prime) == r.GetStringRef())
      return true;

    // If the new variant supports something the old one didn't, accept it.
    if (r.IsEmpty() || r.GetStringRef().equals("<invalid>") ||
        r.GetStringRef().contains("__ObjC.") ||
        r.GetStringRef().contains(" -> ()"))
      return true;

    std::string r_prime =
        std::regex_replace(r.GetStringRef().str(), std::regex("NS"), "");
    if (l.GetStringRef() == llvm::StringRef(r_prime))
      return true;

    // The way it is currently configured, ASTPrinter's always-qualify
    // mode is turned off. In this mode,
    // TypePrinter::shouldPrintFullyQualified() insists on never
    // printing qualifiers for types that come from Clang modules, but
    // the way this is implemented this rule also fires for types from
    // SDK overlays, which are technically Swift modules. Detecting
    // this in TypeSystemSwiftTypeRef is so complicated that it just
    // isn't worth the effort and we accept over-qualified types
    // instead. It would be best to just always qualify types not from
    // the current module.
    l_prime =
        std::regex_replace(l.GetStringRef().str(),
                           std::regex("(CoreGraphics|Foundation|)\\."), "");
    if (llvm::StringRef(l_prime) == r.GetStringRef())
      return true;

#ifndef STRICT_VALIDATION
    return true;
#endif
  }
  return l == r;
}
} // namespace
#endif

// This can be removed once the transition is complete.
#ifndef NDEBUG
#define VALIDATE_AND_RETURN_STATIC(IMPL, REFERENCE)                     \
  do {                                                                         \
    auto result = IMPL();                                                      \
    if (!m_swift_ast_context)                                                  \
      return result;                                                           \
    assert((result == m_swift_ast_context->REFERENCE()) &&                     \
           "TypeSystemSwiftTypeRef diverges from SwiftASTContext");            \
    return result;                                                             \
  } while (0)

#define VALIDATE_AND_RETURN(IMPL, REFERENCE, TYPE, ARGS)                       \
  do {                                                                         \
    auto result = IMPL();                                                      \
    if (!m_swift_ast_context)                                                  \
      return result;                                                           \
    bool equivalent =                                                          \
        !ReconstructType(TYPE) /* missing .swiftmodule */ ||                   \
        (Equivalent(result, m_swift_ast_context->REFERENCE ARGS));             \
    if (!equivalent)                                                           \
      llvm::dbgs() << "failing type was " << (const char *)TYPE << "\n";       \
    assert(equivalent &&                                                       \
           "TypeSystemSwiftTypeRef diverges from SwiftASTContext");            \
    return result;                                                             \
  } while (0)
#else
#define VALIDATE_AND_RETURN_STATIC(IMPL, REFERENCE) return IMPL()
#define VALIDATE_AND_RETURN(IMPL, REFERENCE, TYPE, ARGS) return IMPL()
#endif

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
  VALIDATE_AND_RETURN(impl, IsArrayType, type,
                      (ReconstructType(type), nullptr, nullptr, nullptr));
}

bool TypeSystemSwiftTypeRef::IsAggregateType(opaque_compiler_type_t type) {
  auto impl = [&]() -> bool {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = DemangleCanonicalType(Dem, type);

    if (!node)
      return false;
    switch (node->getKind()) {
    case Node::Kind::Structure:
    case Node::Kind::Class:
    case Node::Kind::Enum:
    case Node::Kind::Tuple:
    case Node::Kind::Protocol:
    case Node::Kind::ProtocolList:
    case Node::Kind::ProtocolListWithClass:
    case Node::Kind::ProtocolListWithAnyObject:
    case Node::Kind::BoundGenericClass:
    case Node::Kind::BoundGenericEnum:
    case Node::Kind::BoundGenericStructure:
    case Node::Kind::BoundGenericProtocol:
    case Node::Kind::BoundGenericOtherNominalType:
    case Node::Kind::BoundGenericTypeAlias:
      return true;
    default:
      return false;
    }
  };
  VALIDATE_AND_RETURN(impl, IsAggregateType, type, (ReconstructType(type)));
}

bool TypeSystemSwiftTypeRef::IsDefined(opaque_compiler_type_t type) {
  auto impl = [&]() -> bool { return type; };
  VALIDATE_AND_RETURN(impl, IsDefined, type, (ReconstructType(type)));
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
  VALIDATE_AND_RETURN(impl, IsFunctionType, type,
                      (ReconstructType(type), nullptr));
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
  VALIDATE_AND_RETURN(impl, GetNumberOfFunctionArguments, type,
                      (ReconstructType(type)));
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
  VALIDATE_AND_RETURN(impl, GetFunctionArgumentAtIndex, type,
                      (ReconstructType(type), index));
}
bool TypeSystemSwiftTypeRef::IsFunctionPointerType(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> bool { return IsFunctionType(type, nullptr); };
  VALIDATE_AND_RETURN(impl, IsFunctionPointerType, type,
                      (ReconstructType(type)));
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
  VALIDATE_AND_RETURN(impl, IsPointerType, type,
                      (ReconstructType(type), pointee_type));
}
bool TypeSystemSwiftTypeRef::IsVoidType(opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = DemangleCanonicalType(Dem, type);
    return node && node->getNumChildren() == 0 &&
           node->getKind() == Node::Kind::Tuple;
  };
  VALIDATE_AND_RETURN(impl, IsVoidType, type, (ReconstructType(type)));
}
// AST related queries
uint32_t TypeSystemSwiftTypeRef::GetPointerByteSize() {
  auto impl = [&]() -> uint32_t {
    if (auto *module = GetModule()) {
      auto &triple = module->GetArchitecture().GetTriple();
      if (triple.isArch64Bit())
        return 8;
      if (triple.isArch32Bit())
        return 4;
      if (triple.isArch16Bit())
        return 2;
    }
    // An expression context has no module. Since it's for expression
    // evaluation we might as well defer to the SwiftASTContext.
    return m_swift_ast_context->GetPointerByteSize();
  };
  VALIDATE_AND_RETURN_STATIC(impl, GetPointerByteSize);
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
        AsMangledName(type), true);
    std::string remangled = mangleNode(print_node);
    return ConstString(SwiftLanguageRuntime::DemangleSymbolAsString(
        remangled, SwiftLanguageRuntime::eTypeName));
  };
  VALIDATE_AND_RETURN(impl, GetTypeName, type, (ReconstructType(type)));
}
ConstString
TypeSystemSwiftTypeRef::GetDisplayTypeName(opaque_compiler_type_t type,
                                           const SymbolContext *sc) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer print_node = GetDemangleTreeForPrinting(
        m_description, GetModule(),
        [&](ClangExternalASTSourceCallbacks *source, unsigned id) {
          return GetAPINotesManager(source, id);
        },
        [&]() { return m_swift_ast_context->GetClangImporter(); }, Dem,
        AsMangledName(type), false);
    std::string remangled = mangleNode(print_node);
    return ConstString(SwiftLanguageRuntime::DemangleSymbolAsString(
        remangled, SwiftLanguageRuntime::eDisplayTypeName, sc));
  };
  VALIDATE_AND_RETURN(impl, GetDisplayTypeName, type,
                      (ReconstructType(type), sc));
}
uint32_t TypeSystemSwiftTypeRef::GetTypeInfo(
    opaque_compiler_type_t type, CompilerType *pointee_or_element_clang_type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer node = Dem.demangleSymbol(AsMangledName(type));
    return collectTypeInfo(GetModule(), Dem, node);
  };
  VALIDATE_AND_RETURN(impl, GetTypeInfo, type,
                      (ReconstructType(type), nullptr));
}
lldb::TypeClass
TypeSystemSwiftTypeRef::GetTypeClass(opaque_compiler_type_t type) {
  auto impl = [&]() {
    uint32_t flags = GetTypeInfo(type, nullptr);
    // The ordering is significant since GetTypeInfo() returns many flags.
    if ((flags & eTypeIsScalar))
      return eTypeClassBuiltin;
    if ((flags & eTypeIsVector))
      return eTypeClassVector;
    if ((flags & eTypeIsTuple))
      return eTypeClassArray;
    if ((flags & eTypeIsEnumeration))
      return eTypeClassUnion;
    if ((flags & eTypeIsProtocol))
      return eTypeClassOther;
    if ((flags & eTypeIsStructUnion))
      return eTypeClassStruct;
    if ((flags & eTypeIsClass))
      return eTypeClassClass;
    if ((flags & eTypeIsReference))
      return eTypeClassReference;
    // This only works because we excluded all other options.
    if ((flags & eTypeIsPointer))
      return eTypeClassFunction;
    return eTypeClassOther;
  };
  VALIDATE_AND_RETURN(impl, GetTypeClass, type, (ReconstructType(type)));
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
  VALIDATE_AND_RETURN(impl, GetArrayElementType, type,
                      (ReconstructType(type), nullptr));
}
CompilerType
TypeSystemSwiftTypeRef::GetCanonicalType(opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler Dem;
    NodePointer canonical =
        GetCanonicalDemangleTree(GetModule(), Dem, AsMangledName(type));
    ConstString mangled(mangleNode(canonical));
    return GetTypeFromMangledTypename(mangled);
  };
  VALIDATE_AND_RETURN(impl, GetCanonicalType, type, (ReconstructType(type)));
}
int TypeSystemSwiftTypeRef::GetFunctionArgumentCount(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> int { return GetNumberOfFunctionArguments(type); };
  VALIDATE_AND_RETURN(impl, GetFunctionArgumentCount, type,
                      (ReconstructType(type)));
}
CompilerType TypeSystemSwiftTypeRef::GetFunctionArgumentTypeAtIndex(
    opaque_compiler_type_t type, size_t idx) {
  auto impl = [&] { return GetFunctionArgumentAtIndex(type, idx); };
  VALIDATE_AND_RETURN(impl, GetFunctionArgumentTypeAtIndex, type,
                      (ReconstructType(type), idx));
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
  VALIDATE_AND_RETURN(impl, GetFunctionReturnType, type,
                      (ReconstructType(type)));
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

  VALIDATE_AND_RETURN(impl, IsReferenceType, type,
                      (ReconstructType(type), pointee_type, is_rvalue));
}
