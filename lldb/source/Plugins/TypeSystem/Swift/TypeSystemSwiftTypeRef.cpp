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
#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Timer.h"

#include "Plugins/ExpressionParser/Clang/ClangExternalASTSourceCallbacks.h"
#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include "swift/AST/ClangModuleLoader.h"
#include "swift/Basic/Version.h"
#include "swift/../../lib/ClangImporter/ClangAdapter.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/Demangler.h"
#include "swift/Strings.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/APINotes/APINotesManager.h"
#include "clang/APINotes/APINotesReader.h"

#include "llvm/ADT/ScopeExit.h"

#include <algorithm>
#include <sstream>

using namespace lldb;
using namespace lldb_private;

char TypeSystemSwift::ID;
char TypeSystemSwiftTypeRef::ID;

TypeSystemSwift::TypeSystemSwift() : TypeSystem() {}

/// Determine wether this demangle tree contains an unresolved type alias.
static bool ContainsUnresolvedTypeAlias(swift::Demangle::NodePointer node) {
  if (!node)
    return false;

  if (node->getKind() == swift::Demangle::Node::Kind::TypeAlias)
    return true;

  for (swift::Demangle::NodePointer child : *node)
    if (ContainsUnresolvedTypeAlias(child))
      return true;

  return false;
}

swift::Demangle::NodePointer
TypeSystemSwiftTypeRef::CanonicalizeSugar(swift::Demangle::Demangler &dem,
                                          swift::Demangle::NodePointer node) {
  using namespace swift::Demangle;
  return TypeSystemSwiftTypeRef::Transform(dem, node, [&](NodePointer n) {
    if ((n->getKind() != Node::Kind::BoundGenericEnum &&
         n->getKind() != Node::Kind::BoundGenericStructure) ||
        n->getNumChildren() != 2)
      return n;
    NodePointer type = n->getChild(0);
    if (!type || type->getNumChildren() != 1)
      return n;
    NodePointer payload = n->getChild(1);
    if (!payload || payload->getKind() != Node::Kind::TypeList ||
        !payload->hasChildren())
      return n;

    NodePointer e = type->getChild(0);
    if (!e || e->getNumChildren() != 2)
      return n;
    NodePointer module = e->getChild(0);
    NodePointer ident = e->getChild(1);
    if (!module || module->getKind() != Node::Kind::Module ||
        !module->hasText() || module->getText() != swift::STDLIB_NAME ||
        !ident || ident->getKind() != Node::Kind::Identifier ||
        !ident->hasText())
      return n;
    auto kind = llvm::StringSwitch<Node::Kind>(ident->getText())
                    .Case("Array", Node::Kind::SugaredArray)
                    .Case("Dictionary", Node::Kind::SugaredDictionary)
                    .Case("Optional", Node::Kind::SugaredOptional)
                    .Default(Node::Kind::Type);
    if (kind == Node::Kind::Type)
      return n;
    NodePointer sugared = dem.createNode(kind);
    for (NodePointer child : *payload)
      sugared->addChild(child, dem);
    return sugared;
  });
}

llvm::StringRef
TypeSystemSwiftTypeRef::GetBaseName(swift::Demangle::NodePointer node) {
  if (!node)
    return {};
    
  using namespace swift::Demangle;
  switch (node->getKind()) {
  case Node::Kind::Structure:
  case Node::Kind::Class: {
    if (node->getNumChildren() != 2)
      return {};
    NodePointer ident = node->getChild(1);
    if (ident && ident->hasText())
      return ident->getText();
    if (ident->getKind() == Node::Kind::PrivateDeclName ||
        ident->getKind() == Node::Kind::LocalDeclName) {
      if (ident->getNumChildren() != 2)
        return {};
      ident = ident->getChild(1);
      if (ident && ident->hasText())
        return ident->getText();
    }
    return {};
  }
  default:
    // Visit the child nodes.
    for (NodePointer child : *node)
      return GetBaseName(child);
    return {};
  }
}

/// Create a mangled name for a type alias node.
static swift::Demangle::ManglingErrorOr<std::string>
GetTypeAlias(swift::Demangle::Demangler &dem,
             swift::Demangle::NodePointer node) {
  using namespace swift::Demangle;
  auto global = dem.createNode(Node::Kind::Global);
  auto type_mangling = dem.createNode(Node::Kind::TypeMangling);
  global->addChild(type_mangling, dem);
  type_mangling->addChild(node, dem);
  return mangleNode(global);
}

/// Find a Clang type by name in the modules in \p module_holder.
static TypeSP LookupClangType(SwiftASTContext *module_holder, StringRef name) {
  auto lookup = [](Module &M, StringRef name) -> TypeSP {
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
  };
  if (!module_holder)
    return {};
  if (auto *M = module_holder->GetModule())
    return lookup(*M, name);
  TargetSP target_sp = module_holder->GetTarget().lock();
  if (!target_sp)
    return {};
  TypeSP result;
  target_sp->GetImages().ForEach([&](const ModuleSP &module) -> bool {
    result = lookup(const_cast<Module &>(*module), name);
    return !result;
  });
  return result;
}

/// Find a Clang type by name in module \p M.
static CompilerType LookupClangForwardType(SwiftASTContext *module_holder,
                                           StringRef name) {
  if (TypeSP type = LookupClangType(module_holder, name))
    return type->GetForwardCompilerType();
  return {};
}

/// Return a demangle tree for an UnsafePointer<Pointee>.
static swift::Demangle::NodePointer
GetPointerTo(swift::Demangle::Demangler &dem,
             swift::Demangle::NodePointer pointee) {
  using namespace swift::Demangle;
  auto *bgs = dem.createNode(Node::Kind::BoundGenericStructure);
  // Construct the first branch of BoundGenericStructure.
  {
    auto *type = dem.createNode(Node::Kind::Type);
    auto *structure = dem.createNode(Node::Kind::Structure);
    structure->addChild(
        dem.createNodeWithAllocatedText(Node::Kind::Module, swift::STDLIB_NAME),
        dem);
    structure->addChild(dem.createNode(Node::Kind::Identifier, "UnsafePointer"),
                        dem);
    type->addChild(structure, dem);
    bgs->addChild(type, dem);
  }

  // Construct the second branch of BoundGenericStructure.
  {
    auto *typelist = dem.createNode(Node::Kind::TypeList);
    auto *type = dem.createNode(Node::Kind::Type);
    typelist->addChild(type, dem);
    type->addChild(pointee, dem);
    bgs->addChild(typelist, dem);
  }
  return bgs;
}

/// Return a demangle tree leaf node representing \p clang_type.
static swift::Demangle::NodePointer
GetClangTypeNode(CompilerType clang_type, swift::Demangle::Demangler &dem,
                 SwiftASTContext *swift_ast_context) {
  using namespace swift;
  using namespace swift::Demangle;
  Node::Kind kind = Node::Kind::Structure;
  llvm::StringRef swift_name;
  llvm::StringRef module_name = swift::MANGLING_MODULE_OBJC;
  CompilerType pointee;
  if (clang_type.IsPointerType(&pointee))
    clang_type = pointee;
  llvm::StringRef clang_name = clang_type.GetTypeName().GetStringRef();
  // FIXME: Create a higher-level entry point for this by generalizing ClangAdapter.
  struct Adapter {
    struct Context {
      swift::ASTContext *AST;
      llvm::StringRef getSwiftName(swift::KnownFoundationEntity entity) {
        if (AST)
          return AST->getSwiftName(entity);
        return "<error: no Swift context>";
      }
      Context(swift::ASTContext *ctx) : AST(ctx){};
    } SwiftContext;
    Adapter(swift::ASTContext *ctx) : SwiftContext(ctx){};
  } Impl(swift_ast_context ? swift_ast_context->GetASTContext() : nullptr);
#define MAP_TYPE(C_TYPE_NAME, C_TYPE_KIND, C_TYPE_BITWIDTH, SWIFT_MODULE_NAME, \
                 SWIFT_TYPE_NAME, CAN_BE_MISSING, C_NAME_MAPPING)              \
  if (clang_name.equals(C_TYPE_NAME)) {                                        \
    module_name = (SWIFT_MODULE_NAME);                                         \
    swift_name = (SWIFT_TYPE_NAME);                                            \
  } else
#include "swift/../../lib/ClangImporter/MappedTypes.def"
#undef MAP_TYPE
  // The last dangling else in the macro is for this switch.
  switch (clang_type.GetTypeClass()) {
  case eTypeClassClass:
  case eTypeClassObjCObjectPointer:
  case eTypeClassObjCInterface:
    // Special cases for CF-bridged classes. (Better way to do this by
    // inspecting the clang:Type?)
    if (clang_name != "NSNumber" && clang_name != "NSValue")
      kind = Node::Kind::Class;
    // Objective-C objects are first-class entities, not pointers.
    pointee = {};
    break;
  case eTypeClassBuiltin: 
    kind = Node::Kind::Structure;
    // Ask ClangImporter about the builtin type's Swift name.
    if (auto *ts = llvm::cast<TypeSystemClang>(clang_type.GetTypeSystem())) {
      if (clang_type == ts->GetPointerSizedIntType(true))
        swift_name = "Int";
      else if (clang_type == ts->GetPointerSizedIntType(false))
        swift_name = "UInt";
      else
        swift_name =
            swift::importer::getClangTypeNameForOmission(
                ts->getASTContext(), ClangUtil::GetQualType(clang_type))
                .Name;
      module_name = swift::STDLIB_NAME;
    }
    break;
  case eTypeClassArray: {
    // Ideally we would refactor ClangImporter::ImportType to use an
    // abstract TypeBuilder and reuse it here.
    auto *array_type = llvm::dyn_cast<clang::ConstantArrayType>(
        ClangUtil::GetQualType(clang_type).getTypePtr());
    if (!array_type)
      break;
    auto elem_type = array_type->getElementType();
    auto size = array_type->getSize().getZExtValue();
    if (size > 4096)
      break;
    auto *tuple = dem.createNode(Node::Kind::Tuple);
    NodePointer element_type = GetClangTypeNode(
        {clang_type.GetTypeSystem(), elem_type.getAsOpaquePtr()}, dem,
        swift_ast_context);
    for (unsigned i = 0; i < size; ++i) {
      NodePointer tuple_element = dem.createNode(Node::Kind::TupleElement);
      NodePointer type = dem.createNode(Node::Kind::Type);
      type->addChild(element_type, dem);
      tuple_element->addChild(type, dem);
      tuple->addChild(tuple_element, dem);
    }
    return tuple;
  }
  case eTypeClassTypedef:
    kind = Node::Kind::TypeAlias;
    pointee = {};
    break;
  default:
    break;
  }
  NodePointer module =
      dem.createNodeWithAllocatedText(Node::Kind::Module, module_name);
  NodePointer identifier = dem.createNodeWithAllocatedText(
      Node::Kind::Identifier, swift_name.empty()
                                  ? clang_name
                                  : swift_name);
  NodePointer nominal = dem.createNode(kind);
  nominal->addChild(module, dem);
  nominal->addChild(identifier, dem);
  return pointee ? GetPointerTo(dem, nominal) : nominal;
}

/// \return the child of the \p Type node.
static swift::Demangle::NodePointer GetType(swift::Demangle::NodePointer n) {
  using namespace swift::Demangle;
  if (!n || n->getKind() != Node::Kind::Global || !n->hasChildren())
    return nullptr;
  n = n->getFirstChild();
  if (!n || n->getKind() != Node::Kind::TypeMangling || !n->hasChildren())
    return nullptr;
  n = n->getFirstChild();
  if (!n || n->getKind() != Node::Kind::Type || !n->hasChildren())
    return nullptr;
  n = n->getFirstChild();
  return n;
}

/// Demangle a mangled type name and return the child of the \p Type node.
static swift::Demangle::NodePointer
GetDemangledType(swift::Demangle::Demangler &dem, StringRef name) {
  LLDB_SCOPED_TIMER();
  return GetType(dem.demangleSymbol(name));
}

/// Return a pair of modulename, type name for the outermost nominal type.
static llvm::Optional<std::pair<StringRef, StringRef>>
GetNominal(swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node) {
  if (!node)
    return {};
  using namespace swift::Demangle;
  switch (node->getKind()) {
    case Node::Kind::Structure:
    case Node::Kind::Class:
    case Node::Kind::Enum:
    case Node::Kind::Protocol:
    case Node::Kind::ProtocolList:
    case Node::Kind::ProtocolListWithClass:
    case Node::Kind::ProtocolListWithAnyObject:
    case Node::Kind::TypeAlias:
    case Node::Kind::BoundGenericClass:
    case Node::Kind::BoundGenericEnum:
    case Node::Kind::BoundGenericStructure:
    case Node::Kind::BoundGenericProtocol:
    case Node::Kind::BoundGenericOtherNominalType:
    case Node::Kind::BoundGenericTypeAlias: {
    if (node->getNumChildren() != 2)
      return {};
    auto *m = node->getChild(0);
    if (!m || m->getKind() != Node::Kind::Module || !m->hasText())
      return {};
    auto *n = node->getChild(1);
    if (!n || n->getKind() != Node::Kind::Identifier || !n->hasText())
      return {};
    return {{m->getText(), n->getText()}};
  }
  default:
    break;
  }
  return {};
}

/// Resolve a type alias node and return a demangle tree for the
/// resolved type. If the type alias resolves to a Clang type, return
/// a Clang CompilerType.
///
/// \param prefer_clang_types if this is true, type aliases in the
///                           __C module are resolved as Clang types.
///
static std::pair<swift::Demangle::NodePointer, CompilerType>
ResolveTypeAlias(SwiftASTContext *module_holder,
                 swift::Demangle::Demangler &dem,
                 swift::Demangle::NodePointer node,
                 bool prefer_clang_types = false) {
  LLDB_SCOPED_TIMER();
  auto resolve_clang_type = [&]() -> CompilerType {
    auto maybe_module_and_type_names = GetNominal(dem, node);
    if (!maybe_module_and_type_names)
      return {};

    auto module_name = maybe_module_and_type_names->first;
    if (module_name != swift::MANGLING_MODULE_OBJC)
      return {};

    // Resolve the typedef within the Clang debug info.
    auto clang_type =
        LookupClangForwardType(module_holder, node->getChild(1)->getText());
    if (!clang_type)
      return {};

    return clang_type.GetCanonicalType();
  };

  using namespace swift::Demangle;
  // Try to look this up as a Swift type alias. For each *Swift*
  // type alias there is a debug info entry that has the mangled
  // name as name and the aliased type as a type.
  auto mangling = GetTypeAlias(dem, node);
  if (!mangling.isSuccess()) {
    LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
              "Failed while mangling type alias (%d:%u)", mangling.error().code,
              mangling.error().line);
    return {{}, {}};
  }
  ConstString mangled(mangling.result());
  TypeList types;
  if (!prefer_clang_types) {
    llvm::DenseSet<SymbolFile *> searched_symbol_files;
    if (auto *M = module_holder->GetModule())
      M->FindTypes({mangled}, false, 1, searched_symbol_files, types);
    else if (TargetSP target_sp = module_holder->GetTarget().lock())
      target_sp->GetImages().FindTypes(nullptr, {mangled},
                                       false, 1, searched_symbol_files, types);
    else {
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "No module. Couldn't resolve type alias %s",
                mangled.AsCString());
      return {{}, {}};
    }
  }
  if (prefer_clang_types || types.Empty()) {
    // No Swift type found -- this could be a Clang typedef.  This
    // check is not done earlier because a Clang typedef that points
    // to a builtin type, e.g., "typedef unsigned uint32_t", could
    // end up pointing to a *Swift* type!
    auto clang_type = resolve_clang_type();
    if (!clang_type)
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "Couldn't resolve type alias %s as a Swift or clang type.",
                mangled.AsCString());
    return {{}, clang_type};
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
    // The name is not mangled, this might be a Clang typedef, try
    // to look it up as a clang type.
    LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
              "Found non-Swift type alias %s, looking it up as clang type.",
              mangled.AsCString());
    auto clang_type = resolve_clang_type();
    if (!clang_type)
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "Could not find a clang type for %s.", mangled.AsCString());
    return {{}, clang_type};
  }
  NodePointer n = GetDemangledType(dem, desugared_name.GetStringRef());
  if (!n) {
    LLDB_LOG(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
             "Unrecognized demangling %s", desugared_name.AsCString());
    return {{}, {}};
  }
  return {n, {}};
}

llvm::Optional<TypeSystemSwift::TupleElement>
TypeSystemSwiftTypeRef::GetTupleElement(lldb::opaque_compiler_type_t type,
                                        size_t idx) {
  TupleElement result;
  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = TypeSystemSwiftTypeRef::DemangleCanonicalType(dem, type);
  if (!node || node->getKind() != Node::Kind::Tuple)
    return {};
  if (node->getNumChildren() < idx)
    return {};
  NodePointer child = node->getChild(idx);
  if (child->getNumChildren() != 1 &&
      child->getKind() != Node::Kind::TupleElement)
    return {};
  for (NodePointer n : *child) {
    switch (n->getKind()) {
    case Node::Kind::Type:
      result.element_type = RemangleAsType(dem, n);
      break;
    case Node::Kind::TupleElementName:
      result.element_name = ConstString(n->getText());
      break;
    default:
      break;
    }
  }
  if (!result.element_name) {
    std::string name;
    llvm::raw_string_ostream(name) << idx;
    result.element_name = ConstString(name);
  }
  return result;
}

swift::Demangle::NodePointer TypeSystemSwiftTypeRef::Transform(
    swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
    std::function<swift::Demangle::NodePointer(swift::Demangle::NodePointer)>
        fn) {
  if (!node)
    return node;
  using namespace swift::Demangle;
  llvm::SmallVector<NodePointer, 2> children;
  bool changed = false;
  for (NodePointer child : *node) {
    NodePointer transformed = Transform(dem, child, fn);
    changed |= (child != transformed);
    assert(transformed && "callback returned a nullptr");
    if (transformed)
      children.push_back(transformed);
  }
  if (changed) {
    // Create a new node with the transformed children.
    auto kind = node->getKind();
    if (node->hasText())
      node = dem.createNodeWithAllocatedText(kind, node->getText());
    else if (node->hasIndex())
      node = dem.createNode(kind, node->getIndex());
    else
      node = dem.createNode(kind);
    for (NodePointer transformed_child : children)
      node->addChild(transformed_child, dem);
  }
  return fn(node);
}

/// Iteratively resolve all type aliases in \p node by looking up their
/// desugared types in the debug info of module \p M.
static swift::Demangle::NodePointer
GetCanonicalNode(SwiftASTContext *module_holder,
                 swift::Demangle::Demangler &dem,
                 swift::Demangle::NodePointer node) {
  using namespace swift::Demangle;
  return TypeSystemSwiftTypeRef::Transform(dem, node, [&](NodePointer node) {
    NodePointer canonical = nullptr;
    auto kind = node->getKind();
    switch (kind) {
    case Node::Kind::SugaredOptional:
      // FIXME: Factor these three cases out.
      assert(node->getNumChildren() == 1);
      if (node->getNumChildren() != 1)
        return node;

      canonical = dem.createNode(Node::Kind::BoundGenericEnum);
      {
        NodePointer type = dem.createNode(Node::Kind::Type);
        NodePointer e = dem.createNode(Node::Kind::Enum);
        NodePointer module = dem.createNodeWithAllocatedText(
            Node::Kind::Module, swift::STDLIB_NAME);
        e->addChild(module, dem);
        NodePointer optional =
            dem.createNodeWithAllocatedText(Node::Kind::Identifier, "Optional");
        e->addChild(optional, dem);
        type->addChild(e, dem);
        canonical->addChild(type, dem);
      }
      {
        NodePointer typelist = dem.createNode(Node::Kind::TypeList);
        NodePointer type = dem.createNode(Node::Kind::Type);
        type->addChild(node->getFirstChild(), dem);
        typelist->addChild(type, dem);
        canonical->addChild(typelist, dem);
      }
      return canonical;
    case Node::Kind::SugaredArray: {
      assert(node->getNumChildren() == 1);
      if (node->getNumChildren() != 1)
        return node;

      canonical = dem.createNode(Node::Kind::BoundGenericStructure);
      {
        NodePointer type = dem.createNode(Node::Kind::Type);
        NodePointer structure = dem.createNode(Node::Kind::Structure);
        NodePointer module = dem.createNodeWithAllocatedText(
            Node::Kind::Module, swift::STDLIB_NAME);
        structure->addChild(module, dem);
        NodePointer array =
            dem.createNodeWithAllocatedText(Node::Kind::Identifier, "Array");
        structure->addChild(array, dem);
        type->addChild(structure, dem);
        canonical->addChild(type, dem);
      }
      {
        NodePointer typelist = dem.createNode(Node::Kind::TypeList);
        NodePointer type = dem.createNode(Node::Kind::Type);
        type->addChild(node->getFirstChild(), dem);
        typelist->addChild(type, dem);
        canonical->addChild(typelist, dem);
      }
      return canonical;
    }
    case Node::Kind::SugaredDictionary:
      // FIXME: This isnt covered by any test.
      assert(node->getNumChildren() == 2);
      if (node->getNumChildren() != 2)
        return node;

      canonical = dem.createNode(Node::Kind::BoundGenericStructure);
      {
        NodePointer type = dem.createNode(Node::Kind::Type);
        NodePointer structure = dem.createNode(Node::Kind::Structure);
        NodePointer module = dem.createNodeWithAllocatedText(
            Node::Kind::Module, swift::STDLIB_NAME);
        structure->addChild(module, dem);
        NodePointer dict = dem.createNodeWithAllocatedText(
            Node::Kind::Identifier, "Dictionary");
        structure->addChild(dict, dem);
        type->addChild(structure, dem);
        canonical->addChild(type, dem);
      }
      {
        NodePointer typelist = dem.createNode(Node::Kind::TypeList);
        {
          NodePointer type = dem.createNode(Node::Kind::Type);
          type->addChild(node->getChild(0), dem);
          typelist->addChild(type, dem);
        }
        {
          NodePointer type = dem.createNode(Node::Kind::Type);
          type->addChild(node->getChild(1), dem);
          typelist->addChild(type, dem);
        }
        canonical->addChild(typelist, dem);
      }
      return canonical;
    case Node::Kind::SugaredParen:
      assert(node->getNumChildren() == 1);
      if (node->getNumChildren() != 1)
        return node;
      return node->getFirstChild();

    case Node::Kind::BoundGenericTypeAlias:
    case Node::Kind::TypeAlias: {
      auto node_clangtype = ResolveTypeAlias(module_holder, dem, node);
      if (CompilerType clang_type = node_clangtype.second)
        return GetClangTypeNode(clang_type, dem, module_holder);
      if (node_clangtype.first)
        return node_clangtype.first;
      return node;
    }
    default:
      break;
    }
    return node;
  });
}

/// Return the demangle tree representation of this type's canonical
/// (type aliases resolved) type.
swift::Demangle::NodePointer TypeSystemSwiftTypeRef::GetCanonicalDemangleTree(
    SwiftASTContext *module_holder, swift::Demangle::Demangler &dem,
    StringRef mangled_name) {
  LLDB_SCOPED_TIMER();
  auto *node = dem.demangleSymbol(mangled_name);
  return GetCanonicalNode(module_holder, dem, node);
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
static swift::Demangle::NodePointer
Desugar(swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
        swift::Demangle::Node::Kind bound_kind,
        swift::Demangle::Node::Kind kind, llvm::StringRef name) {
  LLDB_SCOPED_TIMER();

  using namespace swift::Demangle;
  NodePointer desugared = dem.createNode(bound_kind);
  NodePointer type = dem.createNode(Node::Kind::Type);
  {
    NodePointer concrete = dem.createNode(kind);
    NodePointer swift =
        dem.createNodeWithAllocatedText(Node::Kind::Module, swift::STDLIB_NAME);
    concrete->addChild(swift, dem);
    NodePointer ident =
        dem.createNodeWithAllocatedText(Node::Kind::Identifier, name);
    concrete->addChild(ident, dem);
    type->addChild(concrete, dem);
  }
  NodePointer type_list = dem.createNode(Node::Kind::TypeList);
  {
    NodePointer type = dem.createNode(Node::Kind::Type);
    type->addChild(node->getFirstChild(), dem);
    type_list->addChild(type, dem);
  }
  desugared->addChild(type, dem);
  desugared->addChild(type_list, dem);
  return desugared;
}

/// Helper for \p GetSwiftName.
template <typename ContextInfo>
std::string ExtractSwiftName(
    clang::api_notes::APINotesReader::VersionedInfo<ContextInfo> info) {
  if (auto version = info.getSelected()) {
    ContextInfo context_info = info[*version].second;
    if (!context_info.SwiftName.empty())
      return context_info.SwiftName;
  }
  return {};
};

std::string
TypeSystemSwiftTypeRef::GetSwiftName(const clang::Decl *clang_decl,
                                     TypeSystemClang &clang_typesystem) {
  LLDB_SCOPED_TIMER();
  auto *named_decl = llvm::dyn_cast_or_null<const clang::NamedDecl>(clang_decl);
  if (!named_decl)
    return {};
  StringRef default_name = named_decl->getName();
  unsigned id = clang_decl->getOwningModuleID();
  if (!id)
    return default_name.str();
  auto *ast_source = llvm::dyn_cast_or_null<ClangExternalASTSourceCallbacks>(
      clang_typesystem.getASTContext().getExternalSource());
  auto *apinotes_manager = GetAPINotesManager(ast_source, id);
  if (!apinotes_manager)
    return default_name.str();

  // Read the Swift name from the API notes.
  for (auto reader : apinotes_manager->getCurrentModuleReaders()) {
    std::string swift_name;
    // The order is significant since some of these decl kinds are also TagDecls.
    if (llvm::isa<clang::TypedefNameDecl>(clang_decl))
      swift_name = ExtractSwiftName(reader->lookupTypedef(default_name));
    else if (llvm::isa<clang::EnumConstantDecl>(clang_decl))
      swift_name = ExtractSwiftName(reader->lookupEnumConstant(default_name));
    else if (llvm::isa<clang::ObjCInterfaceDecl>(clang_decl))
      swift_name = ExtractSwiftName(reader->lookupObjCClassInfo(default_name));
    else if (llvm::isa<clang::ObjCProtocolDecl>(clang_decl))
      swift_name =
          ExtractSwiftName(reader->lookupObjCProtocolInfo(default_name));
    else if (llvm::isa<clang::TagDecl>(clang_decl))
      swift_name = ExtractSwiftName(reader->lookupTag(default_name));
    else {
      assert(false && "unhandled clang decl kind");
    }
    if (!swift_name.empty())
      return swift_name;
  }
  // Else we must go through ClangImporter to apply the automatic
  // swiftification rules.
  //
  // TODO: Separate ClangImporter::ImportDecl into a freestanding
  // class, so we don't need a SwiftASTContext for this!
  return m_swift_ast_context->ImportName(named_decl);
}

/// Determine whether \p node is an Objective-C type and return its name.
static StringRef GetObjCTypeName(swift::Demangle::NodePointer node) {
  if (node && node->getNumChildren() == 2 && node->getChild(0)->hasText() &&
      node->getChild(0)->getText() == swift::MANGLING_MODULE_OBJC &&
      node->getChild(1)->hasText())
    return node->getChild(1)->getText();
  return {};
}

swift::Demangle::NodePointer
TypeSystemSwiftTypeRef::GetSwiftified(swift::Demangle::Demangler &dem,
                                      swift::Demangle::NodePointer node,
                                      bool resolve_objc_module) {
  LLDB_SCOPED_TIMER();

  using namespace swift::Demangle;
  StringRef ident = GetObjCTypeName(node);
  if (ident.empty())
    return node;

  // This is an imported Objective-C type; look it up in the
  // debug info.
  TypeSP clang_type = LookupClangType(m_swift_ast_context, ident);
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
      return node;
  } else {
    toplevel_module = swift::MANGLING_MODULE_OBJC;
  }

  // Create a new node with the Clang module instead of "__C".
  NodePointer renamed = dem.createNode(node->getKind());
  NodePointer module = dem.createNode(Node::Kind::Module, toplevel_module);
  renamed->addChild(module, dem);

  // This order is significant, because of `typedef tag`.
  swift::ClangTypeKind kinds[] = {swift::ClangTypeKind::Typedef,
                                  swift::ClangTypeKind::Tag,
                                  swift::ClangTypeKind::ObjCProtocol};
  clang::NamedDecl *clang_decl = nullptr;
  CompilerType compiler_type = clang_type->GetForwardCompilerType();
  auto *clang_ts =
      llvm::dyn_cast_or_null<TypeSystemClang>(compiler_type.GetTypeSystem());
  if (!clang_ts)
    return node;
  clang::QualType qual_type = ClangUtil::GetQualType(compiler_type);
  for (auto kind : kinds) {
    clang_decl = llvm::dyn_cast_or_null<clang::NamedDecl>(
        GetDeclForTypeAndKind(qual_type, kind));
    if (clang_decl)
      break;
  }
  if (!clang_decl)
    return node;

  std::string swift_name = GetSwiftName(clang_decl, *clang_ts);
  NodePointer identifier = dem.createNode(Node::Kind::Identifier, swift_name);
  renamed->addChild(identifier, dem);
  return renamed;
}

swift::Demangle::NodePointer TypeSystemSwiftTypeRef::GetNodeForPrintingImpl(
    swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
    bool resolve_objc_module, bool desugar) {
  using namespace swift::Demangle;
  return Transform(dem, node, [&](NodePointer node) {
    NodePointer canonical = node;
    auto kind = node->getKind();
    switch (kind) {
    case Node::Kind::Class:
    case Node::Kind::Structure:
    case Node::Kind::TypeAlias:
      return GetSwiftified(dem, node, resolve_objc_module);

      //
      // The remaining cases are all about bug-for-bug compatibility
      // with the type dumper and we don't need to carry them forward
      // necessarily.
      //

      // The type dumper doesn't print these.
#define REF_STORAGE(Name, ...)                                                 \
  case Node::Kind::Name:                                                       \
    return (node->getNumChildren() == 1) ? node->getChild(0) : node;
#include "swift/AST/ReferenceStorage.def"

    case Node::Kind::ImplFunctionType: {
      // Rewrite ImplFunctionType nodes as FunctionType nodes.
      NodePointer fnty = dem.createNode(Node::Kind::FunctionType);
      NodePointer args = dem.createNode(Node::Kind::ArgumentTuple);
      NodePointer rett = dem.createNode(Node::Kind::ReturnType);
      NodePointer args_ty = dem.createNode(Node::Kind::Type);
      NodePointer args_tuple = dem.createNode(Node::Kind::Tuple);
      for (NodePointer child : *node) {
        if (child->getKind() == Node::Kind::ImplParameter) {
          for (NodePointer type : *node)
            if (type->getKind() == Node::Kind::Type &&
                type->getNumChildren() == 1)
              rett->addChild(type->getChild(0), dem);
        } else if (child->getKind() == Node::Kind::ImplResult) {
          for (NodePointer type : *node)
            if (type->getKind() == Node::Kind::Type)
              rett->addChild(type, dem);
        }
      }
      args_ty->addChild(args_tuple, dem);
      args->addChild(args_ty, dem);
      fnty->addChild(args, dem);
      if (rett->getNumChildren() != 1)
        rett->addChild(dem.createNode(Node::Kind::Tuple), dem);
      fnty->addChild(rett, dem);
      return fnty;
    }

    case Node::Kind::SugaredOptional:
      // This is particularly silly. The outermost sugared Optional is
      // desugared. See SwiftASTContext::GetTypeName() and remove it there, too!
      if (desugar && node->getNumChildren() == 1) {
        desugar = false;
        return Desugar(dem, node, Node::Kind::BoundGenericEnum,
                       Node::Kind::Enum, "Optional");
      }
      return node;
    case Node::Kind::SugaredArray:
      // See comment on SugaredOptional.
      if (desugar && node->getNumChildren() == 1) {
        desugar = false;
        return Desugar(dem, node, Node::Kind::BoundGenericStructure,
                       Node::Kind::Structure, "Array");
      }
      return node;
    case Node::Kind::SugaredDictionary:
      // See comment on SugaredOptional.
      if (desugar && node->getNumChildren() == 1) {
        desugar = false;
        return Desugar(dem, node, Node::Kind::BoundGenericStructure,
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

            canonical->addChild(module, dem);
            canonical->addChild(identifier, dem);
            return canonical;
          }
    }
    return canonical;
  });
}

/// Return the demangle tree representation with all "__C" module
/// names with their actual Clang module names.
swift::Demangle::NodePointer TypeSystemSwiftTypeRef::GetDemangleTreeForPrinting(
    swift::Demangle::Demangler &dem, const char *mangled_name,
     bool resolve_objc_module) {
  LLDB_SCOPED_TIMER();

  auto *node = dem.demangleSymbol(mangled_name);
  return GetNodeForPrintingImpl(dem, node, resolve_objc_module);
}

/// Determine wether this demangle tree contains an unresolved type alias.
static bool ContainsGenericTypeParameter(swift::Demangle::NodePointer node) {
  if (!node)
    return false;

  if (node->getKind() == swift::Demangle::Node::Kind::DependentGenericParamType)
    return true;

  for (swift::Demangle::NodePointer child : *node)
    if (ContainsGenericTypeParameter(child))
      return true;

  return false;
}

/// Collect TypeInfo flags from a demangle tree. For most attributes
/// this can stop scanning at the outmost type, however in order to
/// determine whether a node is generic or not, it needs to visit all
/// nodes. The \p generic_walk argument specifies that the primary
/// attributes have been collected and that we only look for generics.
static uint32_t collectTypeInfo(SwiftASTContext *module_holder,
                                swift::Demangle::Demangler &dem,
                                swift::Demangle::NodePointer node,
                                bool generic_walk = false) {
  LLDB_SCOPED_TIMER();
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
    if ((type_class & eTypeClassBuiltin)) {
      swift_flags &= ~eTypeIsStructUnion;
      swift_flags |= collectTypeInfo(
          module_holder, dem, GetClangTypeNode(clang_type, dem, module_holder));
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
      swift_flags |= eTypeIsGeneric | eTypeIsBound | eTypeHasChildren | eTypeIsStructUnion;
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
    case Node::Kind::BoundGenericFunction:
      swift_flags |= eTypeIsGeneric | eTypeIsBound;
      LLVM_FALLTHROUGH;
    case Node::Kind::NoEscapeFunctionType:
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
        else if (node->getText() == swift::BUILTIN_TYPE_NAME_BRIDGEOBJECT ||
                 node->getText() == swift::BUILTIN_TYPE_NAME_UNKNOWNOBJECT)
          swift_flags |=
              eTypeHasChildren | eTypeIsPointer | eTypeIsScalar | eTypeIsObjC;
        else if (node->getText().startswith(swift::BUILTIN_TYPE_NAME_FLOAT) ||
                 node->getText().startswith(swift::BUILTIN_TYPE_NAME_FLOAT_PPC))
          swift_flags |= eTypeIsFloat | eTypeIsScalar;
        else if (node->getText().startswith(swift::BUILTIN_TYPE_NAME_VEC))
          swift_flags |= eTypeHasChildren | eTypeIsVector;
        else if (node->getText().startswith(swift::BUILTIN_TYPE_NAME_INT) ||
                 node->getText().startswith(swift::BUILTIN_TYPE_NAME_WORD))
          swift_flags |= eTypeIsInteger | eTypeIsScalar;
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
      if (!ContainsGenericTypeParameter(node->getChild(1)))
        swift_flags |= eTypeHasValue | eTypeHasChildren;
      auto module = node->getChild(0);
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

        // Look up the Clang type in DWARF.
        CompilerType clang_type =
            LookupClangForwardType(module_holder, ident->getText());
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
      auto node_clangtype =
          ResolveTypeAlias(module_holder, dem, node);
      if (CompilerType clang_type = node_clangtype.second) {
        collect_clang_type(clang_type);
        return swift_flags;
      }
      swift_flags |= collectTypeInfo(module_holder, dem, node_clangtype.first,
                                     generic_walk);
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
    swift_flags |= collectTypeInfo(module_holder, dem, node->getChild(i), generic_walk);

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

CompilerType
TypeSystemSwiftTypeRef::GetGenericArgumentType(opaque_compiler_type_t type,
                                               size_t idx) {
  return m_swift_ast_context->GetGenericArgumentType(ReconstructType(type),
                                                     idx);
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

TypeSP TypeSystemSwiftTypeRef::LookupTypeInModule(
    lldb::opaque_compiler_type_t opaque_type) {
  auto *M = GetModule();
  if (!M)
    return {};
  swift::Demangle::Demangler dem;
  auto *node = GetDemangledType(dem, AsMangledName(opaque_type));
  auto module_type = GetNominal(dem, node);
  if (!module_type)
    return {};
  // DW_AT_linkage_name is not part of the accelerator table, so
  // we need to search by module+name.
  ConstString module(module_type->first);
  ConstString type(module_type->second);
  llvm::SmallVector<CompilerContext, 2> decl_context;
  decl_context.push_back({CompilerContextKind::Module, module});
  decl_context.push_back({CompilerContextKind::AnyType, type});
  llvm::DenseSet<SymbolFile *> searched_symbol_files;
  TypeMap types;
  M->FindTypes(decl_context, TypeSystemSwift::GetSupportedLanguagesForTypes(),
               searched_symbol_files, types);
  return types.Empty() ? TypeSP() : types.GetTypeAtIndex(0);
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
  auto mangling = mangleNode(node);
  if (!mangling.isSuccess())
    return false;
  std::string remangled = mangling.result();
  return remangled == std::string(str);
}

#include <regex>
namespace {
template <typename T> bool Equivalent(T l, T r) {
  if (l != r)
    llvm::dbgs() <<  l << " != " << r << "\n";
  return l == r;
}

/// Specialization for GetTypeInfo().
template <> bool Equivalent<uint32_t>(uint32_t l, uint32_t r) {
  if (l != r) {
    // Failure. Dump it for easier debugging.
    llvm::dbgs() << "TypeSystemSwiftTypeRef diverges from SwiftASTContext:\n";
#define HANDLE_ENUM_CASE(VAL, CASE) \
    if (VAL & CASE) llvm::dbgs() << " | " << #CASE

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

/// Determine wether this demangle tree contains a sugar () node.
static bool ContainsSugaredParen(swift::Demangle::NodePointer node) {
  if (!node)
    return false;

  if (node->getKind() == swift::Demangle::Node::Kind::SugaredParen)
    return true;

  for (swift::Demangle::NodePointer child : *node)
    if (ContainsSugaredParen(child))
      return true;

  return false;
}

swift::Demangle::NodePointer
StripPrivateIDs(swift::Demangle::Demangler &dem,
                swift::Demangle::NodePointer node) {
  using namespace swift::Demangle;
  return TypeSystemSwiftTypeRef::Transform(dem, node, [&](NodePointer node) {
    if (node->getKind() != Node::Kind::PrivateDeclName ||
        node->getNumChildren() != 2)
      return node;

    assert(node->getFirstChild()->getKind() == Node::Kind::Identifier);
    assert(node->getLastChild()->getKind() == Node::Kind::Identifier);
    auto *new_node = dem.createNode(Node::Kind::PrivateDeclName);
    auto *ident = dem.createNodeWithAllocatedText(
        Node::Kind::Identifier, node->getLastChild()->getText());
    new_node->addChild(ident, dem);
    return new_node;
  });
}

/// Compare two swift types from different type systems by comparing their
/// (canonicalized) mangled name.
template <> bool Equivalent<CompilerType>(CompilerType l, CompilerType r) {
  if (!l || !r)
    return !l && !r;
  // See comments in SwiftASTContext::ReconstructType(). For
  // SILFunctionTypes the mapping isn't bijective.
  auto *ast_ctx = llvm::cast<SwiftASTContext>(r.GetTypeSystem());
  if (((void *)ast_ctx->ReconstructType(l.GetMangledTypeName())) ==
      r.GetOpaqueQualType())
    return true;
  ConstString lhs = l.GetMangledTypeName();
  ConstString rhs = r.GetMangledTypeName();
  if (lhs == ConstString("$sSiD") && rhs == ConstString("$sSuD"))
    return true;
  // Ignore missing sugar.
  swift::Demangle::Demangler dem;
  auto l_node = GetDemangledType(dem, lhs.GetStringRef());
  auto r_node = GetDemangledType(dem, rhs.GetStringRef());
  if (ContainsUnresolvedTypeAlias(r_node) ||
      ContainsGenericTypeParameter(r_node) || ContainsSugaredParen(r_node))
    return true;
  auto l_mangling = swift::Demangle::mangleNode(StripPrivateIDs(
      dem, TypeSystemSwiftTypeRef::CanonicalizeSugar(dem, l_node)));
  auto r_mangling = swift::Demangle::mangleNode(StripPrivateIDs(
      dem, TypeSystemSwiftTypeRef::CanonicalizeSugar(dem, r_node)));
  if (!l_mangling.isSuccess() || !r_mangling.isSuccess())
    return false;

  if (l_mangling.result() == r_mangling.result())
    return true;

  // SwiftASTContext hardcodes some less-precise types.
  if (rhs.GetStringRef().equals("$sBpD"))
    return true;

  // If the type is a Clang-imported type ignore mismatches. Since we
  // don't have any visibility into Swift overlays of SDK modules we
  // can only present the underlying Clang type. However, we can
  // still swiftify that type later for printing.
  if (auto *ts =
          llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(l.GetTypeSystem()))
    if (ts->IsImportedType(l.GetOpaqueQualType(), nullptr))
      return true;
  if (lhs == rhs)
    return true;

  // Failure. Dump it for easier debugging.
  llvm::dbgs() << "TypeSystemSwiftTypeRef diverges from SwiftASTContext: "
               << lhs.GetStringRef() << " != " << rhs.GetStringRef() << "\n";
  return false;
}
/// This one is particularly taylored for GetTypeName() and
/// GetDisplayTypeName().
///
/// String divergences are mostly cosmetic in nature and usually
/// TypeSystemSwiftTypeRef is returning more accurate results. They only really
/// matter for GetTypeName() and there only if there is a data formatter
/// matching that name.
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
    l_prime = std::regex_replace(
        l.GetStringRef().str(), std::regex("(CoreGraphics|Foundation|)\\."), "");
    if (llvm::StringRef(l_prime) == r.GetStringRef())
      return true;

#ifdef STRICT_VALIDATION
    // Failure. Dump it for easier debugging.
    llvm::dbgs() << "TypeSystemSwiftTypeRef diverges from SwiftASTContext: "
                 << l.GetStringRef() << " != " << r.GetStringRef() << "\n";
#else
    return true;
#endif
  }
  return l == r;
}

/// Version tailored to GetBitSize & friends.
template <typename T>
bool Equivalent(llvm::Optional<T> l, llvm::Optional<T> r) {
  if (l == r)
    return true;
  // There are situations where SwiftASTContext incorrectly returns
  // all Clang-imported members of structs as having a size of 0, we
  // thus assume that a larger number is "better".
  if (l.hasValue() && r.hasValue() && *l > *r)
    return true;
  // Assume that any value is "better" than none.
  if (l.hasValue() && !r.hasValue())
    return true;
  llvm::dbgs() << l << " != " << r << "\n";
  return false;
}

// Introduced for `GetNumChildren`.
template <typename T>
bool Equivalent(llvm::Optional<T> l, T r) {
  return Equivalent(l, llvm::Optional<T>(r));
}

} // namespace
#endif

// This can be removed once the transition is complete.
#define FALLBACK(REFERENCE, ARGS)                                              \
  do {                                                                         \
    if (!ModuleList::GetGlobalModuleListProperties()                           \
             .GetUseSwiftTypeRefTypeSystem())                                  \
      return m_swift_ast_context->REFERENCE ARGS;                              \
  } while (0)

#ifndef NDEBUG
#define VALIDATE_AND_RETURN_STATIC(IMPL, REFERENCE)                            \
  do {                                                                         \
    FALLBACK(REFERENCE, ());                                                   \
    auto result = IMPL();                                                      \
    if (!ModuleList::GetGlobalModuleListProperties()                           \
             .GetSwiftValidateTypeSystem())                                    \
      return result;                                                           \
    if (!m_swift_ast_context)                                                  \
      return result;                                                           \
    assert((result == m_swift_ast_context->REFERENCE()) &&                     \
           "TypeSystemSwiftTypeRef diverges from SwiftASTContext");            \
    return result;                                                             \
  } while (0)

#define VALIDATE_AND_RETURN(IMPL, REFERENCE, TYPE, ARGS, FALLBACK_ARGS)        \
  do {                                                                         \
    FALLBACK(REFERENCE, FALLBACK_ARGS);                                        \
    auto result = IMPL();                                                      \
    if (!ModuleList::GetGlobalModuleListProperties()                           \
             .GetSwiftValidateTypeSystem())                                    \
      return result;                                                           \
    if (!m_swift_ast_context)                                                  \
      return result;                                                           \
    if ((TYPE) && !ReconstructType(TYPE))                                      \
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
#define VALIDATE_AND_RETURN_STATIC(IMPL, REFERENCE)                            \
  FALLBACK(REFERENCE, ());                                                     \
  return IMPL()
#define VALIDATE_AND_RETURN(IMPL, REFERENCE, TYPE, ARGS, FALLBACK_ARGS)        \
  FALLBACK(REFERENCE, FALLBACK_ARGS);                                          \
  return IMPL();
#endif

CompilerType
TypeSystemSwiftTypeRef::RemangleAsType(swift::Demangle::Demangler &dem,
                                       swift::Demangle::NodePointer node) {
  if (!node)
    return {};

  using namespace swift::Demangle;
  assert(node->getKind() == Node::Kind::Type && "expected type node");
  auto global = dem.createNode(Node::Kind::Global);
  auto type_mangling = dem.createNode(Node::Kind::TypeMangling);
  type_mangling->addChild(node, dem);
  global->addChild(type_mangling, dem);
  auto mangling = mangleNode(global);
  if (!mangling.isSuccess())
    return {};
  ConstString mangled_element(mangling.result());
  return GetTypeFromMangledTypename(mangled_element);
}

swift::Demangle::NodePointer TypeSystemSwiftTypeRef::DemangleCanonicalType(
    swift::Demangle::Demangler &dem, opaque_compiler_type_t opaque_type) {
  using namespace swift::Demangle;
  NodePointer node = GetCanonicalDemangleTree(m_swift_ast_context, dem,
                                              AsMangledName(opaque_type));
  return GetType(node);
}

bool TypeSystemSwiftTypeRef::IsArrayType(opaque_compiler_type_t type,
                                         CompilerType *element_type,
                                         uint64_t *size, bool *is_incomplete) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);
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
         node->getChild(1)->getText() != "ContiguousArray" &&
         node->getChild(1)->getText() != "ArraySlice"))
      return false;

    if (elem_node->getNumChildren() != 1 ||
        elem_node->getKind() != Node::Kind::TypeList)
      return false;
    elem_node = elem_node->getFirstChild();
    if (element_type)
      *element_type = RemangleAsType(dem, elem_node);

    if (is_incomplete)
      *is_incomplete = true;
    if (size)
      *size = 0;

    return true;
  };
  VALIDATE_AND_RETURN(
      impl, IsArrayType, type,
      (ReconstructType(type), nullptr, nullptr, nullptr),
      (ReconstructType(type), element_type, size, is_incomplete));
}

bool TypeSystemSwiftTypeRef::IsAggregateType(opaque_compiler_type_t type) {
  auto impl = [&]() -> bool {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);
 
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
  VALIDATE_AND_RETURN(impl, IsAggregateType, type, (ReconstructType(type)),
                      (ReconstructType(type)));
}

bool TypeSystemSwiftTypeRef::IsDefined(opaque_compiler_type_t type) {
  auto impl = [&]() -> bool {
    return type;
  };
  VALIDATE_AND_RETURN(impl, IsDefined, type, (ReconstructType(type)),
                      (ReconstructType(type)));
}

bool TypeSystemSwiftTypeRef::IsFunctionType(opaque_compiler_type_t type) {
  auto impl = [&]() -> bool {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);
    // Note: There are a number of other candidates, and this list may need
    // updating. Ex: `NoEscapeFunctionType`, `ThinFunctionType`, etc.
    return node && (node->getKind() == Node::Kind::FunctionType ||
                    node->getKind() == Node::Kind::NoEscapeFunctionType ||
                    node->getKind() == Node::Kind::ImplFunctionType);
  };
  VALIDATE_AND_RETURN(impl, IsFunctionType, type, (ReconstructType(type)),
                      (ReconstructType(type)));
}
size_t TypeSystemSwiftTypeRef::GetNumberOfFunctionArguments(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> size_t {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);
    if (!node || (node->getKind() != Node::Kind::FunctionType &&
                  node->getKind() != Node::Kind::NoEscapeFunctionType &&
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
                      (ReconstructType(type)), (ReconstructType(type)));
}
CompilerType
TypeSystemSwiftTypeRef::GetFunctionArgumentAtIndex(opaque_compiler_type_t type,
                                                   const size_t index) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);
    if (!node || (node->getKind() != Node::Kind::FunctionType &&
                  node->getKind() != Node::Kind::NoEscapeFunctionType &&
                  node->getKind() != Node::Kind::ImplFunctionType))
      return {};
    unsigned num_args = 0;
    for (NodePointer child : *node) {
      if (child->getKind() == Node::Kind::ImplParameter) {
        if (num_args == index)
          for (NodePointer type : *child)
            if (type->getKind() == Node::Kind::Type)
              return RemangleAsType(dem, type);
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
                return RemangleAsType(dem, type);
              ++num_args;
            }
          }
      }
    }
    return {};
  };
  VALIDATE_AND_RETURN(impl, GetFunctionArgumentAtIndex, type,
                      (ReconstructType(type), index),
                      (ReconstructType(type), index));
}
bool TypeSystemSwiftTypeRef::IsFunctionPointerType(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> bool { return IsFunctionType(type); };
  VALIDATE_AND_RETURN(impl, IsFunctionPointerType, type,
                      (ReconstructType(type)), (ReconstructType(type)));
}

bool TypeSystemSwiftTypeRef::IsPossibleDynamicType(opaque_compiler_type_t type,
                                                   CompilerType *target_type,
                                                   bool check_cplusplus,
                                                   bool check_objc) {
  LLDB_SCOPED_TIMER();

  if (target_type)
    target_type->Clear();

  if (!type)
    return false;

  // This is a discrepancy with `SwiftASTContext`. The `impl` below correctly
  // returns true, but `VALIDATE_AND_RETURN` will assert. This hardcoded
  // handling of `__C.NSNotificationName` can be removed when the
  // `VALIDATE_AND_RETURN` is removed.
  if (GetMangledTypeName(type) == "$sSo18NSNotificationNameaD")
    return true;

  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    auto *node = DemangleCanonicalType(dem, type);
    if (!node)
      return false;

    if (node->getKind() == Node::Kind::TypeAlias) {
      auto resolved = ResolveTypeAlias(m_swift_ast_context, dem, node);
      if (auto *n = std::get<swift::Demangle::NodePointer>(resolved))
        node = n;
    }

    switch (node->getKind()) {
    case Node::Kind::Class:
    case Node::Kind::BoundGenericClass:
    case Node::Kind::Protocol:
    case Node::Kind::ProtocolList:
    case Node::Kind::ProtocolListWithClass:
    case Node::Kind::ProtocolListWithAnyObject:
    case Node::Kind::ExistentialMetatype:
    case Node::Kind::DynamicSelf:
      return true;
    case Node::Kind::BuiltinTypeName: {
      if (!node->hasText())
        return false;
      StringRef name = node->getText();
      return name == swift::BUILTIN_TYPE_NAME_RAWPOINTER ||
             name == swift::BUILTIN_TYPE_NAME_NATIVEOBJECT ||
             name == swift::BUILTIN_TYPE_NAME_BRIDGEOBJECT;
    }
    default:
      return ContainsGenericTypeParameter(node);
    }
  };
  VALIDATE_AND_RETURN(
      impl, IsPossibleDynamicType, type,
      (ReconstructType(type), nullptr, check_cplusplus, check_objc),
      (ReconstructType(type), target_type, check_cplusplus, check_objc));
}

bool TypeSystemSwiftTypeRef::IsPointerType(opaque_compiler_type_t type,
                                           CompilerType *pointee_type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);
    if (!node || node->getKind() != Node::Kind::BuiltinTypeName ||
        !node->hasText())
      return false;
    return ((node->getText() == swift::BUILTIN_TYPE_NAME_RAWPOINTER) ||
            (node->getText() == swift::BUILTIN_TYPE_NAME_UNSAFEVALUEBUFFER) ||
            (node->getText() == swift::BUILTIN_TYPE_NAME_NATIVEOBJECT) ||
            (node->getText() == swift::BUILTIN_TYPE_NAME_BRIDGEOBJECT));
  };
  VALIDATE_AND_RETURN(impl, IsPointerType, type,
                      (ReconstructType(type), nullptr),
                      (ReconstructType(type), pointee_type));
}
bool TypeSystemSwiftTypeRef::IsVoidType(opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);
    return node && node->getNumChildren() == 0 &&
           node->getKind() == Node::Kind::Tuple;
  };
  VALIDATE_AND_RETURN(impl, IsVoidType, type, (ReconstructType(type)),
                      (ReconstructType(type)));
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
    Demangler dem;
    NodePointer print_node =
        GetDemangleTreeForPrinting(dem, AsMangledName(type), true);
    auto mangling = mangleNode(print_node);
    std::string remangled;
    if (mangling.isSuccess())
      remangled = mangling.result();
    else {
      std::ostringstream buf;
      buf << "<mangling error " << mangling.error().code << ":"
          << mangling.error().line << ">";
      remangled = buf.str();
    }
    return ConstString(SwiftLanguageRuntime::DemangleSymbolAsString(
        remangled, SwiftLanguageRuntime::eTypeName));
  };
  VALIDATE_AND_RETURN(impl, GetTypeName, type, (ReconstructType(type)),
                      (ReconstructType(type)));
}
ConstString
TypeSystemSwiftTypeRef::GetDisplayTypeName(opaque_compiler_type_t type,
                                           const SymbolContext *sc) {
  LLDB_SCOPED_TIMER();
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer print_node =
        GetDemangleTreeForPrinting(dem, AsMangledName(type), false);
    auto mangling = mangleNode(print_node);
    std::string remangled;
    if (mangling.isSuccess())
      remangled = mangling.result();
    else {
      std::ostringstream buf;
      buf << "<mangling error " << mangling.error().code << ":"
          << mangling.error().line << ">";
      remangled = buf.str();
    }
    return ConstString(SwiftLanguageRuntime::DemangleSymbolAsString(
        remangled, SwiftLanguageRuntime::eDisplayTypeName, sc));
  };
  VALIDATE_AND_RETURN(impl, GetDisplayTypeName, type,
                      (ReconstructType(type), sc),
                      (ReconstructType(type), sc));
}
uint32_t TypeSystemSwiftTypeRef::GetTypeInfo(
    opaque_compiler_type_t type, CompilerType *pointee_or_element_clang_type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = dem.demangleSymbol(AsMangledName(type));
    return collectTypeInfo(m_swift_ast_context, dem, node);
  };
#ifndef NDEBUG
  // This type has special behavior hardcoded in the Swift frontend
  // that we can't reproduce here.
  if (StringRef(AsMangledName(type)).equals("$sSo18NSNotificationNameaD"))
    return impl();
#endif
  VALIDATE_AND_RETURN(impl, GetTypeInfo, type, (ReconstructType(type), nullptr),
                      (ReconstructType(type), pointee_or_element_clang_type));
}
lldb::TypeClass
TypeSystemSwiftTypeRef::GetTypeClass(opaque_compiler_type_t type) {
  auto impl = [&]() {
    uint32_t flags = GetTypeInfo(type, nullptr);
    // The ordering is significant since GetTypeInfo() returns many flags.
    if ((flags & eTypeIsGenericTypeParam))
      return eTypeClassOther;
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
  VALIDATE_AND_RETURN(impl, GetTypeClass, type, (ReconstructType(type)),
                      (ReconstructType(type)));
}

// Creating related types
CompilerType
TypeSystemSwiftTypeRef::GetArrayElementType(opaque_compiler_type_t type,
                                            ExecutionContextScope *exe_scope) {
  auto impl = [&]() {
    CompilerType element_type;
    IsArrayType(type, &element_type, nullptr, nullptr);
    return element_type;
  };
  VALIDATE_AND_RETURN(impl, GetArrayElementType, type,
                      (ReconstructType(type), exe_scope),
                      (ReconstructType(type), exe_scope));
}

CompilerType
TypeSystemSwiftTypeRef::GetCanonicalType(opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer canonical =
        GetCanonicalDemangleTree(m_swift_ast_context, dem, AsMangledName(type));
    if (ContainsUnresolvedTypeAlias(canonical)) {
      // If this is a typealias defined in the expression evaluator,
      // then we don't have debug info to resolve it from.
      CompilerType ast_type = ReconstructType({this, type}).GetCanonicalType();
      return GetTypeFromMangledTypename(ast_type.GetMangledTypeName());
    }
    auto mangling = mangleNode(canonical);
    if (!mangling.isSuccess())
      return CompilerType();
    ConstString mangled(mangling.result());
    return GetTypeFromMangledTypename(mangled);
  };
  VALIDATE_AND_RETURN(impl, GetCanonicalType, type, (ReconstructType(type)),
                      (ReconstructType(type)));
}
int TypeSystemSwiftTypeRef::GetFunctionArgumentCount(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> int { return GetNumberOfFunctionArguments(type); };
  VALIDATE_AND_RETURN(impl, GetFunctionArgumentCount, type,
                      (ReconstructType(type)), (ReconstructType(type)));
}
CompilerType TypeSystemSwiftTypeRef::GetFunctionArgumentTypeAtIndex(
    opaque_compiler_type_t type, size_t idx) {
  auto impl = [&] { return GetFunctionArgumentAtIndex(type, idx); };
  VALIDATE_AND_RETURN(impl, GetFunctionArgumentTypeAtIndex, type,
                      (ReconstructType(type), idx),
                      (ReconstructType(type), idx));
}
CompilerType
TypeSystemSwiftTypeRef::GetFunctionReturnType(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);
    if (!node || (node->getKind() != Node::Kind::FunctionType &&
                  node->getKind() != Node::Kind::NoEscapeFunctionType &&
                  node->getKind() != Node::Kind::ImplFunctionType))
      return {};
    for (NodePointer child : *node) {
      if (child->getKind() == Node::Kind::ImplResult) {
        for (NodePointer type : *child)
          if (type->getKind() == Node::Kind::Type)
            return RemangleAsType(dem, type);
      }
      if (child->getKind() == Node::Kind::ReturnType &&
          child->getNumChildren() == 1) {
        NodePointer type = child->getFirstChild();
        if (type->getKind() == Node::Kind::Type)
          return RemangleAsType(dem, type);
      }
    }
    // Else this is a void / "()" type.
    NodePointer type = dem.createNode(Node::Kind::Type);
    NodePointer tuple = dem.createNode(Node::Kind::Tuple);
    type->addChild(tuple, dem);
    return RemangleAsType(dem, type);
  };
  VALIDATE_AND_RETURN(impl, GetFunctionReturnType, type,
                      (ReconstructType(type)), (ReconstructType(type)));
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
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler dem;

    // The type that will be wrapped in UnsafePointer.
    auto *pointee_type = GetDemangledType(dem, AsMangledName(type));
    // The UnsafePointer type.
    auto *pointer_type = dem.createNode(Node::Kind::Type);

    auto *bgs = GetPointerTo(dem, pointee_type);
    pointer_type->addChild(bgs, dem);
    return RemangleAsType(dem, pointer_type);
  };
  VALIDATE_AND_RETURN(impl, GetPointerType, type, (ReconstructType(type)),
                      (ReconstructType(type)));
}

// Exploring the type
llvm::Optional<uint64_t>
TypeSystemSwiftTypeRef::GetBitSize(opaque_compiler_type_t type,
                                   ExecutionContextScope *exe_scope) {
  LLDB_SCOPED_TIMER();
  auto impl = [&]() -> llvm::Optional<uint64_t> {
    // Bug-for-bug compatibility. See comment in SwiftASTContext::GetBitSize().
    if (IsFunctionType(type))
      return GetPointerByteSize() * 8;

    // Clang types can be resolved even without a process.
    if (CompilerType clang_type = GetAsClangTypeOrNull(type)) {
      // Swift doesn't know pointers: return the size of the object
      // pointer instead of the underlying object.
      if (Flags(clang_type.GetTypeInfo()).AllSet(eTypeIsObjC | eTypeIsClass))
        return GetPointerByteSize() * 8;
      return clang_type.GetBitSize(exe_scope);
    }
    if (!exe_scope) {
      LLDB_LOGF(
          GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
          "Couldn't compute size of type %s without an execution context.",
          AsMangledName(type));
      return {};
    }
    // The hot code path is to ask the Swift runtime for the size.
    if (auto *runtime =
        SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
      if (auto result = runtime->GetBitSize({this, type}, exe_scope))
        return result;
      // If this is an expression context, perhaps the type was
      // defined in the expression. In that case we don't have debug
      // info for it, so defer to SwiftASTContext.
      if (llvm::isa<SwiftASTContextForExpressions>(m_swift_ast_context))
        return ReconstructType({this, type}).GetBitSize(exe_scope);
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "Couldn't compute size of type %s using SwiftLanguageRuntime.",
                AsMangledName(type));
      return {};
    }

    // If there is no process, we can still try to get the static size
    // information out of DWARF. Because it is stored in the Type
    // object we need to look that up by name again.
    if (TypeSP type_sp = LookupTypeInModule(type)) {
      struct SwiftType : public Type {
        /// Avoid a potential infinite recursion because
        /// Type::GetByteSize() may call into this function again.
        llvm::Optional<uint64_t> GetStaticByteSize() {
          if (m_byte_size_has_value)
            return m_byte_size;
          return {};
        }
      };
      if (auto byte_size =
              reinterpret_cast<SwiftType *>(type_sp.get())->GetStaticByteSize())
        return *byte_size * 8;
      else return {};
    }
    LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
              "Couldn't compute size of type %s using static debug info.",
              AsMangledName(type));
    return {};
  };
  FALLBACK(GetBitSize, (ReconstructType(type), exe_scope));
  if (exe_scope && exe_scope->CalculateProcess()) {
    VALIDATE_AND_RETURN(impl, GetBitSize, type,
                        (ReconstructType(type), exe_scope),
                        (ReconstructType(type), exe_scope));
  } else
    return impl();
}

llvm::Optional<uint64_t>
TypeSystemSwiftTypeRef::GetByteStride(opaque_compiler_type_t type,
                                      ExecutionContextScope *exe_scope) {
  LLDB_SCOPED_TIMER();
  auto impl = [&]() -> llvm::Optional<uint64_t> {
    if (auto *runtime =
            SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
      return runtime->GetByteStride(GetCanonicalType(type));
    }
    return {};
  };
  VALIDATE_AND_RETURN(impl, GetByteStride, type,
                      (ReconstructType(type), exe_scope),
                      (ReconstructType(type), exe_scope));
}

lldb::Encoding TypeSystemSwiftTypeRef::GetEncoding(opaque_compiler_type_t type,
                                                   uint64_t &count) {
  LLDB_SCOPED_TIMER();
  auto impl = [&]() -> lldb::Encoding {
    if (!type)
      return lldb::eEncodingInvalid;

    count = 1;

    using namespace swift::Demangle;
    Demangler dem;
    auto *node = DemangleCanonicalType(dem, type);
    auto kind = node->getKind();

    if (kind == Node::Kind::BuiltinTypeName) {
      assert(node->hasText());
      if (node->getText().startswith(swift::BUILTIN_TYPE_NAME_INT) ||
          node->getText() == swift::BUILTIN_TYPE_NAME_WORD)
        return lldb::eEncodingSint;
      if (node->getText().startswith(swift::BUILTIN_TYPE_NAME_FLOAT) ||
          node->getText().startswith(swift::BUILTIN_TYPE_NAME_FLOAT_PPC))
        return lldb::eEncodingIEEE754;
      if (node->getText() == swift::BUILTIN_TYPE_NAME_RAWPOINTER ||
          node->getText() == swift::BUILTIN_TYPE_NAME_NATIVEOBJECT ||
          node->getText() == swift::BUILTIN_TYPE_NAME_UNSAFEVALUEBUFFER ||
          node->getText() == swift::BUILTIN_TYPE_NAME_BRIDGEOBJECT)
        return lldb::eEncodingUint;
      if (node->getText().startswith(swift::BUILTIN_TYPE_NAME_VEC)) {
        count = 0;
        return lldb::eEncodingInvalid;
      }

      assert(false && "Unhandled builtin");
      count = 0;
      return lldb::eEncodingInvalid;
    }

    switch (kind) {
    case Node::Kind::Class:
    case Node::Kind::BoundGenericClass:
    case Node::Kind::FunctionType:
    case Node::Kind::NoEscapeFunctionType:
    case Node::Kind::ImplFunctionType:
    case Node::Kind::DependentGenericParamType:
    case Node::Kind::Function:
    case Node::Kind::BoundGenericFunction:
    case Node::Kind::Metatype:
    case Node::Kind::ExistentialMetatype:
      return lldb::eEncodingUint;

    case Node::Kind::Unmanaged:
    case Node::Kind::Unowned:
    case Node::Kind::Weak: {
      auto *referent_node = node->getFirstChild();
      assert(referent_node->getKind() == Node::Kind::Type);
      auto referent_type = RemangleAsType(dem, referent_node);
      return referent_type.GetEncoding(count);
    }
    default:
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "No encoding for type %s",
                AsMangledName(type));
      break;
    }

    count = 0;
    return lldb::eEncodingInvalid;
  };

#ifndef NDEBUG
  uint64_t validation_count = 0;
#endif
  VALIDATE_AND_RETURN(impl, GetEncoding, type,
                      (ReconstructType(type), validation_count),
                      (ReconstructType(type), count));
}

lldb::Format TypeSystemSwiftTypeRef::GetFormat(opaque_compiler_type_t type) {
  LLDB_SCOPED_TIMER();
  return m_swift_ast_context->GetFormat(ReconstructType(type));
}

uint32_t
TypeSystemSwiftTypeRef::GetNumChildren(opaque_compiler_type_t type,
                                       bool omit_empty_base_classes,
                                       const ExecutionContext *exe_ctx) {
  LLDB_SCOPED_TIMER();
  FALLBACK(GetNumChildren,
           (ReconstructType(type), omit_empty_base_classes, exe_ctx));
  if (exe_ctx)
    if (auto *exe_scope = exe_ctx->GetBestExecutionContextScope())
      if (auto *runtime =
              SwiftLanguageRuntime::Get(exe_scope->CalculateProcess()))
        if (auto num_children =
                runtime->GetNumChildren(GetCanonicalType(type), nullptr))
          // Use a lambda to intercept and unwrap the `Optional` return value.
          // Optional<uint32_t> uses more lax equivalency function.
          return [&]() -> llvm::Optional<uint32_t> {
            auto impl = [&]() { return num_children; };
            VALIDATE_AND_RETURN(
                impl, GetNumChildren, type,
                (ReconstructType(type), omit_empty_base_classes, exe_ctx),
                (ReconstructType(type), omit_empty_base_classes, exe_ctx));
          }().getValueOr(0);

  LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
            "Using SwiftASTContext::GetNumChildren fallback for type %s",
            AsMangledName(type));

  return m_swift_ast_context->GetNumChildren(ReconstructType(type),
                                             omit_empty_base_classes, exe_ctx);
}

uint32_t TypeSystemSwiftTypeRef::GetNumFields(opaque_compiler_type_t type,
                                              ExecutionContext *exe_ctx) {
  LLDB_SCOPED_TIMER();
  FALLBACK(GetNumFields, (ReconstructType(type), exe_ctx));
  if (exe_ctx)
    if (auto *runtime = SwiftLanguageRuntime::Get(exe_ctx->GetProcessSP()))
      if (auto num_fields =
              runtime->GetNumFields(GetCanonicalType(type), exe_ctx))
        // Use a lambda to intercept & unwrap the `Optional` return value from
        // `SwiftLanguageRuntime::GetNumFields`.
        // Optional<uint32_t> uses more lax equivalency function.
        return [&]() -> llvm::Optional<uint32_t> {
          auto impl = [&]() -> llvm::Optional<uint32_t> {
            if (!type)
              return 0;
            return num_fields;
          };
          VALIDATE_AND_RETURN(impl, GetNumFields, type,
                              (ReconstructType(type), exe_ctx),
                              (ReconstructType(type), exe_ctx));
        }().getValueOr(0);

  LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
            "Using SwiftASTContext::GetNumFields fallback for type %s",
            AsMangledName(type));

  return m_swift_ast_context->GetNumFields(ReconstructType(type), exe_ctx);
}

CompilerType TypeSystemSwiftTypeRef::GetFieldAtIndex(
    opaque_compiler_type_t type, size_t idx, std::string &name,
    uint64_t *bit_offset_ptr, uint32_t *bitfield_bit_size_ptr,
    bool *is_bitfield_ptr) {
  LLDB_SCOPED_TIMER();
  return m_swift_ast_context->GetFieldAtIndex(
      ReconstructType(type), idx, name, bit_offset_ptr, bitfield_bit_size_ptr,
      is_bitfield_ptr);
}

static swift::Demangle::NodePointer
GetClangTypeTypeNode(TypeSystemSwiftTypeRef &ts,
                     swift::Demangle::Demangler &dem, CompilerType clang_type,
                     SwiftASTContext *module_holder) {
  assert(llvm::isa<TypeSystemClang>(clang_type.GetTypeSystem()) &&
         "expected a clang type");
  using namespace swift::Demangle;
  NodePointer type = dem.createNode(Node::Kind::Type);
  type->addChild(GetClangTypeNode(clang_type, dem, module_holder), dem);
  return type;
}

CompilerType TypeSystemSwiftTypeRef::GetChildCompilerTypeAtIndex(
    opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
    bool transparent_pointers, bool omit_empty_base_classes,
    bool ignore_array_bounds, std::string &child_name,
    uint32_t &child_byte_size, int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class, bool &child_is_deref_of_parent,
    ValueObject *valobj, uint64_t &language_flags) {
  LLDB_SCOPED_TIMER();
  child_name = "";
  child_byte_size = 0;
  child_byte_offset = 0;
  child_bitfield_bit_size = 0;
  child_bitfield_bit_offset = 0;
  child_is_base_class = false;
  child_is_deref_of_parent = false;
  language_flags = 0;
  auto fallback = [&]() -> CompilerType {
    LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
              "Had to engage SwiftASTContext fallback for type %s.",
              AsMangledName(type));
    return m_swift_ast_context->GetChildCompilerTypeAtIndex(
        ReconstructType(type), exe_ctx, idx, transparent_pointers,
        omit_empty_base_classes, ignore_array_bounds, child_name,
        child_byte_size, child_byte_offset, child_bitfield_bit_size,
        child_bitfield_bit_offset, child_is_base_class,
        child_is_deref_of_parent, valobj, language_flags);
  };
  FALLBACK(GetChildCompilerTypeAtIndex,
           (ReconstructType(type), exe_ctx, idx, transparent_pointers,
            omit_empty_base_classes, ignore_array_bounds, child_name,
            child_byte_size, child_byte_offset, child_bitfield_bit_size,
            child_bitfield_bit_offset, child_is_base_class,
            child_is_deref_of_parent, valobj, language_flags));
  llvm::Optional<unsigned> ast_num_children;
  auto get_ast_num_children = [&]() {
    if (ast_num_children)
      return *ast_num_children;
    ast_num_children = m_swift_ast_context->GetNumChildren(
        ReconstructType(type), omit_empty_base_classes, exe_ctx);
    return *ast_num_children;
  };
  auto impl = [&]() -> CompilerType {
    ExecutionContextScope *exe_scope = nullptr;
    if (exe_ctx)
      exe_scope = exe_ctx->GetBestExecutionContextScope();
    if (exe_scope) {
      if (auto *runtime =
            SwiftLanguageRuntime::Get(exe_scope->CalculateProcess()))
        if (CompilerType result = runtime->GetChildCompilerTypeAtIndex(
                {this, type}, idx, transparent_pointers,
                omit_empty_base_classes, ignore_array_bounds, child_name,
                child_byte_size, child_byte_offset, child_bitfield_bit_size,
                child_bitfield_bit_offset, child_is_base_class,
                child_is_deref_of_parent, valobj, language_flags)) {
          // This type is treated specially by ClangImporter.  It's really a
          // typedef to NSString *, but ClangImporter introduces an extra
          // layer of indirection that we simulate here.
          if (llvm::StringRef(AsMangledName(type))
                  .endswith("sSo18NSNotificationNameaD"))
            return GetTypeFromMangledTypename(ConstString("$sSo8NSStringCD"));
          if (result.GetMangledTypeName().GetStringRef().count('$') > 1 &&
              get_ast_num_children() ==
                  runtime->GetNumChildren({this, type}, valobj))
            // If available, prefer the AST for private types. Private
            // identifiers are not ABI; the runtime returns anonymous private
            // identifiers (using a '$' prefix) which cannot match identifiers
            // in the AST. Because these private types can't be used in an AST
            // context, prefer the AST type if available.
            if (auto ast_type = fallback())
              return ast_type;
          return result;
        }
    }
    // Clang types can be resolved even without a process.
    bool is_signed;
    if (CompilerType clang_type = GetAsClangTypeOrNull(type)) {
      if (clang_type.IsEnumerationType(is_signed) && idx == 0)
        // C enums get imported into Swift as structs with a "rawValue" field.
        if (auto *ts =
                llvm::dyn_cast<TypeSystemClang>(clang_type.GetTypeSystem()))
          if (clang::EnumDecl *enum_decl = ts->GetAsEnumDecl(clang_type)) {
            swift::Demangle::Demangler dem;
            CompilerType raw_value =
                CompilerType(ts, enum_decl->getIntegerType().getAsOpaquePtr());
            child_name = "rawValue";
            auto bit_size = raw_value.GetBitSize(
                exe_ctx ? exe_ctx->GetBestExecutionContextScope() : nullptr);
            child_byte_size = bit_size.getValueOr(0) / 8;
            child_byte_offset = 0;
            child_bitfield_bit_size = 0;
            child_bitfield_bit_offset = 0;
            child_is_base_class = false;
            child_is_deref_of_parent = false;
            language_flags = 0;
            return RemangleAsType(dem,
                                  GetClangTypeTypeNode(*this, dem, raw_value,
                                                       m_swift_ast_context));
          }
      // Otherwise defer to TypeSystemClang.
      //
      // Swift skips bitfields when counting children. Unfortunately
      // this means we need to do this inefficient linear search here.
      CompilerType clang_child_type;
      for (size_t clang_idx = 0, swift_idx = 0; swift_idx <= idx; ++clang_idx) {
        child_bitfield_bit_size = 0;
        child_bitfield_bit_offset = 0;
        clang_child_type = clang_type.GetChildCompilerTypeAtIndex(
            exe_ctx, clang_idx, transparent_pointers, omit_empty_base_classes,
            ignore_array_bounds, child_name, child_byte_size, child_byte_offset,
            child_bitfield_bit_size, child_bitfield_bit_offset,
            child_is_base_class, child_is_deref_of_parent, valobj,
            language_flags);
        if (!child_bitfield_bit_size && !child_bitfield_bit_offset)
          ++swift_idx;
        // FIXME: Why is this necessary?
        if (clang_child_type.IsTypedefType() &&
            clang_child_type.GetTypeName() ==
                clang_child_type.GetTypedefedType().GetTypeName())
          clang_child_type = clang_child_type.GetTypedefedType();
      }
      if (clang_child_type) {
        std::string prefix;
        swift::Demangle::Demangler dem;
        swift::Demangle::NodePointer node = GetClangTypeTypeNode(
            *this, dem, clang_child_type, m_swift_ast_context);
        switch (node->getChild(0)->getKind()) {
        case swift::Demangle::Node::Kind::Class:
            prefix = "ObjectiveC.";
          break;
        default:
          break;
        }
        child_name = prefix + child_name;
        return RemangleAsType(dem, node);
      }
    }
    // FIXME: SwiftASTContext can sometimes find more Clang types because it
    // imports Clang modules from source. We should be able to replicate this
    // and remove this fallback.
    return fallback();

    if (!exe_scope)
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "Cannot compute the children of type %s without an execution "
                "context.",
                AsMangledName(type));
    else
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "Couldn't compute size of type %s without a process.",
                AsMangledName(type));
    return {};
  };
  // Skip validation when there is no process, because then we also
  // don't have a runtime.
  if (!exe_ctx)
    return impl();
  ExecutionContextScope *exe_scope = exe_ctx->GetBestExecutionContextScope();
  if (!exe_scope)
    return impl();
  auto *runtime = SwiftLanguageRuntime::Get(exe_scope->CalculateProcess());
  if (!runtime)
    return impl();
  // FIXME:
  // No point comparing the results if the reflection data has more
  // information.  There's a nasty chicken & egg problem buried here:
  // Because the API deals out an index into a list of children we
  // can't mix&match between the two typesystems if there is such a
  // divergence. We'll need to replace all calls at once.
  if (get_ast_num_children() <
      runtime->GetNumChildren({this, type}, valobj).getValueOr(0))
    return impl();

#ifndef NDEBUG
  std::string ast_child_name;
  uint32_t ast_child_byte_size = 0;
  int32_t ast_child_byte_offset = 0;
  uint32_t ast_child_bitfield_bit_size = 0;
  uint32_t ast_child_bitfield_bit_offset = 0;
  bool ast_child_is_base_class = false;
  bool ast_child_is_deref_of_parent = false;
  uint64_t ast_language_flags = 0;
  auto defer = llvm::make_scope_exit([&] {
    if (!ModuleList::GetGlobalModuleListProperties()
             .GetSwiftValidateTypeSystem())
      return;
    llvm::StringRef suffix(ast_child_name);
    if (suffix.consume_front("__ObjC."))
      ast_child_name = suffix.str();
    assert((llvm::StringRef(child_name).contains('.') ||
            llvm::StringRef(ast_child_name).contains('.') ||
            Equivalent(child_name, ast_child_name)));
    assert(ast_language_flags ||
           (Equivalent(llvm::Optional<uint64_t>(child_byte_size),
                       llvm::Optional<uint64_t>(ast_child_byte_size))));
    assert(Equivalent(llvm::Optional<uint64_t>(child_byte_offset),
                      llvm::Optional<uint64_t>(ast_child_byte_offset)));
    assert(
        Equivalent(child_bitfield_bit_offset, ast_child_bitfield_bit_offset));
    assert(Equivalent(child_bitfield_bit_size, ast_child_bitfield_bit_size));
    assert(Equivalent(child_is_base_class, ast_child_is_base_class));
    assert(Equivalent(child_is_deref_of_parent, ast_child_is_deref_of_parent));
    assert(Equivalent(language_flags, ast_language_flags));
  });
#endif
  VALIDATE_AND_RETURN(
      impl, GetChildCompilerTypeAtIndex, type,
      (ReconstructType(type), exe_ctx, idx, transparent_pointers,
       omit_empty_base_classes, ignore_array_bounds, ast_child_name,
       ast_child_byte_size, ast_child_byte_offset, ast_child_bitfield_bit_size,
       ast_child_bitfield_bit_offset, ast_child_is_base_class,
       ast_child_is_deref_of_parent, valobj, ast_language_flags),
      (ReconstructType(type), exe_ctx, idx, transparent_pointers,
       omit_empty_base_classes, ignore_array_bounds, child_name,
       child_byte_size, child_byte_offset, child_bitfield_bit_size,
       child_bitfield_bit_offset, child_is_base_class, child_is_deref_of_parent,
       valobj, language_flags));
}

size_t TypeSystemSwiftTypeRef::GetIndexOfChildMemberWithName(
    opaque_compiler_type_t type, const char *name, ExecutionContext *exe_ctx,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  LLDB_SCOPED_TIMER();
  FALLBACK(GetIndexOfChildMemberWithName,
           (ReconstructType(type), name, exe_ctx, omit_empty_base_classes,
            child_indexes));
  if (auto *exe_scope = exe_ctx->GetBestExecutionContextScope())
    if (auto *runtime =
            SwiftLanguageRuntime::Get(exe_scope->CalculateProcess()))
      if (auto index_size = runtime->GetIndexOfChildMemberWithName(
              GetCanonicalType(type), name, exe_ctx, omit_empty_base_classes,
              child_indexes)) {
#ifndef NDEBUG
        // This block is a custom VALIDATE_AND_RETURN implementation to support
        // checking the return value, plus the by-ref `child_indexes`.
        if (!m_swift_ast_context)
          return *index_size;
        auto ast_type = ReconstructType(type);
        if (!ast_type)
          return *index_size;
        std::vector<uint32_t> ast_child_indexes;
        auto ast_index_size = m_swift_ast_context->GetIndexOfChildMemberWithName(
                ast_type, name, exe_ctx, omit_empty_base_classes,
                ast_child_indexes);
        // The runtime has more info than the AST. No useful validation can be
        // done.
        if (*index_size > ast_index_size)
          return *index_size;

        auto fail = [&]() {
          auto join = [](const auto &v) {
            std::ostringstream buf;
            buf << "{";
            for (const auto &item : v)
              buf << item << ",";
            buf.seekp(-1, std::ios_base::end);
            buf << "}";
            return buf.str();
          };
          llvm::dbgs() << join(child_indexes)
                       << " != " << join(ast_child_indexes) << "\n";
          llvm::dbgs() << "failing type was " << (const char *)type
                       << ", member was " << name << "\n";
          assert(false &&
                 "TypeSystemSwiftTypeRef diverges from SwiftASTContext");
        };
        if (*index_size != ast_index_size)
          fail();
        for (unsigned i = 0; i < *index_size; ++i)
          if (child_indexes[i] < ast_child_indexes[i])
            // When the runtime may know know about more children. When this
            // happens, indexes will be larger. But if an index is smaller, that
            // means the runtime has dropped info somehow.
            fail();
#endif
        return *index_size;
      }

  LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
            "Using SwiftASTContext::GetIndexOfChildMemberWithName fallback for "
            "type %s",
            AsMangledName(type));

  return m_swift_ast_context->GetIndexOfChildMemberWithName(
      ReconstructType(type), name, exe_ctx, omit_empty_base_classes,
      child_indexes);
}

size_t
TypeSystemSwiftTypeRef::GetNumTemplateArguments(opaque_compiler_type_t type) {
  auto impl = [&]() -> size_t {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);

    if (!node)
      return 0;

    switch (node->getKind()) {
    case Node::Kind::BoundGenericClass:
    case Node::Kind::BoundGenericEnum:
    case Node::Kind::BoundGenericStructure:
    case Node::Kind::BoundGenericProtocol:
    case Node::Kind::BoundGenericOtherNominalType:
    case Node::Kind::BoundGenericTypeAlias:
    case Node::Kind::BoundGenericFunction: {
      if (node->getNumChildren() > 1) {
        NodePointer child = node->getChild(1);
        if (child && child->getKind() == Node::Kind::TypeList)
          return child->getNumChildren();
      }
    } break;
    default:
      break;
    }
    return 0;
  };
  VALIDATE_AND_RETURN(impl, GetNumTemplateArguments, type,
                      (ReconstructType(type)), (ReconstructType(type)));
}

CompilerType
TypeSystemSwiftTypeRef::GetTypeForFormatters(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType { return {this, type}; };
  VALIDATE_AND_RETURN(impl, GetTypeForFormatters, type,
                      (ReconstructType(type)),
                      (ReconstructType(type)));
}

LazyBool
TypeSystemSwiftTypeRef::ShouldPrintAsOneLiner(opaque_compiler_type_t type,
                                              ValueObject *valobj) {
  auto impl = [&]() {
    if (type) {
      if (IsImportedType(type, nullptr))
        return eLazyBoolNo;
    }
    if (valobj) {
      if (valobj->IsBaseClass())
        return eLazyBoolNo;
      if ((valobj->GetLanguageFlags() & LanguageFlags::eIsIndirectEnumCase) ==
          LanguageFlags::eIsIndirectEnumCase)
        return eLazyBoolNo;
    }

    return eLazyBoolCalculate;
  };
  VALIDATE_AND_RETURN(impl, ShouldPrintAsOneLiner, type,
                      (ReconstructType(type), valobj),
                      (ReconstructType(type), valobj));
}

bool TypeSystemSwiftTypeRef::IsMeaninglessWithoutDynamicResolution(
    opaque_compiler_type_t type) {
  LLDB_SCOPED_TIMER();
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    auto *node = DemangleCanonicalType(dem, type);
    return ContainsGenericTypeParameter(node) && !IsFunctionType(type);
  };
  VALIDATE_AND_RETURN(impl, IsMeaninglessWithoutDynamicResolution, type,
                      (ReconstructType(type)), (ReconstructType(type)));
}

CompilerType TypeSystemSwiftTypeRef::GetAsClangTypeOrNull(
    lldb::opaque_compiler_type_t type) {
  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = GetDemangledType(dem, AsMangledName(type));
  // Directly resolve Clang typedefs into Clang types.  Imported
  // type aliases that point to Clang type that are also Swift builtins, like
  // Swift.Int, otherwise would resolved to Swift types.
  if (node && node->getKind() == Node::Kind::TypeAlias &&
      node->getNumChildren() == 2 && node->getChild(0)->hasText() &&
      node->getChild(0)->getText() == swift::MANGLING_MODULE_OBJC &&
      node->getChild(1)->hasText()) {
    auto node_clangtype = ResolveTypeAlias(m_swift_ast_context, dem, node,
                                           /*prefer_clang_types*/ true);
    if (node_clangtype.second)
      return node_clangtype.second;
  }
  CompilerType clang_type;
  IsImportedType(type, &clang_type);
  return clang_type;
}

bool TypeSystemSwiftTypeRef::IsImportedType(opaque_compiler_type_t type,
                                            CompilerType *original_type) {
  LLDB_SCOPED_TIMER();
  auto impl = [&]() -> bool {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));

    // This is an imported Objective-C type; look it up in the debug info.
    StringRef ident = GetObjCTypeName(node);
    if (ident.empty())
      return {};
    if (original_type)
      if (TypeSP clang_type = LookupClangType(m_swift_ast_context, ident))
        *original_type = clang_type->GetForwardCompilerType();
    return true;
  };
  FALLBACK(IsImportedType, (ReconstructType(type), original_type));
  // We can't validate the result because ReconstructType may call this
  // function, causing an infinite loop.
  return impl();
}

bool TypeSystemSwiftTypeRef::IsExistentialType(
    lldb::opaque_compiler_type_t type) {
  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = DemangleCanonicalType(dem, type);
  if (!node || node->getNumChildren() != 1)
    return false;
  switch (node->getKind()) {
  case Node::Kind::Protocol:
  case Node::Kind::ProtocolList:
    return true;
  default:
    return false;
  }
}

CompilerType TypeSystemSwiftTypeRef::GetRawPointerType() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer raw_ptr = dem.createNode(Node::Kind::BuiltinTypeName,
                                         swift::BUILTIN_TYPE_NAME_RAWPOINTER);
    NodePointer node = dem.createNode(Node::Kind::Type);
    node->addChild(raw_ptr, dem);
    return RemangleAsType(dem, node);
}

CompilerType TypeSystemSwiftTypeRef::GetErrorType() {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    auto *error_type = dem.createNode(Node::Kind::Type);
    auto *parent = error_type;
    NodePointer node;
    node = dem.createNode(Node::Kind::ProtocolList);
    parent->addChild(node, dem);
    parent = node;
    node = dem.createNode(Node::Kind::TypeList);
    parent->addChild(node, dem);
    parent = node;
    node = dem.createNode(Node::Kind::Type);
    parent->addChild(node, dem);
    parent = node;
    node = dem.createNode(Node::Kind::Protocol);
    parent->addChild(node, dem);
    parent = node;

    parent->addChild(
        dem.createNodeWithAllocatedText(Node::Kind::Module, swift::STDLIB_NAME),
        dem);
    parent->addChild(dem.createNode(Node::Kind::Identifier, "Error"), dem);

    return RemangleAsType(dem, error_type);
  };
  VALIDATE_AND_RETURN_STATIC(impl, GetErrorType);
}

CompilerType
TypeSystemSwiftTypeRef::GetReferentType(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));
    if (!node ||
        (node->getKind() != Node::Kind::Unowned &&
         node->getKind() != Node::Kind::Unmanaged) ||
        !node->hasChildren())
      return {this, type};
    node = node->getFirstChild();
    if (!node || node->getKind() != Node::Kind::Type || !node->hasChildren())
      return {this, type};
    return RemangleAsType(dem, node);
  };
  VALIDATE_AND_RETURN(impl, GetReferentType, type, (ReconstructType(type)),
                      (ReconstructType(type)));
}

CompilerType
TypeSystemSwiftTypeRef::GetInstanceType(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);

    if (!node)
      return {};
    if (ContainsUnresolvedTypeAlias(node)) {
      // If we couldn't resolve all type aliases, we might be in a REPL session
      // where getting to the debug information necessary for resolving that
      // type alias isn't possible, or the user might have defined the
      // type alias in the REPL. In these cases, fallback to asking the AST
      // for the canonical type.
      return m_swift_ast_context->GetInstanceType(ReconstructType(type));
    }

    if (node->getKind() == Node::Kind::Metatype) {
      for (NodePointer child : *node)
        if (child->getKind() == Node::Kind::Type)
          return RemangleAsType(dem, child);
      return {};
    }
    return {this, type};
  };
  VALIDATE_AND_RETURN(impl, GetInstanceType, type, (ReconstructType(type)),
                      (ReconstructType(type)));
}

TypeSystemSwift::TypeAllocationStrategy
TypeSystemSwiftTypeRef::GetAllocationStrategy(opaque_compiler_type_t type) {
  return m_swift_ast_context->GetAllocationStrategy(ReconstructType(type));
}

CompilerType TypeSystemSwiftTypeRef::CreateTupleType(
    const std::vector<TupleElement> &elements) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler dem;
    auto *tuple_type = dem.createNode(Node::Kind::Type);
    auto *tuple = dem.createNode(Node::Kind::Tuple);
    tuple_type->addChild(tuple, dem);

    for (const auto &element : elements) {
      auto *tuple_element = dem.createNode(Node::Kind::TupleElement);
      tuple->addChild(tuple_element, dem);

      // Add the element's name, if it has one.
      // Ex: `(Int, Int)` vs `(x: Int, y: Int)`
      if (!element.element_name.IsEmpty()) {
        auto *name = dem.createNode(Node::Kind::TupleElementName,
                                    element.element_name.GetStringRef());
        tuple_element->addChild(name, dem);
      }

      auto *type = dem.createNode(Node::Kind::Type);
      tuple_element->addChild(type, dem);
      auto *element_type = GetDemangledType(
          dem, element.element_type.GetMangledTypeName().GetStringRef());
      type->addChild(element_type, dem);
    }

    return RemangleAsType(dem, tuple_type);
  };

  // The signature of VALIDATE_AND_RETURN doesn't support this function, below
  // is an inlined function-specific variation.
  FALLBACK(CreateTupleType, (elements));
#ifndef NDEBUG
  {
    auto result = impl();
    if (!m_swift_ast_context)
      return result;
    bool equivalent =
        Equivalent(result, m_swift_ast_context->CreateTupleType(elements));
    if (!equivalent)
      llvm::dbgs() << "failing tuple type\n";
    assert(equivalent &&
           "TypeSystemSwiftTypeRef diverges from SwiftASTContext");
    return result;
  }
#else
  return impl();
#endif
}

void TypeSystemSwiftTypeRef::DumpTypeDescription(
    opaque_compiler_type_t type, bool print_help_if_available,
    bool print_extensions_if_available, lldb::DescriptionLevel level) {
  LLDB_SCOPED_TIMER();
  return m_swift_ast_context->DumpTypeDescription(
      ReconstructType(type), print_help_if_available, print_help_if_available,
      level);
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(
    opaque_compiler_type_t type, Stream *s, bool print_help_if_available,
    bool print_extensions_if_available, lldb::DescriptionLevel level) {
  LLDB_SCOPED_TIMER();
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

bool TypeSystemSwiftTypeRef::DumpTypeValue(
    opaque_compiler_type_t type, Stream *s, lldb::Format format,
    const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, ExecutionContextScope *exe_scope,
    bool is_base_class) {
  LLDB_SCOPED_TIMER();
  auto impl = [&]() -> bool {
    if (!type)
      return false;

    using namespace swift::Demangle;
    Demangler dem;
    auto *node = DemangleCanonicalType(dem, type);
    switch (node->getKind()) {
    case Node::Kind::Class:
    case Node::Kind::BoundGenericClass:
      if (is_base_class)
        return false;
      LLVM_FALLTHROUGH;
    case Node::Kind::ExistentialMetatype:
    case Node::Kind::Metatype:
      format = eFormatPointer;
      LLVM_FALLTHROUGH;
    case Node::Kind::BuiltinTypeName:
    case Node::Kind::DependentGenericParamType:
    case Node::Kind::FunctionType:
    case Node::Kind::NoEscapeFunctionType:
    case Node::Kind::CFunctionPointer:
    case Node::Kind::ImplFunctionType: {
      uint32_t item_count = 1;
      // A few formats, we might need to modify our size and count for
      // depending on how we are trying to display the value.
      switch (format) {
      case eFormatChar:
      case eFormatCharPrintable:
      case eFormatCharArray:
      case eFormatBytes:
      case eFormatBytesWithASCII:
        item_count = data_byte_size;
        data_byte_size = 1;
        break;
      case eFormatUnicode16:
        item_count = data_byte_size / 2;
        data_byte_size = 2;
        break;
      case eFormatUnicode32:
        item_count = data_byte_size / 4;
        data_byte_size = 4;
        break;
      case eFormatAddressInfo:
        if (data_byte_size == 0) {
          data_byte_size = exe_scope->CalculateTarget()
                               ->GetArchitecture()
                               .GetAddressByteSize();
          item_count = 1;
        }
        break;
      default:
        break;
      }
      return DumpDataExtractor(data, s, data_offset, format, data_byte_size,
                               item_count, UINT32_MAX, LLDB_INVALID_ADDRESS,
                               bitfield_bit_size, bitfield_bit_offset,
                               exe_scope);
    }
    case Node::Kind::Unmanaged:
    case Node::Kind::Unowned:
    case Node::Kind::Weak: {
      auto *referent_node = node->getFirstChild();
      assert(referent_node->getKind() == Node::Kind::Type);
      auto referent_type = RemangleAsType(dem, referent_node);
      return referent_type.DumpTypeValue(
          s, format, data, data_offset, data_byte_size, bitfield_bit_size,
          bitfield_bit_offset, exe_scope, is_base_class);
    }
    case Node::Kind::BoundGenericStructure:
      return false;
    case Node::Kind::Structure: {
      // In some instances, a swift `structure` wraps an objc enum. The enum
      // case needs to be handled, but structs are no-ops.
      auto resolved = ResolveTypeAlias(m_swift_ast_context, dem, node, true);
      auto clang_type = std::get<CompilerType>(resolved);
      if (!clang_type)
        return false;

      bool is_signed;
      if (!clang_type.IsEnumerationType(is_signed))
        // The type is a clang struct, not an enum.
        return false;

      // The type is an enum imported from clang. Try Swift type metadata first,
      // and failing that fallback to the AST.
      LLVM_FALLTHROUGH;
    }
    case Node::Kind::Enum:
    case Node::Kind::BoundGenericEnum: {
      if (exe_scope)
        if (auto runtime =
                SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
          ExecutionContext exe_ctx;
          exe_scope->CalculateExecutionContext(exe_ctx);
          if (auto case_name =
                  runtime->GetEnumCaseName({this, type}, data, &exe_ctx)) {
            s->PutCString(*case_name);
            return true;
          }
        }

      // No result available from the runtime, fallback to the AST.
      // This can happen in two cases:
      // 1. MultiPayloadEnums not currently supported by Swift reflection
      // 2. Some clang imported enums
      return m_swift_ast_context->DumpTypeValue(
          ReconstructType(type), s, format, data, data_offset, data_byte_size,
          bitfield_bit_size, bitfield_bit_offset, exe_scope, is_base_class);
    }
    default:
      assert(false && "Unhandled node kind");
      LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
                "DumpTypeValue: Unhandled node kind for type %s",
                AsMangledName(type));
      return false;
    }
  };

#ifndef NDEBUG
  FALLBACK(DumpTypeValue,
           (ReconstructType(type), s, format, data, data_offset, data_byte_size,
            bitfield_bit_size, bitfield_bit_offset, exe_scope, is_base_class));
  StreamString ast_s;
  auto defer = llvm::make_scope_exit([&] {
    assert(Equivalent(ConstString(ast_s.GetString()),
                      ConstString(((StreamString *)s)->GetString())) &&
           "TypeSystemSwiftTypeRef diverges from SwiftASTContext");
  });
#endif
  VALIDATE_AND_RETURN(
      impl, DumpTypeValue, type,
      (ReconstructType(type), &ast_s, format, data, data_offset, data_byte_size,
       bitfield_bit_size, bitfield_bit_offset, exe_scope, is_base_class),
      (ReconstructType(type), s, format, data, data_offset, data_byte_size,
       bitfield_bit_size, bitfield_bit_offset, exe_scope, is_base_class));
}

void TypeSystemSwiftTypeRef::DumpTypeDescription(opaque_compiler_type_t type,
                                                 lldb::DescriptionLevel level) {
  LLDB_SCOPED_TIMER();
  return m_swift_ast_context->DumpTypeDescription(ReconstructType(type), level);
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(opaque_compiler_type_t type,
                                                 Stream *s,
                                                 lldb::DescriptionLevel level) {
  LLDB_SCOPED_TIMER();
  return m_swift_ast_context->DumpTypeDescription(ReconstructType(type), s,
                                                  level);
}
void TypeSystemSwiftTypeRef::DumpSummary(opaque_compiler_type_t type,
                                         ExecutionContext *exe_ctx, Stream *s,
                                         const DataExtractor &data,
                                         lldb::offset_t data_offset,
                                         size_t data_byte_size) {
  LLDB_SCOPED_TIMER();
  return m_swift_ast_context->DumpSummary(ReconstructType(type), exe_ctx, s,
                                          data, data_offset, data_byte_size);
}
bool TypeSystemSwiftTypeRef::IsPointerOrReferenceType(
    opaque_compiler_type_t type, CompilerType *pointee_type) {
  auto impl = [&]() {
    return IsPointerType(type, pointee_type) ||
           IsReferenceType(type, pointee_type, nullptr);
  };
  VALIDATE_AND_RETURN(impl, IsPointerOrReferenceType, type,
                      (ReconstructType(type), nullptr),
                      (ReconstructType(type), pointee_type));
}
llvm::Optional<size_t>
TypeSystemSwiftTypeRef::GetTypeBitAlign(opaque_compiler_type_t type,
                                        ExecutionContextScope *exe_scope) {
  LLDB_SCOPED_TIMER();
  FALLBACK(GetTypeBitAlign, (ReconstructType(type), exe_scope));
  // This method doesn't use VALIDATE_AND_RETURN because except for
  // fixed-size types the SwiftASTContext implementation forwards to
  // SwiftLanguageRuntime anyway and for many fixed-size types the
  // fixed layout still returns an incorrect default alignment of 0.
  //
  // Clang types can be resolved even without a process.
  if (CompilerType clang_type = GetAsClangTypeOrNull(type)) {
    // Swift doesn't know pointers: return the size alignment of the
    // object pointer instead of the underlying object.
    if (Flags(clang_type.GetTypeInfo()).AllSet(eTypeIsObjC | eTypeIsClass))
      return GetPointerByteSize() * 8;
    return clang_type.GetTypeBitAlign(exe_scope);
  }
  if (!exe_scope) {
    LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
              "Couldn't compute alignment of type %s without an execution "
              "context.",
              AsMangledName(type));
    return {};
  }
  if (auto *runtime =
          SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
    if (auto result = runtime->GetBitAlignment({this, type}, exe_scope))
      return result;
    // If this is an expression context, perhaps the type was
    // defined in the expression. In that case we don't have debug
    // info for it, so defer to SwiftASTContext.
    if (llvm::isa<SwiftASTContextForExpressions>(m_swift_ast_context))
      return ReconstructType({this, type}).GetTypeBitAlign(exe_scope);
  }

  // If there is no process, we can still try to get the static
  // alignment information out of DWARF. Because it is stored in the
  // Type object we need to look that up by name again.
  if (TypeSP type_sp = LookupTypeInModule(type))
    if (type_sp->GetLayoutCompilerType().GetOpaqueQualType() != type)
      return type_sp->GetLayoutCompilerType().GetTypeBitAlign(exe_scope);
  LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES),
            "Couldn't compute alignment of type %s using static debug info.",
            AsMangledName(type));
  return {};
}

#ifndef NDEBUG
static bool IsSIMDNode(NodePointer node) {
  // A SIMD vector is a clang typealias whose identifier starts with "simd_".
  if (node->getKind() == Node::Kind::TypeAlias && node->getNumChildren() >= 2) {
    NodePointer module = node->getFirstChild();
    NodePointer identifier = node->getChild(1);
    return module->getKind() == Node::Kind::Module &&
           module->getText() == "__C" &&
           identifier->getKind() == Node::Kind::Identifier &&
           identifier->getText().startswith("simd_");
  }
  // A SIMD matrix is a BoundGenericStructure whose inner identifier starts with
  // SIMD.
  if (node->getKind() == Node::Kind::BoundGenericStructure &&
      node->hasChildren()) {
    NodePointer type = node->getFirstChild();
    if (type->getKind() == Node::Kind::Type && node->hasChildren()) {
      NodePointer structure = type->getFirstChild();
      if (structure->getKind() == Node::Kind::Structure &&
          structure->getNumChildren() >= 2) {
        NodePointer identifier = structure->getChild(1);
        return identifier->getKind() == Node::Kind::Identifier &&
               identifier->getText().startswith("SIMD");
      }
    }
  }
  return false;
}
#endif

bool TypeSystemSwiftTypeRef::IsTypedefType(opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));
    return node && (node->getKind() == Node::Kind::TypeAlias ||
                    node->getKind() == Node::Kind::BoundGenericTypeAlias);
  };

#ifndef NDEBUG
  // We skip validation when dealing with a builtin type since builtins are
  // considered type aliases by Swift, which we're deviating from since
  // SwiftASTContext reconstructs Builtin types as TypeAliases pointing to the
  // actual Builtin types, but mangled names always describe the underlying
  // builtins directly.
  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = GetDemangledType(dem, AsMangledName(type));
  if (node && node->getKind() == Node::Kind::BuiltinTypeName)
    return impl();

  // This is a discrepancy with `SwiftASTContext`. `impl` correctly
  // returns true, but `VALIDATE_AND_RETURN` will assert. This hardcoded
  // handling of `__C.NSNotificationName` and `__C.NSDecimal` can be removed
  // when the `VALIDATE_AND_RETURN` is removed.
  auto mangled_name = GetMangledTypeName(type);
  if (mangled_name == "$sSo18NSNotificationNameaD" ||
      mangled_name == "$sSo9NSDecimalaD")
    return impl();

  // SIMD types have some special handling in the compiler, causing divergences
  // on the way SwiftASTContext and TypeSystemSwiftTypeRef view the same type.
  if (IsSIMDNode(node))
    return impl();
#endif
  VALIDATE_AND_RETURN(impl, IsTypedefType, type, (ReconstructType(type)),
                      (ReconstructType(type)));
}

CompilerType
TypeSystemSwiftTypeRef::GetTypedefedType(opaque_compiler_type_t type) {
  LLDB_SCOPED_TIMER();
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));
    if (!node || (node->getKind() != Node::Kind::TypeAlias &&
                  node->getKind() != Node::Kind::BoundGenericTypeAlias))
      return {};
    auto pair = ResolveTypeAlias(m_swift_ast_context, dem, node);
    NodePointer type_node = dem.createNode(Node::Kind::Type);
    if (NodePointer resolved = std::get<swift::Demangle::NodePointer>(pair)) {
      type_node->addChild(resolved, dem);
    } else {
      NodePointer clang_node = GetClangTypeNode(std::get<CompilerType>(pair),
                                                dem, m_swift_ast_context);
      type_node->addChild(clang_node, dem);
    }
    return RemangleAsType(dem, type_node);
  };
#ifndef NDEBUG
  // We skip validation when dealing with a builtin type since builtins are
  // considered type aliases by Swift, which we're deviating from since
  // SwiftASTContext reconstructs Builtin types as TypeAliases pointing to the
  // actual Builtin types, but mangled names always describe the underlying
  // builtins directly.
  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = GetDemangledType(dem, AsMangledName(type));
  if (node && node->getKind() == Node::Kind::BuiltinTypeName)
    return impl();

  // This is a discrepancy with `SwiftASTContext`. `impl` correctly
  // returns true, but `VALIDATE_AND_RETURN` will assert. This hardcoded
  // handling of `__C.NSNotificationName` and `__C.NSDecimal` can be removed
  // when the `VALIDATE_AND_RETURN` is removed.
  auto mangled_name = GetMangledTypeName(type);
  if (mangled_name == "$sSo18NSNotificationNameaD" ||
      mangled_name == "$sSo9NSDecimalaD")
    return impl();

  // SIMD types have some special handling in the compiler, causing divergences
  // on the way SwiftASTContext and TypeSystemSwiftTypeRef view the same type.
  if (IsSIMDNode(node))
    return impl();
#endif
  VALIDATE_AND_RETURN(impl, GetTypedefedType, type, (ReconstructType(type)),
                      (ReconstructType(type)));
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
    auto impl = []() { return CompilerType(); };

  VALIDATE_AND_RETURN(impl, GetLValueReferenceType, type,
                      (ReconstructType(type)), (ReconstructType(type)));
}
CompilerType
TypeSystemSwiftTypeRef::GetRValueReferenceType(opaque_compiler_type_t type) {
  auto impl = []() { return CompilerType(); };

  VALIDATE_AND_RETURN(impl, GetRValueReferenceType, type,
                      (ReconstructType(type)), (ReconstructType(type)));
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
    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);
    if (!node || node->getNumChildren() != 1 ||
        node->getKind() != Node::Kind::InOut)
      return false;

    if (pointee_type) {
      NodePointer referenced = node->getFirstChild();
      auto type = dem.createNode(Node::Kind::Type);
      type->addChild(referenced, dem);
      *pointee_type = RemangleAsType(dem, type);
    }

    if (is_rvalue)
      *is_rvalue = false;

    return true;
  };

  VALIDATE_AND_RETURN(impl, IsReferenceType, type,
                      (ReconstructType(type), nullptr, nullptr),
                      (ReconstructType(type), pointee_type, is_rvalue));
}
