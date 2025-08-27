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
#include "Plugins/TypeSystem/Swift/SwiftDWARFImporterForClangTypes.h"
#include "Plugins/TypeSystem/Swift/SwiftDemangle.h"

#include "Plugins/ExpressionParser/Clang/ClangExternalASTSourceCallbacks.h"
#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/ExpressionParser/Swift/SwiftPersistentExpressionState.h"
#include "Plugins/ExpressionParser/Swift/SwiftUserExpression.h"
#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/SymbolFile/DWARF/DWARFASTParserSwift.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TypeSystemSwiftTypeRef.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Host/StreamFile.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Timer.h"

#include "lldb/lldb-enumerations.h"
#include "swift/../../lib/ClangImporter/ClangAdapter.h"
#include "swift/ClangImporter/ClangImporter.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/Demangler.h"
#include "swift/Demangling/ManglingFlavor.h"
#include "swift/Frontend/Frontend.h"

#include "clang/APINotes/APINotesManager.h"
#include "clang/APINotes/APINotesReader.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"

#include "llvm/ADT/ScopeExit.h"

#include <algorithm>
#include <cerrno>
#include <sstream>
#include <type_traits>

using namespace lldb;
using namespace lldb_private;
using namespace swift_demangle;

char TypeSystemSwift::ID;
char TypeSystemSwiftTypeRef::ID;
char TypeSystemSwiftTypeRefForExpressions::ID;

namespace lldb_private {

/// swift::ASTContext-less Clang name importer.
class ClangNameImporter {
public:
  ClangNameImporter(swift::LangOptions lang_opts)
      : m_source_manager(FileSystem::Instance().GetVirtualFileSystem()),
        m_diagnostic_engine(m_source_manager) {
    m_compiler_invocation.getLangOptions() = lang_opts;
    m_ast_context.reset(swift::ASTContext::get(
        m_compiler_invocation.getLangOptions(),
        m_compiler_invocation.getTypeCheckerOptions(),
        m_compiler_invocation.getSILOptions(),
        m_compiler_invocation.getSearchPathOptions(),
        m_compiler_invocation.getClangImporterOptions(),
        m_compiler_invocation.getSymbolGraphOptions(),
        m_compiler_invocation.getCASOptions(),
        m_compiler_invocation.getSerializationOptions(),
        m_source_manager, m_diagnostic_engine));
    m_clang_importer = swift::ClangImporter::create(*m_ast_context, "", {}, {});
  }
  std::string ImportName(const clang::NamedDecl *decl) {
    swift::DeclName imported_name = m_clang_importer->importName(decl, {});
    return imported_name.getBaseName().userFacingName().str();
  }

  template <typename IntTy>
  llvm::StringRef ProjectEnumCase(const clang::EnumDecl *decl, IntTy val) {
    for (const auto *enumerator : decl->enumerators()) {
      llvm::APSInt case_val = enumerator->getInitVal();
      if ((case_val.isSigned() &&
           llvm::APSInt::isSameValue(case_val, llvm::APSInt::get(val))) ||
          (case_val.isUnsigned() &&
           llvm::APSInt::isSameValue(case_val, llvm::APSInt::getUnsigned(val))))
        return m_clang_importer->getEnumConstantName(enumerator).str();
    }
    return {};
  }

private:
  swift::CompilerInvocation m_compiler_invocation;
  swift::SourceManager m_source_manager;
  swift::DiagnosticEngine m_diagnostic_engine;
  std::unique_ptr<swift::ASTContext> m_ast_context;
  std::unique_ptr<swift::ClangImporter> m_clang_importer;
};
} // namespace lldb_private

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

NodePointer TypeSystemSwiftTypeRef::FindTypeWithModuleAndIdentifierNode(
    swift::Demangle::NodePointer node) {
  if (!node || node->getKind() != Node::Kind::Type)
    return nullptr;

  NodePointer current = node;
  while (current && current->hasChildren() &&
         current->getFirstChild()->getKind() != Node::Kind::Module) {
    current = current->getFirstChild();
  }
  switch (current->getKind()) {
  case Node::Kind::Structure:
  case Node::Kind::Class:
  case Node::Kind::Enum:
  case Node::Kind::BoundGenericStructure:
  case Node::Kind::BoundGenericClass:
  case Node::Kind::BoundGenericEnum:
    return current;
  default:
    return nullptr;
  }
}

std::string TypeSystemSwiftTypeRef::AdjustTypeForOriginallyDefinedInModule(
    llvm::StringRef mangled_typename) {
  if (mangled_typename.empty())
    return {};

  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_typename);
  swift::Demangle::Demangler dem;
  auto *type_node =
      swift_demangle::GetDemangledTypeMangling(dem, mangled_typename);
  if (!type_node)
    return {};

  TargetSP target_sp(GetTargetWP().lock());
  if (!target_sp)
    return {};

  ModuleList &module_list = target_sp->GetImages();

  // A map from the node containing the module and identifier of a specific type
  // to a node with the modified module and identifier of that type. For
  // example, given the following type:
  //
  // Module "a":
  //
  // @available(...)
  // @_originallyDefinedIn(module: "Other", ...)
  // public struct A { ... }
  // The demangle tree of the mangled name stored in DWARF will be:
  //
  // kind=Global
  //   kind=TypeMangling
  //     kind=Type
  //       kind=Structure
  //         kind=Module, text="Other"
  //         kind=Identifier, text="A"
  //
  // This functions needs to construct the following tree:
  //
  // kind=Global
  //   kind=TypeMangling
  //     kind=Type
  //       kind=Structure
  //         kind=Module, text="a"
  //         kind=Identifier, text="A"
  //
  // type_to_renamed_type_nodes is populated with the nodes in the original tree
  // node that need to be replaced mapping to their replacements. In this
  // example that would be:
  //
  // kind=Structure
  //   kind=Module, text="Other"
  //   kind=Identifier, text="A"
  //
  // mapping to:
  //
  // kind=Structure
  //   kind=Module, text="a"
  //   kind=Identifier, text="A"
  //
  // We can't have a map from module nodes to renamed module nodes because those
  // nodes might be reused elsewhere in the tree.
  llvm::DenseMap<NodePointer, NodePointer> type_to_renamed_type_nodes;

  // Visit the demangle tree and populate type_to_renamed_type_nodes.
  PreOrderTraversal(type_node, [&](NodePointer node) {
    // We're visiting the entire tree, but we only need to examine "Type" nodes.
    if (node->getKind() != Node::Kind::Type)
      return true;

    auto compiler_type = RemangleAsType(dem, node, flavor);
    if (!compiler_type)
      return true;

    // Find the node that contains the module and identifier nodes.
    NodePointer node_with_module_and_name =
        FindTypeWithModuleAndIdentifierNode(node);
    if (!node_with_module_and_name)
      return true;

    auto module_name = node_with_module_and_name->getFirstChild()->getText();
    // Clang types couldn't have been renamed.
    if (module_name == swift::MANGLING_MODULE_OBJC)
      return true;

    // If we already processed this node there's nothing to do (this can happen
    // because nodes are shared in the tree).
    if (type_to_renamed_type_nodes.contains(node_with_module_and_name))
      return true;

    // Look for the imported declarations that indicate the type has moved
    // modules.
    std::vector<ImportedDeclaration> decls;
    module_list.FindImportedDeclarations(GetModule(),
                                         compiler_type.GetMangledTypeName(),
                                         decls, /*find_one=*/true);
    // If there are none there's nothing to do.
    if (decls.empty())
      return true;

    std::vector<lldb_private::CompilerContext> declContext =
        decls[0].GetDeclContext();

    lldbassert(!declContext.empty() &&
               "Unexpected decl context for imported declaration!");
    if (declContext.empty())
      return true;

    auto module_context = declContext[0];

    // If the mangled name's module and module context module match then
    // there's nothing to do.
    if (module_name == module_context.name)
      return true;

    // Construct the node tree that will substituted in.
    NodePointer new_node = dem.createNode(node_with_module_and_name->getKind());
    NodePointer new_module_node = dem.createNodeWithAllocatedText(
        Node::Kind::Module, module_context.name);
    new_node->addChild(new_module_node, dem);
    new_node->addChild(node_with_module_and_name->getLastChild(), dem);

    type_to_renamed_type_nodes[node_with_module_and_name] = new_node;
    return true;
  });

  // If there are no renamed modules, there's nothing to do.
  if (type_to_renamed_type_nodes.empty())
    return mangled_typename.str();

  NodePointer transformed = Transform(dem, type_node, [&](NodePointer node) {
    return type_to_renamed_type_nodes.contains(node)
               ? type_to_renamed_type_nodes[node]
               : node;
  });

  auto mangling = mangleNode(swift_demangle::MangleType(dem, transformed));
  assert(mangling.isSuccess());
  if (!mangling.isSuccess()) {
    LLDB_LOG(GetLog(LLDBLog::Types),
             "[AdjustTypeForOriginallyDefinedInModule] Unexpected mangling "
             "error when mangling adjusted node for type with mangled name {0}",
             mangled_typename);

    return {};
  }

  auto str = mangling.result();
  return str;
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

CompilerType TypeSystemSwiftTypeRef::GetTypeFromTypeMetadataNode(
    llvm::StringRef mangled_name) {
  Demangler dem;
  NodePointer node = dem.demangleSymbol(mangled_name);
  NodePointer type = swift_demangle::NodeAtPath(
      node, {Node::Kind::Global, Node::Kind::TypeMetadata, Node::Kind::Type});
  if (!type)
    return {};
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);
  return RemangleAsType(dem, type, flavor);
}

TypeSP TypeSystemSwiftTypeRef::LookupClangType(StringRef name_ref,
                                               SymbolContext sc) {
  llvm::SmallVector<CompilerContext, 2> decl_context;
  // Make up a decl context for non-nested types.
  decl_context.push_back({CompilerContextKind::AnyType, ConstString(name_ref)});
  return LookupClangType(name_ref, decl_context, /*ignore_modules=*/true, sc);
}

/// Look up one Clang type in a module.
static TypeSP LookupClangType(Module &m,
                              llvm::ArrayRef<CompilerContext> decl_context,
                              bool ignore_modules) {
  auto opts = TypeQueryOptions::e_find_one | TypeQueryOptions::e_module_search;
  if (ignore_modules) {
    opts |= TypeQueryOptions::e_ignore_modules;
  }
  TypeQuery query(decl_context, opts);
  query.SetLanguages(TypeSystemClang::GetSupportedLanguagesForTypes());
  TypeResults results;
  m.FindTypes(query, results);
  return results.GetFirstType();
}

TypeSP TypeSystemSwiftTypeRef::LookupClangType(
    StringRef name_ref, llvm::ArrayRef<CompilerContext> decl_context,
    bool ignore_modules, SymbolContext sc) {
  Module *m = sc.module_sp.get();
  if (!m)
    m = GetModule();
  if (!m)
    return {};
  return ::LookupClangType(const_cast<Module &>(*m), decl_context,
                           ignore_modules);
}

TypeSP TypeSystemSwiftTypeRefForExpressions::LookupClangType(
    StringRef name_ref, llvm::ArrayRef<CompilerContext> decl_context,
    bool ignore_modules, SymbolContext sc) {
  // Check the cache first. Negative results are also cached.
  TypeSP result;
  ConstString name(name_ref);
  if (m_clang_type_cache.Lookup(name.AsCString(), result))
    return result;

  ModuleSP cur_module = sc.module_sp;
  auto lookup = [&](const ModuleSP &m) -> bool {
    // Already visited this.
    if (m == cur_module)
      return true;

    // Don't recursively call into LookupClangTypes() to avoid filling
    // hundreds of image caches with negative results.
    result = ::LookupClangType(const_cast<Module &>(*m), decl_context,
                               ignore_modules);
    // Cache it in the expression context.
    if (result)
      m_clang_type_cache.Insert(name.AsCString(), result);
    return !result;
  };

  // Visit the current module first as a performance optimization heuristic.
  if (cur_module)
    if (!lookup(cur_module))
      return result;

  if (TargetSP target_sp = GetTargetWP().lock())
    target_sp->GetImages().ForEach(lookup);

  return result;
}

/// Find a Clang type by name in module \p M.
CompilerType TypeSystemSwiftTypeRef::LookupClangForwardType(
    StringRef name, llvm::ArrayRef<CompilerContext> decl_context,
    bool ignore_modules) {
  if (TypeSP type = LookupClangType(name, decl_context, ignore_modules))
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

NodePointer TypeSystemSwiftTypeRef::CreateBoundGenericStruct(
    llvm::StringRef name, llvm::StringRef module_name,
    llvm::ArrayRef<NodePointer> type_list_elements,
    swift::Demangle::Demangler &dem) {
  NodePointer type_list = dem.createNode(Node::Kind::TypeList);
  for (auto *type_list_element : type_list_elements)
    type_list->addChild(type_list_element, dem);
  NodePointer identifier = dem.createNode(Node::Kind::Identifier, name);
  NodePointer module = dem.createNode(Node::Kind::Module, module_name);
  NodePointer structure = dem.createNode(Node::Kind::Structure);
  structure->addChild(module, dem);
  structure->addChild(identifier, dem);
  NodePointer type = dem.createNode(Node::Kind::Type);
  type->addChild(structure, dem);
  NodePointer signature = dem.createNode(Node::Kind::BoundGenericStructure);
  signature->addChild(type, dem);
  signature->addChild(type_list, dem);
  NodePointer outer_type = dem.createNode(Node::Kind::Type);
  outer_type->addChild(signature, dem);
  return outer_type;
}

CompilerType
TypeSystemSwiftTypeRef::CreateClangStructType(llvm::StringRef name) {
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(name);

  using namespace swift::Demangle;
  Demangler dem;
  NodePointer module = dem.createNodeWithAllocatedText(
      Node::Kind::Module, swift::MANGLING_MODULE_OBJC);
  NodePointer identifier =
      dem.createNodeWithAllocatedText(Node::Kind::Identifier, name);
  NodePointer nominal = dem.createNode(Node::Kind::Structure);
  nominal->addChild(module, dem);
  nominal->addChild(identifier, dem);
  NodePointer type = dem.createNode(Node::Kind::Type);
  type->addChild(nominal, dem);
  return RemangleAsType(dem, type, flavor);
}

/// Return a demangle tree leaf node representing \p clang_type.
swift::Demangle::NodePointer
TypeSystemSwiftTypeRef::GetClangTypeNode(CompilerType clang_type,
                                         swift::Demangle::Demangler &dem) {
  using namespace swift;
  using namespace swift::Demangle;
  Node::Kind kind = Node::Kind::Structure;
  llvm::StringRef swift_name;
  llvm::StringRef module_name = swift::MANGLING_MODULE_OBJC;
  CompilerType pointee;
  if (clang_type.IsPointerType(&pointee)) {
    clang_type = pointee;
    if (clang_type.IsVoidType()) {
      // Sugar (void *) as "UnsafeMutableRawPointer?".
      NodePointer optional = dem.createNode(Node::Kind::SugaredOptional);
      NodePointer type = dem.createNode(Node::Kind::Type);
      NodePointer module = dem.createNodeWithAllocatedText(Node::Kind::Module,
                                                           swift::STDLIB_NAME);
      NodePointer identifier = dem.createNodeWithAllocatedText(
          Node::Kind::Identifier, clang_type.IsConst()
                                      ? "UnsafeRawPointer"
                                      : "UnsafeMutableRawPointer");
      NodePointer nominal = dem.createNode(kind);
      nominal->addChild(module, dem);
      nominal->addChild(identifier, dem);
      type->addChild(nominal, dem);
      optional->addChild(type, dem);
      return optional;
    }
  }
  if (clang_type.IsAnonymousType())
    return nullptr;
  llvm::StringRef clang_name = clang_type.GetTypeName().GetStringRef();
#define MAP_TYPE(C_TYPE_NAME, C_TYPE_KIND, C_TYPE_BITWIDTH, SWIFT_MODULE_NAME, \
                 SWIFT_TYPE_NAME, CAN_BE_MISSING, C_NAME_MAPPING)              \
  if (clang_name == C_TYPE_NAME) {                                             \
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
    if (auto ts =
            clang_type.GetTypeSystem().dyn_cast_or_null<TypeSystemClang>()) {
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
        {clang_type.GetTypeSystem(), elem_type.getAsOpaquePtr()}, dem);
    if (!element_type)
      return nullptr;
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
  case eTypeClassVector: {
    CompilerType element_type;
    uint64_t size;
    bool is_vector = clang_type.IsVectorType(&element_type, &size);
    if (!is_vector)
      break;

    auto qual_type = ClangUtil::GetQualType(clang_type);
    const auto *ptr = qual_type.getTypePtrOrNull();
    if (!ptr)
      break;

    // Check if this is an extended vector type.
    if (!llvm::isa<clang::DependentSizedExtVectorType>(ptr) &&
        !llvm::isa<clang::ExtVectorType>(ptr))
      break;

    NodePointer element_type_node = GetClangTypeNode(element_type, dem);
    if (!element_type_node)
      return nullptr;
    llvm::SmallVector<NodePointer, 1> elements({element_type_node});
    return CreateBoundGenericStruct("SIMD" + std::to_string(size),
                                    swift::STDLIB_NAME, elements, dem);
  }
  default:
    break;
  }
  NodePointer module =
      dem.createNodeWithAllocatedText(Node::Kind::Module, module_name);
  NodePointer identifier = dem.createNodeWithAllocatedText(
      Node::Kind::Identifier, swift_name.empty() ? clang_name : swift_name);
  NodePointer nominal = dem.createNode(kind);
  nominal->addChild(module, dem);
  nominal->addChild(identifier, dem);
  return pointee ? GetPointerTo(dem, nominal) : nominal;
}

bool 
TypeSystemSwiftTypeRef::IsBuiltinType(CompilerType type) {
  assert(type.GetTypeSystem().isa_and_nonnull<TypeSystemSwift>() &&
         "Unexpected type system!");
  Demangler dem;
  auto *node = GetDemangledType(dem, type.GetMangledTypeName());
  if (!node)
    return false;
  return node->getKind() == Node::Kind::BuiltinTypeName;
}

std::optional<std::pair<NodePointer, NodePointer>>
ExtractTypeNode(NodePointer node) {
  // A type node is expected to have two children.
  if (node->getNumChildren() != 2)
    return {};

  auto *first = node->getChild(0);
  auto *second = node->getChild(1);

  // The second child of a type node is expected to be its identifier.
  if (second->getKind() != Node::Kind::Identifier)
    return {};
  return {{first, second}};
}

static std::optional<llvm::StringRef>
ExtractIdentifierFromLocalDeclName(NodePointer node) {
  assert(node->getKind() == Node::Kind::LocalDeclName &&
         "Expected LocalDeclName node!");

  if (node->getNumChildren() != 2) {
    assert(false && "Local decl name node should have 2 children!");
    return {};
  }

  auto *first = node->getChild(0);
  auto *second = node->getChild(1);

  assert(first->getKind() == Node::Kind::Number && "Expected number node!");

  if (second->getKind() == Node::Kind::Identifier) {
    if (!second->hasText()) {
      assert(false && "Identifier should have text!");
      return {};
    }
    return second->getText();
  }
  assert(false && "Expected second child of local decl name to be its "
                  "identifier!"); 
  return {};
}

static std::optional<std::pair<StringRef, StringRef>>
ExtractIdentifiersFromPrivateDeclName(NodePointer node) {
  assert(node->getKind() == Node::Kind::PrivateDeclName && "Expected PrivateDeclName node!");

  if (node->getNumChildren() != 2) {
    assert(false && "Private decl name node should have 2 children!");
    return {};
  }

  auto *private_discriminator = node->getChild(0);
  auto *type_name = node->getChild(1);

  if (private_discriminator->getKind() != Node::Kind::Identifier) {
    assert(
        false &&
        "Expected first child of private decl name node to be an identifier!");
    return {};
  }
  if (type_name->getKind() != Node::Kind::Identifier) {
    assert(
        false &&
        "Expected second child of private decl name node to be an identifier!");
    return {};
  }

  if (!private_discriminator->hasText() || !type_name->hasText()) {
    assert(false && "Identifier nodes should have text!");
    return {};
  }

  return {{private_discriminator->getText(), type_name->getText()}};
}
/// Builds the decl context of a given node. The decl context is the context in
/// where this type is defined. For example, give a module A with the following 
/// code:
///
/// struct B {
///   class C {
///     ...
///   }
/// }
///
/// The decl context of C would be:
/// <Module A>
///   <AnyType B>
///     <AnyType C>
static bool BuildDeclContext(swift::Demangle::NodePointer node,
                             std::vector<CompilerContext> &context) {
  if (!node)
    return false;
  using namespace swift::Demangle;
  switch (node->getKind()) {
  case Node::Kind::Structure:
  case Node::Kind::Class:
  case Node::Kind::Enum:
  case Node::Kind::Protocol:
  case Node::Kind::ProtocolList:
  case Node::Kind::ProtocolListWithClass:
  case Node::Kind::ProtocolListWithAnyObject:
  case Node::Kind::TypeAlias: {
    if (node->getNumChildren() != 2)
      return false;
    auto *first = node->getChild(0);
    if (!first)
      return false;
    if (first->getKind() == Node::Kind::Module) {
      assert(first->hasText() && "Module node should have text!");
      context.push_back(
          {CompilerContextKind::Module, ConstString(first->getText())});
    } else {
      // If the node's kind is not a module, this is probably a nested type, so
      // build the decl context for the parent type first.
      if (!BuildDeclContext(first, context))
        return false;
    }

    auto *second = node->getChild(1);
    if (!second)
      return false;

    if (second->getKind() == Node::Kind::Identifier) {
      assert(second->hasText() && "Identifier node should have text!");
      context.push_back(
          {CompilerContextKind::AnyType, ConstString(second->getText())});
      return true;
    }
    // If the second child is not an identifier, it could be a private decl
    // name or a local decl name.
    return BuildDeclContext(second, context);
  }
  case Node::Kind::BoundGenericClass:
  case Node::Kind::BoundGenericEnum:
  case Node::Kind::BoundGenericStructure:
  case Node::Kind::BoundGenericProtocol:
  case Node::Kind::BoundGenericOtherNominalType:
  case Node::Kind::BoundGenericTypeAlias: {
    if (node->getNumChildren() != 2)
      return {};
    auto *type = node->getChild(0);
    if (!type || type->getKind() != Node::Kind::Type || !type->hasChildren())
      return {};
    return BuildDeclContext(type->getFirstChild(), context);
  }
  case Node::Kind::Function: {
    if (node->getNumChildren() != 3)
      return false;

    auto *first = node->getChild(0);
    auto *second = node->getChild(1);

    if (first->getKind() == Node::Kind::Module) {
      if (!first->hasText()) {
        assert(false && "Module should have text!");
        return false;
      }
      context.push_back(
          {CompilerContextKind::Module, ConstString(first->getText())});
    } else if (!BuildDeclContext(first, context))
      return false;

    if (second->getKind() == Node::Kind::Identifier) {
      if (!second->hasText()) {
        assert(false && "Identifier should have text!");
        return false;
      }
      context.push_back(
          {CompilerContextKind::Function, ConstString(second->getText())});
    } else if (!BuildDeclContext(second, context))
      return false;

    return true;
  }
  case Node::Kind::LocalDeclName: {
    std::optional<llvm::StringRef> identifier =
        ExtractIdentifierFromLocalDeclName(node);
    if (!identifier)
      return false;

    context.push_back(
        {CompilerContextKind::AnyDeclContext, ConstString(*identifier)});
    return true;
  }

  case Node::Kind::PrivateDeclName: {
    auto discriminator_and_type_name =
        ExtractIdentifiersFromPrivateDeclName(node);

    if (!discriminator_and_type_name)
      return false;
    auto &[private_discriminator, type_name] = *discriminator_and_type_name;

    context.push_back(
        {CompilerContextKind::Namespace, ConstString(private_discriminator)});
    context.push_back(
        {CompilerContextKind::AnyDeclContext, ConstString(type_name)});
    return true;
  }
  default:
    break;
  }
  return false;
}

/// Builds the decl context of a given swift type. See the documentation in the
/// other implementation of BuildDeclContext for more details.
static std::optional<std::vector<CompilerContext>>
BuildDeclContext(llvm::StringRef mangled_name,
                 swift::Demangle::Demangler &dem) {
  std::vector<CompilerContext> context;
  auto *node = GetDemangledType(dem, mangled_name);
  
  /// Builtin names belong to the builtin module, and are stored only with their
  /// mangled name.
  if (node->getKind() == Node::Kind::BuiltinTypeName) {
    context.push_back({CompilerContextKind::Module, ConstString("Builtin")});
    context.push_back(
        {CompilerContextKind::AnyType, ConstString(mangled_name)});
    return std::move(context);
  }

  auto success = BuildDeclContext(node, context);
  if (success)
    return std::move(context);
  return {};
}

/// Detect the AnyObject type alias.
static bool IsAnyObjectTypeAlias(swift::Demangle::NodePointer node) {
  using namespace swift::Demangle;
  if (!node || node->getKind() != Node::Kind::TypeAlias)
    return false;
  if (node->getNumChildren() < 2)
    return false;
  NodePointer module = node->getChild(0);
  if (!module || !module->hasText() || module->getText() != swift::STDLIB_NAME)
    return false;
  NodePointer ident = node->getChild(1);
  if (!ident || !ident->hasText() || ident->getText() != "AnyObject")
    return false;
  return true;
}

/// Build a demangle tree for the builtin AnyObject type.
static swift::Demangle::NodePointer
GetBuiltinAnyObjectNode(swift::Demangle::Demangler &dem) {
  auto proto_list_any = dem.createNode(Node::Kind::ProtocolListWithAnyObject);
  auto proto_list = dem.createNode(Node::Kind::ProtocolList);
  auto type_list = dem.createNode(Node::Kind::TypeList);
  proto_list_any->addChild(proto_list, dem);
  proto_list->addChild(type_list, dem);
  return proto_list_any;
}

/// Builds the decl context to look up clang types with.
static bool
IsClangImportedType(NodePointer node,
                    llvm::SmallVectorImpl<CompilerContext> &decl_context,
                    bool &ignore_modules) {
  if (node->getKind() == Node::Kind::Module && node->hasText() &&
      node->getText() == swift::MANGLING_MODULE_OBJC) {
    ignore_modules = true;
    return true;
  }

  if (node->getNumChildren() != 2 || !node->getLastChild()->hasText())
    return false;

  switch (node->getKind()) {
  case Node::Kind::Structure:
  case Node::Kind::Class:
  case Node::Kind::Enum:
  case Node::Kind::TypeAlias:
    if (!IsClangImportedType(node->getFirstChild(), decl_context,
                             ignore_modules))
      return false;

    // When C++ interop is enabled, Swift enums represent Swift namespaces.
    decl_context.push_back({node->getKind() == Node::Kind::Enum
                                ? CompilerContextKind::Namespace
                                : CompilerContextKind::AnyType,
                            ConstString(node->getLastChild()->getText())});
    return true;
  default:
    return false;
  }
}

std::pair<swift::Demangle::NodePointer, CompilerType>
TypeSystemSwiftTypeRef::ResolveTypeAlias(swift::Demangle::Demangler &dem,
                                         swift::Demangle::NodePointer node,
                                         swift::Mangle::ManglingFlavor flavor,
                                         bool prefer_clang_types) {
  // Hardcode that the Swift.AnyObject type alias always resolves to
  // the builtin AnyObject type.
  if (IsAnyObjectTypeAlias(node))
    return {GetBuiltinAnyObjectNode(dem), {}};

  using namespace swift::Demangle;
  // Try to look this up as a Swift type alias. For each *Swift*
  // type alias there is a debug info entry that has the mangled
  // name as name and the aliased type as a type.
  auto mangling = GetMangledName(dem, node, flavor);
  if (!mangling.isSuccess()) {
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "Failed while mangling type alias (%d:%u)", mangling.error().code,
              mangling.error().line);
    return {{}, {}};
  }
  ConstString mangled(mangling.result());

  auto resolve_clang_type = [&]() -> CompilerType {
    // This is an imported Objective-C type; look it up in the debug info.
    llvm::SmallVector<CompilerContext, 2> decl_context;
    bool ignore_modules = false;
    if (!IsClangImportedType(node, decl_context, ignore_modules))
      return {};

    // Resolve the typedef within the Clang debug info.
    auto clang_type = LookupClangForwardType(mangled.GetStringRef(),
                                             decl_context, ignore_modules);
    if (!clang_type)
      return {};

    return clang_type.GetCanonicalType();
  };

  TypeResults results;
  TypeQuery query(mangled.GetStringRef(), TypeQueryOptions::e_find_one);
  if (!prefer_clang_types) {
    // First check if this type has already been parsed from DWARF.
    if (auto cached = m_swift_type_map.Lookup(mangled.AsCString()))
      results.InsertUnique(cached);
    else if (auto *M = GetModule())
      M->FindTypes(query, results);
    else if (TargetSP target_sp = GetTargetWP().lock()) {
      // Look it up using the conformances in the reflection metadata.
      if (auto *runtime =
              SwiftLanguageRuntime::Get(target_sp->GetProcessSP())) {
        auto ty =
            runtime->ResolveTypeAlias(GetTypeFromMangledTypename(mangled));
        if (ty)
          return {GetDemangledType(dem, ty->GetMangledTypeName()), {}};
        LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), ty.takeError(),
                        "Could not resolve type alias {0}: {1}",
                        mangled.AsCString());
      }

      // Do an even more expensive global search.
      target_sp->GetImages().FindTypes(/*search_first=*/nullptr, query,
                                       results);
    } else {
      LLDB_LOGF(GetLog(LLDBLog::Types),
                "No module. Couldn't resolve type alias %s",
                mangled.AsCString());
      return {{}, {}};
    }
  }

  if (prefer_clang_types || !results.Done(query)) {
    // No Swift type found -- this could be a Clang typedef.  This
    // check is not done earlier because a Clang typedef that points
    // to a builtin type, e.g., "typedef unsigned uint32_t", could
    // end up pointing to a *Swift* type!
    auto clang_type = resolve_clang_type();
    if (!clang_type)
      LLDB_LOGF(GetLog(LLDBLog::Types),
                "Couldn't resolve type alias %s as a Swift or clang type.",
                mangled.AsCString());
    return {{}, clang_type};
  }

  TypeSP type = results.GetFirstType();
  if (!type) {
    LLDB_LOGF(GetLog(LLDBLog::Types), "Found empty type alias %s",
              mangled.AsCString());
    return {{}, {}};
  }

  // DWARFASTParserSwift stashes the desugared mangled name of a
  // type alias into the Type's name field.
  ConstString desugared_name = type->GetName();
  if (!isMangledName(desugared_name.GetStringRef())) {
    // The name is not mangled, this might be a Clang typedef, try
    // to look it up as a clang type.
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "Found non-Swift type alias %s, looking it up as clang type.",
              mangled.AsCString());
    auto clang_type = resolve_clang_type();
    if (!clang_type)
      LLDB_LOGF(GetLog(LLDBLog::Types), "Could not find a clang type for %s.",
                mangled.AsCString());
    return {{}, clang_type};
  }
  NodePointer n = GetDemangledType(dem, desugared_name.GetStringRef());
  if (!n) {
    LLDB_LOG(GetLog(LLDBLog::Types), "Unrecognized demangling {0}",
             desugared_name.AsCString());
    return {{}, {}};
  }
  return {n, {}};
}

std::optional<TypeSystemSwift::TupleElement>
TypeSystemSwiftTypeRef::GetTupleElement(lldb::opaque_compiler_type_t type,
                                        size_t idx) {
  TupleElement result;
  using namespace swift::Demangle;
  Demangler dem;
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(AsMangledName(type));
  NodePointer node =
      TypeSystemSwiftTypeRef::DemangleCanonicalOutermostType(dem, type);
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
      result.element_type = RemangleAsType(dem, n, flavor);
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

llvm::Expected<swift::Demangle::NodePointer>
TypeSystemSwiftTypeRef::TryTransform(
    swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
    std::function<llvm::Expected<swift::Demangle::NodePointer>(
        swift::Demangle::NodePointer)>
        fn) {
  if (!node)
    return node;
  using namespace swift::Demangle;
  llvm::SmallVector<NodePointer, 2> children;
  bool changed = false;
  for (NodePointer child : *node) {
    llvm::Expected<NodePointer> transformed_or_err = TryTransform(dem, child, fn);
    if (!transformed_or_err)
      return transformed_or_err.takeError();
    NodePointer transformed = *transformed_or_err;
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

void TypeSystemSwiftTypeRef::PreOrderTraversal(
    swift::Demangle::NodePointer node,
    std::function<bool(swift::Demangle::NodePointer)> visitor) {
  if (!node)
    return;
  if (!visitor(node))
    return;

  for (swift::Demangle::NodePointer child : *node) 
    PreOrderTraversal(child, visitor);
}

/// Desugar a sugared type.
static swift::Demangle::NodePointer
Desugar(swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
        swift::Demangle::Node::Kind bound_kind,
        swift::Demangle::Node::Kind kind, llvm::StringRef name) {
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

  assert(node->getNumChildren() >= 1 && node->getNumChildren() <= 2 &&
         "Sugared types should only have 1 or 2 children");
  for (NodePointer child : *node) {
    NodePointer type = dem.createNode(Node::Kind::Type);
    type->addChild(child, dem);
    type_list->addChild(type, dem);
  }
  desugared->addChild(type, dem);
  desugared->addChild(type_list, dem);
  return desugared;
}

static swift::Demangle::NodePointer Desugar(swift::Demangle::Demangler &dem,
                                            swift::Demangle::NodePointer node) {
  assert(node);
  auto kind = node->getKind();
  switch (kind) {
  case Node::Kind::SugaredOptional:
    // FIXME: Factor these three cases out.
    assert(node->getNumChildren() == 1);
    if (node->getNumChildren() != 1)
      return node;
    return Desugar(dem, node, Node::Kind::BoundGenericEnum, Node::Kind::Enum,
                   "Optional");
  case Node::Kind::SugaredArray: {
    assert(node->getNumChildren() == 1);
    if (node->getNumChildren() != 1)
      return node;
    return Desugar(dem, node, Node::Kind::BoundGenericStructure,
                   Node::Kind::Structure, "Array");
  }
  case Node::Kind::SugaredDictionary:
    // FIXME: This isn't covered by any test.
    assert(node->getNumChildren() == 2);
    if (node->getNumChildren() != 2)
      return node;
    return Desugar(dem, node, Node::Kind::BoundGenericStructure,
                   Node::Kind::Structure, "Dictionary");
  case Node::Kind::SugaredParen:
    assert(node->getNumChildren() == 1);
    if (node->getNumChildren() != 1)
      return node;
    return node->getFirstChild();
  default:
    return node;
  }
}

swift::Demangle::NodePointer
TypeSystemSwiftTypeRef::DesugarNode(swift::Demangle::Demangler &dem,
                                    swift::Demangle::NodePointer node) {
  using namespace swift::Demangle;
  return TypeSystemSwiftTypeRef::Transform(
      dem, node, [&](NodePointer node) { return Desugar(dem, node); });
}

swift::Demangle::NodePointer
TypeSystemSwiftTypeRef::Canonicalize(swift::Demangle::Demangler &dem,
                                     swift::Demangle::NodePointer node,
                                     swift::Mangle::ManglingFlavor flavor) {
  assert(node);
  node = Desugar(dem, node);
  auto kind = node->getKind();
  switch (kind) {
  case Node::Kind::BoundGenericTypeAlias:
  case Node::Kind::TypeAlias: {
    // Safeguard against cyclic aliases.
    for (unsigned alias_depth = 0; alias_depth < 64; ++alias_depth) {
      auto node_clangtype = ResolveTypeAlias(dem, node, flavor);
      if (CompilerType clang_type = node_clangtype.second) {
        if (auto result = GetClangTypeNode(clang_type, dem))
          return result;
        // Failed to convert that clang type into a demangle node.
        return node;
      }
      if (!node_clangtype.first)
        return node;
      if (node_clangtype.first == node)
        return node;
      node = node_clangtype.first;
      if (node->getKind() != Node::Kind::BoundGenericTypeAlias &&
          node->getKind() != Node::Kind::TypeAlias)
        // Resolve any type aliases in the resolved type.
        return GetCanonicalNode(dem, node, flavor);
      // This type alias resolved to another type alias.
    }
    // Hit the safeguard limit.
    return node;
  }
  default:
    return node;
  }
  return node;
}

swift::Demangle::NodePointer
TypeSystemSwiftTypeRef::GetCanonicalNode(swift::Demangle::Demangler &dem,
                                         swift::Demangle::NodePointer node,
                                         swift::Mangle::ManglingFlavor flavor) {
  if (!node)
    return nullptr;
  // This is a pre-order traversal, which is necessary to resolve
  // generic type aliases that bind other type aliases in one go,
  // instead of first resolving the bound type aliases.  Debug Info
  // will have a record for SomeAlias<SomeOtherAlias> but not
  // SomeAlias<WhatSomeOtherAliasResolvesTo> because it tries to
  // preserve all sugar.
  using namespace swift::Demangle;
  node = Canonicalize(dem, node, flavor);

  llvm::SmallVector<NodePointer, 2> children;
  bool changed = false;
  for (NodePointer child : *node) {
    NodePointer transformed_child = GetCanonicalNode(dem, child, flavor);
    changed |= (child != transformed_child);
    children.push_back(transformed_child);
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
  return node;
}

/// Return the demangle tree representation of this type's canonical
/// (type aliases resolved) type.
swift::Demangle::NodePointer TypeSystemSwiftTypeRef::GetCanonicalDemangleTree(
    swift::Demangle::Demangler &dem, StringRef mangled_name) {
  auto *node = dem.demangleSymbol(mangled_name);
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

  return GetCanonicalNode(dem, node, flavor);
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
    // The order is significant since some of these decl kinds are also
    // TagDecls.
    if (llvm::isa<clang::TypedefNameDecl>(clang_decl))
      // FIXME: this should pass the correct context.
      swift_name = ExtractSwiftName(reader->lookupTypedef(default_name));
    else if (llvm::isa<clang::EnumConstantDecl>(clang_decl))
      swift_name = ExtractSwiftName(reader->lookupEnumConstant(default_name));
    else if (llvm::isa<clang::ObjCInterfaceDecl>(clang_decl))
      swift_name = ExtractSwiftName(reader->lookupObjCClassInfo(default_name));
    else if (llvm::isa<clang::ObjCProtocolDecl>(clang_decl))
      swift_name =
          ExtractSwiftName(reader->lookupObjCProtocolInfo(default_name));
    else if (llvm::isa<clang::TagDecl>(clang_decl))
      // FIXME: this should pass the correct context.
      swift_name = ExtractSwiftName(reader->lookupTag(default_name));
    else {
      assert(false && "unhandled clang decl kind");
    }
    if (!swift_name.empty())
      return swift_name;
  }
  // Else we must go through ClangImporter to apply the automatic
  // swiftification rules.
  if (auto *importer = GetNameImporter())
    return importer->ImportName(named_decl);
  return {};
}

static bool IsImportedType(swift::Demangle::NodePointer node) {
  if (!node)
    return false;
  if (node->hasText() && node->getText() == swift::MANGLING_MODULE_OBJC)
    return true;
  if (node->hasChildren())
    return IsImportedType(node->getFirstChild());
  return false;
}

swift::Demangle::NodePointer TypeSystemSwiftTypeRef::GetSwiftified(
    swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
    swift::Mangle::ManglingFlavor flavor, bool resolve_objc_module) {
  auto mangling = GetMangledName(dem, node, flavor);
  if (!mangling.isSuccess()) {
    LLDB_LOGF(GetLog(LLDBLog::Types), "Failed while getting swiftified (%d:%u)",
              mangling.error().code, mangling.error().line);
    return node;
  }

  llvm::SmallVector<CompilerContext, 2> decl_context;
  bool ignore_modules = false;
  if (!IsClangImportedType(node, decl_context, ignore_modules))
    return node;

  // This is an imported Objective-C type; look it up in the
  // debug info.
  TypeSP clang_type =
      LookupClangType(mangling.result(), decl_context, ignore_modules);
  if (!clang_type)
    return node;

  // Extract the toplevel Clang module name from the debug info.
  std::vector<CompilerContext> DeclCtx = clang_type->GetDeclContext();
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
  auto clang_ts =
      compiler_type.GetTypeSystem().dyn_cast_or_null<TypeSystemClang>();
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
    swift::Mangle::ManglingFlavor flavor, bool resolve_objc_module) {
  using namespace swift::Demangle;
  return Transform(dem, node, [&](NodePointer node) {
    NodePointer canonical = node;
    auto kind = node->getKind();
    switch (kind) {
    case Node::Kind::Class:
    case Node::Kind::Structure:
    case Node::Kind::TypeAlias:
      return GetSwiftified(dem, node, flavor, resolve_objc_module);

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
      if (node->getNumChildren() == 1) {
        return Desugar(dem, node, Node::Kind::BoundGenericEnum,
                       Node::Kind::Enum, "Optional");
      }
      return node;
    case Node::Kind::SugaredArray:
      if (node->getNumChildren() == 1) {
        return Desugar(dem, node, Node::Kind::BoundGenericStructure,
                       Node::Kind::Structure, "Array");
      }
      return node;
    case Node::Kind::SugaredDictionary:
      if (node->getNumChildren() == 2) {
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
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

  auto *node = dem.demangleSymbol(mangled_name);
  return GetNodeForPrintingImpl(dem, node, flavor, resolve_objc_module);
}

/// Determine wether this demangle tree contains a node of kind \c kind and with
/// text \c text (if provided).
static bool Contains(swift::Demangle::NodePointer node,
                     swift::Demangle::Node::Kind kind,
                     llvm::StringRef text = "") {
  if (!node)
    return false;

  if (node->getKind() == kind) {
    if (text.empty())
      return true;
    if (!node->hasText())
      return false;
    return node->getText() == text;
  }

  for (swift::Demangle::NodePointer child : *node)
    if (Contains(child, kind, text))
      return true;

  return false;
}

static bool ProtocolCompositionContainsSingleObjcProtocol(
    swift::Demangle::NodePointer node) {
  // Kind=ProtocolList
  // Kind=TypeList
  //   Kind=Type
  //     Kind=Protocol
  //       Kind=Module, text="__C"
  //       Kind=Identifier, text="SomeIdentifier"
  if (node->getKind() != Node::Kind::ProtocolList)
    return false;
  NodePointer type_list = node->getFirstChild();
  if (type_list->getKind() != Node::Kind::TypeList ||
      type_list->getNumChildren() != 1)
    return false;
  NodePointer type = type_list->getFirstChild();
  return Contains(type, Node::Kind::Module, swift::MANGLING_MODULE_OBJC);
}

/// Determine wether this demangle tree contains a generic type parameter.
static bool ContainsGenericTypeParameter(swift::Demangle::NodePointer node) {
  return swift_demangle::FindIf(node, [](NodePointer node) {
    return node->getKind() ==
           swift::Demangle::Node::Kind::DependentGenericParamType;
  });
}

/// Collect TypeInfo flags from a demangle tree. For most attributes
/// this can stop scanning at the outmost type, however in order to
/// determine whether a node is generic or not, it needs to visit all
/// nodes. The \p generic_walk argument specifies that the primary
/// attributes have been collected and that we only look for generics.
uint32_t TypeSystemSwiftTypeRef::CollectTypeInfo(
    swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
    swift::Mangle::ManglingFlavor flavor, bool &unresolved_typealias) {
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
      swift_flags |= CollectTypeInfo(dem, GetClangTypeNode(clang_type, dem),
                                     flavor, unresolved_typealias);
      return;
    }
  };

  using namespace swift::Demangle;
  switch (node->getKind()) {
  case Node::Kind::SugaredOptional:
    swift_flags |= eTypeHasChildren | eTypeHasValue | eTypeIsEnumeration;
    break;
  case Node::Kind::SugaredArray:
  case Node::Kind::SugaredDictionary:
    swift_flags |= eTypeHasChildren | eTypeIsStructUnion;
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
    swift_flags |= eTypeHasValue;
    break;

  case Node::Kind::ImplFunctionType:
    // Bug-for-bug-compatibility. Not sure if this is correct.
    swift_flags |= eTypeIsPointer | eTypeHasValue;
    return swift_flags;
  case Node::Kind::BoundGenericFunction:
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
      else if (node->getText().starts_with(swift::BUILTIN_TYPE_NAME_FLOAT) ||
               node->getText().starts_with(swift::BUILTIN_TYPE_NAME_FLOAT_PPC))
        swift_flags |= eTypeIsFloat | eTypeIsScalar;
      else if (node->getText().starts_with(swift::BUILTIN_TYPE_NAME_VEC))
        swift_flags |= eTypeHasChildren | eTypeIsVector;
      else if (node->getText().starts_with(swift::BUILTIN_TYPE_NAME_INT) ||
               node->getText().starts_with(swift::BUILTIN_TYPE_NAME_WORD))
        swift_flags |= eTypeIsInteger | eTypeIsScalar;
    }
    break;
  case Node::Kind::Tuple:
    swift_flags |= eTypeHasChildren | eTypeIsTuple;
    break;
  case Node::Kind::BoundGenericEnum:
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
    if (module->hasText() && module->getText() == swift::MANGLING_MODULE_OBJC) {
      swift_flags |= eTypeHasValue /*| eTypeIsObjC*/;
    }
    break;
    }
    case Node::Kind::BoundGenericStructure:
    case Node::Kind::Structure: {
    swift_flags |= eTypeHasChildren | eTypeIsStructUnion;
    if (node->getNumChildren() != 2)
      break;
    if (node->getKind() == Node::Kind::Structure) {
      auto module = node->getChild(0);
      auto ident = node->getChild(1);
      // Builtin types.
      if (module->hasText() && module->getText() == swift::STDLIB_NAME) {
        if (ident->hasText() &&
            ident->getText().starts_with(swift::BUILTIN_TYPE_NAME_INT))
          swift_flags |= eTypeIsScalar | eTypeIsInteger;
        else if (ident->hasText() &&
                 ident->getText().starts_with(swift::BUILTIN_TYPE_NAME_FLOAT))
          swift_flags |= eTypeIsScalar | eTypeIsFloat;
      }
    } else {
      // Variadic generic types.
      auto typelist = node->getChild(1);
      bool ignore;
      auto param_flags = CollectTypeInfo(dem, typelist, flavor, ignore);
      swift_flags |= param_flags & eTypeIsPack;
    }

    // Clang-imported types.
    llvm::SmallVector<CompilerContext, 2> decl_context;
    bool ignore_modules = false;
    if (!IsClangImportedType(node, decl_context, ignore_modules))
      break;

    auto mangling = GetMangledName(dem, node, flavor);
    if (!mangling.isSuccess()) {
      LLDB_LOGF(GetLog(LLDBLog::Types),
                "Failed mangling while collecting type infos (%d:%u)",
                mangling.error().code, mangling.error().line);

      return {};
    }
    // Resolve the typedef within the Clang debug info.
    auto clang_type =
        LookupClangForwardType(mangling.result(), decl_context, ignore_modules);
    collect_clang_type(clang_type.GetCanonicalType());
    return swift_flags;
    }
    case Node::Kind::BoundGenericClass:
    case Node::Kind::Class:
      swift_flags |= eTypeHasChildren | eTypeIsClass | eTypeHasValue |
                     eTypeInstanceIsPointer;
      break;

    case Node::Kind::BoundGenericOtherNominalType:
      swift_flags |= eTypeHasValue;
      break;

    case Node::Kind::BoundGenericProtocol:
    case Node::Kind::Protocol:
      swift_flags |= eTypeHasChildren | eTypeIsStructUnion | eTypeIsProtocol;
      break;
    case Node::Kind::ProtocolList:
    case Node::Kind::ProtocolListWithClass:
    case Node::Kind::ProtocolListWithAnyObject:
      swift_flags |= eTypeIsProtocol;
      // Bug-for-bug-compatibility.
      swift_flags |= eTypeHasChildren | eTypeIsStructUnion;
      if (ProtocolCompositionContainsSingleObjcProtocol(node))
        swift_flags |= eTypeIsObjC | eTypeHasValue;
      break;

    case Node::Kind::ExistentialMetatype:
      swift_flags |= eTypeIsProtocol;
      LLVM_FALLTHROUGH;
    case Node::Kind::Metatype:
      swift_flags |= eTypeIsMetatype | eTypeHasValue;
      break;

    case Node::Kind::InOut:
      swift_flags |= eTypeHasChildren | eTypeIsReference | eTypeHasValue;
      break;

    case Node::Kind::BoundGenericTypeAlias:
    case Node::Kind::TypeAlias: {
      // Bug-for-bug compatibility.
      // swift_flags |= eTypeIsTypedef;
      auto node_clangtype = ResolveTypeAlias(dem, node, flavor);
      if (CompilerType clang_type = node_clangtype.second) {
        collect_clang_type(clang_type);
        return swift_flags;
      }
      if (!node_clangtype.first) {
        // If this is a typealias defined in the expression evaluator,
        // then we don't have debug info to resolve it from.
        unresolved_typealias = true;
      }
      swift_flags |= CollectTypeInfo(dem, node_clangtype.first, flavor,
                                     unresolved_typealias);
      return swift_flags;
    }
    case Node::Kind::PackExpansion:
      swift_flags |= eTypeIsPack;
      break;
    case Node::Kind::SILPackDirect:
    case Node::Kind::SILPackIndirect:
      swift_flags |= eTypeIsPack;
      return swift_flags;
    default:
      break;
    }

  // If swift_flags were collected we're done here except for
  // determining whether the type is generic.
  if (swift_flags != eTypeIsSwift) {
    if (ContainsGenericTypeParameter(node))
      swift_flags |= eTypeHasUnboundGeneric;
    if (swift_demangle::FindIf(node, [](NodePointer node) {
          return node->getKind() == swift::Demangle::Node::Kind::DynamicSelf;
        }))
      swift_flags |= eTypeHasDynamicSelf;
    return swift_flags;
  }

  // Visit the child nodes.
  for (auto *child : *node)
    swift_flags |= CollectTypeInfo(dem, child, flavor, unresolved_typealias);

  return swift_flags;
}

CompilerType
TypeSystemSwift::GetInstanceType(CompilerType compiler_type,
                                 ExecutionContextScope *exe_scope) {
  auto ts = compiler_type.GetTypeSystem();
  if (auto tr = ts.dyn_cast_or_null<TypeSystemSwiftTypeRef>())
    return tr->GetInstanceType(compiler_type.GetOpaqueQualType(), exe_scope);
  if (auto ast = ts.dyn_cast_or_null<SwiftASTContext>())
    return ast->GetInstanceType(compiler_type.GetOpaqueQualType(), exe_scope);
  return {};
}

TypeSystemSwiftTypeRef::TypeSystemSwiftTypeRef() {}

TypeSystemSwiftTypeRef::~TypeSystemSwiftTypeRef() {}

TypeSystemSwiftTypeRef::TypeSystemSwiftTypeRef(Module &module) {
  m_module = &module;
  {
    llvm::raw_string_ostream ss(m_description);
    ss << "TypeSystemSwiftTypeRef(\"";
    module.GetDescription(ss, eDescriptionLevelBrief);
    ss << "\")";
  }
  LLDB_LOGF(GetLog(LLDBLog::Types), "%s::TypeSystemSwiftTypeRef()",
            m_description.c_str());
}

TypeSystemSwiftTypeRefForExpressions::TypeSystemSwiftTypeRefForExpressions(
    lldb::LanguageType language, Target &target, bool repl, bool playground,
    const char *extra_options)
    : m_target_wp(target.shared_from_this()),
      m_persistent_state_up(new SwiftPersistentExpressionState) {
  m_description = "TypeSystemSwiftTypeRefForExpressions";
  LLDB_LOGF(GetLog(LLDBLog::Types),
            "%s::TypeSystemSwiftTypeRefForExpressions()",
            m_description.c_str());
  // Is this a REPL or Playground?
  assert(!repl && !playground && !extra_options && "use SetCompilerOptions()");
  if (repl || playground || extra_options) {
    SymbolContext global_sc(target.shared_from_this(),
                            target.GetExecutableModule());
    const char *key = DeriveKeyFor(global_sc);
    m_swift_ast_context_map.insert(
        {key,
         {SwiftASTContext::CreateInstance(
              global_sc,
              *const_cast<TypeSystemSwiftTypeRefForExpressions *>(this), repl,
              playground, extra_options),
          0}});
  }
}

TypeSystemSwiftTypeRefForExpressionsSP
TypeSystemSwiftTypeRefForExpressions::GetForTarget(Target &target) {
  auto type_system_or_err =
      target.GetScratchTypeSystemForLanguage(eLanguageTypeSwift);
  if (!type_system_or_err || !type_system_or_err->get()) {
    llvm::consumeError(type_system_or_err.takeError());
    return {};
  }
  return std::static_pointer_cast<TypeSystemSwiftTypeRefForExpressions>(
      *type_system_or_err);
}

TypeSystemSwiftTypeRefForExpressionsSP
TypeSystemSwiftTypeRefForExpressions::GetForTarget(TargetSP target) {
  return target ? GetForTarget(*target) : nullptr;
}

void TypeSystemSwiftTypeRef::NotifyAllTypeSystems(
    std::function<void(lldb::TypeSystemSP)> fn) {
  // Grab the list of typesystems while holding the lock.
  std::vector<TypeSystemSP> typesystems;
  {
    std::lock_guard<std::mutex> guard(m_swift_ast_context_lock);
    for (auto &it : m_swift_ast_context_map)
      typesystems.push_back(it.second.typesystem);
  }
  // Notify all SwiftASTContexts.
  for (auto &ts_sp : typesystems)
    fn(ts_sp);
}

void TypeSystemSwiftTypeRefForExpressions::ModulesDidLoad(
    ModuleList &module_list) {
  ++m_generation;
  m_clang_type_cache.Clear();
  NotifyAllTypeSystems([&](TypeSystemSP ts_sp) {
    if (auto swift_ast_ctx =
            llvm::dyn_cast_or_null<SwiftASTContextForExpressions>(ts_sp.get()))
      swift_ast_ctx->ModulesDidLoad(module_list);
  });
}

Status TypeSystemSwiftTypeRefForExpressions::PerformCompileUnitImports(
    const SymbolContext &sc) {
  Status status;
  lldb::ProcessSP process_sp;
  if (auto target_sp = sc.target_sp)
    process_sp = target_sp->GetProcessSP();

  if (auto swift_ast_ctx = GetSwiftASTContextOrNull(sc))
    swift_ast_ctx->PerformCompileUnitImports(sc, process_sp, status);
  return status;
}

UserExpression *TypeSystemSwiftTypeRefForExpressions::GetUserExpression(
    llvm::StringRef expr, llvm::StringRef prefix, SourceLanguage language,
    Expression::ResultType desired_type,
    const EvaluateExpressionOptions &options, ValueObject *ctx_obj) {
  TargetSP target_sp = GetTargetWP().lock();
  if (!target_sp)
    return nullptr;
  if (ctx_obj != nullptr) {
    lldbassert(false &&
               "Swift doesn't support 'evaluate in the context of an object'.");
    return nullptr;
  }

  return new SwiftUserExpression(*target_sp.get(), expr, prefix, language,
                                 desired_type, options);
}

PersistentExpressionState *
TypeSystemSwiftTypeRefForExpressions::GetPersistentExpressionState() {
  return m_persistent_state_up.get();
}

ConstString TypeSystemSwiftTypeRef::GetSwiftModuleFor(const SymbolContext &sc) {
  if (sc.function) {
    std::vector<CompilerContext> decl_ctx = sc.function->GetCompilerContext();
    for (auto &ctx : decl_ctx)
      if (ctx.kind == CompilerContextKind::Module)
        return ctx.name;
  }
  return {};
}

const char *TypeSystemSwiftTypeRef::DeriveKeyFor(const SymbolContext &sc) {
  if (sc.comp_unit && sc.comp_unit->GetLanguage() == eLanguageTypeSwift)
    if (ConstString name = GetSwiftModuleFor(sc))
      return name.GetCString();

  // Otherwise create a catch-all context per unique triple.
  if (sc.module_sp)
    return ConstString(sc.module_sp->GetArchitecture().GetTriple().str()).AsCString();

  return nullptr;
}

SymbolContext TypeSystemSwiftTypeRef::GetSymbolContext(
    ExecutionContextScope *exe_scope) const {
  if (!exe_scope)
    return {};
  ExecutionContext exe_ctx;
  exe_scope->CalculateExecutionContext(exe_ctx);
  return GetSymbolContext(&exe_ctx);
}

SymbolContext TypeSystemSwiftTypeRef::GetSymbolContext(
    const ExecutionContext *exe_ctx) const {
  SymbolContext sc;
  if (exe_ctx) {
    // The SymbolContext is a Function, which outlives the stack
    // frame, so not holding on to the shared pointer is safe here.
    auto stack_frame = exe_ctx->GetFramePtr();
    if (stack_frame)
      sc = stack_frame->GetSymbolContext(eSymbolContextFunction);
  }
  if (!sc.module_sp)
    if (auto target_sp = GetTargetWP().lock())
      sc = SymbolContext(target_sp, target_sp->GetExecutableModule());

  return sc;
}

SwiftASTContextSP
TypeSystemSwiftTypeRef::GetSwiftASTContext(const SymbolContext &sc) const {
  if (!sc.module_sp) {
    LLDB_LOGV(GetLog(LLDBLog::Types),
              "Cannot create a SwiftASTContext without an execution context");
    return nullptr;
  }

  std::lock_guard<std::mutex> guard(m_swift_ast_context_lock);
  // There is only one per-module context.
  const char *key = nullptr;
  // Look up the SwiftASTContext in the cache.
  auto it = m_swift_ast_context_map.find(key);
  if (it != m_swift_ast_context_map.end()) {
    // SwiftASTContext::CreateInstance() returns a nullptr on failure,
    // there is no point in trying to initialize when that happens.
    if (!it->second.typesystem)
      return nullptr;
    return std::static_pointer_cast<SwiftASTContext>(it->second.typesystem);
  }

  // Create a new SwiftASTContextForExpressions.
  TypeSystemSP ts = SwiftASTContext::CreateInstance(
      sc, *const_cast<TypeSystemSwiftTypeRef *>(this));
  m_swift_ast_context_map.insert({key, {ts, 0}});

  auto swift_ast_context = std::static_pointer_cast<SwiftASTContext>(ts);
  return swift_ast_context;
}

SwiftASTContextSP TypeSystemSwiftTypeRefForExpressions::GetSwiftASTContext(
    const SymbolContext &sc) const {
  if (!sc.module_sp) {
    LLDB_LOGV(GetLog(LLDBLog::Types),
              "Cannot create a SwiftASTContext without an execution context");
    return nullptr;
  }

  // Compute the cache key.
  const char *key = DeriveKeyFor(sc);
  unsigned char retry_count = 0;

  // Look up the SwiftASTContext in the cache.
  TypeSystemSP ts;
  {
    std::lock_guard<std::mutex> guard(m_swift_ast_context_lock);
    auto it = m_swift_ast_context_map.find(key);
    if (it != m_swift_ast_context_map.end()) {
      retry_count = it->second.retry_count + 1;
      // SwiftASTContext::CreateInstance() returns a nullptr on failure,
      // there is no point in trying to initialize when that happens.
      if (!it->second.typesystem)
        return nullptr;
      auto swift_ast_ctx =
          std::static_pointer_cast<SwiftASTContext>(it->second.typesystem);
      if (!swift_ast_ctx->HasFatalErrors())
        return swift_ast_ctx;

      // Recreate the SwiftASTContext if it has developed fatal errors. Any
      // clients holding on to the old context via a CompilerType will keep its
      // shared_ptr alive.
      if (retry_count > 3) {
        LLDB_LOG(GetLog(LLDBLog::Types), "maximum number of retries reached");
        return nullptr;
      }

      m_swift_ast_context_map.erase(key);
      LLDB_LOG(GetLog(LLDBLog::Types),
               "Recreating SwiftASTContext due to fatal errors (retry #{0}).",
               retry_count);
    }

    // Create a new SwiftASTContextForExpressions.
    ts = SwiftASTContext::CreateInstance(
        sc, *const_cast<TypeSystemSwiftTypeRefForExpressions *>(this), m_repl,
        m_playground, m_compiler_options);
    m_swift_ast_context_map.insert({key, {ts, retry_count}});
  }

  // Now perform the initial imports. This step can be very expensive.
  auto swift_ast_context = std::static_pointer_cast<SwiftASTContext>(ts);
  if (!swift_ast_context)
    return nullptr;
  assert(llvm::isa<SwiftASTContextForExpressions>(swift_ast_context.get()));

  auto perform_initial_import = [&](const SymbolContext &sc) {
    Status error;
    lldb::ProcessSP process_sp;
    TargetSP target_sp = GetTargetWP().lock();
    if (target_sp)
      process_sp = target_sp->GetProcessSP();
    swift_ast_context->PerformCompileUnitImports(sc, process_sp, error);
    if (error.Fail() && target_sp)
      if (StreamSP errs_sp = target_sp->GetDebugger().GetAsyncErrorStream())
        errs_sp->Printf(
            "Could not import Swift modules for translation unit: %s",
            error.AsCString());
  };

  perform_initial_import(sc);
  return swift_ast_context;
}

SwiftASTContextSP TypeSystemSwiftTypeRef::GetSwiftASTContextOrNull(
    const SymbolContext &sc) const {
  std::lock_guard<std::mutex> guard(m_swift_ast_context_lock);
  const char *key = nullptr;
  auto it = m_swift_ast_context_map.find(key);
  if (it != m_swift_ast_context_map.end())
    return std::static_pointer_cast<SwiftASTContext>(it->second.typesystem);
  return {};
}

SwiftASTContextSP TypeSystemSwiftTypeRefForExpressions::GetSwiftASTContextOrNull(
    const SymbolContext &sc) const {
  const char *key = DeriveKeyFor(sc);

  std::lock_guard<std::mutex> guard(m_swift_ast_context_lock);
  auto it = m_swift_ast_context_map.find(key);
  if (it != m_swift_ast_context_map.end())
    return std::static_pointer_cast<SwiftASTContext>(it->second.typesystem);
  return {};
}

SwiftDWARFImporterForClangTypes &
TypeSystemSwiftTypeRef::GetSwiftDWARFImporterForClangTypes() {
  if (!m_dwarf_importer_for_clang_types_up)
    m_dwarf_importer_for_clang_types_up =
        std::make_unique<SwiftDWARFImporterForClangTypes>(*this);
  assert(m_dwarf_importer_for_clang_types_up);
  return *m_dwarf_importer_for_clang_types_up;
}

ClangNameImporter *TypeSystemSwiftTypeRef::GetNameImporter() const {
  if (!m_name_importer_up) {
    swift::LangOptions lang_opts;
    lang_opts.setTarget(GetTriple());
    m_name_importer_up = std::make_unique<ClangNameImporter>(lang_opts);
  }
  assert(m_name_importer_up);
  return m_name_importer_up.get();
}

llvm::Triple TypeSystemSwiftTypeRef::GetTriple() const {
  if (auto *module = GetModule())
    return module->GetArchitecture().GetTriple();
  else if (auto target_sp = GetTargetWP().lock())
    return target_sp->GetArchitecture().GetTriple();
  LLDB_LOGF(
      GetLog(LLDBLog::Types),
      "Cannot determine triple when no Module or no Target is available.");
  return {};
}

void TypeSystemSwiftTypeRef::SetTriple(const SymbolContext &sc,
                                       const llvm::Triple triple) {
  // This function appears to be only called via
  // Module::SetArchitecture(ArchSpec).
  if (auto swift_ast_context = GetSwiftASTContextOrNull(sc))
    swift_ast_context->SetTriple(sc, triple);
}

void TypeSystemSwiftTypeRef::ClearModuleDependentCaches() {
  NotifyAllTypeSystems([&](TypeSystemSP ts_sp) {
    if (auto swift_ast_ctx =
            llvm::dyn_cast_or_null<SwiftASTContext>(ts_sp.get()))
      swift_ast_ctx->ClearModuleDependentCaches();
  });
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

void *TypeSystemSwiftTypeRef::ReconstructType(opaque_compiler_type_t type,
                                              const ExecutionContext *exe_ctx) {
  std::pair<const char *, const char *> key = {
      DeriveKeyFor(GetSymbolContext(exe_ctx)),
      reinterpret_cast<const char *>(type)};

  if (m_dangerous_types.count(key))
    return nullptr;

  // This can crash SwiftASTContext.
  if (swift_demangle::ContainsError(AsMangledName(type)))
    return nullptr;

  auto swift_ast_context = GetSwiftASTContext(GetSymbolContext(exe_ctx));
  if (!swift_ast_context || swift_ast_context->HasFatalErrors())
    return nullptr;
  void *result = llvm::expectedToStdOptional(swift_ast_context->ReconstructType(
                                                 GetMangledTypeName(type)))
                     .value_or(nullptr);

  // This reconstruction likely induced a fatal error.
  if (!result && swift_ast_context->HasFatalErrors())
    m_dangerous_types.insert(key);

  return result;
}

void *TypeSystemSwiftTypeRef::ReconstructType(
    opaque_compiler_type_t type,  ExecutionContextScope *exe_scope) {
  ExecutionContext exe_ctx;
  if (exe_scope)
    exe_scope->CalculateExecutionContext(exe_ctx);
  return ReconstructType(type, &exe_ctx);
}

CompilerType
TypeSystemSwiftTypeRef::ReconstructType(CompilerType type,
                                        const ExecutionContext *exe_ctx) {
  if (auto swift_ast_context = GetSwiftASTContext(GetSymbolContext(exe_ctx)))
    if (auto ts =
            type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>())
      return {{swift_ast_context->weak_from_this()},
              ts->ReconstructType(type.GetOpaqueQualType(), exe_ctx)};
  return {};
}

CompilerType TypeSystemSwiftTypeRef::GetTypeFromMangledTypename(
    ConstString mangled_typename) {
  return {weak_from_this(),
          (opaque_compiler_type_t)mangled_typename.AsCString()};
}

TypeSP TypeSystemSwiftTypeRef::GetCachedType(ConstString mangled) {
  return m_swift_type_map.Lookup(mangled.GetCString());
}

TypeSP TypeSystemSwiftTypeRef::GetCachedType(opaque_compiler_type_t type) {
  return m_swift_type_map.Lookup(AsMangledName(type));
}

void TypeSystemSwiftTypeRef::SetCachedType(ConstString mangled,
                                           const TypeSP &type_sp) {
  m_swift_type_map.Insert(mangled.GetCString(), type_sp);
}

bool TypeSystemSwiftTypeRef::SupportsLanguage(lldb::LanguageType language) {
  return language == eLanguageTypeSwift;
}

Status TypeSystemSwiftTypeRef::IsCompatible() {
  // This is called only from SBModule.
  // Currently basically a noop, since the module isn't being passed in.
  if (auto swift_ast_context = GetSwiftASTContext(SymbolContext()))
    return swift_ast_context->IsCompatible();
  return {};
}

plugin::dwarf::DWARFASTParser *TypeSystemSwiftTypeRef::GetDWARFParser() {
  if (!m_dwarf_ast_parser_up)
    m_dwarf_ast_parser_up.reset(new DWARFASTParserSwift(*this));
  return m_dwarf_ast_parser_up.get();
}

TypeSP
TypeSystemSwiftTypeRef::FindTypeInModule(opaque_compiler_type_t opaque_type) {
  auto *M = GetModule();
  if (!M)
    return {};

  swift::Demangle::Demangler dem;
  auto context = BuildDeclContext(AsMangledName(opaque_type), dem);
  if (!context)
    return {};
  // DW_AT_linkage_name is not part of the accelerator table, so
  // we need to search by decl context.
  auto options =
      TypeQueryOptions::e_find_one | TypeQueryOptions::e_module_search;

  // FIXME: It would be nice to not need this.
  if (context->size() == 2 &&
      context->front().kind == CompilerContextKind::Module &&
      context->front().name == "Builtin") {
    // LLVM cannot nest basic types inside a module.
    context->erase(context->begin());
    options = TypeQueryOptions::e_find_one;
  }
  TypeQuery query(*context, options);
  query.SetLanguages(TypeSystemSwift::GetSupportedLanguagesForTypes());

  TypeResults results;
  M->FindTypes(query, results);
  return results.GetFirstType();
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
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(str);

  auto mangling = mangleNode(node, flavor);
  if (!mangling.isSuccess())
    return false;
  std::string remangled = mangling.result();
  return remangled == std::string(str);
}

#include <regex>
namespace {
template <typename T> bool Equivalent(T l, T r) {
  if (l != r)
    llvm::dbgs() << l << " != " << r << "\n";
  return l == r;
}

/// Specialization for GetTypeInfo().
template <> bool Equivalent<uint32_t>(uint32_t l, uint32_t r) {
  if (l != r) {
    // The expanded pack types only exist in the typeref typesystem.
    if ((l & lldb::eTypeIsPack))
      return true;

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
    HANDLE_ENUM_CASE(l, eTypeHasUnboundGeneric);
    HANDLE_ENUM_CASE(l, eTypeIsPack);
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
    HANDLE_ENUM_CASE(r, eTypeHasUnboundGeneric);
    HANDLE_ENUM_CASE(r, eTypeIsPack);
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

/// Compare two swift types from different type systems by comparing their
/// (canonicalized) mangled name.
template <> bool Equivalent<CompilerType>(CompilerType l, CompilerType r) {
#ifdef STRICT_VALIDATION
  if (!l || !r)
    return !l && !r;
#else
  // We allow the typeref typesystem to return a type where
  // SwiftASTContext fails.
  if (!l || !r) {
    if (l || r)
      llvm::dbgs() << l.GetMangledTypeName() << " != " << r.GetMangledTypeName()
                   << "\n";
    return !r;
  }
#endif

  // See comments in SwiftASTContext::ReconstructType(). For
  // SILFunctionTypes the mapping isn't bijective.
  auto ast_ctx = r.GetTypeSystem().dyn_cast_or_null<SwiftASTContext>();
  if (((void *)llvm::expectedToStdOptional(
           ast_ctx->ReconstructType(l.GetMangledTypeName()))
           .value_or(nullptr)) == r.GetOpaqueQualType())
    return true;

  ConstString lhs = l.GetMangledTypeName();
  ConstString rhs = r.GetMangledTypeName();
  if (lhs == ConstString("$sSiD") && rhs == ConstString("$sSuD"))
    return true;
  if (lhs.GetStringRef() == "$sSPySo0023unnamedstruct_hEEEdhdEaVGSgD" &&
      rhs.GetStringRef() == "$ss13OpaquePointerVSgD")
    return true;
  // Ignore missing sugar.
  swift::Demangle::Demangler dem;
  auto l_node = GetDemangledType(dem, lhs.GetStringRef());
  auto r_node = GetDemangledType(dem, rhs.GetStringRef());
  if (ContainsUnresolvedTypeAlias(r_node) ||
      ContainsGenericTypeParameter(r_node) || ContainsSugaredParen(r_node))
    return true;
  auto l_mangling = swift::Demangle::mangleNode(
      TypeSystemSwiftTypeRef::CanonicalizeSugar(dem, l_node),
      SwiftLanguageRuntime::GetManglingFlavor(l.GetMangledTypeName()));
  auto r_mangling = swift::Demangle::mangleNode(
      TypeSystemSwiftTypeRef::CanonicalizeSugar(dem, r_node),
      SwiftLanguageRuntime::GetManglingFlavor(r.GetMangledTypeName()));
  if (!l_mangling.isSuccess() || !r_mangling.isSuccess()) {
    llvm::dbgs() << "TypeSystemSwiftTypeRef diverges from SwiftASTContext "
                    "(mangle error): "
                 << lhs.GetStringRef() << " != " << rhs.GetStringRef() << "\n";
    return false;
  }

  if (l_mangling.result() == r_mangling.result())
    return true;

  // SwiftASTContext hardcodes some less-precise types.
  if (rhs.GetStringRef() == "$sBpD")
    return true;

  // If the type is a Clang-imported type ignore mismatches. Since we
  // don't have any visibility into Swift overlays of SDK modules we
  // can only present the underlying Clang type. However, we can
  // still swiftify that type later for printing.
  if (auto ts = l.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>())
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
    if (r.IsEmpty() || r.GetStringRef() == "<invalid>" ||
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
bool Equivalent(std::optional<T> l, std::optional<T> r) {
  if (l == r)
    return true;
  // There are situations where SwiftASTContext incorrectly returns
  // all Clang-imported members of structs as having a size of 0, we
  // thus assume that a larger number is "better".
  if (l.has_value() && r.has_value() && *l > *r)
    return true;
  // Assume that any value is "better" than none.
  if (l.has_value() && !r.has_value())
    return true;
  llvm::dbgs() << l << " != " << r << "\n";
  return false;
}

template <>
bool Equivalent(std::optional<CompilerType> l, std::optional<CompilerType> r) {
  if (l == r)
    return true;
  // Assume that any value is "better" than none.
  if (l.has_value() && !r.has_value())
    return true;
  if (!l.has_value() && r.has_value()) {
    llvm::dbgs() << "{} != <some value>\n";
    return false;
  }
  return Equivalent(*l, *r);
}

// Introduced for `GetNumChildren`.
template <typename T> bool Equivalent(std::optional<T> l, T r) {
  return Equivalent(l, std::optional<T>(r));
}

} // namespace
#endif

#ifndef NDEBUG
constexpr ExecutionContextScope *g_no_exe_ctx = nullptr;
#endif

#ifndef NDEBUG
// Due to the lack of a symbol context, this only does the validation
// on TypeSystemSwiftTypeRefForExpressions.
#define VALIDATE_AND_RETURN_STATIC(IMPL, REFERENCE)                            \
  do {                                                                         \
    auto result = IMPL();                                                      \
    if (!ModuleList::GetGlobalModuleListProperties()                           \
             .GetSwiftValidateTypeSystem())                                    \
      return result;                                                           \
    auto target_sp = GetTargetWP().lock();                                     \
    if (!target_sp)                                                            \
      return result;                                                           \
    auto swift_ast_ctx = GetSwiftASTContext(                                   \
        SymbolContext(target_sp, target_sp->GetExecutableModule()));           \
    if (!swift_ast_ctx)                                                        \
      return result;                                                           \
    assert(Equivalent(result, swift_ast_ctx->REFERENCE()) &&                   \
           "TypeSystemSwiftTypeRef diverges from SwiftASTContext");            \
    return result;                                                             \
  } while (0)

#define VALIDATE_AND_RETURN_CUSTOM(IMPL, REFERENCE, TYPE, COMPARISON, EXE_CTX, \
                                   ARGS)                                       \
  do {                                                                         \
    auto result = IMPL();                                                      \
    if (!ModuleList::GetGlobalModuleListProperties()                           \
             .GetSwiftValidateTypeSystem())                                    \
      return result;                                                           \
    ExecutionContext _exe_ctx(EXE_CTX);                                        \
    if (!GetSwiftASTContext(GetSymbolContext(&_exe_ctx)))                      \
      return result;                                                           \
    if (ShouldSkipValidation(TYPE))                                            \
      return result;                                                           \
    if ((TYPE) && !ReconstructType(TYPE))                                      \
      return result;                                                           \
    /* When in the error backstop the sc will point into the stdlib. */        \
    if (auto *frame = _exe_ctx.GetFramePtr())                                  \
      if (frame->GetSymbolContext(eSymbolContextFunction).GetFunctionName() == \
          SwiftLanguageRuntime::GetErrorBackstopName())                        \
        return result;                                                         \
    bool equivalent =                                                          \
        !ReconstructType(TYPE) /* missing .swiftmodule */ ||                   \
        (COMPARISON(                                                           \
            result,                                                            \
            GetSwiftASTContext(GetSymbolContext(&_exe_ctx))->REFERENCE ARGS)); \
    if (!equivalent)                                                           \
      llvm::dbgs() << "failing type was " << (const char *)TYPE << "\n";       \
    assert(equivalent &&                                                       \
           "TypeSystemSwiftTypeRef diverges from SwiftASTContext");            \
    return result;                                                             \
  } while (0)

#define VALIDATE_AND_RETURN(IMPL, REFERENCE, TYPE, EXE_CTX, ARGS)              \
  VALIDATE_AND_RETURN_CUSTOM(IMPL, REFERENCE, TYPE, Equivalent, EXE_CTX, ARGS)

#define VALIDATE_AND_RETURN_EXPECTED(IMPL, REFERENCE, TYPE, EXE_CTX, ARGS)     \
  do {                                                                         \
    auto result = IMPL();                                                      \
    if (!ModuleList::GetGlobalModuleListProperties()                           \
             .GetSwiftValidateTypeSystem())                                    \
      return result;                                                           \
    ExecutionContext _exe_ctx(EXE_CTX);                                        \
    if (!GetSwiftASTContext(GetSymbolContext(&_exe_ctx)))                      \
      return result;                                                           \
    if (ShouldSkipValidation(TYPE))                                            \
      return result;                                                           \
    if ((TYPE) && !ReconstructType(TYPE))                                      \
      return result;                                                           \
    /* When in the error backstop the sc will point into the stdlib. */        \
    if (auto *frame = _exe_ctx.GetFramePtr())                                  \
      if (frame->GetSymbolContext(eSymbolContextFunction).GetFunctionName() == \
          SwiftLanguageRuntime::GetErrorBackstopName())                        \
        return result;                                                         \
    bool equivalent = true;                                                    \
    if (ReconstructType(TYPE)) {                                               \
      equivalent =                                                             \
          (Equivalent(llvm::expectedToStdOptional(std::move(result)),          \
                      llvm::expectedToStdOptional(                             \
                          GetSwiftASTContext(GetSymbolContext(&_exe_ctx))      \
                              ->REFERENCE ARGS)));                             \
    } else { /* missing .swiftmodule */                                        \
      if (!result)                                                             \
        llvm::consumeError(result.takeError());                                \
    }                                                                          \
    if (!equivalent)                                                           \
      llvm::dbgs() << "failing type was " << (const char *)TYPE << "\n";       \
    assert(equivalent &&                                                       \
           "TypeSystemSwiftTypeRef diverges from SwiftASTContext");            \
    return IMPL();                                                             \
  } while (0)

#else
#define VALIDATE_AND_RETURN_STATIC(IMPL, REFERENCE)                            \
  return IMPL()
#define VALIDATE_AND_RETURN(IMPL, REFERENCE, TYPE, EXE_CTX, ARGS)  \
  return IMPL();
#define VALIDATE_AND_RETURN_CUSTOM(IMPL, REFERENCE, TYPE, COMPARISON, EXE_CTX, ARGS)       \
  return IMPL();
#define VALIDATE_AND_RETURN_EXPECTED(IMPL, REFERENCE, TYPE, EXE_CTX, ARGS)     \
  return IMPL();
#endif

#define FORWARD_TO_EXPRAST_ONLY(FUNC, ARGS, DEFAULT_RETVAL)                    \
  do {                                                                         \
    if (auto target_sp = GetTargetWP().lock())                                 \
      if (auto swift_ast_ctx = GetSwiftASTContext(                             \
              SymbolContext(target_sp, target_sp->GetExecutableModule())))     \
        return swift_ast_ctx->FUNC ARGS;                                       \
    return DEFAULT_RETVAL;                                                     \
  } while (0)

bool TypeSystemSwiftTypeRef::UseSwiftASTContextFallback(
    const char *func_name, lldb::opaque_compiler_type_t type) {
  if (IsExpressionEvaluatorDefined(type))
    return true;
  if (!ModuleList::GetGlobalModuleListProperties().GetSwiftTypeSystemFallback())
    return false;

  LLDB_LOGF(GetLog(LLDBLog::Types),
            "TypeSystemSwiftTypeRef::%s(): Engaging SwiftASTContext fallback "
            "for type %s",
            func_name, AsMangledName(type));
  return true;
}

void TypeSystemSwiftTypeRef::DiagnoseSwiftASTContextFallback(
    const char *func_name, lldb::opaque_compiler_type_t type) {
  if (IsExpressionEvaluatorDefined(type))
    return;

  const char *type_name = AsMangledName(type);

  std::optional<lldb::user_id_t> debugger_id;
  if (auto target_sp = GetTargetWP().lock())
    debugger_id = target_sp->GetDebugger().GetID();

  std::string msg;
  llvm::raw_string_ostream(msg)
      << "TypeSystemSwiftTypeRef::" << func_name
      << ": had to engage SwiftASTContext fallback for type " << type_name;
  Debugger::ReportWarning(msg, debugger_id, &m_fallback_warning);

  LLDB_LOGF(GetLog(LLDBLog::Types), "%s", msg.c_str());
}

CompilerType
TypeSystemSwiftTypeRef::RemangleAsType(swift::Demangle::Demangler &dem,
                                       swift::Demangle::NodePointer node,
                                       swift::Mangle::ManglingFlavor flavor) {
  if (!node)
    return {};

  // Guard against an empty opaque type. This can happen when demangling an
  // OpaqueTypeRef (ex `$sBpD`). An empty opaque will assert when mangled.
  if (auto *opaque_type =
          swift_demangle::ChildAtPath(node, {Node::Kind::OpaqueType}))
    if (!opaque_type->hasChildren())
      return {};

  using namespace swift::Demangle;
  if (node->getKind() != Node::Kind::Global) {
    auto global = dem.createNode(Node::Kind::Global);
    if (node->getKind() != Node::Kind::TypeMangling) {
      auto type_mangling = dem.createNode(Node::Kind::TypeMangling);
      type_mangling->addChild(node, dem);
      assert(node->getKind() == Node::Kind::Type);
      node = type_mangling;
    }
    global->addChild(node, dem);
    node = global;
  }
  auto mangling = mangleNode(node, flavor);
  if (!mangling.isSuccess())
    return {};
  ConstString mangled_element(mangling.result());
  return GetTypeFromMangledTypename(mangled_element);
}

swift::Demangle::NodePointer TypeSystemSwiftTypeRef::DemangleCanonicalType(
    swift::Demangle::Demangler &dem, lldb::opaque_compiler_type_t opaque_type) {
  using namespace swift::Demangle;
  CompilerType type = GetCanonicalType(opaque_type);
  return GetDemangledType(dem, type.GetMangledTypeName().GetStringRef());
}

swift::Demangle::NodePointer
TypeSystemSwiftTypeRef::DemangleCanonicalOutermostType(
    swift::Demangle::Demangler &dem, lldb::opaque_compiler_type_t type) {
  using namespace swift::Demangle;
  const auto *mangled_name = AsMangledName(type);
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);
  NodePointer node = GetDemangledType(dem, mangled_name);

  if (!node)
    return nullptr;
  NodePointer canonical = Canonicalize(dem, node, flavor);
  if (canonical &&
      canonical->getKind() == swift::Demangle::Node::Kind::TypeAlias) {
    // If this is a typealias defined in the expression evaluator,
    // then we don't have debug info to resolve it from.
    CompilerType ast_type =
        ReconstructType({weak_from_this(), type}, nullptr).GetCanonicalType();
    return GetDemangledType(dem, ast_type.GetMangledTypeName());
  }
  return canonical;
}

bool TypeSystemSwiftTypeRef::IsExpressionEvaluatorDefined(
    lldb::opaque_compiler_type_t type) {
  using namespace swift::Demangle;
  const auto *mangled_name = AsMangledName(type);
  Demangler dem;
  NodePointer node = GetDemangledType(dem, mangled_name);
  return swift_demangle::FindIf(node, [](NodePointer node) -> bool {
    if (node->getKind() == Node::Kind::Module &&
        node->getText().starts_with("__lldb_expr"))
      return true;
    return false;
  });
}

CompilerType TypeSystemSwiftTypeRef::CreateGenericTypeParamType(
    unsigned int depth, unsigned int index,
    swift::Mangle::ManglingFlavor flavor) {
  Demangler dem;
  NodePointer type_node = dem.createNode(Node::Kind::Type);

  NodePointer dep_type_node =
      dem.createNode(Node::Kind::DependentGenericParamType);
  type_node->addChild(dep_type_node, dem);

  NodePointer depth_node = dem.createNode(Node::Kind::Index, depth);
  NodePointer index_node = dem.createNode(Node::Kind::Index, index);

  dep_type_node->addChild(depth_node, dem);
  dep_type_node->addChild(index_node, dem);
  auto type = RemangleAsType(dem, type_node, flavor);
  return type;
}

bool TypeSystemSwiftTypeRef::IsArrayType(opaque_compiler_type_t type,
                                         CompilerType *element_type,
                                         uint64_t *size, bool *is_incomplete) {
  auto impl = [&]() {
    using namespace swift::Demangle;

    auto mangled_name = AsMangledName(type);
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);
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
         node->getChild(1)->getText() != "InlineArray" &&
         node->getChild(1)->getText() != "ArraySlice"))
      return false;

    if (elem_node->getKind() != Node::Kind::TypeList)
      return false;
    if (node->getChild(1)->getText() == "InlineArray") {
      if (elem_node->getNumChildren() != 2)
        return false;
      elem_node = elem_node->getChild(1);
    } else {
      if (elem_node->getNumChildren() != 1)
        return false;
      elem_node = elem_node->getFirstChild();
    }
    if (element_type)
      // FIXME: This expensive canonicalization is only there for
      // SwiftASTContext compatibility.
      *element_type = RemangleAsType(dem, elem_node, flavor).GetCanonicalType();

    if (is_incomplete)
      *is_incomplete = true;
    if (size)
      *size = 0;

    return true;
  };
  VALIDATE_AND_RETURN(impl, IsArrayType, type, g_no_exe_ctx,
                      (ReconstructType(type), nullptr, nullptr, nullptr));
}

bool TypeSystemSwiftTypeRef::IsAggregateType(opaque_compiler_type_t type) {
  auto impl = [&]() -> bool {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);

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
  VALIDATE_AND_RETURN(impl, IsAggregateType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

bool TypeSystemSwiftTypeRef::IsDefined(opaque_compiler_type_t type) {
  auto impl = [&]() -> bool { return type; };
  VALIDATE_AND_RETURN(impl, IsDefined, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

bool TypeSystemSwiftTypeRef::IsFunctionType(opaque_compiler_type_t type) {
  auto impl = [&]() -> bool {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);
    // Note: There are a number of other candidates, and this list may need
    // updating. Ex: `NoEscapeFunctionType`, `ThinFunctionType`, etc.
    return node && (node->getKind() == Node::Kind::FunctionType ||
                    node->getKind() == Node::Kind::NoEscapeFunctionType ||
                    node->getKind() == Node::Kind::ImplFunctionType);
  };
  VALIDATE_AND_RETURN(impl, IsFunctionType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}
size_t TypeSystemSwiftTypeRef::GetNumberOfFunctionArguments(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> size_t {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);
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
  VALIDATE_AND_RETURN(impl, GetNumberOfFunctionArguments, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}
CompilerType
TypeSystemSwiftTypeRef::GetFunctionArgumentAtIndex(opaque_compiler_type_t type,
                                                   const size_t index) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;

    const auto *mangled_name = AsMangledName(type);
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);
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
              return RemangleAsType(dem, type, flavor);
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
                return RemangleAsType(dem, type, flavor);
              ++num_args;
            }
          }
      }
    }
    return {};
  };
  VALIDATE_AND_RETURN(impl, GetFunctionArgumentAtIndex, type, g_no_exe_ctx,
                      (ReconstructType(type), index));
}
bool TypeSystemSwiftTypeRef::IsFunctionPointerType(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> bool { return IsFunctionType(type); };
  VALIDATE_AND_RETURN(impl, IsFunctionPointerType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

bool TypeSystemSwiftTypeRef::IsPossibleDynamicType(opaque_compiler_type_t type,
                                                   CompilerType *target_type,
                                                   bool check_cplusplus,
                                                   bool check_objc) {
  if (target_type)
    target_type->Clear();

  if (!type)
    return false;

  const char *mangled_name = AsMangledName(type);
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    std::function<bool(NodePointer)> is_possible_dynamic =
        [&](NodePointer node) -> bool {
      if (!node)
        return false;

      if (node->getKind() == Node::Kind::TypeAlias) {
        auto resolved = ResolveTypeAlias(dem, node, flavor);
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
      case Node::Kind::OpaqueType:
      case Node::Kind::Enum:
      case Node::Kind::BoundGenericEnum:
        return true;
      case Node::Kind::BoundGenericStructure: {
        if (node->getNumChildren() < 2)
          return false;
        NodePointer type_list = node->getLastChild();
        if (type_list->getKind() != Node::Kind::TypeList)
          return false;
        for (NodePointer child : *type_list) {
          if (child->getKind() == Node::Kind::Type) {
            child = child->getFirstChild();
            if (is_possible_dynamic(child))
              return true;
          }
        }
        return false;
      }
      case Node::Kind::ImplFunctionType:
        return false;
      case Node::Kind::BuiltinTypeName: {
        if (!node->hasText())
          return false;
        StringRef name = node->getText();
        return name == swift::BUILTIN_TYPE_NAME_RAWPOINTER ||
               name == swift::BUILTIN_TYPE_NAME_NATIVEOBJECT ||
               name == swift::BUILTIN_TYPE_NAME_BRIDGEOBJECT ||
               name == swift::BUILTIN_TYPE_NAME_UNKNOWNOBJECT;
      }
      default:
        return ContainsGenericTypeParameter(node);
      }
    };

    auto *node = DemangleCanonicalType(dem, type);
    return is_possible_dynamic(node);
  };
  VALIDATE_AND_RETURN(
      impl, IsPossibleDynamicType, type, g_no_exe_ctx,
      (ReconstructType(type), nullptr, check_cplusplus, check_objc));
}

bool TypeSystemSwiftTypeRef::IsPointerType(opaque_compiler_type_t type,
                                           CompilerType *pointee_type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);
    if (!node || node->getKind() != Node::Kind::BuiltinTypeName ||
        !node->hasText())
      return false;
    return ((node->getText() == swift::BUILTIN_TYPE_NAME_RAWPOINTER) ||
            (node->getText() == swift::BUILTIN_TYPE_NAME_UNSAFEVALUEBUFFER) ||
            (node->getText() == swift::BUILTIN_TYPE_NAME_NATIVEOBJECT) ||
            (node->getText() == swift::BUILTIN_TYPE_NAME_BRIDGEOBJECT));
  };
  VALIDATE_AND_RETURN(impl, IsPointerType, type, g_no_exe_ctx,
                      (ReconstructType(type), nullptr));
}
bool TypeSystemSwiftTypeRef::IsVoidType(opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);
    return node && node->getNumChildren() == 0 &&
           node->getKind() == Node::Kind::Tuple;
  };
  VALIDATE_AND_RETURN(impl, IsVoidType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}
// AST related queries
uint32_t TypeSystemSwiftTypeRef::GetPointerByteSize() {
  auto impl = [&]() -> uint32_t {
    llvm::Triple triple = GetTriple();
    if (triple.isArch64Bit())
      return 8;
    if (triple.isArch32Bit())
      return 4;
    if (triple.isArch16Bit())
      return 2;

    return 0;
  };
  VALIDATE_AND_RETURN_STATIC(impl, GetPointerByteSize);
}
// Accessors
ConstString TypeSystemSwiftTypeRef::GetTypeName(opaque_compiler_type_t type,
                                                bool BaseOnly) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer print_node =
        GetDemangleTreeForPrinting(dem, AsMangledName(type), true);
    auto mangling = mangleNode(
        print_node,
        SwiftLanguageRuntime::GetManglingFlavor(AsMangledName(type)));
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
  VALIDATE_AND_RETURN(impl, GetTypeName, type, g_no_exe_ctx,
                      (ReconstructType(type), false));
}
ConstString
TypeSystemSwiftTypeRef::GetDisplayTypeName(opaque_compiler_type_t type,
                                           const SymbolContext *sc) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(AsMangledName(type));
    Demangler dem;
    NodePointer print_node =
        GetDemangleTreeForPrinting(dem, AsMangledName(type), false);
    auto mangling = mangleNode(print_node, flavor);
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
  VALIDATE_AND_RETURN(impl, GetDisplayTypeName, type, g_no_exe_ctx,
                      (ReconstructType(type), sc));
}

uint32_t TypeSystemSwiftTypeRef::GetTypeInfo(
    opaque_compiler_type_t type, CompilerType *pointee_or_element_clang_type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    const char *mangled_name = AsMangledName(type);
    NodePointer node = dem.demangleSymbol(mangled_name);
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    bool unresolved_typealias = false;
    uint32_t flags = CollectTypeInfo(dem, node, flavor, unresolved_typealias);
    if (unresolved_typealias)
      if (auto target_sp = GetTargetWP().lock())
        if (auto swift_ast_ctx = GetSwiftASTContext(
                SymbolContext(target_sp, target_sp->GetExecutableModule())))
          // If this is a typealias defined in the expression evaluator,
          // then we don't have debug info to resolve it from.
          return swift_ast_ctx->GetTypeInfo(ReconstructType(type),
                                            pointee_or_element_clang_type);
    return flags;
  };

  VALIDATE_AND_RETURN(impl, GetTypeInfo, type, g_no_exe_ctx,
                      (ReconstructType(type), nullptr));
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
  VALIDATE_AND_RETURN(impl, GetTypeClass, type, g_no_exe_ctx,
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
  VALIDATE_AND_RETURN(impl, GetArrayElementType, type, exe_scope,
                      (ReconstructType(type, exe_scope), exe_scope));
}

CompilerType
TypeSystemSwiftTypeRef::GetCanonicalType(opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer canonical = GetCanonicalDemangleTree(dem, AsMangledName(type));
    if (ContainsUnresolvedTypeAlias(canonical)) {
      // If this is a typealias defined in the expression evaluator,
      // then we don't have debug info to resolve it from.
      CompilerType ast_type =
        ReconstructType({weak_from_this(), type}, nullptr).GetCanonicalType();
      LLDB_LOG(GetLog(LLDBLog::Types),
               "Cannot resolve type alias in type \"{0}\"",
               AsMangledName(type));
      if (!ast_type)
        return CompilerType();
      CompilerType result =
          GetTypeFromMangledTypename(ast_type.GetMangledTypeName());
      if (result && !llvm::isa<TypeSystemSwiftTypeRefForExpressions>(this))
        DiagnoseSwiftASTContextFallback(__FUNCTION__, type);
      return result;
    }
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(AsMangledName(type));
    auto mangling = mangleNode(canonical, flavor);
    if (!mangling.isSuccess())
      return CompilerType();
    ConstString mangled(mangling.result());
    return GetTypeFromMangledTypename(mangled);
  };
  VALIDATE_AND_RETURN(impl, GetCanonicalType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}
int TypeSystemSwiftTypeRef::GetFunctionArgumentCount(
    opaque_compiler_type_t type) {
  auto impl = [&]() -> int { return GetNumberOfFunctionArguments(type); };
  VALIDATE_AND_RETURN(impl, GetFunctionArgumentCount, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}
CompilerType TypeSystemSwiftTypeRef::GetFunctionArgumentTypeAtIndex(
    opaque_compiler_type_t type, size_t idx) {
  auto impl = [&] { return GetFunctionArgumentAtIndex(type, idx); };
  VALIDATE_AND_RETURN(impl, GetFunctionArgumentTypeAtIndex, type, g_no_exe_ctx,
                      (ReconstructType(type), idx));
}
CompilerType
TypeSystemSwiftTypeRef::GetFunctionReturnType(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;

    const char *mangled_name = AsMangledName(type);
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);
    if (!node || (node->getKind() != Node::Kind::FunctionType &&
                  node->getKind() != Node::Kind::NoEscapeFunctionType &&
                  node->getKind() != Node::Kind::ImplFunctionType))
      return {};
    for (NodePointer child : *node) {
      if (child->getKind() == Node::Kind::ImplResult) {
        for (NodePointer type : *child)
          if (type->getKind() == Node::Kind::Type)
            return RemangleAsType(dem, type, flavor);
      }
      if (child->getKind() == Node::Kind::ReturnType &&
          child->getNumChildren() == 1) {
        NodePointer type = child->getFirstChild();
        if (type->getKind() == Node::Kind::Type)
          return RemangleAsType(dem, type, flavor);
      }
    }
    // Else this is a void / "()" type.
    NodePointer type = dem.createNode(Node::Kind::Type);
    NodePointer tuple = dem.createNode(Node::Kind::Tuple);
    type->addChild(tuple, dem);
    return RemangleAsType(dem, type, flavor);
  };
  VALIDATE_AND_RETURN(impl, GetFunctionReturnType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

size_t
TypeSystemSwiftTypeRef::GetNumMemberFunctions(opaque_compiler_type_t type) {
  // We forward the call to SwiftASTContext because an implementation of
  // this function would require it to have an execution context being passed
  // in. Given the purpose of TypeSystemSwiftTypeRef, it's unlikely this
  // function will be called much.
  FORWARD_TO_EXPRAST_ONLY(GetNumMemberFunctions, (ReconstructType(type)), {});
}

TypeMemberFunctionImpl
TypeSystemSwiftTypeRef::GetMemberFunctionAtIndex(opaque_compiler_type_t type,
                                                 size_t idx) {
  // We forward the call to SwiftASTContext because an implementation of
  // this function would require it to have an execution context being passed
  // in. Given the purpose of TypeSystemSwiftTypeRef, it's unlikely this
  // function will be called much.
  FORWARD_TO_EXPRAST_ONLY(GetMemberFunctionAtIndex, (ReconstructType(type),
                                                     idx), {});
}

CompilerType
TypeSystemSwiftTypeRef::GetPointeeType(opaque_compiler_type_t type) {
  auto impl = []() { return CompilerType(); };
  VALIDATE_AND_RETURN(impl, GetPointeeType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

CompilerType
TypeSystemSwiftTypeRef::GetPointerType(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    const auto *mangled_name = AsMangledName(type);
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    Demangler dem;
    // The type that will be wrapped in UnsafePointer.
    auto *pointee_type = GetDemangledType(dem, AsMangledName(type));
    if (!pointee_type)
      return {};

    // The UnsafePointer type.
    auto *pointer_type = dem.createNode(Node::Kind::Type);

    auto *bgs = GetPointerTo(dem, pointee_type);
    pointer_type->addChild(bgs, dem);
    return RemangleAsType(dem, pointer_type, flavor);
  };
  VALIDATE_AND_RETURN(impl, GetPointerType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

CompilerType TypeSystemSwiftTypeRef::GetVoidFunctionType() {
  using namespace swift::Demangle;
  Demangler dem;
  NodePointer type = dem.createNode(Node::Kind::Type);
  NodePointer fnty = dem.createNode(Node::Kind::FunctionType);
  type->addChild(fnty, dem);
  NodePointer args = dem.createNode(Node::Kind::ArgumentTuple);
  fnty->addChild(args, dem);
  NodePointer args_ty = dem.createNode(Node::Kind::Type);
  args->addChild(args_ty, dem);
  NodePointer args_tuple = dem.createNode(Node::Kind::Tuple);
  args_ty->addChild(args_tuple, dem);
  NodePointer rett = dem.createNode(Node::Kind::ReturnType);
  fnty->addChild(rett, dem);
  NodePointer ret_ty = dem.createNode(Node::Kind::Type);
  rett->addChild(ret_ty, dem);
  NodePointer ret_tuple = dem.createNode(Node::Kind::Tuple);
  ret_ty->addChild(ret_tuple, dem);
  return RemangleAsType(dem, type, GetManglingFlavor());
}

// Exploring the type
llvm::Expected<uint64_t>
TypeSystemSwiftTypeRef::GetBitSize(opaque_compiler_type_t type,
                                   ExecutionContextScope *exe_scope) {
  auto impl = [&]() -> llvm::Expected<uint64_t> {
    auto get_static_size = [&](bool cached_only) -> std::optional<uint64_t> {
      if (IsMeaninglessWithoutDynamicResolution(type))
        return {};

      // If there is no process, we can still try to get the static size
      // information out of DWARF. Because it is stored in the Type
      // object we need to look that up by name again.
      TypeSP static_type =
          cached_only ? GetCachedType(type) : FindTypeInModule(type);

      struct SwiftType : public Type {
        /// Avoid a potential infinite recursion because
        /// Type::GetByteSize() may call into this function again.
        std::optional<uint64_t> GetStaticByteSize() {
          if (m_byte_size_has_value)
            return uint64_t(m_byte_size);
          return {};
        }
      };
      if (static_type)
        if (auto byte_size = reinterpret_cast<SwiftType *>(static_type.get())
                                 ->GetStaticByteSize())
          return *byte_size * 8;
      return {};
    };

    // Bug-for-bug compatibility. See comment in
    // SwiftASTContext::GetBitSize().
    if (IsFunctionType(type))
      return GetPointerByteSize() * 8;

    if (IsSILPackType({weak_from_this(), type}))
      return GetPointerByteSize() * 8;

    // Clang types can be resolved even without a process.
    if (CompilerType clang_type = GetAsClangTypeOrNull(type)) {
      // Swift doesn't know pointers: return the size of the object
      // pointer instead of the underlying object.
      if (Flags(clang_type.GetTypeInfo()).AllSet(eTypeIsObjC | eTypeIsClass))
        return GetPointerByteSize() * 8;
      auto clang_size = clang_type.GetBitSize(exe_scope);
      return clang_size;
    }
    if (!exe_scope)
      return llvm::createStringError(
          "Cannot compute size of type %s without an execution context.",
          AsMangledName(type));
    // The hot code path is to ask the Swift runtime for the size.
    if (auto *runtime =
            SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
      auto result_or_err =
          runtime->GetBitSize({weak_from_this(), type}, exe_scope);
      if (result_or_err)
        return *result_or_err;
      LLDB_LOG_ERROR(
          GetLog(LLDBLog::Types), result_or_err.takeError(),
          "Couldn't compute size of type {1} using Swift language runtime: {0}",
          AsMangledName(type));

      // Runtime failed, fallback to SwiftASTContext.
      if (UseSwiftASTContextFallback(__FUNCTION__, type)) {
        if (auto swift_ast_context =
                GetSwiftASTContext(GetSymbolContext(exe_scope))) {
          auto result = swift_ast_context->GetBitSize(
              ReconstructType(type, exe_scope), exe_scope);
          if (result)
            DiagnoseSwiftASTContextFallback(__FUNCTION__, type);
          return result;
        }
      }
    }

    // FIXME: Move this to the top. Currently this causes VALIDATE
    // errors on resilient types, and Foundation overlay types. These
    // are most likely bugs in the Swift compiler that need to be
    // resolved first.

    // If we have already parsed this as an lldb::Type from DWARF,
    // return its static size.
    if (auto cached_type_static_size = get_static_size(true))
      return *cached_type_static_size;

    // If we are here, we probably are in a target with no process and
    // inspect a gloabl variable.  Do an (expensive) search for the
    // static type in the debug info.
    if (auto static_size = get_static_size(false))
      return *static_size;
    return llvm::createStringError(
        "Cannot compute size of type %s using static debug info.",
        AsMangledName(type));
  };
  if (exe_scope && exe_scope->CalculateProcess()) {
    VALIDATE_AND_RETURN_EXPECTED(impl, GetBitSize, type, exe_scope,
                                 (ReconstructType(type, exe_scope), exe_scope));
  } else
    return impl();
}

std::optional<uint64_t>
TypeSystemSwiftTypeRef::GetByteStride(opaque_compiler_type_t type,
                                      ExecutionContextScope *exe_scope) {
  auto impl = [&]() -> std::optional<uint64_t> {
    if (auto *runtime =
            SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
      if (auto stride = runtime->GetByteStride(GetCanonicalType(type)))
        return stride;
    }
    // Runtime failed, fallback to SwiftASTContext.
    if (UseSwiftASTContextFallback(__FUNCTION__, type)) {
      if (auto swift_ast_context =
              GetSwiftASTContext(GetSymbolContext(exe_scope))) {
        auto result =
            swift_ast_context->GetByteStride(ReconstructType(type), exe_scope);
        if (result)
          DiagnoseSwiftASTContextFallback(__FUNCTION__, type);
        return result;
      }
    }
    return {};
  };
  VALIDATE_AND_RETURN(impl, GetByteStride, type, exe_scope,
                      (ReconstructType(type, exe_scope), exe_scope));
}

lldb::Encoding TypeSystemSwiftTypeRef::GetEncoding(opaque_compiler_type_t type,
                                                   uint64_t &count) {
  auto impl = [&]() -> lldb::Encoding {
    if (!type)
      return lldb::eEncodingInvalid;

    count = 1;

    using namespace swift::Demangle;
    Demangler dem;
    auto *node = DemangleCanonicalOutermostType(dem, type);
    if (!node)
      return lldb::eEncodingInvalid;
    auto kind = node->getKind();

    if (kind == Node::Kind::BuiltinTypeName) {
      assert(node->hasText());
      if (node->getText().starts_with(swift::BUILTIN_TYPE_NAME_INT) ||
          node->getText() == swift::BUILTIN_TYPE_NAME_WORD)
        return lldb::eEncodingSint;
      if (node->getText().starts_with(swift::BUILTIN_TYPE_NAME_FLOAT) ||
          node->getText().starts_with(swift::BUILTIN_TYPE_NAME_FLOAT_PPC))
        return lldb::eEncodingIEEE754;
      if (node->getText() == swift::BUILTIN_TYPE_NAME_RAWPOINTER ||
          node->getText() == swift::BUILTIN_TYPE_NAME_NATIVEOBJECT ||
          node->getText() == swift::BUILTIN_TYPE_NAME_UNSAFEVALUEBUFFER ||
          node->getText() == swift::BUILTIN_TYPE_NAME_BRIDGEOBJECT ||
          node->getText() == swift::BUILTIN_TYPE_NAME_RAWUNSAFECONTINUATION)
        return lldb::eEncodingUint;
      if (node->getText().starts_with(swift::BUILTIN_TYPE_NAME_VEC)) {
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

      const auto *mangled_name = AsMangledName(type);
      auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);
      auto referent_type = RemangleAsType(dem, referent_node, flavor);
      return referent_type.GetEncoding(count);
    }
    default:
      LLDB_LOGF(GetLog(LLDBLog::Types), "No encoding for type %s",
                AsMangledName(type));
      break;
    }

    count = 0;
    return lldb::eEncodingInvalid;
  };

#ifndef NDEBUG
  uint64_t validation_count = 0;
#endif
  VALIDATE_AND_RETURN(impl, GetEncoding, type, g_no_exe_ctx,
                      (ReconstructType(type), validation_count));
}

llvm::Expected<uint32_t>
TypeSystemSwiftTypeRef::GetNumChildren(opaque_compiler_type_t type,
                                       bool omit_empty_base_classes,
                                       const ExecutionContext *exe_ctx) {
  auto impl = [&]() -> llvm::Expected<uint32_t> {
    if (exe_ctx)
      if (auto *exe_scope = exe_ctx->GetBestExecutionContextScope())
        if (auto *runtime =
                SwiftLanguageRuntime::Get(exe_scope->CalculateProcess()))
          return runtime->GetNumChildren(GetCanonicalType(type), exe_scope);

    if (CompilerType clang_type = GetAsClangTypeOrNull(type)) {
      bool is_signed;
      // Clang-imported enum types always have one child in Swift.
      if (clang_type.IsEnumerationType(is_signed))
        return 1;
      return clang_type.GetNumChildren(omit_empty_base_classes, exe_ctx);
    }
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "incomplete type information");
  };
  llvm::Expected<uint32_t> num_children = impl();
  if (num_children) {
    ExecutionContext exe_ctx_obj;
    if (exe_ctx)
      exe_ctx_obj = *exe_ctx;
    VALIDATE_AND_RETURN_EXPECTED(
        impl, GetNumChildren, type, exe_ctx_obj,
        (ReconstructType(type, exe_ctx), omit_empty_base_classes, exe_ctx));
  }
  // Runtime failed, fallback to SwiftASTContext.
  if (UseSwiftASTContextFallback(__FUNCTION__, type)) {
    if (auto swift_ast_context = GetSwiftASTContext(GetSymbolContext(exe_ctx)))
      if (auto n =
              llvm::expectedToStdOptional(swift_ast_context->GetNumChildren(
                  ReconstructType(type, exe_ctx), omit_empty_base_classes,
                  exe_ctx))) {
        LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), num_children.takeError(),
                        "SwiftLanguageRuntime::GetNumChildren() failed: {0}");
        DiagnoseSwiftASTContextFallback(__FUNCTION__, type);
        return *n;
      }
  }

  // Otherwise return the error from the runtime.
  return num_children.takeError();
}

uint32_t TypeSystemSwiftTypeRef::GetNumFields(opaque_compiler_type_t type,
                                              ExecutionContext *exe_ctx) {
  auto impl = [&]() -> std::optional<uint32_t> {
    if (exe_ctx)
      if (auto *runtime = SwiftLanguageRuntime::Get(exe_ctx->GetProcessSP())) {
        auto num_fields_or_err =
            runtime->GetNumFields(GetCanonicalType(type), exe_ctx);
        if (num_fields_or_err)
          return *num_fields_or_err;
        LLDB_LOG_ERROR(GetLog(LLDBLog::Types), num_fields_or_err.takeError(),
                       "GetNumFields failed for type {1}: {0}",
                       AsMangledName(type));
      }

    bool is_imported = false;
    if (auto clang_type = GetAsClangTypeOrNull(type, &is_imported)) {
      switch (clang_type.GetTypeClass()) {
      case lldb::eTypeClassObjCObject:
      case lldb::eTypeClassObjCInterface:
        // Imported ObjC types are treated as having no fields.
        return 0;
      default:
        return clang_type.GetNumFields(exe_ctx);
      }
    } else if (is_imported) {
      // A known imported type, but where clang has no info. Return early to
      // avoid loading Swift ASTContexts, only to return the same zero value.
      LLDB_LOGF(GetLog(LLDBLog::Types),
                "No CompilerType for imported Clang type %s",
                AsMangledName(type));
      return 0;
    }
    return {};
  };
  if (auto num_fields = impl()) {
    // Use a lambda to intercept and unwrap the `Optional` return value.
    // Optional<uint32_t> uses more lax equivalency function.
    return [&]() -> std::optional<uint32_t> {
      auto impl = [&]() { return num_fields; };
      ExecutionContext exe_ctx_obj;
      if (exe_ctx)
        exe_ctx_obj = *exe_ctx;
      VALIDATE_AND_RETURN(impl, GetNumFields, type, exe_ctx_obj,
                          (ReconstructType(type, exe_ctx), exe_ctx));
    }()
                        .value_or(0);
  }

  // Runtime failed, fallback to SwiftASTContext.
  if (UseSwiftASTContextFallback(__FUNCTION__, type))
    if (auto swift_ast_context =
            GetSwiftASTContext(GetSymbolContext(exe_ctx))) {
      auto result = swift_ast_context->GetNumFields(
          ReconstructType(type, exe_ctx), exe_ctx);
      if (result)
        DiagnoseSwiftASTContextFallback(__FUNCTION__, type);
      return result;
    }
  return {};
}

CompilerType TypeSystemSwiftTypeRef::GetFieldAtIndex(
    opaque_compiler_type_t type, size_t idx, std::string &name,
    uint64_t *bit_offset_ptr, uint32_t *bitfield_bit_size_ptr,
    bool *is_bitfield_ptr) {
  // We forward the call to SwiftASTContext because an implementation of
  // this function would require it to have an execution context being passed
  // in. Given the purpose of TypeSystemSwiftTypeRef, it's unlikely this
  // function will be called much.
  FORWARD_TO_EXPRAST_ONLY(GetFieldAtIndex,
                          (ReconstructType(type), idx, name, bit_offset_ptr,
                           bitfield_bit_size_ptr, is_bitfield_ptr),
                          {});
}

swift::reflection::DescriptorFinder *
TypeSystemSwiftTypeRef::GetDescriptorFinder() {
  return llvm::cast<DWARFASTParserSwift>(GetDWARFParser());
}

swift::Demangle::NodePointer
TypeSystemSwiftTypeRef::GetClangTypeTypeNode(swift::Demangle::Demangler &dem,
                                             CompilerType clang_type) {
  assert(clang_type.GetTypeSystem().isa_and_nonnull<TypeSystemClang>() &&
         "expected a clang type");
  using namespace swift::Demangle;
  NodePointer node = GetClangTypeNode(clang_type, dem);
  if (!node)
    return nullptr;
  NodePointer type = dem.createNode(Node::Kind::Type);
  type->addChild(node, dem);
  return type;
}

CompilerType
TypeSystemSwiftTypeRef::ConvertClangTypeToSwiftType(CompilerType clang_type) {
  assert(clang_type.GetTypeSystem().isa_and_nonnull<TypeSystemClang>() &&
         "Unexpected type system!");

  if (!clang_type.GetTypeSystem().isa_and_nonnull<TypeSystemClang>())
    return {};

  swift::Demangle::Demangler dem;
  swift::Demangle::NodePointer node = GetClangTypeTypeNode(dem, clang_type);
  return RemangleAsType(dem, node, GetManglingFlavor());
}

llvm::Expected<CompilerType>
TypeSystemSwiftTypeRef::GetChildCompilerTypeAtIndex(
    opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
    bool transparent_pointers, bool omit_empty_base_classes,
    bool ignore_array_bounds, std::string &child_name,
    uint32_t &child_byte_size, int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class, bool &child_is_deref_of_parent,
    ValueObject *valobj, uint64_t &language_flags) {
  child_name = "";
  child_byte_size = 0;
  child_byte_offset = 0;
  child_bitfield_bit_size = 0;
  child_bitfield_bit_offset = 0;
  child_is_base_class = false;
  child_is_deref_of_parent = false;
  language_flags = 0;
  auto fallback = [&]() -> llvm::Expected<CompilerType> {
    LLDB_LOG(GetLog(LLDBLog::Types),
             "Had to engage SwiftASTContext fallback for type {0}, field #{1}.",
             AsMangledName(type), idx);
    if (auto swift_ast_context =
            GetSwiftASTContext(GetSymbolContext(exe_ctx)))
      return swift_ast_context->GetChildCompilerTypeAtIndex(
          ReconstructType(type, exe_ctx), exe_ctx, idx, transparent_pointers,
          omit_empty_base_classes, ignore_array_bounds, child_name,
          child_byte_size, child_byte_offset, child_bitfield_bit_size,
          child_bitfield_bit_offset, child_is_base_class,
          child_is_deref_of_parent, valobj, language_flags);
    return llvm::createStringError("no SwiftASTContext");
  };
  std::optional<unsigned> ast_num_children;
  auto get_ast_num_children = [&]() {
    if (ast_num_children)
      return *ast_num_children;
    if (auto swift_ast_context =
            GetSwiftASTContext(GetSymbolContext(exe_ctx)))
      ast_num_children = llvm::expectedToStdOptional(
          swift_ast_context->GetNumChildren(ReconstructType(type, exe_ctx),
                                            omit_empty_base_classes, exe_ctx));
    return ast_num_children.value_or(0);
  };
  auto impl = [&]() -> llvm::Expected<CompilerType> {
    std::string error = "unknown error";
    ExecutionContextScope *exe_scope = nullptr;
    if (exe_ctx)
      exe_scope = exe_ctx->GetBestExecutionContextScope();
    if (exe_scope) {
      if (auto *runtime =
              SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
        auto result = runtime->GetChildCompilerTypeAtIndex(
            {weak_from_this(), type}, idx, transparent_pointers,
            omit_empty_base_classes, ignore_array_bounds, child_name,
            child_byte_size, child_byte_offset, child_bitfield_bit_size,
            child_bitfield_bit_offset, child_is_base_class,
            child_is_deref_of_parent, valobj, language_flags);
        if (result && *result) {
          // This type is treated specially by ClangImporter.  It's really a
          // typedef to NSString *, but ClangImporter introduces an extra
          // layer of indirection that we simulate here.
          if (llvm::StringRef(AsMangledName(type))
                  .ends_with("sSo18NSNotificationNameaD"))
            return GetTypeFromMangledTypename(ConstString("$sSo8NSStringCD"));
          return result;
        }
        if (!result)
          error = llvm::toString(result.takeError());
      }
      // Clang types can be resolved even without a process.
      bool is_signed;
      if (CompilerType clang_type = GetAsClangTypeOrNull(type)) {
        if (clang_type.IsEnumerationType(is_signed) && idx == 0)
          // C enums get imported into Swift as structs with a "rawValue" field.
          if (auto ts = clang_type.GetTypeSystem()
                            .dyn_cast_or_null<TypeSystemClang>())
            if (clang::EnumDecl *enum_decl = ts->GetAsEnumDecl(clang_type)) {
              swift::Demangle::Demangler dem;
              CompilerType raw_value = CompilerType(
                  ts, enum_decl->getIntegerType().getAsOpaquePtr());
              child_name = "rawValue";
              auto bit_size = raw_value.GetBitSize(
                  exe_ctx ? exe_ctx->GetBestExecutionContextScope() : nullptr);
              if (!bit_size)
                return bit_size.takeError();
              child_byte_size = *bit_size / 8;
              child_byte_offset = 0;
              child_bitfield_bit_size = 0;
              child_bitfield_bit_offset = 0;
              child_is_base_class = false;
              child_is_deref_of_parent = false;
              language_flags = 0;
              return RemangleAsType(dem, GetClangTypeTypeNode(dem, raw_value),
                                    GetManglingFlavor(exe_ctx));
            }
        // Otherwise defer to TypeSystemClang.
        //
        // Swift skips bitfields when counting children. Unfortunately
        // this means we need to do this inefficient linear search here.
        CompilerType clang_child_type;
        for (size_t clang_idx = 0, swift_idx = 0; swift_idx <= idx;
             ++clang_idx) {
          child_bitfield_bit_size = 0;
          child_bitfield_bit_offset = 0;
          auto clang_child_type_or_err = clang_type.GetChildCompilerTypeAtIndex(
              exe_ctx, clang_idx, transparent_pointers, omit_empty_base_classes,
              ignore_array_bounds, child_name, child_byte_size,
              child_byte_offset, child_bitfield_bit_size,
              child_bitfield_bit_offset, child_is_base_class,
              child_is_deref_of_parent, valobj, language_flags);
          if (!clang_child_type_or_err)
            LLDB_LOG_ERROR(
                GetLog(LLDBLog::Types), clang_child_type_or_err.takeError(),
                "could not find child {1} using clang: {0}", clang_idx);
          else
            clang_child_type = *clang_child_type_or_err;
          if (!child_bitfield_bit_size && !child_bitfield_bit_offset)
            ++swift_idx;
          // FIXME: Why is this necessary?
          if (clang_child_type.IsTypedefType() &&
              clang_child_type.GetTypeName() ==
                  clang_child_type.GetTypedefedType().GetTypeName())
            clang_child_type = clang_child_type.GetTypedefedType();
        }
        if (clang_child_type) {
          // TypeSystemSwiftTypeRef can't properly handle C anonymous types, as
          // the only identifier CompilerTypes backed by this type system carry
          // is the type's mangled name. This is problematic for anonymous
          // types, as sibling anonymous types will share the exact same mangled
          // name, making it impossible to diferentiate between them. For
          // example, the following two anonymous structs in "MyStruct" share
          // the same name (which is MyStruct::(anonymous struct)):
          //
          // struct MyStruct {
          //         struct {
          //             float x;
          //             float y;
          //             float z;
          //         };
          //         struct {
          //           int a;
          //         };
          // };
          //
          // For this reason, forward any lookups of anonymous types to
          // TypeSystemClang instead, as that type system carries enough
          // information to handle anonymous types properly.
          auto ts_clang = clang_child_type.GetTypeSystem()
                              .dyn_cast_or_null<TypeSystemClang>();
          if (ts_clang &&
              ts_clang->IsAnonymousType(clang_child_type.GetOpaqueQualType()))
            return clang_child_type;

          std::string prefix;
          swift::Demangle::Demangler dem;
          swift::Demangle::NodePointer node =
              GetClangTypeTypeNode(dem, clang_child_type);
          if (!node)
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "object has no address");

          switch (node->getChild(0)->getKind()) {
          case swift::Demangle::Node::Kind::Class:
            prefix = "ObjectiveC.";
            break;
          default:
            break;
          }
          child_name = prefix + child_name;
          return RemangleAsType(dem, node, GetManglingFlavor(exe_ctx));
        }
      }
    }
    if (!exe_scope)
      LLDB_LOGF(GetLog(LLDBLog::Types),
                "Cannot compute the children of type %s without an execution "
                "context.",
                AsMangledName(type));

    // Runtime failed, fallback to SwiftASTContext.
    if (UseSwiftASTContextFallback(__FUNCTION__, type)) {
      // FIXME: SwiftASTContext can sometimes find more Clang types because it
      // imports Clang modules from source. We should be able to replicate this
      // and remove this fallback.
      auto result = fallback();
      if (result)
        DiagnoseSwiftASTContextFallback(__FUNCTION__, type);
      return result;
    }
    return llvm::createStringError(llvm::inconvertibleErrorCode(), error);
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
#ifndef NDEBUG
  auto result = impl();
  if (ShouldSkipValidation(type))
    return result;

  if (!ModuleList::GetGlobalModuleListProperties().GetSwiftValidateTypeSystem())
    return result;

  // FIXME:
  // No point comparing the results if the reflection data has more
  // information.  There's a nasty chicken & egg problem buried here:
  // Because the API deals out an index into a list of children we
  // can't mix&match between the two typesystems if there is such a
  // divergence. We'll need to replace all calls at once.
  if (get_ast_num_children() <
      llvm::expectedToStdOptional(
          runtime->GetNumChildren({weak_from_this(), type}, exe_scope))
          .value_or(0))
    return result;
  // When the child compiler type is an anonymous clang type,
  // GetChildCompilerTypeAtIndex will return the clang type directly. In this
  // case validation will fail as it can't correctly compare the mangled 
  // clang and Swift names, so return early.
  if (result) {
    if (auto ts_clang =
            result->GetTypeSystem().dyn_cast_or_null<TypeSystemClang>())
      if (ts_clang->IsAnonymousType(result->GetOpaqueQualType()))
        return result;
  } else {
    llvm::consumeError(result.takeError());
  }
  std::string ast_child_name;
  uint32_t ast_child_byte_size = 0;
  int32_t ast_child_byte_offset = 0;
  uint32_t ast_child_bitfield_bit_size = 0;
  uint32_t ast_child_bitfield_bit_offset = 0;
  bool ast_child_is_base_class = false;
  bool ast_child_is_deref_of_parent = false;
  uint64_t ast_language_flags = 0;
  auto defer = llvm::make_scope_exit([&] {
    // Ignore if SwiftASTContext got no result.
    if (ast_child_name.empty())
      return;
    llvm::StringRef suffix(ast_child_name);
    if (suffix.consume_front("__ObjC."))
      ast_child_name = suffix.str();
    assert((llvm::StringRef(child_name).contains('.') ||
            llvm::StringRef(ast_child_name).contains('.') ||
            llvm::StringRef(ast_child_name).starts_with("_") ||
            Equivalent(child_name, ast_child_name)));
    assert(ast_language_flags ||
           (Equivalent(std::optional<uint64_t>(child_byte_size),
                       std::optional<uint64_t>(ast_child_byte_size))));
    assert(Equivalent(std::optional<uint64_t>(child_byte_offset),
                      std::optional<uint64_t>(ast_child_byte_offset)));
    assert(
        Equivalent(child_bitfield_bit_offset, ast_child_bitfield_bit_offset));
    assert(Equivalent(child_bitfield_bit_size, ast_child_bitfield_bit_size));
    assert(Equivalent(child_is_base_class, ast_child_is_base_class));
    assert(Equivalent(child_is_deref_of_parent, ast_child_is_deref_of_parent));
    // There are cases where only the runtime correctly detects an indirect enum.
    ast_language_flags |= (language_flags & LanguageFlags::eIsIndirectEnumCase);
    assert(Equivalent(language_flags, ast_language_flags));
  });
#endif
  VALIDATE_AND_RETURN_EXPECTED(
      impl, GetChildCompilerTypeAtIndex, type, exe_ctx,
      (ReconstructType(type, exe_ctx), exe_ctx, idx, transparent_pointers,
       omit_empty_base_classes, ignore_array_bounds, ast_child_name,
       ast_child_byte_size, ast_child_byte_offset, ast_child_bitfield_bit_size,
       ast_child_bitfield_bit_offset, ast_child_is_base_class,
       ast_child_is_deref_of_parent, valobj, ast_language_flags));
}

size_t TypeSystemSwiftTypeRef::GetIndexOfChildMemberWithName(
    opaque_compiler_type_t type, StringRef name, ExecutionContext *exe_ctx,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  if (auto *exe_scope = exe_ctx->GetBestExecutionContextScope())
    if (auto *runtime =
            SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
      auto found_numidx = runtime->GetIndexOfChildMemberWithName(
          GetCanonicalType(type), name, exe_ctx, omit_empty_base_classes,
          child_indexes);
      // Only use the SwiftASTContext fallback if there was an
      // error. If the runtime had complete type info and couldn't
      // find a result, don't waste time retrying.
      if (found_numidx.first != SwiftLanguageRuntime::eError) {
        size_t index_size = found_numidx.second.value_or(0);
#ifndef NDEBUG
        // This block is a custom VALIDATE_AND_RETURN implementation to support
        // checking the return value, plus the by-ref `child_indexes`.
        if (!ModuleList::GetGlobalModuleListProperties()
                 .GetSwiftValidateTypeSystem())
          return index_size;
        if (!GetSwiftASTContext(GetSymbolContext(exe_ctx)))
          return index_size;
        auto ast_type = ReconstructType(type, exe_ctx);
        if (!ast_type)
          return index_size;
        std::vector<uint32_t> ast_child_indexes;
        auto ast_index_size =
            GetSwiftASTContext(GetSymbolContext(exe_ctx))
                ->GetIndexOfChildMemberWithName(ast_type, name, exe_ctx,
                                                omit_empty_base_classes,
                                                ast_child_indexes);
        // The runtime has more info than the AST. No useful validation can be
        // done.
        if (index_size > ast_index_size)
          return index_size;

        auto fail = [&]() {
          llvm::dbgs() << "{";
          llvm::interleaveComma(child_indexes, llvm::dbgs());
          llvm::dbgs() << "} != {";
          llvm::interleaveComma(ast_child_indexes, llvm::dbgs());
          llvm::dbgs() << "}\n";
          llvm::dbgs() << "failing type was " << (const char *)type
                       << ", member was " << name << "\n";
          assert(false &&
                 "TypeSystemSwiftTypeRef diverges from SwiftASTContext");
        };
        if (index_size != ast_index_size)
          fail();
        for (unsigned i = 0; i < index_size; ++i)
          if (child_indexes[i] < ast_child_indexes[i])
            // When the runtime may know know about more children. When this
            // happens, indexes will be larger. But if an index is smaller, that
            // means the runtime has dropped info somehow.
            fail();
#endif
        return index_size;
      }
      // If we're here, the runtime didn't find type info.
      assert(!found_numidx.first);
    }

  LLDB_LOGF(GetLog(LLDBLog::Types),
            "Using SwiftASTContext::GetIndexOfChildMemberWithName fallback for "
            "type %s",
            AsMangledName(type));

  // Runtime failed, fallback to SwiftASTContext.
  if (UseSwiftASTContextFallback(__FUNCTION__, type))
    if (auto swift_ast_context =
            GetSwiftASTContext(GetSymbolContext(exe_ctx))) {
      auto result = swift_ast_context->GetIndexOfChildMemberWithName(
          ReconstructType(type, exe_ctx), name, exe_ctx,
          omit_empty_base_classes, child_indexes);
      if (result)
        DiagnoseSwiftASTContextFallback(__FUNCTION__, type);
      return result;
    }
  return {};
}

size_t
TypeSystemSwiftTypeRef::GetNumTemplateArguments(opaque_compiler_type_t type,
                                                bool expand_pack) {
  auto impl = [&]() -> size_t {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);

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
  VALIDATE_AND_RETURN(impl, GetNumTemplateArguments, type, g_no_exe_ctx,
                      (ReconstructType(type), expand_pack));
}

CompilerType
TypeSystemSwiftTypeRef::GetTypeForFormatters(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType { return {weak_from_this(), type}; };
  VALIDATE_AND_RETURN(impl, GetTypeForFormatters, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

LazyBool
TypeSystemSwiftTypeRef::ShouldPrintAsOneLiner(opaque_compiler_type_t type,
                                              ValueObject *valobj) {
  auto impl = [&]() {
    if (type) {
      auto canonical = GetCanonicalType(type);
      if (canonical)
        if (IsImportedType(canonical.GetOpaqueQualType(), nullptr))
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
  VALIDATE_AND_RETURN(impl, ShouldPrintAsOneLiner, type, g_no_exe_ctx,
                      (ReconstructType(type), valobj));
}

bool TypeSystemSwiftTypeRef::IsMeaninglessWithoutDynamicResolution(
    opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    auto *node = DemangleCanonicalType(dem, type);
    return ContainsGenericTypeParameter(node) && !IsFunctionType(type);
  };
  VALIDATE_AND_RETURN(impl, IsMeaninglessWithoutDynamicResolution, type,
                      g_no_exe_ctx, (ReconstructType(type)));
}

CompilerType
TypeSystemSwiftTypeRef::GetAsClangTypeOrNull(lldb::opaque_compiler_type_t type,
                                             bool *is_imported) {
  using namespace swift::Demangle;
  Demangler dem;
  const char *mangled_name = AsMangledName(type);
  NodePointer node = GetDemangledType(dem, mangled_name);
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);
  // Directly resolve Clang typedefs into Clang types.  Imported
  // type aliases that point to Clang type that are also Swift builtins, like
  // Swift.Int, otherwise would resolved to Swift types.
  if (node && node->getKind() == Node::Kind::TypeAlias &&
      node->getNumChildren() == 2 && node->getChild(0)->hasText() &&
      node->getChild(0)->getText() == swift::MANGLING_MODULE_OBJC &&
      node->getChild(1)->hasText()) {
    auto node_clangtype = ResolveTypeAlias(dem, node, flavor,
                                           /*prefer_clang_types*/ true);
    if (node_clangtype.second)
      return node_clangtype.second;
  }
  CompilerType clang_type;
  bool imported = IsImportedType(type, &clang_type);
  if (is_imported)
    *is_imported = imported;
  return clang_type;
}

bool TypeSystemSwiftTypeRef::IsImportedType(opaque_compiler_type_t type,
                                            CompilerType *original_type) {
  auto impl = [&]() -> bool {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));

    auto *log = GetLog(LLDBLog::Types);
    // Types with generic parameters have to be resolved before calling
    // IsImportedType.
    if (log && ContainsGenericTypeParameter(node))
      LLDB_LOGF(log,
                "Checking if type %s which contains a generic parameter is "
                "an imported type",
                AsMangledName(type));

    if (!::IsImportedType(node))
      return false;
    // Early return if we don't need to look up the original type.
    if (!original_type)
      return true;

    // This is an imported Objective-C type; look it up in the debug info.
    llvm::SmallVector<CompilerContext, 2> decl_context;
    bool ignore_modules = false;
    if (!IsClangImportedType(node, decl_context, ignore_modules))
      return false;
    if (original_type)
      if (TypeSP clang_type = LookupClangType(AsMangledName(type), decl_context,
                                              ignore_modules))
        *original_type = clang_type->GetForwardCompilerType();
    return true;
  };
  // We can't validate the result because ReconstructType may call this
  // function, causing an infinite loop.
  return impl();
}

bool TypeSystemSwiftTypeRef::IsExistentialType(
    lldb::opaque_compiler_type_t type) {
  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = DemangleCanonicalOutermostType(dem, type);
  if (!node || node->getNumChildren() != 1)
    return false;
  switch (node->getKind()) {
  case Node::Kind::ExistentialMetatype:
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
  return RemangleAsType(dem, node, GetManglingFlavor());
}

bool TypeSystemSwiftTypeRef::IsErrorType(opaque_compiler_type_t type) {
  auto impl = [&]() -> bool {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer protocol_list = DemangleCanonicalOutermostType(dem, type);
    if (protocol_list && protocol_list->getKind() == Node::Kind::ProtocolList)
      for (auto type_list : *protocol_list)
        if (type_list && type_list->getKind() == Node::Kind::TypeList)
          for (auto type : *type_list)
            if (type && type->getKind() == Node::Kind::Type)
              for (auto protocol : *type)
                if (protocol->getKind() == Node::Kind::Protocol &&
                    protocol->getNumChildren() == 2) {
                  auto module = protocol->getChild(0);
                  auto identifier = protocol->getChild(1);
                  if (module->getKind() == Node::Kind::Module &&
                      module->getText() == swift::STDLIB_NAME &&
                      identifier->getKind() == Node::Kind::Identifier &&
                      identifier->getText() == "Error")
                    return true;
                }
    return false;
  };
  VALIDATE_AND_RETURN(impl, IsErrorType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
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

    return RemangleAsType(dem, error_type, GetManglingFlavor());
  };
  VALIDATE_AND_RETURN_STATIC(impl, GetErrorType);
}

CompilerType
TypeSystemSwiftTypeRef::GetWeakReferent(opaque_compiler_type_t type) {
  // FIXME: This is very similar to TypeSystemSwiftTypeRef::GetReferentType().
  using namespace swift::Demangle;
  Demangler dem;
  auto mangled = AsMangledName(type);
  NodePointer n = dem.demangleSymbol(mangled);
  if (!n || n->getKind() != Node::Kind::Global || !n->hasChildren())
    return {};
  n = n->getFirstChild();
  if (!n || n->getKind() != Node::Kind::TypeMangling || !n->hasChildren())
    return {};
  n = n->getFirstChild();
  if (!n || n->getKind() != Node::Kind::Type || !n->hasChildren())
    return {};
  n = n->getFirstChild();
  if (!n ||
      (n->getKind() != Node::Kind::Weak &&
       n->getKind() != Node::Kind::Unowned &&
       n->getKind() != Node::Kind::Unmanaged) ||
      !n->hasChildren())
    return {};
  n = n->getFirstChild();
  if (!n || n->getKind() != Node::Kind::Type || !n->hasChildren())
    return {};
  // FIXME: We only need to canonicalize this node, not the entire type.
  n = CanonicalizeSugar(dem, n->getFirstChild());
  if (!n || n->getKind() != Node::Kind::SugaredOptional || !n->hasChildren())
    return {};
  n = n->getFirstChild();
  return RemangleAsType(dem, n,
                        SwiftLanguageRuntime::GetManglingFlavor(mangled));
}

CompilerType
TypeSystemSwiftTypeRef::GetReferentType(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    auto mangled_name = AsMangledName(type);
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));
    if (!node ||
        (node->getKind() != Node::Kind::Unowned &&
         node->getKind() != Node::Kind::Unmanaged) ||
        !node->hasChildren())
      return {weak_from_this(), type};
    node = node->getFirstChild();
    if (!node || node->getKind() != Node::Kind::Type || !node->hasChildren())
      return {weak_from_this(), type};
    return RemangleAsType(dem, node, flavor);
  };
  VALIDATE_AND_RETURN(impl, GetReferentType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

swift::Demangle::NodePointer
TypeSystemSwiftTypeRef::GetStaticSelfType(swift::Demangle::Demangler &dem,
                                          swift::Demangle::NodePointer node) {
  using namespace swift::Demangle;
  return TypeSystemSwiftTypeRef::Transform(dem, node, [](NodePointer node) {
    if (node->getKind() != Node::Kind::DynamicSelf)
      return node;
    // Substitute the static type for dynamic self.
    assert(node->getNumChildren() == 1);
    if (node->getNumChildren() != 1)
      return node;
    NodePointer type = node->getChild(0);
    if (type->getKind() != Node::Kind::Type || type->getNumChildren() != 1)
      return node;
    return type->getChild(0);
  });
}

CompilerType
TypeSystemSwiftTypeRef::GetStaticSelfType(lldb::opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    auto mangled_name = AsMangledName(type);
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));
    auto *type_node = dem.createNode(Node::Kind::Type);
    type_node->addChild(GetStaticSelfType(dem, node), dem);
    return RemangleAsType(dem, type_node, flavor);
  };
  VALIDATE_AND_RETURN(impl, GetStaticSelfType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

CompilerType
TypeSystemSwiftTypeRef::GetInstanceType(opaque_compiler_type_t type,
                                        ExecutionContextScope *exe_scope) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    auto mangled_name = AsMangledName(type);
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    Demangler dem;
    NodePointer node = DemangleCanonicalType(dem, type);

    if (!node || ContainsUnresolvedTypeAlias(node)) {
      // If we couldn't resolve all type aliases, we might be in a REPL session
      // where getting to the debug information necessary for resolving that
      // type alias isn't possible, or the user might have defined the
      // type alias in the REPL. In these cases, fallback to asking the AST
      // for the canonical type.
      // Runtime failed, fallback to SwiftASTContext.
      if (UseSwiftASTContextFallback(__FUNCTION__, type))
        if (auto swift_ast_context =
                GetSwiftASTContext(GetSymbolContext(exe_scope))) {
          auto result = swift_ast_context->GetInstanceType(
              ReconstructType(type, exe_scope), exe_scope);
          if (result)
            DiagnoseSwiftASTContextFallback(__FUNCTION__, type);
          return result;
        }
      return {};
    }

    if (node->getKind() == Node::Kind::Metatype) {
      for (NodePointer child : *node)
        if (child->getKind() == Node::Kind::Type)
          return RemangleAsType(dem, child, flavor);
      return {};
    }
    return {weak_from_this(), type};
  };
  VALIDATE_AND_RETURN(impl, GetInstanceType, type, exe_scope,
                      (ReconstructType(type, exe_scope), exe_scope));
}

CompilerType TypeSystemSwiftTypeRef::CreateSILPackType(CompilerType type,
                                                       bool indirect) {
  using namespace swift::Demangle;
  auto mangled_name = type.GetMangledTypeName().GetStringRef();
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

  Demangler dem;
  NodePointer node =
      GetDemangledType(dem, type.GetMangledTypeName().GetStringRef());
  if (!node)
    return {};
  NodePointer pack_type = dem.createNode(indirect ? Node::Kind::SILPackIndirect
                                                  : Node::Kind::SILPackDirect);
  pack_type->addChild(node, dem);
  NodePointer type_node = dem.createNode(Node::Kind::Type);
  type_node->addChild(pack_type, dem);
  return RemangleAsType(dem, type_node, flavor);
}

static std::optional<TypeSystemSwiftTypeRef::PackTypeInfo>
decodeSILPackType(swift::Demangle::Demangler &dem, CompilerType type,
                  swift::Demangle::NodePointer &node) {
  TypeSystemSwiftTypeRef::PackTypeInfo info;
  node = GetDemangledType(dem, type.GetMangledTypeName().GetStringRef());
  if (!node)
    return {};
  switch (node->getKind()) {
  case Node::Kind::SILPackIndirect:
    info.indirect = true;
    break;
  case Node::Kind::SILPackDirect:
    info.indirect = false;
    break;
  default:
    return {};
  }
  if (node->getNumChildren() != 1)
    return {};
  node = node->getFirstChild();
  if (node->getKind() != Node::Kind::Type)
    return {};
  node = node->getFirstChild();
  info.expanded = node->getKind() == Node::Kind::Tuple;
  if (info.expanded)
    info.count = node->getNumChildren();
  return info;
}

std::optional<TypeSystemSwiftTypeRef::PackTypeInfo>
TypeSystemSwiftTypeRef::IsSILPackType(CompilerType type) {
  NodePointer node;
  swift::Demangle::Demangler dem;
  return decodeSILPackType(dem, type, node);
}

CompilerType TypeSystemSwiftTypeRef::GetSILPackElementAtIndex(CompilerType type,
                                                              unsigned i) {
  using namespace swift::Demangle;

  auto mangled_name = type.GetMangledTypeName().GetStringRef();
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

  Demangler dem;
  NodePointer node;
  if (auto Info = decodeSILPackType(dem, type, node)) {
    if (node->getNumChildren() < i)
      return {};
    node = node->getChild(i);
    if (!node || node->getKind() != Node::Kind::TupleElement)
      return {};
    node = node->getFirstChild();
    if (!node || node->getKind() != Node::Kind::Type)
      return {};
    return RemangleAsType(dem, node, flavor);
  }
  return {};
}

CompilerType TypeSystemSwiftTypeRef::CreateTupleType(
    const std::vector<TupleElement> &elements) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler dem;
    auto *tuple_type = dem.createNode(Node::Kind::Type);
    auto *tuple = dem.createNode(Node::Kind::Tuple);
    tuple_type->addChild(tuple, dem);
    if (elements.empty())
      return RemangleAsType(dem, tuple_type, GetManglingFlavor());

    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(
        elements.front().element_type.GetMangledTypeName());

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
      if (!element_type)
        return {};
      type->addChild(element_type, dem);
      assert(flavor == SwiftLanguageRuntime::GetManglingFlavor(
                           element.element_type.GetMangledTypeName()));
    }

    return RemangleAsType(dem, tuple_type, flavor);
  };

  // The signature of VALIDATE_AND_RETURN doesn't support this function, below
  // is an inlined function-specific variation.
#ifndef NDEBUG
  if (ModuleList::GetGlobalModuleListProperties()
          .GetSwiftValidateTypeSystem()) {
    auto result = impl();
    auto target_sp = GetTargetWP().lock();
    if (!target_sp)
      return result;
    auto swift_ast_ctx = GetSwiftASTContext(
        SymbolContext(target_sp, target_sp->GetExecutableModule()));
    if (!swift_ast_ctx)
      return result;
    std::vector<TupleElement> ast_elements;
    std::transform(elements.begin(), elements.end(),
                   std::back_inserter(ast_elements), [&](TupleElement element) {
                     return TupleElement(
                         element.element_name,
                         ReconstructType(element.element_type, nullptr));
                   });
    bool equivalent =
        Equivalent(result, swift_ast_ctx->CreateTupleType(ast_elements));
    if (!equivalent) {
      result.dump();
      auto a = swift_ast_ctx->CreateTupleType(elements);
      llvm::dbgs() << "AST type: " << a.GetMangledTypeName() << "\n";
      llvm::dbgs() << "failing tuple type\n";
    }

    assert(equivalent &&
           "TypeSystemSwiftTypeRef diverges from SwiftASTContext");
    return result;
  }
#endif
  return impl();
}

bool TypeSystemSwiftTypeRef::IsTupleType(lldb::opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));
    return node && node->getKind() == Node::Kind::Tuple;
  };
  VALIDATE_AND_RETURN(impl, IsTupleType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

std::optional<TypeSystemSwift::NonTriviallyManagedReferenceKind>
TypeSystemSwiftTypeRef::GetNonTriviallyManagedReferenceKind(
    lldb::opaque_compiler_type_t type) {
  auto impl = [&]()
      -> std::optional<TypeSystemSwift::NonTriviallyManagedReferenceKind> {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));
    if (!node)
      return {};
    switch (node->getKind()) {
    default:
      return {};
    case Node::Kind::Unmanaged:
      return NonTriviallyManagedReferenceKind::eUnmanaged;
    case Node::Kind::Unowned:
      return NonTriviallyManagedReferenceKind::eUnowned;
    case Node::Kind::Weak:
      return NonTriviallyManagedReferenceKind::eWeak;
    }
  };
  VALIDATE_AND_RETURN(impl, GetNonTriviallyManagedReferenceKind, type,
                      g_no_exe_ctx, (ReconstructType(type)));
}

void TypeSystemSwiftTypeRef::DumpTypeDescription(
    opaque_compiler_type_t type, lldb::DescriptionLevel level,
    ExecutionContextScope *exe_scope) {

  StreamFile s(stdout, false);
  DumpTypeDescription(type, &s, level, exe_scope);
}

void TypeSystemSwiftTypeRef::DumpTypeDescription(
    opaque_compiler_type_t type, Stream &s, lldb::DescriptionLevel level,
    ExecutionContextScope *exe_scope) {
  DumpTypeDescription(type, &s, false, true, level, exe_scope);
}

void TypeSystemSwiftTypeRef::DumpTypeDescription(
    opaque_compiler_type_t type, bool print_help_if_available,
    bool print_extensions_if_available, lldb::DescriptionLevel level,
    ExecutionContextScope *exe_scope) {
  StreamFile s(stdout, false);
  DumpTypeDescription(type, &s, print_help_if_available,
                      print_extensions_if_available, level, exe_scope);
}

void TypeSystemSwiftTypeRef::DumpTypeDescription(
    opaque_compiler_type_t type, Stream *s, bool print_help_if_available,
    bool print_extensions_if_available, lldb::DescriptionLevel level,
    ExecutionContextScope *exe_scope) {
  // Currently, we need an execution scope so we can access the runtime, which
  // in turn owns the reflection context, which is used to read the typeref. If
  // we were to decouple the reflection context from the runtime, we'd be able
  // to read the typeref without needing an execution scope.
  if (exe_scope) {
    if (auto *runtime =
            SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
      const auto initial_written_bytes = s->GetWrittenBytes();
      s->Printf("Swift Reflection Metadata:\n");
      runtime->DumpTyperef({weak_from_this(), type}, this, s);
      if (s->GetWrittenBytes() == initial_written_bytes)
        s->Printf("<could not resolve type>\n");
    }
  }

  // Also dump the swift ast context info, as this functions should not be in
  // any critical path.
  if (auto swift_ast_context =
          GetSwiftASTContext(GetSymbolContext(exe_scope))) {
    s->PutCString("Source code info:\n");
    swift_ast_context->DumpTypeDescription(
        ReconstructType(type, exe_scope), s, print_help_if_available,
        print_extensions_if_available, level);
  }
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
    opaque_compiler_type_t type, Stream &s, lldb::Format format,
    const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, ExecutionContextScope *exe_scope,
    bool is_base_class) {
  if (!type)
    return false;
  const char *mangled_name = AsMangledName(type);
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

  auto impl = [&]() -> bool {
    using namespace swift::Demangle;
    Demangler dem;    
    auto *node = DemangleCanonicalType(dem, type);
    if (!node)
      return false;
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
      return DumpDataExtractor(data, &s, data_offset, format, data_byte_size,
                               item_count, UINT32_MAX, LLDB_INVALID_ADDRESS,
                               bitfield_bit_size, bitfield_bit_offset,
                               exe_scope);
    }
    case Node::Kind::DynamicSelf:
    case Node::Kind::Unmanaged:
    case Node::Kind::Unowned:
    case Node::Kind::Weak: {
      auto *referent_node = node->getFirstChild();
      assert(referent_node->getKind() == Node::Kind::Type);
      auto referent_type = RemangleAsType(dem, referent_node, flavor);
      return referent_type.DumpTypeValue(
          &s, format, data, data_offset, data_byte_size, bitfield_bit_size,
          bitfield_bit_offset, exe_scope, is_base_class);
    }
    case Node::Kind::BoundGenericStructure:
      return false;
    case Node::Kind::Structure: {
      // In some instances, a swift `structure` wraps an objc enum. The enum
      // case needs to be handled, but structs are no-ops.
      auto resolved = ResolveTypeAlias(dem, node, flavor, true);
      auto resolved_type = std::get<CompilerType>(resolved);
      if (!resolved_type)
        return false;

      bool is_signed;
      if (!resolved_type.IsEnumerationType(is_signed))
        // The type is a clang struct, not an enum.
        return false;

      if (!resolved_type.GetTypeSystem().isa_and_nonnull<TypeSystemClang>())
        return false;

      // The type is an enum imported from clang.
      auto qual_type = ClangUtil::GetQualType(resolved_type);
      auto *enum_type =
          llvm::dyn_cast_or_null<clang::EnumType>(qual_type.getTypePtrOrNull());
      if (!enum_type)
        return false;
      auto *importer = GetNameImporter();
      if (!importer)
        return false;
      if (!data_byte_size)
        return false;
      StringRef case_name;
      if (is_signed) {
        int64_t val = data.GetMaxS64(&data_offset, data_byte_size);
        case_name = importer->ProjectEnumCase(enum_type->getDecl(), val);
      } else {
        uint64_t val = data.GetMaxU64(&data_offset, data_byte_size);
        case_name = importer->ProjectEnumCase(enum_type->getDecl(), val);
      }
      if (case_name.empty())
        return false;
      s << case_name;
      return true;
    }
    case Node::Kind::Enum:
    case Node::Kind::BoundGenericEnum: {
      std::string error;
      if (exe_scope)
        if (auto runtime =
                SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
          ExecutionContext exe_ctx;
          exe_scope->CalculateExecutionContext(exe_ctx);
          auto case_name = runtime->GetEnumCaseName({weak_from_this(), type},
                                                    data, &exe_ctx);
          if (case_name && !case_name->empty()) {
            // The syntactic sugar for `.none` is `nil`.
            if (*case_name == "none" && IsOptionalType(type))
              s << "nil";
            else
              s << *case_name;
            return true;
          }
          if (!case_name)
            error = toString(case_name.takeError());
        }

      s << error;
      return false;
    }
    case Node::Kind::TypeAlias:
    case Node::Kind::BoundGenericTypeAlias: {
      // This means we have an unresolved type alias that even
      // SwiftASTContext couldn't resolve. This happens for ObjC
      // typedefs such as CFString in the REPL. More investigation is
      // needed.
      if (auto swift_ast_context =
              GetSwiftASTContext(GetSymbolContext(exe_scope)))
        return swift_ast_context->DumpTypeValue(
            ReconstructType(type, exe_scope), s, format, data, data_offset,
            data_byte_size, bitfield_bit_size, bitfield_bit_offset, exe_scope,
            is_base_class);
      return false;
    }
    case Node::Kind::ProtocolList:
      return false;
    default:
      assert(false && "Unhandled node kind");
      LLDB_LOGF(GetLog(LLDBLog::Types),
                "DumpTypeValue: Unhandled node kind for type %s",
                AsMangledName(type));
      return false;
    }
  };

  {
    // If this is a typealias defined in the expression evaluator,
    // then we don't have debug info to resolve it from.
    using namespace swift::Demangle;
    Demangler dem;
    auto *node = DemangleCanonicalType(dem, type);
    bool unresolved_typealias = false;
    CollectTypeInfo(dem, node, flavor, unresolved_typealias);
    if (!node || unresolved_typealias) {
      if (auto swift_ast_ctx = GetSwiftASTContext(GetSymbolContext(exe_scope)))
        return swift_ast_ctx->DumpTypeValue(
            ReconstructType(type, exe_scope), s, format, data, data_offset,
            data_byte_size, bitfield_bit_size, bitfield_bit_offset, exe_scope,
            is_base_class);
      return false;
    }
  }

#ifndef NDEBUG
  StreamString ast_s;
  auto defer = llvm::make_scope_exit([&] {
    assert(Equivalent(ConstString(ast_s.GetString()),
                      ConstString(((StreamString *)&s)->GetString())) &&
           "TypeSystemSwiftTypeRef diverges from SwiftASTContext");
  });
#endif

  auto better_or_equal = [](bool a, bool b) -> bool {
    if (a || a == b)
      return true;

    llvm::dbgs() << "TypeSystemSwiftTypeRef: " << a << " SwiftASTContext: " << b
                 << "\n";
    return false;
  };
  VALIDATE_AND_RETURN_CUSTOM(
      impl, DumpTypeValue, type, better_or_equal, exe_scope,
      (ReconstructType(type, exe_scope), ast_s, format, data, data_offset,
       data_byte_size, bitfield_bit_size, bitfield_bit_offset, exe_scope,
       is_base_class));
}

bool TypeSystemSwiftTypeRef::IsPointerOrReferenceType(
    opaque_compiler_type_t type, CompilerType *pointee_type) {
  auto impl = [&]() {
    return IsPointerType(type, pointee_type) ||
           IsReferenceType(type, pointee_type, nullptr);
  };
  VALIDATE_AND_RETURN(impl, IsPointerOrReferenceType, type, g_no_exe_ctx,
                      (ReconstructType(type), nullptr));
}
std::optional<size_t>
TypeSystemSwiftTypeRef::GetTypeBitAlign(opaque_compiler_type_t type,
                                        ExecutionContextScope *exe_scope) {
  // This method doesn't use VALIDATE_AND_RETURN because except for
  // fixed-size types the SwiftASTContext implementation forwards to
  // SwiftLanguageRuntime anyway and for many fixed-size types the
  // fixed layout still returns an incorrect default alignment of 0.

  // Look up static alignment in the debug info if we have already
  // parsed this type.
  if (TypeSP type_sp = GetCachedType(type))
    if (type_sp->GetLayoutCompilerType().GetOpaqueQualType() != type)
      return type_sp->GetLayoutCompilerType().GetTypeBitAlign(exe_scope);

  // Clang types can be resolved even without a process.
  if (CompilerType clang_type = GetAsClangTypeOrNull(type)) {
    // Swift doesn't know pointers: return the size alignment of the
    // object pointer instead of the underlying object.
    if (Flags(clang_type.GetTypeInfo()).AllSet(eTypeIsObjC | eTypeIsClass))
      return GetPointerByteSize() * 8;
    if (auto clang_align = clang_type.GetTypeBitAlign(exe_scope))
      return clang_align;
  }
  if (!exe_scope) {
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "Couldn't compute alignment of type %s without an execution "
              "context.",
              AsMangledName(type));
    return {};
  }
  if (auto *runtime =
          SwiftLanguageRuntime::Get(exe_scope->CalculateProcess())) {
    if (auto result =
            runtime->GetBitAlignment({weak_from_this(), type}, exe_scope))
      return result;
    // If this is an expression context, perhaps the type was
    // defined in the expression. In that case we don't have debug
    // info for it, so defer to SwiftASTContext.
    if (llvm::isa<TypeSystemSwiftTypeRefForExpressions>(this)) {
      ExecutionContext exe_ctx;
      if (exe_scope)
        exe_scope->CalculateExecutionContext(exe_ctx);
      return ReconstructType({weak_from_this(), type}, &exe_ctx)
          .GetTypeBitAlign(exe_scope);
    }
  }

  // If there is no process, we can still try to get the static
  // alignment information out of DWARF. Because it is stored in the
  // Type object we need to look that up by name again.
  if (TypeSP type_sp = FindTypeInModule(type))
    if (type_sp->GetLayoutCompilerType().GetOpaqueQualType() != type)
      return type_sp->GetLayoutCompilerType().GetTypeBitAlign(exe_scope);
  LLDB_LOGF(GetLog(LLDBLog::Types),
            "Couldn't compute alignment of type %s using static debug info.",
            AsMangledName(type));
  return {};
}

bool TypeSystemSwiftTypeRef::IsSIMDType(CompilerType type) {
  using namespace swift::Demangle;
  Demangler dem;
  swift::Demangle::NodePointer global =
      dem.demangleSymbol(type.GetMangledTypeName().GetStringRef());
  using Kind = swift::Demangle::Node::Kind;
  auto *simd_storage = swift_demangle::ChildAtPath(
      global, {Kind::TypeMangling, Kind::Type, Kind::Structure});
  if (!simd_storage || simd_storage->getNumChildren() != 2)
    return false;
  auto base_type = simd_storage->getFirstChild();
  auto wrapper = simd_storage->getLastChild();
  return wrapper && wrapper->getKind() == Kind::Identifier &&
         wrapper->hasText() && wrapper->getText().starts_with("SIMD") &&
         base_type && base_type->getKind() == Kind::Structure;
}

#ifndef NDEBUG
static bool IsSIMDNode(NodePointer node) {
  // A SIMD vector is a clang typealias whose identifier starts with "simd_".
  if (node->getKind() == Node::Kind::TypeAlias && node->getNumChildren() >= 2) {
    NodePointer module = node->getFirstChild();
    NodePointer identifier = node->getChild(1);
    return module->getKind() == Node::Kind::Module &&
           module->getText() == swift::MANGLING_MODULE_OBJC &&
           identifier->getKind() == Node::Kind::Identifier &&
           identifier->getText().starts_with("simd_");
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
               identifier->getText().starts_with("SIMD");
      }
    }
  }
  return false;
}
#endif

bool TypeSystemSwiftTypeRef::IsOptionalType(lldb::opaque_compiler_type_t type) {
  using namespace swift::Demangle;
  Demangler dem;
  swift::Demangle::NodePointer global = dem.demangleSymbol(AsMangledName(type));
  using Kind = swift::Demangle::Node::Kind;
  swift::Demangle::NodePointer node = swift_demangle::ChildAtPath(
      global, {Kind::TypeMangling, Kind::Type, Kind::BoundGenericEnum,
               Kind::Type, Kind::Enum});
  if (!node || node->getNumChildren() != 2)
    return false;
  NodePointer module = node->getChild(0);
  NodePointer identifier = node->getChild(1);
  return module->getKind() == Node::Kind::Module &&
         module->getText() == swift::STDLIB_NAME &&
         identifier->getKind() == Node::Kind::Identifier &&
         identifier->getText() == "Optional";
}

CompilerType TypeSystemSwiftTypeRef::GetOptionalType(CompilerType type) {
  using namespace swift::Demangle;
  Demangler dem;
  swift::Demangle::NodePointer global =
      dem.demangleSymbol(type.GetMangledTypeName().GetStringRef());
  using Kind = swift::Demangle::Node::Kind;
  auto *bge = swift_demangle::ChildAtPath(
      global, {Kind::TypeMangling, Kind::Type, Kind::BoundGenericEnum});
  if (!bge || bge->getNumChildren() != 2)
    return {};
  auto optional_type = swift_demangle::NodeAtPath(bge->getChild(1),
                                                  {Kind::TypeList, Kind::Type});
  if (!optional_type)
    return {};
  auto ts = type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
  if (!ts)
    return {};
  return ts->RemangleAsType(
      dem, optional_type,
      SwiftLanguageRuntime::GetManglingFlavor(type.GetMangledTypeName()));
}

bool TypeSystemSwiftTypeRef::IsTypedefType(opaque_compiler_type_t type) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));
    return node && (node->getKind() == Node::Kind::TypeAlias ||
                    node->getKind() == Node::Kind::BoundGenericTypeAlias);
  };

#ifndef NDEBUG
  {
    // Sometimes SwiftASTContext returns the resolved AnyObject type.
    Demangler dem;
    NodePointer node = GetDemangledType(dem, AsMangledName(type));
    if (IsAnyObjectTypeAlias(node))
      return impl();
  }
#endif

  VALIDATE_AND_RETURN(impl, IsTypedefType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

CompilerType
TypeSystemSwiftTypeRef::GetTypedefedType(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType {
    using namespace swift::Demangle;
    Demangler dem;
    const char *mangled_name = AsMangledName(type);
    NodePointer node = GetDemangledType(dem, mangled_name);
    if (!node || (node->getKind() != Node::Kind::TypeAlias &&
                  node->getKind() != Node::Kind::BoundGenericTypeAlias))
      return {};
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    auto pair = ResolveTypeAlias(dem, node, flavor);
    NodePointer type_node = dem.createNode(Node::Kind::Type);
    if (NodePointer resolved = std::get<swift::Demangle::NodePointer>(pair)) {
      type_node->addChild(resolved, dem);
    } else {
      NodePointer clang_node =
          GetClangTypeNode(std::get<CompilerType>(pair), dem);
      if (!clang_node)
        return {};
      type_node->addChild(clang_node, dem);
    }
    return RemangleAsType(dem, type_node, flavor);
  };
  VALIDATE_AND_RETURN(impl, GetTypedefedType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}

CompilerType
TypeSystemSwiftTypeRef::GetFullyUnqualifiedType(opaque_compiler_type_t type) {
  auto impl = [&]() -> CompilerType { return {weak_from_this(), type}; };

  VALIDATE_AND_RETURN(impl, GetFullyUnqualifiedType, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}
uint32_t
TypeSystemSwiftTypeRef::GetNumDirectBaseClasses(opaque_compiler_type_t type) {
  auto impl = [&]() -> uint32_t {
    CompilerType class_ty(weak_from_this(), type);
    if (auto target_sp = GetTargetWP().lock())
      if (auto *runtime = SwiftLanguageRuntime::Get(target_sp->GetProcessSP()))
        if (runtime->GetBaseClass(class_ty))
          return 1;
    return 0;
  };

  VALIDATE_AND_RETURN(impl, GetNumDirectBaseClasses, type, g_no_exe_ctx,
                      (ReconstructType(type)));
}
CompilerType TypeSystemSwiftTypeRef::GetDirectBaseClassAtIndex(
    opaque_compiler_type_t type, size_t idx, uint32_t *bit_offset_ptr) {
  auto impl = [&]() {
    if (idx != 0)
      return CompilerType();
    CompilerType class_ty(weak_from_this(), type);
    if (auto target_sp = GetTargetWP().lock())
      if (auto *runtime = SwiftLanguageRuntime::Get(target_sp->GetProcessSP()))
        return runtime->GetBaseClass(class_ty);
    return CompilerType();
  };

  VALIDATE_AND_RETURN(impl, GetDirectBaseClassAtIndex, type, g_no_exe_ctx,
                      (ReconstructType(type), idx, nullptr));
}
bool TypeSystemSwiftTypeRef::IsReferenceType(opaque_compiler_type_t type,
                                             CompilerType *pointee_type,
                                             bool *is_rvalue) {
  auto impl = [&]() {
    using namespace swift::Demangle;
    const auto *mangled_name = AsMangledName(type);
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);
    if (!node || node->getNumChildren() != 1 ||
        node->getKind() != Node::Kind::InOut)
      return false;

    if (pointee_type) {
      NodePointer referenced = node->getFirstChild();
      auto *type = dem.createNode(Node::Kind::Type);
      type->addChild(referenced, dem);
      *pointee_type = RemangleAsType(dem, type, flavor);
    }

    if (is_rvalue)
      *is_rvalue = false;

    return true;
  };

  VALIDATE_AND_RETURN(impl, IsReferenceType, type, g_no_exe_ctx,
                      (ReconstructType(type), nullptr, nullptr));
}

CompilerType
TypeSystemSwiftTypeRef::GetGenericArgumentType(opaque_compiler_type_t type,
                                               size_t idx) {
  auto impl = [&]() -> CompilerType {
    const auto *mangled_name = AsMangledName(type);
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

    Demangler dem;
    NodePointer node = DemangleCanonicalOutermostType(dem, type);
    if (!node || node->getNumChildren() != 2)
      return {};

    if (node->getKind() != Node::Kind::BoundGenericClass &&
        node->getKind() != Node::Kind::BoundGenericStructure &&
        node->getKind() != Node::Kind::BoundGenericEnum &&
        node->getKind() != Node::Kind::BoundGenericFunction &&
        node->getKind() != Node::Kind::BoundGenericProtocol &&
        node->getKind() != Node::Kind::BoundGenericTypeAlias &&
        node->getKind() != Node::Kind::BoundGenericProtocol)
      return {};

    NodePointer type_list = node->getChild(1);
    if (!type_list || type_list->getNumChildren() <= idx)
      return {};

    NodePointer generic_argument_type = type_list->getChild(idx);
    if (!generic_argument_type)
      return {};

    return RemangleAsType(dem, generic_argument_type, flavor);
  };

  VALIDATE_AND_RETURN(impl, GetGenericArgumentType, type, g_no_exe_ctx,
                      (ReconstructType(type), idx));
}

llvm::SmallVector<std::pair<int, int>, 1>
TypeSystemSwiftTypeRef::GetDependentGenericParamListForType(
    llvm::StringRef type) {
  llvm::SmallVector<std::pair<int, int>, 1> dependent_params;
  Demangler dem;
  NodePointer type_node = GetDemangledType(dem, type);
  if (!type_node)
    return dependent_params;
  if (type_node->getNumChildren() != 2)
    return dependent_params;

  NodePointer type_list = type_node->getLastChild();
  for (auto *child_type : *type_list) {
    if (child_type->getKind() != Node::Kind::Type)
      continue;
    if (child_type->getNumChildren() != 1)
      continue;

    NodePointer dep_gen_param_type =  child_type->getFirstChild();
    if (dep_gen_param_type->getKind() != Node::Kind::DependentGenericParamType)
      continue;
    if (dep_gen_param_type->getNumChildren() != 2)
      continue;
    
    NodePointer depth = dep_gen_param_type->getFirstChild();
    NodePointer index = dep_gen_param_type->getLastChild();

    if (!depth->hasIndex() || !index->hasIndex())
      continue;
    dependent_params.emplace_back(depth->getIndex(), index->getIndex());
  }
  return dependent_params;
}

swift::Mangle::ManglingFlavor
TypeSystemSwiftTypeRef::GetManglingFlavor(ExecutionContext *exe_ctx) {
  auto sc = GetSymbolContext(exe_ctx);
  auto *cu = sc.comp_unit;
  // Cache the result for the last recently used CU.
  if (cu != m_lru_is_embedded.first)
    m_lru_is_embedded = {cu, ShouldEnableEmbeddedSwift(sc.comp_unit)
                                 ? swift::Mangle::ManglingFlavor::Embedded
                                 : swift::Mangle::ManglingFlavor::Default};
  return m_lru_is_embedded.second;
}

#ifndef NDEBUG
bool TypeSystemSwiftTypeRef::ShouldSkipValidation(opaque_compiler_type_t type) {
  auto mangled_name = GetMangledTypeName(type);
  // NSNotificationName is a typedef to a NSString in clang type, but it's a
  // struct in SwiftASTContext. Skip validation in this case.
  if (mangled_name == "$sSo18NSNotificationNameaD")
    return true;

  // $s10Foundation12NotificationV4NameaD is a typealias to NSNotificationName,
  // so we skip validation in that casse as well.
  if (mangled_name == "$s10Foundation12NotificationV4NameaD")
    return true;

  if (mangled_name ==  "$s10Foundation12NotificationVD")
    return true;

  if (mangled_name == "$sSo9NSDecimalaD")
    return true;
  // Reconstruct($sSo11CFStringRefaD) returns a non-typealias type, breaking
  // isTypedef().
  if (mangled_name == "$sSo11CFStringRefaD")
    return true;

  if (mangled_name == "$sSo7CGPointVD")
    return true;

  if (mangled_name == "$sSo6CGSizeVD")
    return true;

  // We skip validation when dealing with a builtin type since builtins are
  // considered type aliases by Swift, which we're deviating from since
  // SwiftASTContext reconstructs Builtin types as TypeAliases pointing to the
  // actual Builtin types, but mangled names always describe the underlying
  // builtins directly.
  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = GetDemangledType(dem, AsMangledName(type));
  if (node && node->getKind() == Node::Kind::BuiltinTypeName)
    return true;

  // SIMD types have some special handling in the compiler, causing divergences
  // on the way SwiftASTContext and TypeSystemSwiftTypeRef view the same type.
  if (node && IsSIMDNode(node))
    return true;

  return false;
}
#endif
