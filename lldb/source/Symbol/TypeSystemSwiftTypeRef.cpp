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
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Utility/Log.h"

#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/Demangler.h"
#include "swift/Strings.h"

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
      // TODO: No Swift type found -- this could be a Clang typdef.
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

const char *TypeSystemSwiftTypeRef::AsMangledName(void *type) {
  assert(type && *reinterpret_cast<const char *>(type) == '$' &&
         "wrong type system");
  return reinterpret_cast<const char *>(type);
}

ConstString TypeSystemSwiftTypeRef::GetMangledTypeName(void *type) {
  // FIXME: Suboptimal performance, because the ConstString is looked up again.
  return ConstString(AsMangledName(type));
}

void *TypeSystemSwiftTypeRef::ReconstructType(void *type) {
  Status error;
  return m_swift_ast_context->ReconstructType(GetMangledTypeName(type), error);
}

CompilerType TypeSystemSwiftTypeRef::ReconstructType(CompilerType type) {
  return {m_swift_ast_context, ReconstructType(type.GetOpaqueQualType())};
}

CompilerType TypeSystemSwiftTypeRef::GetTypeFromMangledTypename(
    ConstString mangled_typename) {
  return {this, (void *)mangled_typename.AsCString()};
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
bool TypeSystemSwiftTypeRef::Verify(lldb::opaque_compiler_type_t type) {
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

namespace {
template <typename T> bool Equivalent(T l, T r) { return l == r; }
/// Compare two swift types from different type systems by comparing their
/// (canonicalized) mangled name.
template <> bool Equivalent<CompilerType>(CompilerType l, CompilerType r) {
  return l.GetMangledTypeName() == r.GetMangledTypeName();
}
}
#endif

// This can be removed once the transition is complete.
#define VALIDATE_AND_RETURN(IMPL, EXPECTED)                                    \
  do {                                                                         \
    auto result = IMPL();                                                      \
    if (m_swift_ast_context)                                                   \
      assert(Equivalent(result, (EXPECTED)) &&                                 \
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

swift::Demangle::NodePointer
TypeSystemSwiftTypeRef::DemangleCanonicalType(swift::Demangle::Demangler &Dem,
                                              void *opaque_type) {
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

bool TypeSystemSwiftTypeRef::IsArrayType(void *type, CompilerType *element_type,
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
bool TypeSystemSwiftTypeRef::IsAggregateType(void *type) {
  return m_swift_ast_context->IsAggregateType(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsDefined(void *type) {
  return m_swift_ast_context->IsDefined(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsFloatingPointType(void *type, uint32_t &count,
                                                 bool &is_complex) {
  return m_swift_ast_context->IsFloatingPointType(ReconstructType(type), count,
                                                  is_complex);
}

bool TypeSystemSwiftTypeRef::IsFunctionType(void *type, bool *is_variadic_ptr) {
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
size_t TypeSystemSwiftTypeRef::GetNumberOfFunctionArguments(void *type) {
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
        if (node->getNumChildren() != 1 ||
            node->getKind() != Node::Kind::Type)
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
TypeSystemSwiftTypeRef::GetFunctionArgumentAtIndex(void *type,
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
        if (node->getNumChildren() != 1 ||
            node->getKind() != Node::Kind::Type)
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
bool TypeSystemSwiftTypeRef::IsFunctionPointerType(void *type) {
  auto impl = [&]() -> bool { return IsFunctionType(type, nullptr); };
  VALIDATE_AND_RETURN(
      impl, m_swift_ast_context->IsFunctionPointerType(ReconstructType(type)));
}
bool TypeSystemSwiftTypeRef::IsIntegerType(void *type, bool &is_signed) {
  return m_swift_ast_context->IsIntegerType(ReconstructType(type), is_signed);
}
bool TypeSystemSwiftTypeRef::IsPossibleDynamicType(void *type,
                                                   CompilerType *target_type,
                                                   bool check_cplusplus,
                                                   bool check_objc) {
  return m_swift_ast_context->IsPossibleDynamicType(
      ReconstructType(type), target_type, check_cplusplus, check_objc);
}
bool TypeSystemSwiftTypeRef::IsPointerType(void *type,
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
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->IsPointerType(ReconstructType(type),
                                                               pointee_type));
}
bool TypeSystemSwiftTypeRef::IsScalarType(void *type) {
  return m_swift_ast_context->IsScalarType(ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsVoidType(void *type) {
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
bool TypeSystemSwiftTypeRef::GetCompleteType(void *type) {
  return m_swift_ast_context->GetCompleteType(ReconstructType(type));
}
// AST related queries
uint32_t TypeSystemSwiftTypeRef::GetPointerByteSize() {
  return m_swift_ast_context->GetPointerByteSize();
}
// Accessors
ConstString TypeSystemSwiftTypeRef::GetTypeName(void *type) {
  return m_swift_ast_context->GetTypeName(ReconstructType(type));
}
ConstString
TypeSystemSwiftTypeRef::GetDisplayTypeName(void *type,
                                           const SymbolContext *sc) {
  return m_swift_ast_context->GetDisplayTypeName(ReconstructType(type), sc);
}
uint32_t TypeSystemSwiftTypeRef::GetTypeInfo(
    void *type, CompilerType *pointee_or_element_clang_type) {
  return m_swift_ast_context->GetTypeInfo(ReconstructType(type),
                                          pointee_or_element_clang_type);
}
lldb::LanguageType TypeSystemSwiftTypeRef::GetMinimumLanguage(void *type) {
  return m_swift_ast_context->GetMinimumLanguage(ReconstructType(type));
}
lldb::TypeClass TypeSystemSwiftTypeRef::GetTypeClass(void *type) {
  return m_swift_ast_context->GetTypeClass(ReconstructType(type));
}

// Creating related types
CompilerType TypeSystemSwiftTypeRef::GetArrayElementType(void *type,
                                                         uint64_t *stride) {
  auto impl = [&]() {
    CompilerType element_type;
    IsArrayType(type, &element_type, nullptr, nullptr);
    return element_type;
  };
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->GetArrayElementType(
                                ReconstructType(type), nullptr));
}
CompilerType TypeSystemSwiftTypeRef::GetCanonicalType(void *type) {
  return m_swift_ast_context->GetCanonicalType(ReconstructType(type));
}
int TypeSystemSwiftTypeRef::GetFunctionArgumentCount(void *type) {
  auto impl = [&]() -> int { return GetNumberOfFunctionArguments(type); };
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->GetFunctionArgumentCount(
                                ReconstructType(type)));
}
CompilerType
TypeSystemSwiftTypeRef::GetFunctionArgumentTypeAtIndex(void *type, size_t idx) {
  auto impl = [&] { return GetFunctionArgumentAtIndex(type, idx); };
  VALIDATE_AND_RETURN(impl, m_swift_ast_context->GetFunctionArgumentTypeAtIndex(
                                ReconstructType(type), idx));
}
CompilerType TypeSystemSwiftTypeRef::GetFunctionReturnType(void *type) {
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
size_t TypeSystemSwiftTypeRef::GetNumMemberFunctions(void *type) {
  return m_swift_ast_context->GetNumMemberFunctions(ReconstructType(type));
}
TypeMemberFunctionImpl
TypeSystemSwiftTypeRef::GetMemberFunctionAtIndex(void *type, size_t idx) {
  return m_swift_ast_context->GetMemberFunctionAtIndex(ReconstructType(type),
                                                       idx);
}
CompilerType TypeSystemSwiftTypeRef::GetPointeeType(void *type) {
  return m_swift_ast_context->GetPointeeType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetPointerType(void *type) {
  return m_swift_ast_context->GetPointerType(ReconstructType(type));
}

// Exploring the type
llvm::Optional<uint64_t>
TypeSystemSwiftTypeRef::GetBitSize(lldb::opaque_compiler_type_t type,
                                   ExecutionContextScope *exe_scope) {
  return m_swift_ast_context->GetBitSize(ReconstructType(type), exe_scope);
}
llvm::Optional<uint64_t>
TypeSystemSwiftTypeRef::GetByteStride(lldb::opaque_compiler_type_t type,
                                      ExecutionContextScope *exe_scope) {
  return m_swift_ast_context->GetByteStride(ReconstructType(type), exe_scope);
}
lldb::Encoding TypeSystemSwiftTypeRef::GetEncoding(void *type,
                                                   uint64_t &count) {
  return m_swift_ast_context->GetEncoding(ReconstructType(type), count);
}
lldb::Format TypeSystemSwiftTypeRef::GetFormat(void *type) {
  return m_swift_ast_context->GetFormat(ReconstructType(type));
}
uint32_t
TypeSystemSwiftTypeRef::GetNumChildren(void *type, bool omit_empty_base_classes,
                                       const ExecutionContext *exe_ctx) {
  return m_swift_ast_context->GetNumChildren(ReconstructType(type),
                                             omit_empty_base_classes, exe_ctx);
}
uint32_t TypeSystemSwiftTypeRef::GetNumFields(void *type) {
  return m_swift_ast_context->GetNumFields(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetFieldAtIndex(
    void *type, size_t idx, std::string &name, uint64_t *bit_offset_ptr,
    uint32_t *bitfield_bit_size_ptr, bool *is_bitfield_ptr) {
  return m_swift_ast_context->GetFieldAtIndex(
      ReconstructType(type), idx, name, bit_offset_ptr, bitfield_bit_size_ptr,
      is_bitfield_ptr);
}
CompilerType TypeSystemSwiftTypeRef::GetChildCompilerTypeAtIndex(
    void *type, ExecutionContext *exe_ctx, size_t idx,
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
TypeSystemSwiftTypeRef::GetIndexOfChildWithName(void *type, const char *name,
                                                bool omit_empty_base_classes) {
  return m_swift_ast_context->GetIndexOfChildWithName(
      ReconstructType(type), name, omit_empty_base_classes);
}
size_t TypeSystemSwiftTypeRef::GetIndexOfChildMemberWithName(
    void *type, const char *name, bool omit_empty_base_classes,
    std::vector<uint32_t> &child_indexes) {
  return m_swift_ast_context->GetIndexOfChildMemberWithName(
      ReconstructType(type), name, omit_empty_base_classes, child_indexes);
}
size_t TypeSystemSwiftTypeRef::GetNumTemplateArguments(void *type) {
  return m_swift_ast_context->GetNumTemplateArguments(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetTypeForFormatters(void *type) {
  return m_swift_ast_context->GetTypeForFormatters(ReconstructType(type));
}
LazyBool TypeSystemSwiftTypeRef::ShouldPrintAsOneLiner(void *type,
                                                       ValueObject *valobj) {
  return m_swift_ast_context->ShouldPrintAsOneLiner(ReconstructType(type),
                                                    valobj);
}
bool TypeSystemSwiftTypeRef::IsMeaninglessWithoutDynamicResolution(void *type) {
  return m_swift_ast_context->IsMeaninglessWithoutDynamicResolution(
      ReconstructType(type));
}
bool TypeSystemSwiftTypeRef::IsImportedType(CompilerType type,
                                            CompilerType *original_type) {
  return m_swift_ast_context->IsImportedType(
      {m_swift_ast_context, ReconstructType(type.GetOpaqueQualType())},
      original_type);
}
bool TypeSystemSwiftTypeRef::IsErrorType(CompilerType compiler_type) {
  return m_swift_ast_context->IsErrorType(
      {m_swift_ast_context,
       ReconstructType(compiler_type.GetOpaqueQualType())});
}
CompilerType TypeSystemSwiftTypeRef::GetErrorType() {
  return m_swift_ast_context->GetErrorType();
}

CompilerType
TypeSystemSwiftTypeRef::GetReferentType(CompilerType compiler_type) {
  return m_swift_ast_context->GetReferentType(
      {m_swift_ast_context,
       ReconstructType(compiler_type.GetOpaqueQualType())});
}

CompilerType TypeSystemSwiftTypeRef::GetInstanceType(void *type) {
  return m_swift_ast_context->GetInstanceType(ReconstructType(type));
}
TypeSystemSwift::TypeAllocationStrategy
TypeSystemSwiftTypeRef::GetAllocationStrategy(CompilerType type) {
  return m_swift_ast_context->GetAllocationStrategy(
      {m_swift_ast_context, ReconstructType(type.GetOpaqueQualType())});
}
CompilerType TypeSystemSwiftTypeRef::CreateTupleType(
    const std::vector<TupleElement> &elements) {
  return m_swift_ast_context->CreateTupleType(elements);
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(
    void *type, bool print_help_if_available,
    bool print_extensions_if_available) {
  return m_swift_ast_context->DumpTypeDescription(
      ReconstructType(type), print_help_if_available, print_help_if_available);
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(
    void *type, Stream *s, bool print_help_if_available,
    bool print_extensions_if_available) {
  return m_swift_ast_context->DumpTypeDescription(
      ReconstructType(type), s, print_help_if_available,
      print_extensions_if_available);
}

// Dumping types
#ifndef NDEBUG
/// Convenience LLVM-style dump method for use in the debugger only.
LLVM_DUMP_METHOD void
TypeSystemSwiftTypeRef::dump(lldb::opaque_compiler_type_t type) const {
  llvm::dbgs() << reinterpret_cast<const char *>(type) << "\n";
}
#endif

void TypeSystemSwiftTypeRef::DumpValue(
    void *type, ExecutionContext *exe_ctx, Stream *s, lldb::Format format,
    const DataExtractor &data, lldb::offset_t data_offset,
    size_t data_byte_size, uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset, bool show_types, bool show_summary,
    bool verbose, uint32_t depth) {
  return m_swift_ast_context->DumpValue(
      ReconstructType(type), exe_ctx, s, format, data, data_offset,
      data_byte_size, bitfield_bit_size, bitfield_bit_offset, show_types,
      show_summary, verbose, depth);
}

bool TypeSystemSwiftTypeRef::DumpTypeValue(
    void *type, Stream *s, lldb::Format format, const DataExtractor &data,
    lldb::offset_t data_offset, size_t data_byte_size,
    uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
    ExecutionContextScope *exe_scope, bool is_base_class) {
  return m_swift_ast_context->DumpTypeValue(
      ReconstructType(type), s, format, data, data_offset, data_byte_size,
      bitfield_bit_size, bitfield_bit_offset, exe_scope, is_base_class);
}

void TypeSystemSwiftTypeRef::DumpTypeDescription(void *type) {
  return m_swift_ast_context->DumpTypeDescription(ReconstructType(type));
}
void TypeSystemSwiftTypeRef::DumpTypeDescription(void *type, Stream *s) {
  return m_swift_ast_context->DumpTypeDescription(ReconstructType(type), s);
}
void TypeSystemSwiftTypeRef::DumpSummary(void *type, ExecutionContext *exe_ctx,
                                         Stream *s, const DataExtractor &data,
                                         lldb::offset_t data_offset,
                                         size_t data_byte_size) {
  return m_swift_ast_context->DumpSummary(ReconstructType(type), exe_ctx, s,
                                          data, data_offset, data_byte_size);
}
bool TypeSystemSwiftTypeRef::IsPointerOrReferenceType(
    void *type, CompilerType *pointee_type) {
  return m_swift_ast_context->IsPointerOrReferenceType(ReconstructType(type),
                                                       pointee_type);
}
llvm::Optional<size_t>
TypeSystemSwiftTypeRef::GetTypeBitAlign(void *type,
                                        ExecutionContextScope *exe_scope) {
  return m_swift_ast_context->GetTypeBitAlign(ReconstructType(type), exe_scope);
}
bool TypeSystemSwiftTypeRef::IsTypedefType(void *type) {
  return m_swift_ast_context->IsTypedefType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetTypedefedType(void *type) {
  return m_swift_ast_context->GetTypedefedType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetFullyUnqualifiedType(void *type) {
  return m_swift_ast_context->GetFullyUnqualifiedType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetNonReferenceType(void *type) {
  return m_swift_ast_context->GetNonReferenceType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetLValueReferenceType(void *type) {
  return m_swift_ast_context->GetLValueReferenceType(ReconstructType(type));
}
CompilerType TypeSystemSwiftTypeRef::GetRValueReferenceType(void *type) {
  return m_swift_ast_context->GetRValueReferenceType(ReconstructType(type));
}
uint32_t TypeSystemSwiftTypeRef::GetNumDirectBaseClasses(void *type) {
  return m_swift_ast_context->GetNumDirectBaseClasses(ReconstructType(type));
}
CompilerType
TypeSystemSwiftTypeRef::GetDirectBaseClassAtIndex(void *type, size_t idx,
                                                  uint32_t *bit_offset_ptr) {
  return m_swift_ast_context->GetDirectBaseClassAtIndex(ReconstructType(type),
                                                        idx, bit_offset_ptr);
}
bool TypeSystemSwiftTypeRef::IsReferenceType(void *type,
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
    lldb::opaque_compiler_type_t type) {
  return m_swift_ast_context->ShouldTreatScalarValueAsAddress(
      ReconstructType(type));
}
