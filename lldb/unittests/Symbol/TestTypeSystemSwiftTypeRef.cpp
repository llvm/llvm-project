//===-- TestTypeSystemSwiftTypeRef.cpp ------------------------------------===//
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

#include "gtest/gtest.h"

#include "lldb/Symbol/SwiftASTContext.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/Demangler.h"
#include "swift/Strings.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

struct TestTypeSystemSwiftTypeRef : public testing::Test {
  TypeSystemSwiftTypeRef m_swift_ts = TypeSystemSwiftTypeRef(nullptr);

  CompilerType GetCompilerType(std::string mangled_name) {
    ConstString internalized(mangled_name);
    return m_swift_ts.GetTypeFromMangledTypename(internalized);
  }
};

/// Helper class to conveniently construct demangle tree hierarchies.
class NodeBuilder {
  using NodePointer = swift::Demangle::NodePointer;
  using Kind = swift::Demangle::Node::Kind;
  
  swift::Demangle::Demangler &m_dem;

public:
  NodeBuilder(swift::Demangle::Demangler &dem) : m_dem(dem) {}
  NodePointer Node(Kind kind, StringRef text) {
    return m_dem.createNode(kind, text);
  }
  NodePointer NodeWithIndex(Kind kind, swift::Demangle::Node::IndexType index) {
    return m_dem.createNode(kind, index);
  }
  NodePointer Node(Kind kind, NodePointer child0 = nullptr,
                   NodePointer child1 = nullptr,
                   NodePointer child2 = nullptr,
                   NodePointer child3 = nullptr) {
    NodePointer node = m_dem.createNode(kind);

    if (child0)
      node->addChild(child0, m_dem);
    if (child1)
      node->addChild(child1, m_dem);
    if (child2)
      node->addChild(child2, m_dem);
    if (child3)
      node->addChild(child3, m_dem);
    return node;
  }
  NodePointer IntType() {
    return Node(
        Node::Kind::Type,
        Node(Node::Kind::Structure,
             Node(Node::Kind::Module, swift::STDLIB_NAME),
             Node(Node::Kind::Identifier, swift::BUILTIN_TYPE_NAME_INT)));
  }
  NodePointer GlobalTypeMangling(NodePointer type) {
    assert(type && type->getKind() == Node::Kind::Type);
    return Node(Node::Kind::Global, Node(Node::Kind::TypeMangling, type));
  }
  NodePointer GlobalType(NodePointer type) {
    assert(type && type->getKind() != Node::Kind::Type &&
           type->getKind() != Node::Kind::TypeMangling &&
           type->getKind() != Node::Kind::Global);
    return GlobalTypeMangling(Node(Node::Kind::Type, type));
  }

  std::string Mangle(NodePointer node) { return mangleNode(node); }
};

TEST_F(TestTypeSystemSwiftTypeRef, Array) {
  using namespace swift::Demangle;
  Demangler dem;
  NodeBuilder b(dem);
  NodePointer n = b.GlobalType(
      b.Node(Node::Kind::BoundGenericStructure,
             b.Node(Node::Kind::Type,
                    b.Node(Node::Kind::Structure,
                           b.Node(Node::Kind::Module, swift::STDLIB_NAME),
                           b.Node(Node::Kind::Identifier, "Array"))),
             b.Node(Node::Kind::TypeList, b.IntType())));
  CompilerType int_array = GetCompilerType(b.Mangle(n));
  ASSERT_TRUE(int_array.IsArrayType(nullptr, nullptr, nullptr));
  NodePointer int_node = b.GlobalTypeMangling(b.IntType());
  CompilerType int_type = GetCompilerType(b.Mangle(int_node));
  ASSERT_EQ(int_array.GetArrayElementType(nullptr), int_type);
}

TEST_F(TestTypeSystemSwiftTypeRef, Function) {
  using namespace swift::Demangle;
  Demangler dem;
  NodeBuilder b(dem);
  NodePointer int_node = b.GlobalTypeMangling(b.IntType());
  CompilerType int_type = GetCompilerType(b.Mangle(int_node));
  NodePointer void_node = b.GlobalType(b.Node(Node::Kind::Tuple));
  CompilerType void_type = GetCompilerType(b.Mangle(void_node));
  {
    NodePointer n = b.GlobalType(
        b.Node(Node::Kind::FunctionType,
               b.Node(Node::Kind::ArgumentTuple,
                      b.Node(Node::Kind::Type, b.Node(Node::Kind::Tuple))),
               b.Node(Node::Kind::ReturnType,
                      b.Node(Node::Kind::Type, b.Node(Node::Kind::Tuple)))));
    CompilerType void_void = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(void_void.IsFunctionType(nullptr));
    ASSERT_TRUE(void_void.IsFunctionPointerType());
    ASSERT_EQ(void_void.GetNumberOfFunctionArguments(), 0);
  }
  {
    NodePointer n = b.GlobalType(
        b.Node(Node::Kind::ImplFunctionType, b.Node(Node::Kind::ImplEscaping),
               b.Node(Node::Kind::ImplConvention, "@callee_guaranteed")));
    CompilerType impl_void_void = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(impl_void_void.IsFunctionType(nullptr));
    ASSERT_EQ(impl_void_void.GetNumberOfFunctionArguments(), 0);
  }
  {
    NodePointer n = b.GlobalType(b.Node(
        Node::Kind::ImplFunctionType, b.Node(Node::Kind::ImplEscaping),
        b.Node(Node::Kind::ImplConvention, "@callee_guaranteed"),
        b.Node(Node::Kind::ImplParameter,
               b.Node(Node::Kind::ImplConvention, "@unowned"), b.IntType()),
        b.Node(Node::Kind::ImplParameter,
               b.Node(Node::Kind::ImplConvention, "@unowned"),
               b.Node(Node::Kind::Tuple))));
    CompilerType impl_two_args = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(impl_two_args.IsFunctionType(nullptr));
    ASSERT_EQ(impl_two_args.GetNumberOfFunctionArguments(), 2);
    ASSERT_EQ(impl_two_args.GetFunctionArgumentAtIndex(0), int_type);
    ASSERT_EQ(impl_two_args.GetFunctionArgumentAtIndex(1), void_type);
    ASSERT_EQ(impl_two_args.GetFunctionArgumentTypeAtIndex(0), int_type);
    ASSERT_EQ(impl_two_args.GetFunctionArgumentTypeAtIndex(1), void_type);
    ASSERT_EQ(impl_two_args.GetFunctionReturnType(), void_type);
  }
  {
    NodePointer n = b.GlobalType(b.Node(
        Node::Kind::FunctionType,
        b.Node(Node::Kind::ArgumentTuple,
               b.Node(Node::Kind::Type,
                      b.Node(Node::Kind::Tuple,
                             b.Node(Node::Kind::TupleElement, b.IntType()),
                             b.Node(Node::Kind::TupleElement,
                                    b.Node(Node::Kind::Tuple))))),
        b.Node(Node::Kind::ReturnType,
               b.Node(Node::Kind::Type, b.Node(Node::Kind::Tuple)))));
    CompilerType two_args = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(two_args.IsFunctionType(nullptr));
    ASSERT_EQ(two_args.GetNumberOfFunctionArguments(), 2);
    ASSERT_EQ(two_args.GetFunctionArgumentAtIndex(0), int_type);
    ASSERT_EQ(two_args.GetFunctionArgumentAtIndex(1), void_type);
    ASSERT_EQ(two_args.GetFunctionArgumentTypeAtIndex(0), int_type);
    ASSERT_EQ(two_args.GetFunctionArgumentTypeAtIndex(1), void_type);
    ASSERT_EQ(two_args.GetFunctionReturnType(), void_type);
  }
  {
    NodePointer n = b.GlobalType(b.Node(
        Node::Kind::FunctionType,
        b.Node(Node::Kind::ArgumentTuple,
               b.Node(Node::Kind::Type, b.Node(Node::Kind::Tuple))),
        b.Node(Node::Kind::ReturnType, b.Node(Node::Kind::Type, b.IntType()))));
    CompilerType void_int = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(void_int.IsFunctionType(nullptr));
    ASSERT_EQ(void_int.GetFunctionReturnType(), int_type);
  }
  {
    NodePointer n = b.GlobalType(b.Node(
        Node::Kind::ImplFunctionType, b.Node(Node::Kind::ImplEscaping),
        b.Node(Node::Kind::ImplConvention, "@callee_guaranteed"),
        b.Node(Node::Kind::ImplResult,
               b.Node(Node::Kind::ImplConvention, "@unowned"), b.IntType())));
    CompilerType impl_void_int = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(impl_void_int.IsFunctionType(nullptr));
    ASSERT_EQ(impl_void_int.GetFunctionReturnType(), int_type);
  }
}

TEST_F(TestTypeSystemSwiftTypeRef, Pointer) {
  using namespace swift::Demangle;
  Demangler dem;
  NodeBuilder b(dem);
  {
    NodePointer n = b.GlobalType(b.Node(Node::Kind::BuiltinTypeName,
                                        swift::BUILTIN_TYPE_NAME_RAWPOINTER));
    CompilerType p = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(p.IsPointerType(nullptr));
  }
  {
    NodePointer n =
        b.GlobalType(b.Node(Node::Kind::BuiltinTypeName,
                            swift::BUILTIN_TYPE_NAME_UNSAFEVALUEBUFFER));
    CompilerType p = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(p.IsPointerType(nullptr));
  }
  {
    NodePointer n = b.GlobalType(b.Node(Node::Kind::BuiltinTypeName,
                                        swift::BUILTIN_TYPE_NAME_NATIVEOBJECT));
    CompilerType p = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(p.IsPointerType(nullptr));
  }
  {
    NodePointer n = b.GlobalType(b.Node(Node::Kind::BuiltinTypeName,
                                        swift::BUILTIN_TYPE_NAME_BRIDGEOBJECT));
    CompilerType p = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(p.IsPointerType(nullptr));
  }
}

TEST_F(TestTypeSystemSwiftTypeRef, Void) {
  using namespace swift::Demangle;
  Demangler dem;
  NodeBuilder b(dem);
  {
    NodePointer n = b.GlobalType(b.Node(Node::Kind::Tuple));
    CompilerType v = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(v.IsVoidType());
  }
}

TEST_F(TestTypeSystemSwiftTypeRef, Reference) {
  using namespace swift::Demangle;
  Demangler dem;
  NodeBuilder b(dem);
  {
    NodePointer n = b.GlobalType(b.Node(Node::Kind::InOut, b.IntType()));
    CompilerType ref = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(ref.IsReferenceType(nullptr, nullptr));
    CompilerType pointee;
    bool is_rvalue = true;
    ASSERT_TRUE(ref.IsReferenceType(&pointee, &is_rvalue));
    NodePointer int_node = b.GlobalTypeMangling(b.IntType());
    CompilerType int_type = GetCompilerType(b.Mangle(int_node));
    ASSERT_EQ(int_type, pointee);
    ASSERT_FALSE(is_rvalue);
  }
}

TEST_F(TestTypeSystemSwiftTypeRef, Aggregate) {
  using namespace swift::Demangle;
  Demangler dem;
  NodeBuilder b(dem);
  {
    NodePointer n = b.GlobalType(b.Node(Node::Kind::Tuple));
    CompilerType tuple = GetCompilerType(b.Mangle(n));
    ASSERT_TRUE(tuple.IsAggregateType());
    // Yes, Int is a struct.
    NodePointer int_node = b.GlobalTypeMangling(b.IntType());
    CompilerType int_type = GetCompilerType(b.Mangle(int_node));
    ASSERT_TRUE(int_type.IsAggregateType());
    NodePointer t = b.GlobalType(b.Node(Node::Kind::DependentGenericParamType,
                                        b.NodeWithIndex(Node::Kind::Index, 0),
                                        b.NodeWithIndex(Node::Kind::Index, 0)));
    CompilerType tau = GetCompilerType(b.Mangle(t));
    ASSERT_FALSE(tau.IsAggregateType());
  }
}
