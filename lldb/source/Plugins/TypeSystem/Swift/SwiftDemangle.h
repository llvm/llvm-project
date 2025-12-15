//===-- SwiftDemangle.h ---------------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftDemangle_h_
#define liblldb_SwiftDemangle_h_

#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/Demangler.h"
#include "swift/Demangling/ManglingFlavor.h"
#include "llvm/ADT/ArrayRef.h"
#include <initializer_list>

namespace lldb_private {
namespace swift_demangle {

using NodePointer = swift::Demangle::NodePointer;
using Node = swift::Demangle::Node;

/// Returns the first child of `node` whose kind is in `kinds`.
inline NodePointer GetFirstChildOfKind(NodePointer node,
                                       llvm::ArrayRef<Node::Kind> kinds) {
  if (!node)
    return nullptr;
  for (auto *child : *node)
    if (llvm::is_contained(kinds, child->getKind()))
      return child;
  return nullptr;
}

/// Assumes that `to_replace` is a child of `parent`, and replaces it with
/// `new_child`. The Nodes must all be owned by the same context.
inline void ReplaceChildWith(Node &parent, Node &to_replace, Node &new_child) {
  for (unsigned idx = 0; idx < parent.getNumChildren(); idx++) {
    auto *child = parent.getChild(idx);
    if (child == &to_replace) {
      parent.replaceChild(idx, &new_child);
      return;
    }
  }
  llvm_unreachable("invalid child passed to replaceChildWith");
}

/// Access an inner node by following the given Node::Kind path.
///
/// Note: The Node::Kind path is relative to the given root node. The root
/// node's Node::Kind must not be included in the path.
inline swift::Demangle::NodePointer
ChildAtPath(swift::Demangle::NodePointer root,
            llvm::ArrayRef<swift::Demangle::Node::Kind> kind_path) {
  if (!root)
    return nullptr;

  auto *node = root;
  for (auto kind : kind_path) {
    bool found = false;
    for (auto *child : *node) {
      assert(child && "swift::Demangle::Node has null child");
      if (child && child->getKind() == kind) {
        node = child;
        found = true;
        break;
      }
    }
    // The current step (`kind`) of the path was not found in the children of
    // the current `node`. The requested path does not exist.
    if (!found)
      return nullptr;
  }

  return node;
}

/// Like \ref childAtPath, but starts the comparison at \c root.
inline swift::Demangle::NodePointer
NodeAtPath(swift::Demangle::NodePointer root,
           llvm::ArrayRef<swift::Demangle::Node::Kind> kind_path) {
  if (!root || !kind_path.size() || root->getKind() != kind_path.front())
    return nullptr;
  return ChildAtPath(root, kind_path.drop_front());
}

/// Find the first child node of root that satisfies cond.
inline swift::Demangle::NodePointer
FindIf(swift::Demangle::NodePointer root,
       std::function<bool(swift::Demangle::NodePointer)> cond) {
  if (!root)
    return nullptr;

  auto *node = root;
  if (cond(node))
    return node;
  for (auto *child : *node) {
    assert(child && "swift::Demangle::Node has null child");
    if (auto *found = FindIf(child, cond))
      return found;
  }

  return nullptr;
}

/// \return the child of the TypeMangling node.
static swift::Demangle::NodePointer
GetTypeMangling(swift::Demangle::NodePointer n) {
  using namespace swift::Demangle;
  if (!n || n->getKind() != Node::Kind::Global)
    return nullptr;
  n = n->getFirstChild();
  if (!n || n->getKind() != Node::Kind::TypeMangling || !n->hasChildren())
    return nullptr;
  n = n->getFirstChild();
  return n;
}

/// \return the child of the \p Type node.
static swift::Demangle::NodePointer GetType(swift::Demangle::NodePointer n) {
  using namespace swift::Demangle;
  n = GetTypeMangling(n);
  if (!n || n->getKind() != Node::Kind::Type || !n->hasChildren())
    return nullptr;
  n = n->getFirstChild();
  return n;
}

/// Demangle a mangled type name and return the child of the \p Type node.
inline swift::Demangle::NodePointer
GetDemangledType(swift::Demangle::Demangler &dem, llvm::StringRef name) {
  return GetType(dem.demangleSymbol(name));
}

/// Demangle a mangled type name and return the child of the \p TypeMangling
/// node.
inline swift::Demangle::NodePointer
GetDemangledTypeMangling(swift::Demangle::Demangler &dem,
                         llvm::StringRef name) {
  return GetTypeMangling(dem.demangleSymbol(name));
}

/// Create a parent-child chain of demangle nodes for the specified node kinds.
///
/// \param initial[in] The starting node to construct the chain from.
/// \param kinds[in] The node kind for each created node.
/// \param factory[in] The factory to create nodes.
/// \return The tail leaf node created, whose kind is the last kind in \c kinds.
inline NodePointer
MakeNodeChain(NodePointer initial,
              std::initializer_list<swift::Demangle::Node::Kind> kinds,
              NodeFactory &factory) {
  auto *current = initial;
  for (auto kind : kinds) {
    if (!current)
      return nullptr;
    auto *child = factory.createNode(kind);
    current->addChild(child, factory);
    current = child;
  }
  return current;
}

/// Create a parent-child chain of demangle nodes for the specified node kinds.
///
/// \param initial[in] The starting node to construct the chain from.
/// \param kinds[in] The node kind for each created node.
/// \param factory[in] The factory to create nodes.
/// \return A pair of nodes: the head and tail of the constructed nodes.
inline std::pair<NodePointer, NodePointer>
MakeNodeChain(std::initializer_list<swift::Demangle::Node::Kind> kinds,
              NodeFactory &factory) {
  auto *root = factory.createNode(Node::Kind::Global);
  auto *leaf = MakeNodeChain(root, kinds, factory);
  return {root, leaf};
}

/// Wrap node in Global/TypeMangling/Type.
inline swift::Demangle::NodePointer
MangleType(swift::Demangle::Demangler &dem,
           swift::Demangle::NodePointer type_node) {
  auto *global = dem.createNode(Node::Kind::Global);
  auto *type_mangling = dem.createNode(Node::Kind::TypeMangling);
  global->addChild(type_mangling, dem);
  auto *type = dem.createNode(Node::Kind::Type);
  type_mangling->addChild(type, dem);
  type->addChild(type_node, dem);
  return global;
}

/// Produce a type mangle tree for a nominal type.
inline swift::Demangle::NodePointer
CreateNominal(swift::Demangle::Demangler &dem, swift::Demangle::Node::Kind kind,
              llvm::StringRef module_name, llvm::StringRef type_name) {
  auto *nominal = dem.createNode(kind);
  auto *m = dem.createNodeWithAllocatedText(Node::Kind::Module, module_name);
  auto *id = dem.createNodeWithAllocatedText(Node::Kind::Identifier, type_name);
  nominal->addChild(m, dem);
  nominal->addChild(id, dem);
  return nominal;
}

/// Produce a type mangling for a class.
inline ManglingErrorOr<std::string>
MangleClass(swift::Demangle::Demangler &dem, llvm::StringRef module_name,
            llvm::StringRef class_name, swift::Mangle::ManglingFlavor flavor) {
  auto *node = CreateNominal(dem, Node::Kind::Class, module_name, class_name);
  return mangleNode(MangleType(dem, node), flavor);
}

/// Create a mangled name for a type node.
inline swift::Demangle::ManglingErrorOr<std::string>
GetMangledName(swift::Demangle::Demangler &dem,
               swift::Demangle::NodePointer node,
               swift::Mangle::ManglingFlavor flavor) {
  using namespace swift::Demangle;
  auto global = dem.createNode(Node::Kind::Global);
  auto type_mangling = dem.createNode(Node::Kind::TypeMangling);
  global->addChild(type_mangling, dem);
  type_mangling->addChild(node, dem);
  return mangleNode(global, flavor);
}

/// Returns true if this type contains an error node anywhere.
inline bool ContainsError(StringRef mangled_name) {
  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = GetDemangledType(dem, mangled_name);
  return swift_demangle::FindIf(node, [](NodePointer node) -> bool {
    // This node is an in-band error, not a Swift.Error type, which is a
    // protocol.
    if (node->getKind() == Node::Kind::ErrorType)
      return true;
    return false;
  });
}

} // namespace swift_demangle
} // namespace lldb_private

#endif
