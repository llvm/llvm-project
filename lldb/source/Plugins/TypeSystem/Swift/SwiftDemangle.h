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
#include "llvm/ADT/ArrayRef.h"

namespace lldb_private {
namespace swift_demangle {

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

/// \return the child of the \p Type node.
static swift::Demangle::NodePointer GetType(swift::Demangle::NodePointer n) {
  using namespace swift::Demangle;
  if (!n || n->getKind() != Node::Kind::Global)
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
inline swift::Demangle::NodePointer
GetDemangledType(swift::Demangle::Demangler &dem, llvm::StringRef name) {
  return GetType(dem.demangleSymbol(name));
}

} // namespace swift_demangle
} // namespace lldb_private

#endif
