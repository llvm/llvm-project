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
#include "llvm/ADT/ArrayRef.h"

namespace lldb_private {
namespace swift_demangle {

/// Access an inner node by following the given Node::Kind path.
///
/// Note: The Node::Kind path is relative to the given root node. The root
/// node's Node::Kind must not be included in the path.
inline swift::Demangle::NodePointer
nodeAtPath(swift::Demangle::NodePointer root,
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

} // namespace swift_demangle
} // namespace lldb_private

#endif
