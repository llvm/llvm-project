//===-- SwiftMangled.h ------------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftMangled_h_
#define liblldb_SwiftMangled_h_

#include "lldb/Core/DemangledNameInfo.h"
#include "swift/Demangling/Demangle.h"

using namespace swift::Demangle;

/// A NodePrinter class with range tracking capabilities.
///
/// When used instead of a regular NodePrinter, this class will store additional
/// range information of the demangled name in the `info` attribute, such as the
/// range of the name of a method.
class TrackingNodePrinter : public NodePrinter {
public:
  TrackingNodePrinter(DemangleOptions options) : NodePrinter(options) {}

  lldb_private::DemangledNameInfo getInfo() { return info; }

private:
  lldb_private::DemangledNameInfo info;
  std::optional<unsigned> parametersDepth;
  std::optional<unsigned> genericsSignatureDepth;

  void startName() {
    if (!info.hasBasename())
      info.BasenameRange.first = getStreamLength();
  }

  void endName() {
    if (!info.hasBasename())
      info.BasenameRange.second = getStreamLength();
  }

  void startGenericSignature(unsigned depth) {
    if (genericsSignatureDepth || !info.hasBasename() ||
        info.TemplateArgumentsRange.first <
            info.TemplateArgumentsRange.second) {
      return;
    }
    info.TemplateArgumentsRange.first = getStreamLength();
    genericsSignatureDepth = depth;
  }

  void endGenericSignature(unsigned depth) {
    if (!genericsSignatureDepth || *genericsSignatureDepth != depth ||
        info.TemplateArgumentsRange.first <
            info.TemplateArgumentsRange.second) {
      return;
    }
    info.TemplateArgumentsRange.second = getStreamLength();
  }

  void startParameters(unsigned depth) {
    if (parametersDepth || !info.hasBasename() ||
        info.ArgumentsRange.first < info.ArgumentsRange.second) {
      return;
    }
    info.ArgumentsRange.first = getStreamLength();
    parametersDepth = depth;
  }

  void endParameters(unsigned depth) {
    if (!parametersDepth || *parametersDepth != depth ||
        info.ArgumentsRange.first < info.ArgumentsRange.second) {
      return;
    }
    info.ArgumentsRange.second = getStreamLength();
  }

  bool shouldTrackNameRange(NodePointer Node) const {
    assert(Node);
    switch (Node->getKind()) {
    case Node::Kind::Function:
    case Node::Kind::Constructor:
    case Node::Kind::Allocator:
    case Node::Kind::ExplicitClosure:
      return true;
    default:
      return false;
    }
  }

  void printFunctionName(bool hasName, llvm::StringRef &OverwriteName,
                         llvm::StringRef &ExtraName, bool MultiWordName,
                         int &ExtraIndex, NodePointer Entity,
                         unsigned int depth) override {
    if (shouldTrackNameRange(Entity))
      startName();
    NodePrinter::printFunctionName(hasName, OverwriteName, ExtraName,
                                   MultiWordName, ExtraIndex, Entity, depth);
    if (shouldTrackNameRange(Entity))
      endName();
  }

  void printGenericSignature(NodePointer Node, unsigned depth) override {
    startGenericSignature(depth);
    NodePrinter::printGenericSignature(Node, depth);
    endGenericSignature(depth);
  }

  void printFunctionParameters(NodePointer LabelList, NodePointer ParameterType,
                               unsigned depth, bool showTypes) override {
    startParameters(depth);
    NodePrinter::printFunctionParameters(LabelList, ParameterType, depth,
                                         showTypes);
    endParameters(depth);
  }
};

#endif // liblldb_SwiftMangled_h_
