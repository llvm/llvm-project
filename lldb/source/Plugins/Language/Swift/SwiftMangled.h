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

class TrackingNodePrinter : public NodePrinter {
public:
  TrackingNodePrinter(DemangleOptions options) : NodePrinter(options) {}

  lldb_private::DemangledNameInfo takeInfo() { return std::move(info); }

private:
  lldb_private::DemangledNameInfo info;
  std::optional<unsigned> parametersDepth;

  void startName() {
    if (!info.hasBasename())
      info.BasenameRange.first = getStreamLength();
  }

  void endName() {
    if (!info.hasBasename())
      info.BasenameRange.second = getStreamLength();
  }

  void startParameters(unsigned depth) {
    if (parametersDepth || !info.hasBasename() || info.hasArguments()) {
      return;
    }
    info.ArgumentsRange.first = getStreamLength();
    parametersDepth = depth;
  }

  void endParameters(unsigned depth) {
    if (!parametersDepth || *parametersDepth != depth || info.hasArguments()) {
      return;
    }
    info.ArgumentsRange.second = getStreamLength();
  }

  bool shouldTrackNameRange(NodePointer Node) const {
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

  void printFunctionParameters(NodePointer LabelList, NodePointer ParameterType,
                               unsigned depth, bool showTypes) override {
    startParameters(depth);
    NodePrinter::printFunctionParameters(LabelList, ParameterType, depth,
                                         showTypes);
    endParameters(depth);
  }
};

#endif // liblldb_SwiftMangled_h_
