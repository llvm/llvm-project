//===-- MsgPackDocument.cpp - MsgPack Document --------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements a class that exposes a simple in-memory representation
/// of a document of MsgPack objects, that can be read from MsgPack, written to
/// MsgPack, and inspected and modified in memory. This is intended to be a
/// lighter-weight (in terms of memory allocations) replacement for
/// MsgPackTypes.
///
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/BinaryFormat/MsgPackWriter.h"

using namespace llvm;
using namespace msgpack;

// Convert this DocNode into an empty array.
void DocNode::convertToArray() { *this = getDocument()->getArrayNode(); }

// Convert this DocNode into an empty map.
void DocNode::convertToMap() { *this = getDocument()->getMapNode(); }

/// Find the key in the MapDocNode.
DocNode::MapTy::iterator MapDocNode::find(StringRef S) {
  return find(getDocument()->getNode(S));
}

/// Member access for MapDocNode. The string data must remain valid for the
/// lifetime of the Document.
DocNode &MapDocNode::operator[](StringRef S) {
  return (*this)[getDocument()->getNode(S)];
}

/// Member access for MapDocNode.
DocNode &MapDocNode::operator[](DocNode Key) {
  assert(!Key.isEmpty());
  MapTy::value_type Entry(Key, DocNode());
  auto ItAndInserted = Map->insert(Entry);
  if (ItAndInserted.second) {
    // Ensure a new element has its KindAndDoc initialized.
    ItAndInserted.first->second = getDocument()->getNode();
  }
  return ItAndInserted.first->second;
}

/// Array element access. This extends the array if necessary.
DocNode &ArrayDocNode::operator[](size_t Index) {
  if (size() <= Index) {
    // Ensure new elements have their KindAndDoc initialized.
    Array->resize(Index + 1, getDocument()->getNode());
  }
  return (*Array)[Index];
}

// A level in the document reading stack.
struct StackLevel {
  DocNode Node;
  size_t Length;
  // Points to map entry when we have just processed a map key.
  DocNode *MapEntry;
};

// Read a document from a binary msgpack blob.
// The blob data must remain valid for the lifetime of this Document (because a
// string object in the document contains a StringRef into the original blob).
// If Multi, then this sets root to an array and adds top-level objects to it.
// If !Multi, then it only reads a single top-level object, even if there are
// more, and sets root to that.
// Returns false if failed due to illegal format.
bool Document::readFromBlob(StringRef Blob, bool Multi) {
  msgpack::Reader MPReader(Blob);
  SmallVector<StackLevel, 4> Stack;
  if (Multi) {
    // Create the array for multiple top-level objects.
    Root = getArrayNode();
    Stack.push_back(StackLevel({Root, (size_t)-1, nullptr}));
  }
  do {
    // On to next element (or key if doing a map key next).
    // Read the value.
    Object Obj;
    if (!MPReader.read(Obj)) {
      if (Multi && Stack.size() == 1) {
        // OK to finish here as we've just done a top-level element with Multi
        break;
      }
      return false; // Finished too early
    }
    // Convert it into a DocNode.
    DocNode Node;
    switch (Obj.Kind) {
    case Type::Nil:
      Node = getNode();
      break;
    case Type::Int:
      Node = getNode(Obj.Int);
      break;
    case Type::UInt:
      Node = getNode(Obj.UInt);
      break;
    case Type::Boolean:
      Node = getNode(Obj.Bool);
      break;
    case Type::Float:
      Node = getNode(Obj.Float);
      break;
    case Type::String:
      Node = getNode(Obj.Raw);
      break;
    case Type::Map:
      Node = getMapNode();
      break;
    case Type::Array:
      Node = getArrayNode();
      break;
    default:
      return false; // Raw and Extension not supported
    }

    // Store it.
    if (Stack.empty())
      Root = Node;
    else if (Stack.back().Node.getKind() == Type::Array) {
      // Reading an array entry.
      auto &Array = Stack.back().Node.getArray();
      Array.push_back(Node);
    } else {
      auto &Map = Stack.back().Node.getMap();
      if (!Stack.back().MapEntry) {
        // Reading a map key.
        Stack.back().MapEntry = &Map[Node];
      } else {
        // Reading the value for the map key read in the last iteration.
        *Stack.back().MapEntry = Node;
        Stack.back().MapEntry = nullptr;
      }
    }

    // See if we're starting a new array or map.
    switch (Node.getKind()) {
    case msgpack::Type::Array:
    case msgpack::Type::Map:
      Stack.push_back(StackLevel({Node, Obj.Length, nullptr}));
      break;
    default:
      break;
    }

    // Pop finished stack levels.
    while (!Stack.empty()) {
      if (Stack.back().Node.getKind() == msgpack::Type::Array) {
        if (Stack.back().Node.getArray().size() != Stack.back().Length)
          break;
      } else {
        if (Stack.back().MapEntry ||
            Stack.back().Node.getMap().size() != Stack.back().Length)
          break;
      }
      Stack.pop_back();
    }
  } while (!Stack.empty());
  return true;
}

struct WriterStackLevel {
  DocNode Node;
  DocNode::MapTy::iterator MapIt;
  DocNode::ArrayTy::iterator ArrayIt;
  bool OnKey;
};

/// Write a MsgPack document to a binary MsgPack blob.
void Document::writeToBlob(std::string &Blob) {
  Blob.clear();
  raw_string_ostream OS(Blob);
  msgpack::Writer MPWriter(OS);
  SmallVector<WriterStackLevel, 4> Stack;
  DocNode Node = getRoot();
  for (;;) {
    switch (Node.getKind()) {
    case Type::Array:
      MPWriter.writeArraySize(Node.getArray().size());
      Stack.push_back(
          {Node, DocNode::MapTy::iterator(), Node.getArray().begin(), false});
      break;
    case Type::Map:
      MPWriter.writeMapSize(Node.getMap().size());
      Stack.push_back(
          {Node, Node.getMap().begin(), DocNode::ArrayTy::iterator(), true});
      break;
    case Type::Nil:
      MPWriter.writeNil();
      break;
    case Type::Boolean:
      MPWriter.write(Node.getBool());
      break;
    case Type::Int:
      MPWriter.write(Node.getInt());
      break;
    case Type::UInt:
      MPWriter.write(Node.getUInt());
      break;
    case Type::String:
      MPWriter.write(Node.getString());
      break;
    default:
      llvm_unreachable("unhandled msgpack object kind");
    }
    // Pop finished stack levels.
    while (!Stack.empty()) {
      if (Stack.back().Node.getKind() == Type::Map) {
        if (Stack.back().MapIt != Stack.back().Node.getMap().end())
          break;
      } else {
        if (Stack.back().ArrayIt != Stack.back().Node.getArray().end())
          break;
      }
      Stack.pop_back();
    }
    if (Stack.empty())
      break;
    // Get the next value.
    if (Stack.back().Node.getKind() == Type::Map) {
      if (Stack.back().OnKey) {
        // Do the key of a key,value pair in a map.
        Node = Stack.back().MapIt->first;
        Stack.back().OnKey = false;
      } else {
        Node = Stack.back().MapIt->second;
        ++Stack.back().MapIt;
        Stack.back().OnKey = true;
      }
    } else {
      Node = *Stack.back().ArrayIt;
      ++Stack.back().ArrayIt;
    }
  }
}

