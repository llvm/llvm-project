//===-- llvm/Support/DataFlow.h - dataflow as graphs ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines specializations of GraphTraits that allows Use-Def and
// Def-Use relations to be treated as proper graphs for generic algorithms.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DATAFLOW_H
#define LLVM_SUPPORT_DATAFLOW_H

#include "llvm/User.h"
#include "llvm/ADT/GraphTraits.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// Provide specializations of GraphTraits to be able to treat def-use/use-def
// chains as graphs

template <> struct GraphTraits<const Value*> {
  typedef const Value NodeType;
  typedef Value::const_use_iterator ChildIteratorType;

  static NodeType *getEntryNode(const Value *G) {
    return G;
  }

  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->use_begin();
  }

  static inline ChildIteratorType child_end(NodeType *N) {
    return N->use_end();
  }
};

template <> struct GraphTraits<Value*> {
  typedef Value NodeType;
  typedef Value::use_iterator ChildIteratorType;

  static NodeType *getEntryNode(Value *G) {
    return G;
  }

  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->use_begin();
  }

  static inline ChildIteratorType child_end(NodeType *N) {
    return N->use_end();
  }
};

template <> struct GraphTraits<Inverse<const User*> > {
  typedef const Value NodeType;
  typedef User::const_op_iterator ChildIteratorType;

  static NodeType *getEntryNode(Inverse<const User*> G) {
    return G.Graph;
  }

  static inline ChildIteratorType child_begin(NodeType *N) {
    if (const User *U = dyn_cast<User>(N))
      return U->op_begin();
    return NULL;
  }

  static inline ChildIteratorType child_end(NodeType *N) {
    if(const User *U = dyn_cast<User>(N))
      return U->op_end();
    return NULL;
  }
};

template <> struct GraphTraits<Inverse<User*> > {
  typedef Value NodeType;
  typedef User::op_iterator ChildIteratorType;

  static NodeType *getEntryNode(Inverse<User*> G) {
    return G.Graph;
  }

  static inline ChildIteratorType child_begin(NodeType *N) {
    if (User *U = dyn_cast<User>(N))
      return U->op_begin();
    return NULL;
  }

  static inline ChildIteratorType child_end(NodeType *N) {
    if (User *U = dyn_cast<User>(N))
      return U->op_end();
    return NULL;
  }
};

}
#endif
