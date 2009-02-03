//===- llvm/ADT/DepthFirstIterator.h - Depth First iterator -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file builds on the ADT/GraphTraits.h file to build generic depth
// first graph iterator.  This file exposes the following functions/types:
//
// df_begin/df_end/df_iterator
//   * Normal depth-first iteration - visit a node and then all of its children.
//
// idf_begin/idf_end/idf_iterator
//   * Depth-first iteration on the 'inverse' graph.
//
// df_ext_begin/df_ext_end/df_ext_iterator
//   * Normal depth-first iteration - visit a node and then all of its children.
//     This iterator stores the 'visited' set in an external set, which allows
//     it to be more efficient, and allows external clients to use the set for
//     other purposes.
//
// idf_ext_begin/idf_ext_end/idf_ext_iterator
//   * Depth-first iteration on the 'inverse' graph.
//     This iterator stores the 'visited' set in an external set, which allows
//     it to be more efficient, and allows external clients to use the set for
//     other purposes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DEPTHFIRSTITERATOR_H
#define LLVM_ADT_DEPTHFIRSTITERATOR_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <vector>
#include <set>

namespace llvm {

// df_iterator_storage - A private class which is used to figure out where to
// store the visited set.
template<class SetType, bool External>   // Non-external set
class df_iterator_storage {
public:
  SetType Visited;
};

template<class SetType>
class df_iterator_storage<SetType, true> {
public:
  df_iterator_storage(SetType &VSet) : Visited(VSet) {}
  df_iterator_storage(const df_iterator_storage &S) : Visited(S.Visited) {}
  SetType &Visited;
};


// Generic Depth First Iterator
template<class GraphT,
class SetType = llvm::SmallPtrSet<typename GraphTraits<GraphT>::NodeType*, 8>,
         bool ExtStorage = false, class GT = GraphTraits<GraphT> >
class df_iterator : public forward_iterator<typename GT::NodeType, ptrdiff_t>,
                    public df_iterator_storage<SetType, ExtStorage> {
  typedef forward_iterator<typename GT::NodeType, ptrdiff_t> super;

  typedef typename GT::NodeType          NodeType;
  typedef typename GT::ChildIteratorType ChildItTy;

  // VisitStack - Used to maintain the ordering.  Top = current block
  // First element is node pointer, second is the 'next child' to visit
  std::vector<std::pair<NodeType *, ChildItTy> > VisitStack;
private:
  inline df_iterator(NodeType *Node) {
    this->Visited.insert(Node);
    VisitStack.push_back(std::make_pair(Node, GT::child_begin(Node)));
  }
  inline df_iterator() { /* End is when stack is empty */ }

  inline df_iterator(NodeType *Node, SetType &S)
    : df_iterator_storage<SetType, ExtStorage>(S) {
    if (!S.count(Node)) {
      this->Visited.insert(Node);
      VisitStack.push_back(std::make_pair(Node, GT::child_begin(Node)));
    }
  }
  inline df_iterator(SetType &S)
    : df_iterator_storage<SetType, ExtStorage>(S) {
    // End is when stack is empty
  }

public:
  typedef typename super::pointer pointer;
  typedef df_iterator<GraphT, SetType, ExtStorage, GT> _Self;

  // Provide static begin and end methods as our public "constructors"
  static inline _Self begin(const GraphT& G) {
    return _Self(GT::getEntryNode(G));
  }
  static inline _Self end(const GraphT& G) { return _Self(); }

  // Static begin and end methods as our public ctors for external iterators
  static inline _Self begin(const GraphT& G, SetType &S) {
    return _Self(GT::getEntryNode(G), S);
  }
  static inline _Self end(const GraphT& G, SetType &S) { return _Self(S); }

  inline bool operator==(const _Self& x) const {
    return VisitStack.size() == x.VisitStack.size() &&
           VisitStack == x.VisitStack;
  }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }

  inline pointer operator*() const {
    return VisitStack.back().first;
  }

  // This is a nonstandard operator-> that dereferences the pointer an extra
  // time... so that you can actually call methods ON the Node, because
  // the contained type is a pointer.  This allows BBIt->getTerminator() f.e.
  //
  inline NodeType *operator->() const { return operator*(); }

  inline _Self& operator++() {   // Preincrement
    do {
      std::pair<NodeType *, ChildItTy> &Top = VisitStack.back();
      NodeType *Node = Top.first;
      ChildItTy &It  = Top.second;

      while (It != GT::child_end(Node)) {
        NodeType *Next = *It++;
        if (!this->Visited.count(Next)) {  // Has our next sibling been visited?
          // No, do it now.
          this->Visited.insert(Next);
          VisitStack.push_back(std::make_pair(Next, GT::child_begin(Next)));
          return *this;
        }
      }

      // Oops, ran out of successors... go up a level on the stack.
      VisitStack.pop_back();
    } while (!VisitStack.empty());
    return *this;
  }

  inline _Self operator++(int) { // Postincrement
    _Self tmp = *this; ++*this; return tmp;
  }

  // nodeVisited - return true if this iterator has already visited the
  // specified node.  This is public, and will probably be used to iterate over
  // nodes that a depth first iteration did not find: ie unreachable nodes.
  //
  inline bool nodeVisited(NodeType *Node) const {
    return this->Visited.count(Node) != 0;
  }
};


// Provide global constructors that automatically figure out correct types...
//
template <class T>
df_iterator<T> df_begin(const T& G) {
  return df_iterator<T>::begin(G);
}

template <class T>
df_iterator<T> df_end(const T& G) {
  return df_iterator<T>::end(G);
}

// Provide global definitions of external depth first iterators...
template <class T, class SetTy = std::set<typename GraphTraits<T>::NodeType*> >
struct df_ext_iterator : public df_iterator<T, SetTy, true> {
  df_ext_iterator(const df_iterator<T, SetTy, true> &V)
    : df_iterator<T, SetTy, true>(V) {}
};

template <class T, class SetTy>
df_ext_iterator<T, SetTy> df_ext_begin(const T& G, SetTy &S) {
  return df_ext_iterator<T, SetTy>::begin(G, S);
}

template <class T, class SetTy>
df_ext_iterator<T, SetTy> df_ext_end(const T& G, SetTy &S) {
  return df_ext_iterator<T, SetTy>::end(G, S);
}


// Provide global definitions of inverse depth first iterators...
template <class T,
  class SetTy = llvm::SmallPtrSet<typename GraphTraits<T>::NodeType*, 8>,
          bool External = false>
struct idf_iterator : public df_iterator<Inverse<T>, SetTy, External> {
  idf_iterator(const df_iterator<Inverse<T>, SetTy, External> &V)
    : df_iterator<Inverse<T>, SetTy, External>(V) {}
};

template <class T>
idf_iterator<T> idf_begin(const T& G) {
  Inverse<T> DummyG;
  return idf_iterator<T>::begin(DummyG);
}

template <class T>
idf_iterator<T> idf_end(const T& G){
  Inverse<T> DummyG;
  return idf_iterator<T>::end(DummyG);
}

// Provide global definitions of external inverse depth first iterators...
template <class T, class SetTy = std::set<typename GraphTraits<T>::NodeType*> >
struct idf_ext_iterator : public idf_iterator<T, SetTy, true> {
  idf_ext_iterator(const idf_iterator<T, SetTy, true> &V)
    : idf_iterator<T, SetTy, true>(V) {}
  idf_ext_iterator(const df_iterator<Inverse<T>, SetTy, true> &V)
    : idf_iterator<T, SetTy, true>(V) {}
};

template <class T, class SetTy>
idf_ext_iterator<T, SetTy> idf_ext_begin(const T& G, SetTy &S) {
  Inverse<T> DummyG(G);
  return idf_ext_iterator<T, SetTy>::begin(DummyG, S);
}

template <class T, class SetTy>
idf_ext_iterator<T, SetTy> idf_ext_end(const T& G, SetTy &S) {
  Inverse<T> DummyG(G);
  return idf_ext_iterator<T, SetTy>::end(DummyG, S);
}

} // End llvm namespace

#endif
