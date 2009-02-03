//===-- Support/SCCIterator.h - Strongly Connected Comp. Iter. --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This builds on the llvm/ADT/GraphTraits.h file to find the strongly connected
// components (SCCs) of a graph in O(N+E) time using Tarjan's DFS algorithm.
//
// The SCC iterator has the important property that if a node in SCC S1 has an
// edge to a node in SCC S2, then it visits S1 *after* S2.
//
// To visit S1 *before* S2, use the scc_iterator on the Inverse graph.
// (NOTE: This requires some simple wrappers and is not supported yet.)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SCCITERATOR_H
#define LLVM_ADT_SCCITERATOR_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/iterator.h"
#include <vector>
#include <map>

namespace llvm {

//===----------------------------------------------------------------------===//
///
/// scc_iterator - Enumerate the SCCs of a directed graph, in
/// reverse topological order of the SCC DAG.
///
template<class GraphT, class GT = GraphTraits<GraphT> >
class scc_iterator
  : public forward_iterator<std::vector<typename GT::NodeType>, ptrdiff_t> {
  typedef typename GT::NodeType          NodeType;
  typedef typename GT::ChildIteratorType ChildItTy;
  typedef std::vector<NodeType*> SccTy;
  typedef forward_iterator<SccTy, ptrdiff_t> super;
  typedef typename super::reference reference;
  typedef typename super::pointer pointer;

  // The visit counters used to detect when a complete SCC is on the stack.
  // visitNum is the global counter.
  // nodeVisitNumbers are per-node visit numbers, also used as DFS flags.
  unsigned visitNum;
  std::map<NodeType *, unsigned> nodeVisitNumbers;

  // SCCNodeStack - Stack holding nodes of the SCC.
  std::vector<NodeType *> SCCNodeStack;

  // CurrentSCC - The current SCC, retrieved using operator*().
  SccTy CurrentSCC;

  // VisitStack - Used to maintain the ordering.  Top = current block
  // First element is basic block pointer, second is the 'next child' to visit
  std::vector<std::pair<NodeType *, ChildItTy> > VisitStack;

  // MinVistNumStack - Stack holding the "min" values for each node in the DFS.
  // This is used to track the minimum uplink values for all children of
  // the corresponding node on the VisitStack.
  std::vector<unsigned> MinVisitNumStack;

  // A single "visit" within the non-recursive DFS traversal.
  void DFSVisitOne(NodeType* N) {
    ++visitNum;                         // Global counter for the visit order
    nodeVisitNumbers[N] = visitNum;
    SCCNodeStack.push_back(N);
    MinVisitNumStack.push_back(visitNum);
    VisitStack.push_back(std::make_pair(N, GT::child_begin(N)));
    //DOUT << "TarjanSCC: Node " << N <<
    //      " : visitNum = " << visitNum << "\n";
  }

  // The stack-based DFS traversal; defined below.
  void DFSVisitChildren() {
    assert(!VisitStack.empty());
    while (VisitStack.back().second != GT::child_end(VisitStack.back().first)) {
      // TOS has at least one more child so continue DFS
      NodeType *childN = *VisitStack.back().second++;
      if (!nodeVisitNumbers.count(childN)) {
        // this node has never been seen
        DFSVisitOne(childN);
      } else {
        unsigned childNum = nodeVisitNumbers[childN];
        if (MinVisitNumStack.back() > childNum)
          MinVisitNumStack.back() = childNum;
      }
    }
  }

  // Compute the next SCC using the DFS traversal.
  void GetNextSCC() {
    assert(VisitStack.size() == MinVisitNumStack.size());
    CurrentSCC.clear();                 // Prepare to compute the next SCC
    while (!VisitStack.empty()) {
      DFSVisitChildren();
      assert(VisitStack.back().second ==GT::child_end(VisitStack.back().first));
      NodeType* visitingN = VisitStack.back().first;
      unsigned minVisitNum = MinVisitNumStack.back();
      VisitStack.pop_back();
      MinVisitNumStack.pop_back();
      if (!MinVisitNumStack.empty() && MinVisitNumStack.back() > minVisitNum)
        MinVisitNumStack.back() = minVisitNum;

      //DOUT << "TarjanSCC: Popped node " << visitingN <<
      //      " : minVisitNum = " << minVisitNum << "; Node visit num = " <<
      //      nodeVisitNumbers[visitingN] << "\n";

      if (minVisitNum == nodeVisitNumbers[visitingN]) {
        // A full SCC is on the SCCNodeStack!  It includes all nodes below
          // visitingN on the stack.  Copy those nodes to CurrentSCC,
          // reset their minVisit values, and return (this suspends
          // the DFS traversal till the next ++).
          do {
            CurrentSCC.push_back(SCCNodeStack.back());
            SCCNodeStack.pop_back();
            nodeVisitNumbers[CurrentSCC.back()] = ~0U;
          } while (CurrentSCC.back() != visitingN);
          return;
        }
    }
  }

  inline scc_iterator(NodeType *entryN) : visitNum(0) {
    DFSVisitOne(entryN);
    GetNextSCC();
  }
  inline scc_iterator() { /* End is when DFS stack is empty */ }

public:
  typedef scc_iterator<GraphT, GT> _Self;

  // Provide static "constructors"...
  static inline _Self begin(GraphT& G) { return _Self(GT::getEntryNode(G)); }
  static inline _Self end  (GraphT& G) { return _Self(); }

  // Direct loop termination test (I.fini() is more efficient than I == end())
  inline bool fini() const {
    assert(!CurrentSCC.empty() || VisitStack.empty());
    return CurrentSCC.empty();
  }

  inline bool operator==(const _Self& x) const {
    return VisitStack == x.VisitStack && CurrentSCC == x.CurrentSCC;
  }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }

  // Iterator traversal: forward iteration only
  inline _Self& operator++() {          // Preincrement
    GetNextSCC();
    return *this;
  }
  inline _Self operator++(int) {        // Postincrement
    _Self tmp = *this; ++*this; return tmp;
  }

  // Retrieve a reference to the current SCC
  inline const SccTy &operator*() const {
    assert(!CurrentSCC.empty() && "Dereferencing END SCC iterator!");
    return CurrentSCC;
  }
  inline SccTy &operator*() {
    assert(!CurrentSCC.empty() && "Dereferencing END SCC iterator!");
    return CurrentSCC;
  }

  // hasLoop() -- Test if the current SCC has a loop.  If it has more than one
  // node, this is trivially true.  If not, it may still contain a loop if the
  // node has an edge back to itself.
  bool hasLoop() const {
    assert(!CurrentSCC.empty() && "Dereferencing END SCC iterator!");
    if (CurrentSCC.size() > 1) return true;
    NodeType *N = CurrentSCC.front();
    for (ChildItTy CI = GT::child_begin(N), CE=GT::child_end(N); CI != CE; ++CI)
      if (*CI == N)
        return true;
    return false;
  }
};


// Global constructor for the SCC iterator.
template <class T>
scc_iterator<T> scc_begin(T G) {
  return scc_iterator<T>::begin(G);
}

template <class T>
scc_iterator<T> scc_end(T G) {
  return scc_iterator<T>::end(G);
}

} // End llvm namespace

#endif
