//===- llvm/ADT/Trie.h ---- Generic trie structure --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class defines a generic trie structure. The trie structure
// is immutable after creation, but the payload contained within it is not.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_TRIE_H
#define LLVM_ADT_TRIE_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/DOTGraphTraits.h"

#include <vector>

namespace llvm {

// FIXME:
// - Labels are usually small, maybe it's better to use SmallString
// - Should we use char* during construction?
// - Should we templatize Empty with traits-like interface?

template<class Payload>
class Trie {
  friend class GraphTraits<Trie<Payload> >;
  friend class DOTGraphTraits<Trie<Payload> >;
public:
  class Node {
    friend class Trie;

  public:
    typedef std::vector<Node*> NodeVectorType;
    typedef typename NodeVectorType::iterator iterator;
    typedef typename NodeVectorType::const_iterator const_iterator;

  private:
    enum QueryResult {
      Same           = -3,
      StringIsPrefix = -2,
      LabelIsPrefix  = -1,
      DontMatch      = 0,
      HaveCommonPart
    };

    struct NodeCmp {
      bool operator() (Node* N1, Node* N2) {
        return (N1->Label[0] < N2->Label[0]);
      }
      bool operator() (Node* N, char Id) {
        return (N->Label[0] < Id);
      }
    };

    std::string Label;
    Payload Data;
    NodeVectorType Children;

    // Do not implement
    Node(const Node&);
    Node& operator=(const Node&);

    inline void addEdge(Node* N) {
      if (Children.empty())
        Children.push_back(N);
      else {
        iterator I = std::lower_bound(Children.begin(), Children.end(),
                                      N, NodeCmp());
        // FIXME: no dups are allowed
        Children.insert(I, N);
      }
    }

    inline void setEdge(Node* N) {
      char Id = N->Label[0];
      iterator I = std::lower_bound(Children.begin(), Children.end(),
                                     Id, NodeCmp());
      assert(I != Children.end() && "Node does not exists!");
      *I = N;
    }

    QueryResult query(const std::string& s) const {
      unsigned i, l;
      unsigned l1 = s.length();
      unsigned l2 = Label.length();

      // Find the length of common part
      l = std::min(l1, l2);
      i = 0;
      while ((i < l) && (s[i] == Label[i]))
        ++i;

      if (i == l) { // One is prefix of another, find who is who
        if (l1 == l2)
          return Same;
        else if (i == l1)
          return StringIsPrefix;
        else
          return LabelIsPrefix;
      } else // s and Label have common (possible empty) part, return its length
        return (QueryResult)i;
    }

  public:
    inline explicit Node(const Payload& data, const std::string& label = ""):
        Label(label), Data(data) { }

    inline const Payload& data() const { return Data; }
    inline void setData(const Payload& data) { Data = data; }

    inline const std::string& label() const { return Label; }

#if 0
    inline void dump() {
      std::cerr << "Node: " << this << "\n"
                << "Label: " << Label << "\n"
                << "Children:\n";

      for (iterator I = Children.begin(), E = Children.end(); I != E; ++I)
        std::cerr << (*I)->Label << "\n";
    }
#endif

    inline Node* getEdge(char Id) {
      Node* fNode = NULL;
      iterator I = std::lower_bound(Children.begin(), Children.end(),
                                          Id, NodeCmp());
      if (I != Children.end() && (*I)->Label[0] == Id)
        fNode = *I;

      return fNode;
    }

    inline iterator       begin()       { return Children.begin(); }
    inline const_iterator begin() const { return Children.begin(); }
    inline iterator       end  ()       { return Children.end();   }
    inline const_iterator end  () const { return Children.end();   }

    inline size_t         size () const { return Children.size();  }
    inline bool           empty() const { return Children.empty(); }
    inline const Node*   &front() const { return Children.front(); }
    inline       Node*   &front()       { return Children.front(); }
    inline const Node*   &back()  const { return Children.back();  }
    inline       Node*   &back()        { return Children.back();  }

  };

private:
  std::vector<Node*> Nodes;
  Payload Empty;

  inline Node* addNode(const Payload& data, const std::string label = "") {
    Node* N = new Node(data, label);
    Nodes.push_back(N);
    return N;
  }

  inline Node* splitEdge(Node* N, char Id, size_t index) {
    Node* eNode = N->getEdge(Id);
    assert(eNode && "Node doesn't exist");

    const std::string &l = eNode->Label;
    assert(index > 0 && index < l.length() && "Trying to split too far!");
    std::string l1 = l.substr(0, index);
    std::string l2 = l.substr(index);

    Node* nNode = addNode(Empty, l1);
    N->setEdge(nNode);

    eNode->Label = l2;
    nNode->addEdge(eNode);

    return nNode;
  }

  // Do not implement
  Trie(const Trie&);
  Trie& operator=(const Trie&);

public:
  inline explicit Trie(const Payload& empty):Empty(empty) {
    addNode(Empty);
  }
  inline ~Trie() {
    for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
      delete Nodes[i];
  }

  inline Node* getRoot() const { return Nodes[0]; }

  bool addString(const std::string& s, const Payload& data);
  const Payload& lookup(const std::string& s) const;

};

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template<class Payload>
bool Trie<Payload>::addString(const std::string& s, const Payload& data) {
  Node* cNode = getRoot();
  Node* tNode = NULL;
  std::string s1(s);

  while (tNode == NULL) {
    char Id = s1[0];
    if (Node* nNode = cNode->getEdge(Id)) {
      typename Node::QueryResult r = nNode->query(s1);

      switch (r) {
      case Node::Same:
      case Node::StringIsPrefix:
        // Currently we don't allow to have two strings in the trie one
        // being a prefix of another. This should be fixed.
        assert(0 && "FIXME!");
        return false;
      case Node::DontMatch:
        assert(0 && "Impossible!");
        return false;
      case Node::LabelIsPrefix:
        s1 = s1.substr(nNode->label().length());
        cNode = nNode;
        break;
      default:
        nNode = splitEdge(cNode, Id, r);
        tNode = addNode(data, s1.substr(r));
        nNode->addEdge(tNode);
      }
    } else {
      tNode = addNode(data, s1);
      cNode->addEdge(tNode);
    }
  }

  return true;
}

template<class Payload>
const Payload& Trie<Payload>::lookup(const std::string& s) const {
  Node* cNode = getRoot();
  Node* tNode = NULL;
  std::string s1(s);

  while (tNode == NULL) {
    char Id = s1[0];
    if (Node* nNode = cNode->getEdge(Id)) {
      typename Node::QueryResult r = nNode->query(s1);

      switch (r) {
      case Node::Same:
        tNode = nNode;
        break;
      case Node::StringIsPrefix:
        return Empty;
      case Node::DontMatch:
        assert(0 && "Impossible!");
        return Empty;
      case Node::LabelIsPrefix:
        s1 = s1.substr(nNode->label().length());
        cNode = nNode;
        break;
      default:
        return Empty;
      }
    } else
      return Empty;
  }

  return tNode->data();
}

template<class Payload>
struct GraphTraits<Trie<Payload> > {
  typedef Trie<Payload> TrieType;
  typedef typename TrieType::Node NodeType;
  typedef typename NodeType::iterator ChildIteratorType;

  static inline NodeType *getEntryNode(const TrieType& T) { return T.getRoot(); }

  static inline ChildIteratorType child_begin(NodeType *N) { return N->begin(); }
  static inline ChildIteratorType child_end(NodeType *N) { return N->end(); }

  typedef typename std::vector<NodeType*>::const_iterator nodes_iterator;

  static inline nodes_iterator nodes_begin(const TrieType& G) {
    return G.Nodes.begin();
  }
  static inline nodes_iterator nodes_end(const TrieType& G) {
    return G.Nodes.end();
  }

};

template<class Payload>
struct DOTGraphTraits<Trie<Payload> > : public DefaultDOTGraphTraits {
  typedef typename Trie<Payload>::Node NodeType;
  typedef typename GraphTraits<Trie<Payload> >::ChildIteratorType EdgeIter;

  static std::string getGraphName(const Trie<Payload>& T) {
    return "Trie";
  }

  static std::string getNodeLabel(NodeType* Node, const Trie<Payload>& T) {
    if (T.getRoot() == Node)
      return "<Root>";
    else
      return Node->label();
  }

  static std::string getEdgeSourceLabel(NodeType* Node, EdgeIter I) {
    NodeType* N = *I;
    return N->label().substr(0, 1);
  }

  static std::string getNodeAttributes(const NodeType* Node,
                                       const Trie<Payload>& T) {
    if (Node->data() != T.Empty)
      return "color=blue";

    return "";
  }

};

} // end of llvm namespace

#endif // LLVM_ADT_TRIE_H
