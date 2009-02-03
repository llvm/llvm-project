//===--- CompilationGraph.h - The LLVM Compiler Driver ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Compilation graph - definition.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMC2_COMPILATION_GRAPH_H
#define LLVM_TOOLS_LLVMC2_COMPILATION_GRAPH_H

#include "llvm/CompilerDriver/Tool.h"

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/System/Path.h"

#include <cassert>
#include <string>

namespace llvmc {

  class CompilationGraph;
  typedef llvm::StringSet<> InputLanguagesSet;

  /// LanguageMap - Maps from extensions to language names.
  class LanguageMap : public llvm::StringMap<std::string> {
  public:

    /// GetLanguage -  Find the language name corresponding to a given file.
    const std::string& GetLanguage(const llvm::sys::Path&) const;
  };

  /// Edge - Represents an edge of the compilation graph.
  class Edge : public llvm::RefCountedBaseVPTR<Edge> {
  public:
    Edge(const std::string& T) : ToolName_(T) {}
    virtual ~Edge() {};

    const std::string& ToolName() const { return ToolName_; }
    virtual unsigned Weight(const InputLanguagesSet& InLangs) const = 0;
  private:
    std::string ToolName_;
  };

  /// SimpleEdge - An edge that has no properties.
  class SimpleEdge : public Edge {
  public:
    SimpleEdge(const std::string& T) : Edge(T) {}
    unsigned Weight(const InputLanguagesSet&) const { return 1; }
  };

  /// Node - A node (vertex) of the compilation graph.
  struct Node {
    // A Node holds a list of the outward edges.
    typedef llvm::SmallVector<llvm::IntrusiveRefCntPtr<Edge>, 3> container_type;
    typedef container_type::iterator iterator;
    typedef container_type::const_iterator const_iterator;

    Node() : OwningGraph(0), InEdges(0) {}
    Node(CompilationGraph* G) : OwningGraph(G), InEdges(0) {}
    Node(CompilationGraph* G, Tool* T) :
      OwningGraph(G), ToolPtr(T), InEdges(0) {}

    bool HasChildren() const { return !OutEdges.empty(); }
    const std::string Name() const
    { return ToolPtr ? ToolPtr->Name() : "root"; }

    // Iteration.
    iterator EdgesBegin() { return OutEdges.begin(); }
    const_iterator EdgesBegin() const { return OutEdges.begin(); }
    iterator EdgesEnd() { return OutEdges.end(); }
    const_iterator EdgesEnd() const { return OutEdges.end(); }

    /// AddEdge - Add an outward edge. Takes ownership of the provided
    /// Edge object.
    void AddEdge(Edge* E);

    // Inward edge counter. Used to implement topological sort.
    void IncrInEdges() { ++InEdges; }
    void DecrInEdges() { --InEdges; }
    bool HasNoInEdges() const { return InEdges == 0; }

    // Needed to implement NodeChildIterator/GraphTraits
    CompilationGraph* OwningGraph;
    // The corresponding Tool.
    // WARNING: ToolPtr can be NULL (for the root node).
    llvm::IntrusiveRefCntPtr<Tool> ToolPtr;
    // Links to children.
    container_type OutEdges;
    // Inward edge counter. Updated in
    // CompilationGraph::insertEdge(). Used for topological sorting.
    unsigned InEdges;
  };

  class NodesIterator;

  /// CompilationGraph - The compilation graph itself.
  class CompilationGraph {
    /// nodes_map_type - The main data structure.
    typedef llvm::StringMap<Node> nodes_map_type;
    /// tools_vector_type, tools_map_type - Data structures used to
    /// map from language names to tools. (We can have several tools
    /// associated with each language name, hence the need for a
    /// vector.)
    typedef
    llvm::SmallVector<llvm::IntrusiveRefCntPtr<Edge>, 3> tools_vector_type;
    typedef llvm::StringMap<tools_vector_type> tools_map_type;

    /// ToolsMap - Map from language names to lists of tool names.
    tools_map_type ToolsMap;
    /// NodesMap - Map from tool names to Tool objects.
    nodes_map_type NodesMap;

  public:

    typedef nodes_map_type::iterator nodes_iterator;
    typedef nodes_map_type::const_iterator const_nodes_iterator;

    CompilationGraph();

    /// insertNode - Insert a new node into the graph. Takes
    /// ownership of the object.
    void insertNode(Tool* T);

    /// insertEdge - Insert a new edge into the graph. Takes ownership
    /// of the Edge object.
    void insertEdge(const std::string& A, Edge* E);

    /// Build - Build target(s) from the input file set. Command-line
    /// options are passed implicitly as global variables.
    int Build(llvm::sys::Path const& TempDir, const LanguageMap& LangMap);

    /// Check - Check the compilation graph for common errors like
    /// cycles, input/output language mismatch and multiple default
    /// edges. Prints error messages and in case it finds any errors.
    int Check();

    /// getNode - Return a reference to the node correponding to the
    /// given tool name. Throws std::runtime_error.
    Node& getNode(const std::string& ToolName);
    const Node& getNode(const std::string& ToolName) const;

    /// viewGraph - This function is meant for use from the debugger.
    /// You can just say 'call G->viewGraph()' and a ghostview window
    /// should pop up from the program, displaying the compilation
    /// graph. This depends on there being a 'dot' and 'gv' program
    /// in your path.
    void viewGraph();

    /// writeGraph - Write a compilation-graph.dot file.
    void writeGraph();

    // GraphTraits support.
    friend NodesIterator GraphBegin(CompilationGraph*);
    friend NodesIterator GraphEnd(CompilationGraph*);

  private:
    // Helper functions.

    /// getToolsVector - Return a reference to the list of tool names
    /// corresponding to the given language name. Throws
    /// std::runtime_error.
    const tools_vector_type& getToolsVector(const std::string& LangName) const;

    /// PassThroughGraph - Pass the input file through the toolchain
    /// starting at StartNode.
    void PassThroughGraph (const llvm::sys::Path& In, const Node* StartNode,
                           const InputLanguagesSet& InLangs,
                           const llvm::sys::Path& TempDir,
                           const LanguageMap& LangMap) const;

    /// FindToolChain - Find head of the toolchain corresponding to
    /// the given file.
    const Node* FindToolChain(const llvm::sys::Path& In,
                              const std::string* ForceLanguage,
                              InputLanguagesSet& InLangs,
                              const LanguageMap& LangMap) const;

    /// BuildInitial - Traverse the initial parts of the toolchains.
    void BuildInitial(InputLanguagesSet& InLangs,
                      const llvm::sys::Path& TempDir,
                      const LanguageMap& LangMap);

    /// TopologicalSort - Sort the nodes in topological order.
    void TopologicalSort(std::vector<const Node*>& Out);
    /// TopologicalSortFilterJoinNodes - Call TopologicalSort and
    /// filter the resulting list to include only Join nodes.
    void TopologicalSortFilterJoinNodes(std::vector<const Node*>& Out);

    // Functions used to implement Check().

    /// CheckLanguageNames - Check that output/input language names
    /// match for all nodes.
    int CheckLanguageNames() const;
    /// CheckMultipleDefaultEdges - check that there are no multiple
    /// default default edges.
    int CheckMultipleDefaultEdges() const;
    /// CheckCycles - Check that there are no cycles in the graph.
    int CheckCycles();

  };

  // GraphTraits support code.

  /// NodesIterator - Auxiliary class needed to implement GraphTraits
  /// support. Can be generalised to something like value_iterator
  /// for map-like containers.
  class NodesIterator : public CompilationGraph::nodes_iterator {
    typedef CompilationGraph::nodes_iterator super;
    typedef NodesIterator ThisType;
    typedef Node* pointer;
    typedef Node& reference;

  public:
    NodesIterator(super I) : super(I) {}

    inline reference operator*() const {
      return super::operator->()->second;
    }
    inline pointer operator->() const {
      return &super::operator->()->second;
    }
  };

  inline NodesIterator GraphBegin(CompilationGraph* G) {
    return NodesIterator(G->NodesMap.begin());
  }

  inline NodesIterator GraphEnd(CompilationGraph* G) {
    return NodesIterator(G->NodesMap.end());
  }


  /// NodeChildIterator - Another auxiliary class needed by GraphTraits.
  class NodeChildIterator : public bidirectional_iterator<Node, ptrdiff_t> {
    typedef NodeChildIterator ThisType;
    typedef Node::container_type::iterator iterator;

    CompilationGraph* OwningGraph;
    iterator EdgeIter;
  public:
    typedef Node* pointer;
    typedef Node& reference;

    NodeChildIterator(Node* N, iterator I) :
      OwningGraph(N->OwningGraph), EdgeIter(I) {}

    const ThisType& operator=(const ThisType& I) {
      assert(OwningGraph == I.OwningGraph);
      EdgeIter = I.EdgeIter;
      return *this;
    }

    inline bool operator==(const ThisType& I) const
    { return EdgeIter == I.EdgeIter; }
    inline bool operator!=(const ThisType& I) const
    { return EdgeIter != I.EdgeIter; }

    inline pointer operator*() const {
      return &OwningGraph->getNode((*EdgeIter)->ToolName());
    }
    inline pointer operator->() const {
      return &OwningGraph->getNode((*EdgeIter)->ToolName());
    }

    ThisType& operator++() { ++EdgeIter; return *this; } // Preincrement
    ThisType operator++(int) { // Postincrement
      ThisType tmp = *this;
      ++*this;
      return tmp;
    }

    inline ThisType& operator--() { --EdgeIter; return *this; }  // Predecrement
    inline ThisType operator--(int) { // Postdecrement
      ThisType tmp = *this;
      --*this;
      return tmp;
    }

  };
}

namespace llvm {
  template <>
  struct GraphTraits<llvmc::CompilationGraph*> {
    typedef llvmc::CompilationGraph GraphType;
    typedef llvmc::Node NodeType;
    typedef llvmc::NodeChildIterator ChildIteratorType;

    static NodeType* getEntryNode(GraphType* G) {
      return &G->getNode("root");
    }

    static ChildIteratorType child_begin(NodeType* N) {
      return ChildIteratorType(N, N->OutEdges.begin());
    }
    static ChildIteratorType child_end(NodeType* N) {
      return ChildIteratorType(N, N->OutEdges.end());
    }

    typedef llvmc::NodesIterator nodes_iterator;
    static nodes_iterator nodes_begin(GraphType *G) {
      return GraphBegin(G);
    }
    static nodes_iterator nodes_end(GraphType *G) {
      return GraphEnd(G);
    }
  };

}

#endif // LLVM_TOOLS_LLVMC2_COMPILATION_GRAPH_H
