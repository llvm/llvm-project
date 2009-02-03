//===-- llvm/Support/GraphWriter.h - Write graph to a .dot file -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple interface that can be used to print out generic
// LLVM graphs to ".dot" files.  "dot" is a tool that is part of the AT&T
// graphviz package (http://www.research.att.com/sw/tools/graphviz/) which can
// be used to turn the files output by this interface into a variety of
// different graphics formats.
//
// Graphs do not need to implement any interface past what is already required
// by the GraphTraits template, but they can choose to implement specializations
// of the DOTGraphTraits template if they want to customize the graphs output in
// any way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GRAPHWRITER_H
#define LLVM_SUPPORT_GRAPHWRITER_H

#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/System/Path.h"
#include <fstream>
#include <vector>

namespace llvm {

namespace DOT {  // Private functions...
  inline std::string EscapeString(const std::string &Label) {
    std::string Str(Label);
    for (unsigned i = 0; i != Str.length(); ++i)
      switch (Str[i]) {
      case '\n':
        Str.insert(Str.begin()+i, '\\');  // Escape character...
        ++i;
        Str[i] = 'n';
        break;
      case '\t':
        Str.insert(Str.begin()+i, ' ');  // Convert to two spaces
        ++i;
        Str[i] = ' ';
        break;
      case '\\':
        if (i+1 != Str.length())
          switch (Str[i+1]) {
            case 'l': continue; // don't disturb \l
            case '|': case '{': case '}':
               Str.erase(Str.begin()+i); continue;
            default: break;
          }
      case '{': case '}':
      case '<': case '>':
      case '|': case '"':
        Str.insert(Str.begin()+i, '\\');  // Escape character...
        ++i;  // don't infinite loop
        break;
      }
    return Str;
  }
}

void DisplayGraph(const sys::Path& Filename);
  
template<typename GraphType>
class GraphWriter {
  std::ostream &O;
  const GraphType &G;

  typedef DOTGraphTraits<GraphType>           DOTTraits;
  typedef GraphTraits<GraphType>              GTraits;
  typedef typename GTraits::NodeType          NodeType;
  typedef typename GTraits::nodes_iterator    node_iterator;
  typedef typename GTraits::ChildIteratorType child_iterator;
public:
  GraphWriter(std::ostream &o, const GraphType &g) : O(o), G(g) {}

  void writeHeader(const std::string &Name) {
    std::string GraphName = DOTTraits::getGraphName(G);

    if (!Name.empty())
      O << "digraph \"" << DOT::EscapeString(Name) << "\" {\n";
    else if (!GraphName.empty())
      O << "digraph \"" << DOT::EscapeString(GraphName) << "\" {\n";
    else
      O << "digraph unnamed {\n";

    if (DOTTraits::renderGraphFromBottomUp())
      O << "\trankdir=\"BT\";\n";

    if (!Name.empty())
      O << "\tlabel=\"" << DOT::EscapeString(Name) << "\";\n";
    else if (!GraphName.empty())
      O << "\tlabel=\"" << DOT::EscapeString(GraphName) << "\";\n";
    O << DOTTraits::getGraphProperties(G);
    O << "\n";
  }

  void writeFooter() {
    // Finish off the graph
    O << "}\n";
  }

  void writeNodes() {
    // Loop over the graph, printing it out...
    for (node_iterator I = GTraits::nodes_begin(G), E = GTraits::nodes_end(G);
         I != E; ++I)
      writeNode(*I);
  }
  
  void writeNode(NodeType& Node) {
    writeNode(&Node);
  }

  void writeNode(NodeType *const *Node) {
    writeNode(*Node);
  }

  void writeNode(NodeType *Node) {
    std::string NodeAttributes = DOTTraits::getNodeAttributes(Node, G);

    O << "\tNode" << static_cast<const void*>(Node) << " [shape=record,";
    if (!NodeAttributes.empty()) O << NodeAttributes << ",";
    O << "label=\"{";

    if (!DOTTraits::renderGraphFromBottomUp()) {
      O << DOT::EscapeString(DOTTraits::getNodeLabel(Node, G));

      // If we should include the address of the node in the label, do so now.
      if (DOTTraits::hasNodeAddressLabel(Node, G))
        O << "|" << (void*)Node;
    }

    // Print out the fields of the current node...
    child_iterator EI = GTraits::child_begin(Node);
    child_iterator EE = GTraits::child_end(Node);
    if (EI != EE) {
      if (!DOTTraits::renderGraphFromBottomUp()) O << "|";
      O << "{";

      for (unsigned i = 0; EI != EE && i != 64; ++EI, ++i) {
        if (i) O << "|";
        O << "<s" << i << ">" << DOTTraits::getEdgeSourceLabel(Node, EI);
      }

      if (EI != EE)
        O << "|<s64>truncated...";
      O << "}";
      if (DOTTraits::renderGraphFromBottomUp()) O << "|";
    }

    if (DOTTraits::renderGraphFromBottomUp()) {
      O << DOT::EscapeString(DOTTraits::getNodeLabel(Node, G));

      // If we should include the address of the node in the label, do so now.
      if (DOTTraits::hasNodeAddressLabel(Node, G))
        O << "|" << (void*)Node;
    }

    if (DOTTraits::hasEdgeDestLabels()) {
      O << "|{";

      unsigned i = 0, e = DOTTraits::numEdgeDestLabels(Node);
      for (; i != e && i != 64; ++i) {
        if (i) O << "|";
        O << "<d" << i << ">" << DOTTraits::getEdgeDestLabel(Node, i);
      }

      if (i != e)
        O << "|<d64>truncated...";
      O << "}";
    }

    O << "}\"];\n";   // Finish printing the "node" line

    // Output all of the edges now
    EI = GTraits::child_begin(Node);
    for (unsigned i = 0; EI != EE && i != 64; ++EI, ++i)
      writeEdge(Node, i, EI);
    for (; EI != EE; ++EI)
      writeEdge(Node, 64, EI);
  }

  void writeEdge(NodeType *Node, unsigned edgeidx, child_iterator EI) {
    if (NodeType *TargetNode = *EI) {
      int DestPort = -1;
      if (DOTTraits::edgeTargetsEdgeSource(Node, EI)) {
        child_iterator TargetIt = DOTTraits::getEdgeTarget(Node, EI);

        // Figure out which edge this targets...
        unsigned Offset =
          (unsigned)std::distance(GTraits::child_begin(TargetNode), TargetIt);
        DestPort = static_cast<int>(Offset);
      }

      emitEdge(static_cast<const void*>(Node), edgeidx,
               static_cast<const void*>(TargetNode), DestPort,
               DOTTraits::getEdgeAttributes(Node, EI));
    }
  }

  /// emitSimpleNode - Outputs a simple (non-record) node
  void emitSimpleNode(const void *ID, const std::string &Attr,
                      const std::string &Label, unsigned NumEdgeSources = 0,
                      const std::vector<std::string> *EdgeSourceLabels = 0) {
    O << "\tNode" << ID << "[ ";
    if (!Attr.empty())
      O << Attr << ",";
    O << " label =\"";
    if (NumEdgeSources) O << "{";
    O << DOT::EscapeString(Label);
    if (NumEdgeSources) {
      O << "|{";

      for (unsigned i = 0; i != NumEdgeSources; ++i) {
        if (i) O << "|";
        O << "<g" << i << ">";
        if (EdgeSourceLabels) O << (*EdgeSourceLabels)[i];
      }
      O << "}}";
    }
    O << "\"];\n";
  }

  /// emitEdge - Output an edge from a simple node into the graph...
  void emitEdge(const void *SrcNodeID, int SrcNodePort,
                const void *DestNodeID, int DestNodePort,
                const std::string &Attrs) {
    if (SrcNodePort  > 64) return;             // Eminating from truncated part?
    if (DestNodePort > 64) DestNodePort = 64;  // Targetting the truncated part?

    O << "\tNode" << SrcNodeID;
    if (SrcNodePort >= 0)
      O << ":s" << SrcNodePort;
    O << " -> Node" << DestNodeID;
    if (DestNodePort >= 0)
      O << ":d" << DestNodePort;

    if (!Attrs.empty())
      O << "[" << Attrs << "]";
    O << ";\n";
  }
};

template<typename GraphType>
std::ostream &WriteGraph(std::ostream &O, const GraphType &G,
                         const std::string &Name = "",
                         const std::string &Title = "") {
  // Start the graph emission process...
  GraphWriter<GraphType> W(O, G);

  // Output the header for the graph...
  W.writeHeader(Title);

  // Emit all of the nodes in the graph...
  W.writeNodes();

  // Output any customizations on the graph
  DOTGraphTraits<GraphType>::addCustomGraphFeatures(G, W);

  // Output the end of the graph
  W.writeFooter();
  return O;
}

template<typename GraphType>
sys::Path WriteGraph(const GraphType &G,
                     const std::string& Name, 
                     const std::string& Title = "") {
  std::string ErrMsg;
  sys::Path Filename = sys::Path::GetTemporaryDirectory(&ErrMsg);
  if (Filename.isEmpty()) {
    cerr << "Error: " << ErrMsg << "\n";
    return Filename;
  }
  Filename.appendComponent(Name + ".dot");
  if (Filename.makeUnique(true,&ErrMsg)) {
    cerr << "Error: " << ErrMsg << "\n";
    return sys::Path();
  }

  cerr << "Writing '" << Filename << "'... ";
  
  std::ofstream O(Filename.c_str());

  if (O.good()) {
    WriteGraph(O, G, Name, Title);
    cerr << " done. \n";

    O.close();
  } else {
    cerr << "error opening file for writing!\n";
    Filename.clear();
  }
  
  return Filename;
}
  
/// ViewGraph - Emit a dot graph, run 'dot', run gv on the postscript file,
/// then cleanup.  For use from the debugger.
///
template<typename GraphType>
void ViewGraph(const GraphType& G, 
               const std::string& Name, 
               const std::string& Title = "") {
  sys::Path Filename =  WriteGraph(G, Name, Title);

  if (Filename.isEmpty()) {
    return;
  }
  
  DisplayGraph(Filename);
}

} // End llvm namespace

#endif
