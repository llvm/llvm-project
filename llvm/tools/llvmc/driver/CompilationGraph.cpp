//===--- CompilationGraph.cpp - The LLVM Compiler Driver --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Compilation graph - implementation.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "llvm/CompilerDriver/CompilationGraph.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <queue>
#include <stdexcept>

using namespace llvm;
using namespace llvmc;

extern cl::list<std::string> InputFilenames;
extern cl::list<std::string> Languages;

namespace llvmc {

  const std::string& LanguageMap::GetLanguage(const sys::Path& File) const {
    LanguageMap::const_iterator Lang = this->find(File.getSuffix());
    if (Lang == this->end())
      throw std::runtime_error("Unknown suffix: " + File.getSuffix());
    return Lang->second;
  }
}

namespace {

  /// ChooseEdge - Return the edge with the maximum weight.
  template <class C>
  const Edge* ChooseEdge(const C& EdgesContainer,
                         const InputLanguagesSet& InLangs,
                         const std::string& NodeName = "root") {
    const Edge* MaxEdge = 0;
    unsigned MaxWeight = 0;
    bool SingleMax = true;

    for (typename C::const_iterator B = EdgesContainer.begin(),
           E = EdgesContainer.end(); B != E; ++B) {
      const Edge* e = B->getPtr();
      unsigned EW = e->Weight(InLangs);
      if (EW > MaxWeight) {
        MaxEdge = e;
        MaxWeight = EW;
        SingleMax = true;
      } else if (EW == MaxWeight) {
        SingleMax = false;
      }
    }

    if (!SingleMax)
      throw std::runtime_error("Node " + NodeName +
                               ": multiple maximal outward edges found!"
                               " Most probably a specification error.");
    if (!MaxEdge)
      throw std::runtime_error("Node " + NodeName +
                               ": no maximal outward edge found!"
                               " Most probably a specification error.");
    return MaxEdge;
  }

}

void Node::AddEdge(Edge* Edg) {
  // If there already was an edge between two nodes, modify it instead
  // of adding a new edge.
  const std::string& ToolName = Edg->ToolName();
  for (container_type::iterator B = OutEdges.begin(), E = OutEdges.end();
       B != E; ++B) {
    if ((*B)->ToolName() == ToolName) {
      llvm::IntrusiveRefCntPtr<Edge>(Edg).swap(*B);
      return;
    }
  }
  OutEdges.push_back(llvm::IntrusiveRefCntPtr<Edge>(Edg));
}

CompilationGraph::CompilationGraph() {
  NodesMap["root"] = Node(this);
}

Node& CompilationGraph::getNode(const std::string& ToolName) {
  nodes_map_type::iterator I = NodesMap.find(ToolName);
  if (I == NodesMap.end())
    throw std::runtime_error("Node " + ToolName + " is not in the graph");
  return I->second;
}

const Node& CompilationGraph::getNode(const std::string& ToolName) const {
  nodes_map_type::const_iterator I = NodesMap.find(ToolName);
  if (I == NodesMap.end())
    throw std::runtime_error("Node " + ToolName + " is not in the graph!");
  return I->second;
}

// Find the tools list corresponding to the given language name.
const CompilationGraph::tools_vector_type&
CompilationGraph::getToolsVector(const std::string& LangName) const
{
  tools_map_type::const_iterator I = ToolsMap.find(LangName);
  if (I == ToolsMap.end())
    throw std::runtime_error("No tool corresponding to the language "
                             + LangName + " found");
  return I->second;
}

void CompilationGraph::insertNode(Tool* V) {
  if (NodesMap.count(V->Name()) == 0)
    NodesMap[V->Name()] = Node(this, V);
}

void CompilationGraph::insertEdge(const std::string& A, Edge* Edg) {
  Node& B = getNode(Edg->ToolName());
  if (A == "root") {
    const char** InLangs = B.ToolPtr->InputLanguages();
    for (;*InLangs; ++InLangs)
      ToolsMap[*InLangs].push_back(IntrusiveRefCntPtr<Edge>(Edg));
    NodesMap["root"].AddEdge(Edg);
  }
  else {
    Node& N = getNode(A);
    N.AddEdge(Edg);
  }
  // Increase the inward edge counter.
  B.IncrInEdges();
}

// Pass input file through the chain until we bump into a Join node or
// a node that says that it is the last.
void CompilationGraph::PassThroughGraph (const sys::Path& InFile,
                                         const Node* StartNode,
                                         const InputLanguagesSet& InLangs,
                                         const sys::Path& TempDir,
                                         const LanguageMap& LangMap) const {
  sys::Path In = InFile;
  const Node* CurNode = StartNode;

  while(true) {
    Tool* CurTool = CurNode->ToolPtr.getPtr();

    if (CurTool->IsJoin()) {
      JoinTool& JT = dynamic_cast<JoinTool&>(*CurTool);
      JT.AddToJoinList(In);
      break;
    }

    Action CurAction = CurTool->GenerateAction(In, CurNode->HasChildren(),
                                               TempDir, InLangs, LangMap);

    if (int ret = CurAction.Execute())
      throw error_code(ret);

    if (CurAction.StopCompilation())
      return;

    CurNode = &getNode(ChooseEdge(CurNode->OutEdges,
                                  InLangs,
                                  CurNode->Name())->ToolName());
    In = CurAction.OutFile();
  }
}

// Find the head of the toolchain corresponding to the given file.
// Also, insert an input language into InLangs.
const Node* CompilationGraph::
FindToolChain(const sys::Path& In, const std::string* ForceLanguage,
              InputLanguagesSet& InLangs, const LanguageMap& LangMap) const {

  // Determine the input language.
  const std::string& InLanguage =
    ForceLanguage ? *ForceLanguage : LangMap.GetLanguage(In);

  // Add the current input language to the input language set.
  InLangs.insert(InLanguage);

  // Find the toolchain for the input language.
  const tools_vector_type& TV = getToolsVector(InLanguage);
  if (TV.empty())
    throw std::runtime_error("No toolchain corresponding to language "
                             + InLanguage + " found");
  return &getNode(ChooseEdge(TV, InLangs)->ToolName());
}

// Helper function used by Build().
// Traverses initial portions of the toolchains (up to the first Join node).
// This function is also responsible for handling the -x option.
void CompilationGraph::BuildInitial (InputLanguagesSet& InLangs,
                                     const sys::Path& TempDir,
                                     const LanguageMap& LangMap) {
  // This is related to -x option handling.
  cl::list<std::string>::const_iterator xIter = Languages.begin(),
    xBegin = xIter, xEnd = Languages.end();
  bool xEmpty = true;
  const std::string* xLanguage = 0;
  unsigned xPos = 0, xPosNext = 0, filePos = 0;

  if (xIter != xEnd) {
    xEmpty = false;
    xPos = Languages.getPosition(xIter - xBegin);
    cl::list<std::string>::const_iterator xNext = llvm::next(xIter);
    xPosNext = (xNext == xEnd) ? std::numeric_limits<unsigned>::max()
      : Languages.getPosition(xNext - xBegin);
    xLanguage = (*xIter == "none") ? 0 : &(*xIter);
  }

  // For each input file:
  for (cl::list<std::string>::const_iterator B = InputFilenames.begin(),
         CB = B, E = InputFilenames.end(); B != E; ++B) {
    sys::Path In = sys::Path(*B);

    // Code for handling the -x option.
    // Output: std::string* xLanguage (can be NULL).
    if (!xEmpty) {
      filePos = InputFilenames.getPosition(B - CB);

      if (xPos < filePos) {
        if (filePos < xPosNext) {
          xLanguage = (*xIter == "none") ? 0 : &(*xIter);
        }
        else { // filePos >= xPosNext
          // Skip xIters while filePos > xPosNext
          while (filePos > xPosNext) {
            ++xIter;
            xPos = xPosNext;

            cl::list<std::string>::const_iterator xNext = llvm::next(xIter);
            if (xNext == xEnd)
              xPosNext = std::numeric_limits<unsigned>::max();
            else
              xPosNext = Languages.getPosition(xNext - xBegin);
            xLanguage = (*xIter == "none") ? 0 : &(*xIter);
          }
        }
      }
    }

    // Find the toolchain corresponding to this file.
    const Node* N = FindToolChain(In, xLanguage, InLangs, LangMap);
    // Pass file through the chain starting at head.
    PassThroughGraph(In, N, InLangs, TempDir, LangMap);
  }
}

// Sort the nodes in topological order.
void CompilationGraph::TopologicalSort(std::vector<const Node*>& Out) {
  std::queue<const Node*> Q;
  Q.push(&getNode("root"));

  while (!Q.empty()) {
    const Node* A = Q.front();
    Q.pop();
    Out.push_back(A);
    for (Node::const_iterator EB = A->EdgesBegin(), EE = A->EdgesEnd();
         EB != EE; ++EB) {
      Node* B = &getNode((*EB)->ToolName());
      B->DecrInEdges();
      if (B->HasNoInEdges())
        Q.push(B);
    }
  }
}

namespace {
  bool NotJoinNode(const Node* N) {
    return N->ToolPtr ? !N->ToolPtr->IsJoin() : true;
  }
}

// Call TopologicalSort and filter the resulting list to include
// only Join nodes.
void CompilationGraph::
TopologicalSortFilterJoinNodes(std::vector<const Node*>& Out) {
  std::vector<const Node*> TopSorted;
  TopologicalSort(TopSorted);
  std::remove_copy_if(TopSorted.begin(), TopSorted.end(),
                      std::back_inserter(Out), NotJoinNode);
}

int CompilationGraph::Build (const sys::Path& TempDir,
                             const LanguageMap& LangMap) {

  InputLanguagesSet InLangs;

  // Traverse initial parts of the toolchains and fill in InLangs.
  BuildInitial(InLangs, TempDir, LangMap);

  std::vector<const Node*> JTV;
  TopologicalSortFilterJoinNodes(JTV);

  // For all join nodes in topological order:
  for (std::vector<const Node*>::iterator B = JTV.begin(), E = JTV.end();
       B != E; ++B) {

    const Node* CurNode = *B;
    JoinTool* JT = &dynamic_cast<JoinTool&>(*CurNode->ToolPtr.getPtr());

    // Are there any files in the join list?
    if (JT->JoinListEmpty())
      continue;

    Action CurAction = JT->GenerateAction(CurNode->HasChildren(),
                                          TempDir, InLangs, LangMap);

    if (int ret = CurAction.Execute())
      throw error_code(ret);

    if (CurAction.StopCompilation())
      return 0;

    const Node* NextNode = &getNode(ChooseEdge(CurNode->OutEdges, InLangs,
                                               CurNode->Name())->ToolName());
    PassThroughGraph(sys::Path(CurAction.OutFile()), NextNode,
                     InLangs, TempDir, LangMap);
  }

  return 0;
}

int CompilationGraph::CheckLanguageNames() const {
  int ret = 0;
  // Check that names for output and input languages on all edges do match.
  for (const_nodes_iterator B = this->NodesMap.begin(),
         E = this->NodesMap.end(); B != E; ++B) {

    const Node & N1 = B->second;
    if (N1.ToolPtr) {
      for (Node::const_iterator EB = N1.EdgesBegin(), EE = N1.EdgesEnd();
           EB != EE; ++EB) {
        const Node& N2 = this->getNode((*EB)->ToolName());

        if (!N2.ToolPtr) {
          ++ret;
          std::cerr << "Error: there is an edge from '" << N1.ToolPtr->Name()
                    << "' back to the root!\n\n";
          continue;
        }

        const char* OutLang = N1.ToolPtr->OutputLanguage();
        const char** InLangs = N2.ToolPtr->InputLanguages();
        bool eq = false;
        for (;*InLangs; ++InLangs) {
          if (std::strcmp(OutLang, *InLangs) == 0) {
            eq = true;
            break;
          }
        }

        if (!eq) {
          ++ret;
          std::cerr << "Error: Output->input language mismatch in the edge '" <<
            N1.ToolPtr->Name() << "' -> '" << N2.ToolPtr->Name() << "'!\n";

          std::cerr << "Expected one of { ";

          InLangs = N2.ToolPtr->InputLanguages();
          for (;*InLangs; ++InLangs) {
            std::cerr << '\'' << *InLangs << (*(InLangs+1) ? "', " : "'");
          }

          std::cerr << " }, but got '" << OutLang << "'!\n\n";
        }

      }
    }
  }

  return ret;
}

int CompilationGraph::CheckMultipleDefaultEdges() const {
  int ret = 0;
  InputLanguagesSet Dummy;

  // For all nodes, just iterate over the outgoing edges and check if there is
  // more than one edge with maximum weight.
  for (const_nodes_iterator B = this->NodesMap.begin(),
         E = this->NodesMap.end(); B != E; ++B) {
    const Node& N = B->second;
    unsigned MaxWeight = 0;

    // Ignore the root node.
    if (!N.ToolPtr)
      continue;

    for (Node::const_iterator EB = N.EdgesBegin(), EE = N.EdgesEnd();
         EB != EE; ++EB) {
      unsigned EdgeWeight = (*EB)->Weight(Dummy);
      if (EdgeWeight > MaxWeight) {
        MaxWeight = EdgeWeight;
      }
      else if (EdgeWeight == MaxWeight) {
        ++ret;
        std::cerr
          << "Error: there are multiple maximal edges stemming from the '"
          << N.ToolPtr->Name() << "' node!\n\n";
        break;
      }
    }
  }

  return ret;
}

int CompilationGraph::CheckCycles() {
  unsigned deleted = 0;
  std::queue<Node*> Q;
  Q.push(&getNode("root"));

  // Try to delete all nodes that have no ingoing edges, starting from the
  // root. If there are any nodes left after this operation, then we have a
  // cycle. This relies on '--check-graph' not performing the topological sort.
  while (!Q.empty()) {
    Node* A = Q.front();
    Q.pop();
    ++deleted;

    for (Node::iterator EB = A->EdgesBegin(), EE = A->EdgesEnd();
         EB != EE; ++EB) {
      Node* B = &getNode((*EB)->ToolName());
      B->DecrInEdges();
      if (B->HasNoInEdges())
        Q.push(B);
    }
  }

  if (deleted != NodesMap.size()) {
    std::cerr << "Error: there are cycles in the compilation graph!\n"
              << "Try inspecting the diagram produced by "
      "'llvmc --view-graph'.\n\n";
    return 1;
  }

  return 0;
}

int CompilationGraph::Check () {
  // We try to catch as many errors as we can in one go.
  int ret = 0;

  // Check that output/input language names match.
  ret += this->CheckLanguageNames();

  // Check for multiple default edges.
  ret += this->CheckMultipleDefaultEdges();

  // Check for cycles.
  ret += this->CheckCycles();

  return ret;
}

// Code related to graph visualization.

namespace llvm {
  template <>
  struct DOTGraphTraits<llvmc::CompilationGraph*>
    : public DefaultDOTGraphTraits
  {

    template<typename GraphType>
    static std::string getNodeLabel(const Node* N, const GraphType&)
    {
      if (N->ToolPtr)
        if (N->ToolPtr->IsJoin())
          return N->Name() + "\n (join" +
            (N->HasChildren() ? ")"
             : std::string(": ") + N->ToolPtr->OutputLanguage() + ')');
        else
          return N->Name();
      else
        return "root";
    }

    template<typename EdgeIter>
    static std::string getEdgeSourceLabel(const Node* N, EdgeIter I) {
      if (N->ToolPtr) {
        return N->ToolPtr->OutputLanguage();
      }
      else {
        const char** InLangs = I->ToolPtr->InputLanguages();
        std::string ret;

        for (; *InLangs; ++InLangs) {
          if (*(InLangs + 1)) {
            ret += *InLangs;
            ret +=  ", ";
          }
          else {
            ret += *InLangs;
          }
        }

        return ret;
      }
    }
  };

}

void CompilationGraph::writeGraph() {
  std::ofstream O("compilation-graph.dot");

  if (O.good()) {
    llvm::WriteGraph(this, "compilation-graph");
    O.close();
  }
  else {
    throw std::runtime_error("Error opening file 'compilation-graph.dot'"
                             " for writing!");
  }
}

void CompilationGraph::viewGraph() {
  llvm::ViewGraph(this, "compilation-graph");
}
