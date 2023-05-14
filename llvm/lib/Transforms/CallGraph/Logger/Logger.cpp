#include <unordered_map>
#include <iostream>
#include <cassert>
#include <fstream>

/// Class for implementing Parsing and Changing dot file with graph
/// Singleton implementation
class GraphEditor {
private:
  /// In that unordered map we store edges info:
  /// key -- addr of caller
  /// value -- map of all callees
  /// value::key -- addr of callee
  /// value::value -- calls amnt
  std::unordered_map<int64_t, std::map<int64_t, int64_t>> Graph;

  /// Set with already created nodes for graphviz
  mutable std::unordered_set<int64_t> Nodes;

  void writeGraph() const;
  /// Method for traversing source dot file and collecting pair from <addr,
  /// name> for every func
  void fillNameMap(std::unordered_map<int64_t, std::string> &NameMap) const;

  GraphEditor() {}

public:
  static GraphEditor &getInstance() {
    static GraphEditor Object;
    return Object;
  }

  void addCall(int64_t Caller, int64_t Callee);

  // We will change the dot file in the destructor, because we want to do it
  // only one time
  ~GraphEditor() { writeGraph(); }
};

void GraphEditor::addCall(int64_t Caller, int64_t Callee) {
  Graph[Caller][Callee]++;
}

void GraphEditor::writeGraph() const {
  std::unordered_map<int64_t, std::string> NameMap;
  fillNameMap(NameMap);

  std::ofstream OutFile;
  OutFile.open("DynamicGraph.dot");

  OutFile << "digraph G {\n";
  // Now need traverse Graph map and print edges to dot file
  for (auto &Edges : Graph) {
    for (auto &Child : Edges.second) {
      // Edge.first - addr of caller
      // Edge.second - map Children (<callee, calls_amnt>)
      // Child - pair <callee_addr, calls_amnt>

      int64_t CallerAddr = Edges.first;
      int64_t CalleeAddr = Child.first;

      if (Nodes.find(CallerAddr) == Nodes.end()) {
        if (NameMap[CallerAddr].empty())
          continue;

        OutFile << CallerAddr << " [label = \"" << NameMap[CallerAddr]
                << "\" ]\n";
        Nodes.insert(CallerAddr);
      }
      if (Nodes.find(CalleeAddr) == Nodes.end()) {
        if (NameMap[CalleeAddr].empty())
          continue;

        OutFile << CalleeAddr << " [label = \"" << NameMap[CalleeAddr]
                << "\" ]\n";
        Nodes.insert(CalleeAddr);
      }

      OutFile << CallerAddr << " -> " << CalleeAddr
              << " [label = " << Child.second << "];\n";
    }
  }

  OutFile << "}";
  OutFile.close();
}

void GraphEditor::fillNameMap(
    std::unordered_map<int64_t, std::string> &NameMap) const {
  FILE *SrcFile = std::fopen("OutFile.dot", "r");
  assert(SrcFile != nullptr);

  char Name[128] = {};
  int64_t Addr = 0;

  fscanf(SrcFile, "digraph G { \n");

  while (fscanf(SrcFile, " {} %ld [label = \" %s \" ] ", &Addr, Name) == 2) {
    auto &OldName = NameMap[Addr];
    if (OldName.empty())
      OldName = Name;

    // Skip lines with edges for graphviz
    while (fscanf(SrcFile, " %ld -> %ld ", &Addr, &Addr))
      continue;
  }
}

void Logger()
{
    uint64_t callee_addr, caller_arrd;  // Dummies. Replace them with return values
                                        // of llvm.returnaddress intrinsic.
    auto graph = GraphEditor::getInstance();
    graph.addCall(caller_arrd, callee_addr);
}
