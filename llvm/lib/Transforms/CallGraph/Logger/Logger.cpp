#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <cassert>
#include <fstream>
#include <map>

#define PRINT_LINE fprintf(stderr, "[%s:%s:%d]\n", __FILE__, __func__, __LINE__)

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

  void addCall(int64_t *Caller, int64_t *Callee);

  // We will change the dot file in the destructor, because we want to do it
  // only one time
  ~GraphEditor() { writeGraph(); }
};

void GraphEditor::addCall(int64_t *Caller, int64_t *Callee) {
    std::cout << "Caller = " << (uint64_t) Caller << ", Callee = " << (uint64_t) Callee << '\n';
  Graph[(uint64_t) Caller][(uint64_t) Callee]++;
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

      //OutFile << Child.first << " --> " << Edges.first << '\n';
        printf("%lx --> %lx\n", Child.first, Edges.first);
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
    PRINT_LINE;
    int64_t *callee_addr, *caller_arrd;  // Dummies. Replace them with return values
                                        // of llvm.returnaddress intrinsic.
    GraphEditor &graph = GraphEditor::getInstance();
    graph.addCall(caller_arrd, callee_addr);
}
