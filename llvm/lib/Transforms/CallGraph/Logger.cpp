#include <cassert>
#include <fstream>
#include <cstdio>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

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

  void setCall(int64_t Caller, int64_t Callee);

  // We will change the dot file in the destructor, because we want to do it
  // only one time
  ~GraphEditor() { writeGraph(); }
};

void Logger(int64_t caller_ptr, int64_t callee_ptr) {
  GraphEditor &Graph = GraphEditor::getInstance();
  Graph.setCall(caller_ptr, callee_ptr);
}

void GraphEditor::setCall(int64_t Caller, int64_t Callee) {
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
      OutFile << NameMap[Edges.first] << Edges.first << " -> "
              << NameMap[Child.first] << "[label = " << Child.second << "];\n";
    }
  }

  OutFile << "}";
  OutFile.close();
}

void GraphEditor::fillNameMap(
    std::unordered_map<int64_t, std::string> &NameMap) const {
  FILE *SrcFile = std::fopen("OutFile.txt", "r");
  assert(SrcFile != nullptr);

  char CallerName[128], CalleeName[128];
  int64_t CallerAddr, CalleeAddr;

  int SymCnt = fscanf(SrcFile, "digraph G { \n");
  assert(SymCnt != 0);

  while (fscanf(SrcFile, "%s%ld -> %s%ld \n", CallerName, &CallerAddr,
                CalleeName, &CalleeAddr)) {
    auto &CallerOldName = NameMap[CallerAddr];
    if (CallerOldName.empty())
      CallerOldName = CallerName;

    auto &CalleeOldName = NameMap[CalleeAddr];
    if (CalleeOldName.empty())
      CalleeOldName = CalleeName;
  }

  SymCnt = fscanf(SrcFile, " } ");
  assert(SymCnt != 0);
}
