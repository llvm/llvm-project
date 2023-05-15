#include "../../../../../elf-parser/include/parser.hpp"
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

  // The vector for representing ranges where the various code segments
  // were loaded
  // {range_begin, range_end, offset in elf src file}
  // We use it in case of PIC
  std::vector<std::array<uint64_t, 3>> Ranges;

  // shows is a file was compiled with -fPIC option
  std::optional<bool> isPIC;

  void writeGraph() const;
  void parseMapsFile();

  // Ctor will dump /proc/self/maps to the maps.txt files
  // It is necessary, because we support -fPIC flag
  GraphEditor() {
    auto pid = getpid();
    std::string system_text =
        "cat /proc/" + std::to_string(pid) + "/maps > maps.txt";
    system(system_text.c_str());
  };

public:
  static GraphEditor &getInstance() {
    static GraphEditor Object;
    return Object;
  }

  void addCall(int64_t *Caller, int64_t *Callee);

  // We will write to file in the destructor, because we want to do it
  // only one time
  ~GraphEditor() { writeGraph(); }
};

void GraphEditor::addCall(int64_t *Caller, int64_t *Callee) {
  Graph[(uint64_t)Caller][(uint64_t)Callee]++;
}

void GraphEditor::writeGraph() const {
  std::ofstream OutFile;
  OutFile.open("Graph.txt");

  // Now need traverse Graph map and print edges to dot file
  for (auto &Edges : Graph) {
    for (auto &Child : Edges.second) {
      // Edge.first - addr of caller
      // Edge.second - map Children (<callee, calls_amnt>)
      // Child - pair <callee_addr, calls_amnt>
      OutFile << std::hex << Edges.first << ' ' << Child.first << ' '
              << std::dec << Child.second << '\n';
    }
  }
  OutFile.close();
}

void GraphEditor::parseMapsFile() {
  /*FILE *MapFile = fopen("maps.txt", "r");
  if (!MapFile)
    return;

  std::array<uint64_t, 3> Range;
  char *Permissions = new char[32];
  char *FileName = new char[256];

  while (strcmp(FileName, "[heap]")) {
    if (!isPIC)
      *isPIC = ::isPIC(FileName);

    fscanf(MapFile, "%lu-%lu %s %lu %*d:%*d %*lu %s ", &Range[0], &Range[1],
           Permissions, &Range[2], FileName);

    if (!strcmp(Permissions, "[r-xp]"))
      Ranges.push_back(Range);
  }

  delete[] Permissions;
  delete[] FileName;
*/}

  void Logger() {
    int64_t *callee_addr,
        *caller_arrd; // Dummies. Replace them with return
                      // values of llvm.returnaddress intrinsic.
    GraphEditor &graph = GraphEditor::getInstance();
    graph.addCall(caller_arrd, callee_addr);
  }
