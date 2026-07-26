#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

struct Event {
  uint64_t ID;
  std::string Module;
  uint64_t InstID;
  std::vector<std::string> Details;
};

struct Edge {
  uint64_t From;
  uint64_t To;
  bool IsMemory;
};

static std::string escapeDotString(const std::string &Text) {
  std::string Result;

  for (char C : Text) {
    if (C == '\\' || C == '"') {
      Result += '\\';
    }

    Result += C;
  }

  return Result;
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 4) {
    std::cerr
        << "Usage: " << argv[0]
        << " <trace-file> [-o <dot-file>]\n";
    return 1;
  }

  std::string InputPath = argv[1];
  std::string OutputPath = "graph.dot";

  if (argc == 4) {
    if (std::string(argv[2]) != "-o") {
      std::cerr << "Expected -o before output filename\n";
      return 1;
    }

    OutputPath = argv[3];
  }

  std::ifstream Input(InputPath);

  if (!Input.is_open()) {
    std::cerr << "Could not open input file: "
              << InputPath << '\n';
    return 1;
  }

  std::map<uint64_t, Event> Events;
  std::vector<Edge> Edges;
  std::optional<uint64_t> CurrentEventID;

  std::string Line;

  while (std::getline(Input, Line)) {
    if (Line.empty()) {
      continue;
    }

    std::istringstream LineStream(Line);
    std::string RecordType;

    LineStream >> RecordType;

    if (RecordType == "EVENT") {
      uint64_t EventID;
      uint64_t InstID;
      std::string ModuleWord;
      std::string Module;
      std::string InstWord;

      if (!(LineStream >> EventID
                       >> ModuleWord
                       >> Module
                       >> InstWord
                       >> InstID)) {
        std::cerr << "Invalid EVENT line: "
                  << Line << '\n';
        return 1;
      }

      if (ModuleWord != "MODULE" ||
          InstWord != "INST") {
        std::cerr << "Invalid EVENT format: "
                  << Line << '\n';
        return 1;
      }

      Events[EventID] =
          Event{EventID, Module, InstID, {}};

      CurrentEventID = EventID;

    } else if (RecordType == "EDGE" ||
               RecordType == "MEM_EDGE") {
      uint64_t From;
      uint64_t To;
      std::string Arrow;

      if (!(LineStream >> From >> Arrow >> To) ||
          Arrow != "->") {
        std::cerr << "Invalid edge line: "
                  << Line << '\n';
        return 1;
      }

      Edges.push_back(
          Edge{From, To, RecordType == "MEM_EDGE"});

    } else if (RecordType == "STORE" ||
               RecordType == "LOAD") {
      std::string Address;
      uint64_t Size;

      if (!(LineStream >> Address >> Size)) {
        std::cerr << "Invalid memory line: "
                  << Line << '\n';
        return 1;
      }

      if (!CurrentEventID.has_value()) {
        std::cerr
            << "Memory operation without preceding EVENT: "
            << Line << '\n';
        return 1;
      }

      auto EventIt = Events.find(*CurrentEventID);

      if (EventIt == Events.end()) {
        std::cerr << "Current event was not found\n";
        return 1;
      }

      std::ostringstream Detail;

      Detail << RecordType
             << " " << Address
             << " size=" << Size;

      EventIt->second.Details.push_back(Detail.str());

    } else {
      std::cerr << "Unknown trace record: "
                << Line << '\n';
      return 1;
    }
  }

  std::ofstream Output(OutputPath);

  if (!Output.is_open()) {
    std::cerr << "Could not open output file: "
              << OutputPath << '\n';
    return 1;
  }

  Output << "digraph DefUse {\n";
  Output << "  rankdir=TB;\n";
  Output << "  node [shape=box, fontname=\"monospace\"];\n";
  Output << "  edge [fontname=\"monospace\"];\n\n";

  for (const auto &[EventID, EventData] : Events) {
    std::ostringstream Label;

    Label << "Event " << EventData.ID
          << "\\nModule " << EventData.Module
          << "\\nInst " << EventData.InstID;

    for (const std::string &Detail : EventData.Details) {
      Label << "\\n" << Detail;
    }

    Output << "  n" << EventID
           << " [label=\""
           << escapeDotString(Label.str())
           << "\"];\n";
  }

  Output << '\n';

  for (const Edge &GraphEdge : Edges) {
    Output << "  n" << GraphEdge.From
          << " -> n" << GraphEdge.To;

    if (GraphEdge.IsMemory) {
      Output << " [label=\"memory\", style=dashed]";
    }

    Output << ";\n";
  }

  Output << "}\n";

  std::cout << "Wrote " << Events.size()
            << " nodes and " << Edges.size()
            << " edges to " << OutputPath << '\n';

  return 0;
}
