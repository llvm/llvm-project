//===- llvm-cas-dump.cpp - Tool for printing MC CAS objects ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCCASPrinter.h"
#include "StatsCollector.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/CASUtil/Utils.h"
#include "llvm/MCCAS/MCCASObjectV1.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::mccasformats::v1;

cl::opt<std::string> CASPath("cas", cl::Required, cl::desc("Path to CAS."));
cl::list<std::string> InputStrings(cl::Positional,
                                   cl::desc("CAS ID of the object to print"));
cl::opt<bool> CASIDFile("casid-file", cl::desc("Treat inputs as CASID files"));
cl::opt<bool> DwarfSectionsOnly("dwarf-sections-only",
                                cl::desc("Only print DWARF related sections"));
cl::opt<bool> DwarfDump("dwarf-dump",
                        cl::desc("Print the contents of DWARF sections"));
cl::opt<bool>
    HexDump("hex-dump",
            cl::desc("Print out a hex dump of every cas blocks contents"));

cl::opt<bool> HexDumpOneLine("hex-dump-one-line",
                             cl::desc("Print out the hex dump in one line"));
cl::opt<bool> Verbose("v", cl::desc("Enable verbse output in the dwarfdump"));
cl::opt<bool> DIERefs("die-refs",
                      cl::desc("Print out the DIERef block structure"));
cl::opt<std::string>
    ComputeStats("object-stats",
                 cl::desc("Compute and print out stats. Use '-' to print to "
                          "stdout, otherwise provide path"));

cl::opt<StatsCollector::FormatType> ObjectStatsFormat(
    "object-stats-format", cl::desc("choose object stats format:"),
    cl::values(
        clEnumValN(StatsCollector::FormatType::Pretty, "pretty",
                   "object stats formatted in a readable format (default)"),
        clEnumValN(StatsCollector::FormatType::CSV, "csv",
                   "object stats formatted in a CSV format")),
    cl::init(StatsCollector::FormatType::Pretty));

cl::opt<bool>
    AnalysisCASTree("analysis-only",
                    cl::desc("analyze converted objects from cas tree"));

/// If the input is a file (--casid-file), open the file given by `InputStr`
/// and get the ID from the file buffer.
/// Otherwise parse `InputStr` as a CASID.
CASID getCASIDFromInput(ObjectStore &CAS, StringRef InputStr) {
  ExitOnError ExitOnErr;
  ExitOnErr.setBanner((InputStr + ": ").str());

  if (!CASIDFile)
    return ExitOnErr(CAS.parseID(InputStr));

  auto ObjBuffer =
      ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(InputStr)));
  return ExitOnErr(readCASIDBuffer(CAS, ObjBuffer->getMemBufferRef()));
}

static void computeStats(ObjectStore &CAS, ArrayRef<ObjectProxy> TopLevels,
                         raw_ostream &StatOS) {
  ExitOnError ExitOnErr;
  ExitOnErr.setBanner("llvm-cas-object-format: compute-stats: ");

  llvm::errs() << "Collecting object stats...\n";

  // In the first traversal, just collect a POT. Use NumPaths as a "Seen" list.
  StatsCollector Collector(CAS, ObjectStatsFormat);
  auto &Nodes = Collector.Nodes;
  struct WorklistItem {
    ObjectRef ID;
    bool Visited;
    StringRef Path;
    const NodeSchema *Schema = nullptr;
  };
  SmallVector<WorklistItem> Worklist;
  SmallVector<POTItem> POT;
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  auto push = [&](ObjectRef ID, StringRef Path,
                  const NodeSchema *Schema = nullptr) {
    auto &Node = Nodes[ID];
    if (!Node.Done)
      Worklist.push_back({ID, false, Path, Schema});
  };
  for (auto ID : TopLevels)
    push(ID.getRef(), "/");
  while (!Worklist.empty()) {
    ObjectRef ID = Worklist.back().ID;
    auto Name = Worklist.back().Path;
    if (Worklist.back().Visited) {
      Nodes[ID].Done = true;
      POT.push_back({ID, Worklist.back().Schema});
      Worklist.pop_back();
      continue;
    }
    if (Nodes.lookup(ID).Done) {
      Worklist.pop_back();
      continue;
    }

    Worklist.back().Visited = true;

    // FIXME: Maybe this should just assert?
    ObjectProxy Object = ExitOnErr(CAS.getProxy(ID));

    TreeSchema TSchema(CAS);
    if (TSchema.isNode(Object)) {
      TreeProxy Tree = ExitOnErr(TSchema.load(Object));
      ExitOnErr(Tree.forEachEntry([&](const NamedTreeEntry &Entry) {
        SmallString<128> PathStorage = Name;
        sys::path::append(PathStorage, sys::path::Style::posix,
                          Entry.getName());
        push(Entry.getRef(), Saver.save(StringRef(PathStorage)));
        return Error::success();
      }));
      continue;
    }

    const NodeSchema *&Schema = Worklist.back().Schema;

    // Update the schema.
    if (!Schema) {
      for (auto &S : Collector.Schemas)
        if (S.first->isRootNode(Object))
          Schema = S.first;
    } else if (!Schema->isNode(Object)) {
      Schema = nullptr;
    }

    ExitOnErr(Object.forEachReference([&, Schema](ObjectRef Child) {
      push(Child, "", Schema);
      return Error::success();
    }));
  }

  Collector.visitPOT(ExitOnErr, TopLevels, POT);
  Collector.printToOuts(TopLevels, StatOS);
}

int main(int argc, char *argv[]) {
  ExitOnError ExitOnErr;
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  cl::ParseCommandLineOptions(argc, argv);
  PrinterOptions Options = {DwarfSectionsOnly, DwarfDump, HexDump,
                            HexDumpOneLine,    Verbose,   DIERefs};

  std::shared_ptr<ObjectStore> CAS =
      ExitOnErr(createCASFromIdentifier(CASPath));
  MCCASPrinter Printer(Options, *CAS, llvm::outs());

  StringMap<ObjectRef> Files;
  SmallVector<ObjectProxy> SummaryIDs;
  for (StringRef InputStr : InputStrings) {
    // Print object file name dumping cas contents, but not when dumping object
    // stats.
    if (ComputeStats.empty())
      outs() << "CASID File Name: " << InputStr << "\n";

    auto ID = getCASIDFromInput(*CAS, InputStr);

    if (AnalysisCASTree) {
      auto Proxy = ExitOnErr(CAS->getProxy(ID));
      SummaryIDs.emplace_back(Proxy);
      continue;
    }

    auto Ref = CAS->getReference(ID);
    if (!Ref) {
      llvm::errs() << "ID is invalid for this CAS\n";
      return 1;
    }

    if (!ComputeStats.empty()) {
      Files.try_emplace(InputStr, *Ref);
      continue;
    }

    // Do one pass over all CASObjectRefs to discover debug info section
    // contents
    auto Obj = Printer.discoverDwarfSections(*Ref);
    if (!Obj)
      ExitOnErr(Obj.takeError());

    ExitOnErr(Printer.printMCObject(*Ref, *Obj));
  }

  if (!ComputeStats.empty()) {
    if (!Files.empty()) {
      llvm::errs() << "Making summary object...\n";
      HierarchicalTreeBuilder Builder;
      for (auto &File : Files) {
        Builder.push(File.second, TreeEntry::Regular, File.first());
      }
      ObjectProxy SummaryRef = ExitOnErr(Builder.create(*CAS));
      SummaryIDs.emplace_back(SummaryRef);
      llvm::errs() << "summary tree: ";
      outs() << SummaryRef.getID() << "\n";
    }

    bool PrintToStdout = false;
    if (StringRef(ComputeStats) == "-")
      PrintToStdout = true;

    raw_ostream *StatOS = &outs();
    std::optional<raw_fd_ostream> StatsFile;

    if (!PrintToStdout) {
      std::error_code EC;
      StatsFile.emplace(ComputeStats, EC, sys::fs::OF_None);
      ExitOnErr(errorCodeToError(EC));
      StatOS = &*StatsFile;
    }

    computeStats(*CAS, SummaryIDs, *StatOS);
  }
  return 0;
}
