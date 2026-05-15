//===------------------- CLIHandler.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Client/CLI/CLIHandler.h"
#include "Client/HTTP/HTTPServer.h"
#include "Utils/JSON.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::advisor;

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

static std::string resolveRelativeToBin(StringRef Argv0, StringRef RelPath) {
  SmallString<256> Bin;
  if (sys::path::is_absolute(Argv0)) {
    Bin = Argv0;
  } else {
    sys::fs::current_path(Bin);
    sys::path::append(Bin, Argv0);
  }
  sys::path::remove_filename(Bin);
  sys::path::append(Bin, RelPath);
  return std::string(Bin);
}

static std::string defaultCapabilityDir(StringRef Argv0) {
  // Installed layout: <prefix>/bin/llvm-advisor →
  // <prefix>/share/llvm-advisor/config/capabilities
  std::string Installed =
      resolveRelativeToBin(Argv0, "../share/llvm-advisor/config/capabilities");
  if (sys::fs::is_directory(Installed))
    return Installed;
  // In-tree build layout: build/bin/llvm-advisor →
  // llvm/tools/llvm-advisor/config/capabilities
  std::string InTree = resolveRelativeToBin(
      Argv0, "../../llvm/tools/llvm-advisor/config/capabilities");
  if (sys::fs::is_directory(InTree))
    return InTree;
  return "llvm/tools/llvm-advisor/config/capabilities";
}

// Return the platform-specific default advisor store path.
// Resolution order:
//   1. $LLVM_ADVISOR_STORE (environment override)
//   2. XDG_DATA_HOME/llvm-advisor (Linux)
//   3. ~/Library/Application Support/llvm-advisor (macOS)
//   4. %APPDATA%/llvm-advisor (Windows)
//   5. ~/.llvm-advisor (fallback)
static std::string getDefaultStorePath() {
  if (const char *Env = std::getenv("LLVM_ADVISOR_STORE"))
    return Env;

#ifdef _WIN32
  if (const char *AppData = std::getenv("APPDATA"))
    return (Twine(AppData) + "/llvm-advisor").str();
#elif defined(__APPLE__)
  if (const char *Home = std::getenv("HOME"))
    return (Twine(Home) + "/Library/Application Support/llvm-advisor").str();
#else
  if (const char *XdgData = std::getenv("XDG_DATA_HOME")) {
    return (Twine(XdgData) + "/llvm-advisor").str();
  }
  if (const char *Home = std::getenv("HOME"))
    return (Twine(Home) + "/.local/share/llvm-advisor").str();
#endif
  // Final fallback: current directory
  return ".llvm-advisor";
}

// Walk upward from Dir looking for compile_commands.json; return the directory
// that contains it, or empty string if not found.
// Also checks common build subdirectories (build/, out/, etc.).
static std::string findBuildDir(StringRef StartDir) {
  // First: check common build subdirectories from StartDir
  static const char *CommonBuildDirs[] = {
      "build",
      "out",
      "cmake-build-debug",
      "cmake-build-release",
      "cmake-build-relwithdebinfo",
      "cmake-build-minsizerel",
      "Debug",
      "Release",
      "RelWithDebInfo",
      "MinSizeRel",
      ".build",
      "_build",
      "builddir",
  };
  for (const char *Sub : CommonBuildDirs) {
    SmallString<256> Candidate(StartDir);
    sys::path::append(Candidate, Sub, "compile_commands.json");
    if (sys::fs::exists(Candidate)) {
      sys::path::remove_filename(Candidate);
      return std::string(Candidate);
    }
  }

  // Second: walk upward looking for compile_commands.json in each dir
  SmallString<256> Dir(StartDir);
  for (int Depth = 0; Depth < 16; ++Depth) {
    SmallString<256> Candidate(Dir);
    sys::path::append(Candidate, "compile_commands.json");
    if (sys::fs::exists(Candidate))
      return std::string(Dir);
    StringRef Parent = sys::path::parent_path(Dir);
    if (Parent == Dir)
      break;
    Dir = Parent;
  }
  return {};
}

// Return the 8-character short prefix of a snapshot/unit ID.
static StringRef shortID(StringRef ID) {
  return ID.size() > 8 ? ID.substr(0, 8) : ID;
}

// Resolve "latest" to the most-recently-created snapshot ID, or return the
// string unchanged if it is already a concrete ID or empty.
static std::string resolveSnapshot(const CoreClient &Client, StringRef ID) {
  if (ID != "latest")
    return std::string(ID);
  SmallVector<SnapshotRecord, 16> Snapshots = Client.listSnapshots();
  if (Snapshots.empty())
    return std::string(ID);
  const SnapshotRecord *Best = &Snapshots[0];
  for (const SnapshotRecord &S : Snapshots)
    if (S.CreatedUnix > Best->CreatedUnix)
      Best = &S;
  return Best->ID;
}

// ---------------------------------------------------------------------------
// Command-line options
// ---------------------------------------------------------------------------

namespace {
cl::OptionCategory AdvisorCategory("llvm-advisor options");
cl::opt<std::string>
    StoreRoot("store",
              cl::desc("Advisor store directory (default: platform-specific)"),
              cl::init(""), cl::cat(AdvisorCategory));
cl::opt<std::string>
    CapabilityDir("capability-dir",
                  cl::desc("Directory containing capability JSON specs"),
                  cl::init(""), cl::cat(AdvisorCategory));
cl::list<std::string>
    PluginPaths("plugin",
                cl::desc("Path to an external advisor plugin "
                         "(.so/.dylib/.dll); may be repeated"),
                cl::ZeroOrMore, cl::cat(AdvisorCategory));

cl::SubCommand HealthCmd("health", "Print service health");
cl::SubCommand CaptureCmd("capture",
                          "Snapshot a build (auto-detects build root)");
cl::SubCommand ListCmd("list", "List snapshots or units in a snapshot");
cl::SubCommand QueryCmd("query", "Run capabilities for a unit or snapshot");
cl::SubCommand InspectCmd("inspect", "Inspect a capability result");
cl::SubCommand CompareCmd("compare", "Compare two snapshots");
cl::SubCommand CapabilitiesCmd("capabilities", "List declared capabilities");
cl::SubCommand InspectStorageCmd("inspect-storage", "Inspect storage state");
cl::SubCommand MaintenanceCmd("maintenance-compact", "Compact CAS storage");
cl::SubCommand ServeCmd("serve", "Run the embedded HTTP server");
cl::SubCommand RuntimeIngestCmd("runtime-ingest",
                                "Ingest runtime execution data");
cl::SubCommand RuntimeCorrelateCmd("runtime-correlate",
                                   "Correlate runtime data to captured units");

cl::opt<std::string> CaptureSourceRoot("source-root",
                                       cl::desc("Source root (default: auto)"),
                                       cl::init(""), cl::sub(CaptureCmd));
cl::opt<std::string> CaptureBuildRoot(
    "build-root",
    cl::desc("Build root containing compile_commands.json (default: auto)"),
    cl::init(""), cl::sub(CaptureCmd));
cl::opt<std::string> CaptureProfile("profile",
                                    cl::desc("Capture profile JSON path"),
                                    cl::init(""), cl::sub(CaptureCmd));
cl::list<std::string> CaptureCapabilities(
    "capability",
    cl::desc("Capability ID; may be repeated; overrides --profile"),
    cl::ZeroOrMore, cl::sub(CaptureCmd));

cl::opt<std::string> ListSnapshot(
    "snapshot",
    cl::desc("Snapshot ID (or 'latest'); if omitted, lists all snapshots"),
    cl::sub(ListCmd));

cl::opt<std::string> QueryUnit("unit", cl::desc("Unit ID"), cl::sub(QueryCmd));
cl::opt<std::string> QuerySnapshot("snapshot",
                                   cl::desc("Snapshot ID (or 'latest')"),
                                   cl::sub(QueryCmd));
cl::list<std::string> QueryCapabilities("capability", cl::desc("Capability ID"),
                                        cl::OneOrMore, cl::sub(QueryCmd));

cl::opt<std::string> InspectSnapshot("snapshot",
                                     cl::desc("Snapshot ID (or 'latest')"),
                                     cl::sub(InspectCmd));
cl::opt<std::string> InspectMode(cl::Positional,
                                 cl::desc("<mode|capability-inspect>"),
                                 cl::init("capability"), cl::sub(InspectCmd));
cl::opt<std::string> InspectUnit("unit",
                                 cl::desc("Unit ID or source path suffix"),
                                 cl::sub(InspectCmd));
cl::opt<std::string> InspectCapability("capability", cl::desc("Capability ID"),
                                       cl::sub(InspectCmd));
cl::opt<std::string>
    InspectBaseline("baseline",
                    cl::desc("Baseline snapshot ID for compare inspection"),
                    cl::init(""), cl::sub(InspectCmd));
cl::opt<std::string> InspectFunction("function",
                                     cl::desc("Function name filter"),
                                     cl::init(""), cl::sub(InspectCmd));
cl::opt<std::string> InspectPass("pass", cl::desc("Pass name filter"),
                                 cl::init(""), cl::sub(InspectCmd));
cl::opt<std::string> InspectSeverity("severity",
                                     cl::desc("Severity/type filter"),
                                     cl::init(""), cl::sub(InspectCmd));
cl::opt<std::string> InspectFile("file", cl::desc("File path filter"),
                                 cl::init(""), cl::sub(InspectCmd));
cl::opt<int> InspectLine("line", cl::desc("Line filter"), cl::init(-1),
                         cl::sub(InspectCmd));
cl::opt<int> InspectIndex("index", cl::desc("0-based item index"), cl::init(-1),
                          cl::sub(InspectCmd));
cl::opt<std::string> InspectOutputFormat("output-format",
                                         cl::desc("json or text"),
                                         cl::init("text"), cl::sub(InspectCmd));

cl::opt<std::string> CompareBefore("before",
                                   cl::desc("Before snapshot ID (or 'latest')"),
                                   cl::sub(CompareCmd));
cl::opt<std::string> CompareAfter("after",
                                  cl::desc("After snapshot ID (or 'latest')"),
                                  cl::sub(CompareCmd));
cl::opt<unsigned> ServePort("port", cl::desc("HTTP listen port"),
                            cl::init(8080), cl::sub(ServeCmd));
cl::opt<std::string> RuntimeSnapshot("snapshot",
                                     cl::desc("Snapshot ID (or 'latest')"),
                                     cl::sub(RuntimeIngestCmd),
                                     cl::sub(RuntimeCorrelateCmd));
cl::opt<std::string> RuntimeKind(
    "kind",
    cl::desc("Runtime kind: pgo-instr, pgo-sample, memprof, coverage, xray, "
             "sanitizer, sancov, offload"),
    cl::sub(RuntimeIngestCmd));
cl::opt<std::string> RuntimeData("data", cl::desc("Runtime data path"),
                                 cl::sub(RuntimeIngestCmd));

// Insight commands
cl::SubCommand InsightListCmd("insight-list", "List available insights");
cl::SubCommand InsightCmd("insight", "Run insight analysis");
cl::opt<std::string> InsightName("name", cl::desc("Insight name"),
                                 cl::sub(InsightCmd));
cl::opt<std::string> InsightSnapshot("snapshot",
                                     cl::desc("Snapshot ID (or 'latest')"),
                                     cl::sub(InsightCmd),
                                     cl::sub(InsightListCmd));
cl::opt<std::string> InsightUnit("unit", cl::desc("Unit ID"),
                                 cl::sub(InsightCmd), cl::sub(InsightListCmd));
cl::opt<std::string>
    InsightBaseline("baseline", cl::desc("Baseline snapshot ID (or 'latest')"),
                    cl::sub(InsightCmd));
} // namespace

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

static int printError(Error Err) {
  errs() << "error: " << toString(std::move(Err)) << '\n';
  return 1;
}

static Expected<SmallVector<std::string, 8>>
loadProfileCapabilities(StringRef Path) {
  Expected<json::Value> Value = parseJSONFile(Path);
  if (!Value)
    return Value.takeError();
  const json::Object *Obj = Value->getAsObject();
  if (!Obj)
    return createStringError(inconvertibleErrorCode(),
                             "profile is not a JSON object");
  return getStringArray(*Obj, "capabilities");
}

static std::string inspectModeToCapability(StringRef Mode, StringRef Explicit) {
  if (!Explicit.empty())
    return Explicit.str();
  if (Mode.empty() || Mode == "capability")
    return Explicit.str();
  if (Mode == "ir")
    return "llvm.ir.view";
  if (Mode == "ir-diff")
    return "llvm.ir.diff";
  if (Mode == "cfg")
    return "llvm.cfg";
  if (Mode == "dom")
    return "llvm.dom_tree";
  if (Mode == "callgraph")
    return "llvm.call_graph";
  if (Mode == "dag")
    return "llvm.selection_dag";
  if (Mode == "mir")
    return "llvm.machine_ir";
  if (Mode == "asm")
    return "llvm.asm.view";
  if (Mode == "remarks")
    return "llvm.remarks.detail";
  if (Mode == "debug")
    return "llvm.debug.detail";
  if (Mode == "passes")
    return "llvm.ir.passes.list";
  if (Mode == "loop")
    return "llvm.loop_info";
  if (Mode == "mca")
    return "llvm.mca.report";
  if (Mode == "exegesis")
    return "llvm.exegesis";
  if (Mode == "offload")
    return "offload.binary.inspect";
  return Explicit.str();
}

static void printInspectionValueText(raw_ostream &OS,
                                     const json::Object &Value) {
  if (std::optional<StringRef> IR = Value.getString("ir")) {
    OS << *IR;
    if (!IR->ends_with('\n'))
      OS << '\n';
    return;
  }
  if (const json::Array *Remarks = Value.getArray("remarks")) {
    for (const json::Value &Item : *Remarks) {
      const json::Object *Remark = Item.getAsObject();
      if (!Remark)
        continue;
      const json::Object *Loc = Remark->getObject("location");
      OS << Remark->getString("type").value_or("remark") << " ";
      OS << Remark->getString("pass").value_or("") << " ";
      OS << Remark->getString("function").value_or("") << " ";
      if (Loc)
        OS << Loc->getString("file").value_or("") << ":"
           << Loc->getInteger("line").value_or(0) << " ";
      OS << "- " << Remark->getString("message").value_or("") << '\n';
    }
    return;
  }
  if (const json::Array *Diagnostics = Value.getArray("diagnostics")) {
    for (const json::Value &Item : *Diagnostics) {
      const json::Object *Diag = Item.getAsObject();
      if (!Diag)
        continue;
      OS << Diag->getString("severity")
                .value_or(Diag->getString("level").value_or("info"))
         << " ";
      OS << Diag->getString("file").value_or("") << ":"
         << Diag->getInteger("line").value_or(0) << ":"
         << Diag->getInteger("column").value_or(0) << " ";
      OS << "- " << Diag->getString("message").value_or("") << '\n';
    }
    return;
  }
  json::Object Copy = Value;
  OS << json::Value(std::move(Copy)) << '\n';
}

static void printInspectionText(raw_ostream &OS, const json::Object &Result) {
  if (const json::Object *Value = Result.getObject("value")) {
    printInspectionValueText(OS, *Value);
    return;
  }
  if (const json::Object *Baseline = Result.getObject("baseline");
      Baseline && Result.getObject("candidate")) {
    OS << "Baseline\n";
    if (const json::Object *BaselineValue = Baseline->getObject("value"))
      printInspectionValueText(OS, *BaselineValue);
    OS << "\nCandidate\n";
    if (const json::Object *CandidateValue =
            Result.getObject("candidate")->getObject("value"))
      printInspectionValueText(OS, *CandidateValue);
    if (const json::Object *Diff = Result.getObject("diff")) {
      OS << "\nDiff\n";
      json::Object Copy = *Diff;
      OS << json::Value(std::move(Copy)) << '\n';
    }
    return;
  }
  json::Object Copy = Result;
  OS << json::Value(std::move(Copy)) << '\n';
}

// Print a human-readable snapshot table.
static void printSnapshotTable(ArrayRef<SnapshotRecord> Snapshots) {
  if (Snapshots.empty()) {
    outs() << "No snapshots found. Run 'llvm-advisor capture' to create one.\n";
    return;
  }
  outs() << formatv("{0,-10} {1,-20} {2}\n", "ID", "Created", "Source Root");
  outs() << std::string(60, '-') << '\n';
  for (const SnapshotRecord &S : Snapshots) {
    // Format unix timestamp as YYYY-MM-DD HH:MM
    std::string Ts;
    if (S.CreatedUnix) {
      time_t T = static_cast<time_t>(S.CreatedUnix);
      struct tm TM;
      gmtime_r(&T, &TM);
      char Buf[20];
      strftime(Buf, sizeof(Buf), "%Y-%m-%d %H:%M", &TM);
      Ts = Buf;
    } else {
      Ts = "unknown";
    }
    outs() << formatv("{0,-10} {1,-20} {2}\n", shortID(S.ID), Ts, S.SourceRoot);
  }
}

// Print a human-readable unit table.
static void printUnitTable(ArrayRef<UnitRecord> Units) {
  if (Units.empty()) {
    outs() << "No units in this snapshot.\n";
    return;
  }
  outs() << formatv("{0,-10} {1,-12} {2}\n", "ID", "Lang", "Source");
  outs() << std::string(60, '-') << '\n';
  for (const UnitRecord &U : Units)
    outs() << formatv("{0,-10} {1,-12} {2}\n", shortID(U.ID), U.Language,
                      U.SourcePath);
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

int CLIHandler::run(int argc, char **argv) {
  cl::HideUnrelatedOptions(AdvisorCategory);
  cl::ParseCommandLineOptions(argc, argv, "LLVM Advisor\n");

  StringRef Argv0(argv[0]);
  std::string ResolvedCapabilityDir = CapabilityDir.empty()
                                          ? defaultCapabilityDir(Argv0)
                                          : CapabilityDir.getValue();
  std::string ResolvedStoreRoot =
      StoreRoot.empty() ? getDefaultStorePath() : StoreRoot.getValue();
  Expected<std::unique_ptr<CoreClient>> Client =
      CoreClient::create(ResolvedStoreRoot, ResolvedCapabilityDir);
  if (!Client)
    return printError(Client.takeError());

  for (const std::string &P : PluginPaths) {
    if (Error Err = (*Client)->loadPlugin(P))
      return printError(std::move(Err));
  }

  if (HealthCmd) {
    outs() << toJSON((*Client)->health()) << '\n';
    return 0;
  }

  if (CaptureCmd) {
    // Auto-detect source root and build root when not provided.
    SmallString<256> CWD;
    sys::fs::current_path(CWD);

    std::string SrcRoot = CaptureSourceRoot.empty()
                              ? std::string(CWD)
                              : CaptureSourceRoot.getValue();
    std::string BldRoot;
    if (!CaptureBuildRoot.empty()) {
      BldRoot = CaptureBuildRoot;
    } else {
      BldRoot = findBuildDir(CWD);
      if (BldRoot.empty()) {
        errs() << "error: could not find compile_commands.json\n\n"
               << "  llvm-advisor needs a compile_commands.json to analyze "
                  "your build.\n"
               << "  Common ways to generate it:\n\n"
               << "    cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\n"
               << "    ninja -C build\n"
               << "    ln -s build/compile_commands.json .\n\n"
               << "  Or pass --build-root <dir> if it's in a non-standard "
                  "location.\n";
        return 1;
      }
    }

    SmallVector<std::string, 8> Capabilities;
    for (const std::string &C : CaptureCapabilities)
      Capabilities.push_back(C);
    if (Capabilities.empty() && !CaptureProfile.empty()) {
      Expected<SmallVector<std::string, 8>> ProfileCaps =
          loadProfileCapabilities(CaptureProfile);
      if (!ProfileCaps)
        return printError(ProfileCaps.takeError());
      Capabilities = std::move(*ProfileCaps);
    }

    Expected<SnapshotRecord> Snapshot =
        (*Client)->createSnapshot(SrcRoot, BldRoot, Capabilities);
    if (!Snapshot)
      return printError(Snapshot.takeError());

    SmallVector<UnitRecord, 64> Units = (*Client)->listUnits(Snapshot->ID);
    outs() << formatv("Snapshot {0} created — {1} unit(s)\n",
                      shortID(Snapshot->ID), Units.size());
    outs() << formatv("  Source: {0}\n  Build:  {1}\n  Full ID: {2}\n",
                      Snapshot->SourceRoot, Snapshot->BuildRoot, Snapshot->ID);
    return 0;
  }

  if (ListCmd) {
    if (!ListSnapshot.empty()) {
      std::string SnapshotID = resolveSnapshot(**Client, ListSnapshot);
      printUnitTable((*Client)->listUnits(SnapshotID));
      return 0;
    }
    printSnapshotTable((*Client)->listSnapshots());
    return 0;
  }

  if (QueryCmd) {
    std::string SnapshotID = resolveSnapshot(**Client, QuerySnapshot);
    Expected<json::Array> Results =
        QueryUnit.empty()
            ? (*Client)->querySnapshot(SnapshotID, QueryCapabilities)
            : (*Client)->queryUnit(QueryUnit, QueryCapabilities);
    if (!Results)
      return printError(Results.takeError());
    outs() << json::Value(std::move(*Results)) << '\n';
    return 0;
  }

  if (InspectCmd) {
    StringRef Mode = InspectMode;
    if (Mode == "storage") {
      outs() << (*Client)->inspectStorage() << '\n';
      return 0;
    }
    if (Mode == "snapshot") {
      if (InspectSnapshot.empty()) {
        errs() << "error: inspect snapshot requires --snapshot\n";
        return 1;
      }
      std::string SnapshotID = resolveSnapshot(**Client, InspectSnapshot);
      Expected<SnapshotRecord> Snapshot =
          (*Client)->storage().metadata().getSnapshot(SnapshotID);
      if (!Snapshot)
        return printError(Snapshot.takeError());
      outs() << toJSON(*Snapshot) << '\n';
      return 0;
    }
    if (Mode == "unit") {
      if (InspectSnapshot.empty() || InspectUnit.empty()) {
        errs() << "error: inspect unit requires --snapshot and --unit\n";
        return 1;
      }
      std::string SnapshotID = resolveSnapshot(**Client, InspectSnapshot);
      Expected<std::string> UnitID =
          (*Client)->resolveUnitID(SnapshotID, InspectUnit);
      if (!UnitID)
        return printError(UnitID.takeError());
      Expected<UnitRecord> Unit =
          (*Client)->storage().metadata().getUnit(*UnitID);
      if (!Unit)
        return printError(Unit.takeError());
      outs() << toJSON(*Unit) << '\n';
      return 0;
    }
    if (Mode == "jobs") {
      json::Array Jobs;
      for (const JobRecord &Job : (*Client)->listJobs())
        Jobs.push_back(toJSON(Job));
      outs() << json::Value(std::move(Jobs)) << '\n';
      return 0;
    }
    if (Mode == "capability" && InspectCapability.empty()) {
      errs() << "error: inspect capability requires --capability\n";
      return 1;
    }
    if (Mode == "capability" && InspectSnapshot.empty() &&
        InspectUnit.empty()) {
      for (const CapabilitySpec &Spec : (*Client)->listCapabilities()) {
        if (Spec.ID == InspectCapability) {
          outs() << toJSON(Spec) << '\n';
          return 0;
        }
      }
      errs() << "error: unknown capability: " << InspectCapability << '\n';
      return 1;
    }
    if (Mode == "list") {
      json::Array Items;
      const auto AddMode = [&](StringRef Name, StringRef Capability) {
        for (const CapabilitySpec &Spec : (*Client)->listCapabilities()) {
          if (Spec.ID != Capability)
            continue;
          Items.push_back(json::Object{{"mode", Name.str()},
                                       {"capability", Capability.str()},
                                       {"readiness", Spec.Readiness}});
          return;
        }
      };
      AddMode("ir", "llvm.ir.view");
      AddMode("ir-diff", "llvm.ir.diff");
      AddMode("cfg", "llvm.cfg");
      AddMode("dom", "llvm.dom_tree");
      AddMode("callgraph", "llvm.call_graph");
      AddMode("loop", "llvm.loop_info");
      AddMode("dag", "llvm.selection_dag");
      AddMode("mir", "llvm.machine_ir");
      AddMode("asm", "llvm.asm.view");
      AddMode("remarks", "llvm.remarks.detail");
      AddMode("debug", "llvm.debug.detail");
      AddMode("passes", "llvm.ir.passes.list");
      AddMode("mca", "llvm.mca.report");
      AddMode("exegesis", "llvm.exegesis");
      AddMode("offload", "offload.binary.inspect");
      Items.push_back(json::Object{{"mode", "signals"},
                                   {"capability", "multi-capability"}});
      outs() << json::Value(std::move(Items)) << '\n';
      return 0;
    }

    if (Mode == "signals") {
      if (InspectSnapshot.empty() || InspectUnit.empty()) {
        errs() << "error: inspect signals requires --snapshot and --unit\n";
        return 1;
      }
      InspectionFilter Filter;
      Filter.Function = InspectFunction;
      Filter.Pass = InspectPass;
      Filter.Severity = InspectSeverity;
      Filter.File = InspectFile;
      Filter.Line = InspectLine;
      Filter.Index = InspectIndex;
      Expected<json::Object> Result = (*Client)->inspectSignals(
          resolveSnapshot(**Client, InspectSnapshot), InspectUnit, Filter);
      if (!Result)
        return printError(Result.takeError());
      outs() << json::Value(std::move(*Result)) << '\n';
      return 0;
    }

    std::string Capability = inspectModeToCapability(Mode, InspectCapability);
    if (Capability.empty() || InspectSnapshot.empty() || InspectUnit.empty()) {
      errs() << "error: inspect requires --snapshot, --unit, and an inspect "
                "mode or --capability\n";
      return 1;
    }
    InspectionFilter Filter;
    Filter.Function = InspectFunction;
    Filter.Pass = InspectPass;
    Filter.Severity = InspectSeverity;
    Filter.File = InspectFile;
    Filter.Line = InspectLine;
    Filter.Index = InspectIndex;
    std::string SnapshotID = resolveSnapshot(**Client, InspectSnapshot);
    Expected<json::Object> Result =
        InspectBaseline.empty()
            ? (*Client)->inspect(SnapshotID, InspectUnit, Capability, Filter)
            : (*Client)->inspectCompare(
                  resolveSnapshot(**Client, InspectBaseline), SnapshotID,
                  InspectUnit, Capability, Filter);
    if (!Result)
      return printError(Result.takeError());
    if (InspectOutputFormat == "json")
      outs() << json::Value(std::move(*Result)) << '\n';
    else
      printInspectionText(outs(), *Result);
    return 0;
  }

  if (CapabilitiesCmd) {
    SmallVector<CapabilitySpec, 32> Specs = (*Client)->listCapabilities();
    if (Specs.empty()) {
      outs() << "No capabilities found in: " << ResolvedCapabilityDir << '\n';
      return 0;
    }
    outs() << formatv("{0,-40} {1,-30} {2}\n", "Capability ID", "Name",
                      "Readiness");
    outs() << std::string(75, '-') << '\n';
    for (const CapabilitySpec &Spec : Specs)
      outs() << formatv("{0,-40} {1,-30} {2}\n", Spec.ID, Spec.Name,
                        Spec.Readiness);
    return 0;
  }

  if (CompareCmd) {
    std::string Before = resolveSnapshot(**Client, CompareBefore);
    std::string After = resolveSnapshot(**Client, CompareAfter);
    outs() << (*Client)->compare(Before, After) << '\n';
    return 0;
  }

  if (InspectStorageCmd) {
    outs() << (*Client)->inspectStorage() << '\n';
    return 0;
  }

  if (MaintenanceCmd) {
    if (Error Err = (*Client)->compactStorage())
      return printError(std::move(Err));
    outs() << "Storage compacted successfully.\n";
    return 0;
  }

  if (RuntimeIngestCmd) {
    if (RuntimeSnapshot.empty() || RuntimeKind.empty() || RuntimeData.empty()) {
      errs()
          << "error: runtime-ingest requires --snapshot, --kind and --data\n";
      return 1;
    }
    std::string SnapshotID = resolveSnapshot(**Client, RuntimeSnapshot);
    Expected<json::Value> Result =
        (*Client)->ingestRuntime(SnapshotID, RuntimeKind, RuntimeData);
    if (!Result)
      return printError(Result.takeError());
    outs() << *Result << '\n';
    return 0;
  }

  if (RuntimeCorrelateCmd) {
    if (RuntimeSnapshot.empty()) {
      errs() << "error: runtime-correlate requires --snapshot\n";
      return 1;
    }
    std::string SnapshotID = resolveSnapshot(**Client, RuntimeSnapshot);
    Expected<json::Value> Result = (*Client)->correlateRuntime(SnapshotID);
    if (!Result)
      return printError(Result.takeError());
    outs() << *Result << '\n';
    return 0;
  }

  if (InsightListCmd) {
    if (InsightSnapshot.empty()) {
      errs() << "error: insight-list requires --snapshot\n";
      return 1;
    }
    std::string SnapshotID = resolveSnapshot(**Client, InsightSnapshot);
    Expected<json::Array> Result =
        (*Client)->listInsights(SnapshotID, InsightUnit);
    if (!Result)
      return printError(Result.takeError());
    outs() << json::Value(std::move(*Result)) << '\n';
    return 0;
  }

  if (InsightCmd) {
    if (InsightName.empty() || InsightSnapshot.empty()) {
      errs() << "error: insight requires --name and --snapshot\n";
      return 1;
    }
    std::string SnapshotID = resolveSnapshot(**Client, InsightSnapshot);
    std::string Baseline = resolveSnapshot(**Client, InsightBaseline);
    Expected<json::Object> Result =
        (*Client)->runInsight(InsightName, SnapshotID, InsightUnit, Baseline);
    if (!Result)
      return printError(Result.takeError());
    outs() << json::Value(std::move(*Result)) << '\n';
    return 0;
  }

  if (ServeCmd) {
    outs() << formatv("Starting HTTP server on port {0}...\n",
                      ServePort.getValue());
    HTTPServer Server(**Client, ServePort);
    if (Error Err = Server.run())
      return printError(std::move(Err));
    return 0;
  }

  cl::PrintHelpMessage();
  return 0;
}
