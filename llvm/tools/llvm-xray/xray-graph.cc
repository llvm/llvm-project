//===-- xray-graph.c - XRay Function Call Graph Renderer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Generate a DOT file to represent the function call graph encountered in
// the trace.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <cassert>
#include <system_error>
#include <utility>

#include "xray-extract.h"
#include "xray-graph.h"
#include "xray-registry.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/XRay/Trace.h"
#include "llvm/XRay/YAMLXRayRecord.h"

using namespace llvm;
using namespace xray;

// Setup llvm-xray graph subcommand and its options.
static cl::SubCommand Graph("graph", "Generate function-call graph");
static cl::opt<std::string> GraphInput(cl::Positional,
                                       cl::desc("<xray log file>"),
                                       cl::Required, cl::sub(Graph));

static cl::opt<std::string>
    GraphOutput("output", cl::value_desc("Output file"), cl::init("-"),
                cl::desc("output file; use '-' for stdout"), cl::sub(Graph));
static cl::alias GraphOutput2("o", cl::aliasopt(GraphOutput),
                              cl::desc("Alias for -output"), cl::sub(Graph));

static cl::opt<std::string> GraphInstrMap(
    "instr_map", cl::desc("binary with the instrumrntation map, or "
                          "a separate instrumentation map"),
    cl::value_desc("binary with xray_instr_map"), cl::sub(Graph), cl::init(""));
static cl::alias GraphInstrMap2("m", cl::aliasopt(GraphInstrMap),
                                cl::desc("alias for -instr_map"),
                                cl::sub(Graph));

static cl::opt<InstrumentationMapExtractor::InputFormats> InstrMapFormat(
    "instr-map-format", cl::desc("format of instrumentation map"),
    cl::values(clEnumValN(InstrumentationMapExtractor::InputFormats::ELF, "elf",
                          "instrumentation map in an ELF header"),
               clEnumValN(InstrumentationMapExtractor::InputFormats::YAML,
                          "yaml", "instrumentation map in YAML")),
    cl::sub(Graph), cl::init(InstrumentationMapExtractor::InputFormats::ELF));
static cl::alias InstrMapFormat2("t", cl::aliasopt(InstrMapFormat),
                                 cl::desc("Alias for -instr-map-format"),
                                 cl::sub(Graph));

static cl::opt<bool> GraphDeduceSiblingCalls(
    "deduce-sibling-calls",
    cl::desc("Deduce sibling calls when unrolling function call stacks"),
    cl::sub(Graph), cl::init(false));
static cl::alias
    GraphDeduceSiblingCalls2("d", cl::aliasopt(GraphDeduceSiblingCalls),
                             cl::desc("Alias for -deduce-sibling-calls"),
                             cl::sub(Graph));

static cl::opt<GraphRenderer::StatType>
    GraphEdgeLabel("edge-label",
                   cl::desc("Output graphs with edges labeled with this field"),
                   cl::value_desc("field"), cl::sub(Graph),
                   cl::init(GraphRenderer::StatType::COUNT),
                   cl::values(clEnumValN(GraphRenderer::StatType::COUNT,
                                         "count", "function call counts"),
                              clEnumValN(GraphRenderer::StatType::MIN, "min",
                                         "minimum function durations"),
                              clEnumValN(GraphRenderer::StatType::MED, "med",
                                         "median function durations"),
                              clEnumValN(GraphRenderer::StatType::PCT90, "90p",
                                         "90th percentile durations"),
                              clEnumValN(GraphRenderer::StatType::PCT99, "99p",
                                         "99th percentile durations"),
                              clEnumValN(GraphRenderer::StatType::MAX, "max",
                                         "maximum function durations"),
                              clEnumValN(GraphRenderer::StatType::SUM, "sum",
                                         "sum of call durations")));
static cl::alias GraphEdgeLabel2("e", cl::aliasopt(GraphEdgeLabel),
                                 cl::desc("Alias for -edge-label"),
                                 cl::sub(Graph));

namespace {
template <class T> T diff(T L, T R) { return std::max(L, R) - std::min(L, R); }

void updateStat(GraphRenderer::TimeStat &S, int64_t lat) {
  S.Count++;
  if (S.Min > lat || S.Min == 0)
    S.Min = lat;
  if (S.Max < lat)
    S.Max = lat;
  S.Sum += lat;
}
}

// Evaluates an XRay record and performs accounting on it, creating and
// decorating a function call graph as it does so. It does this by maintaining
// a call stack on a per-thread basis and adding edges and verticies to the
// graph as they are seen for the first time.
//
// There is an immaginary root for functions at the top of their stack with
// FuncId 0.
//
// FIXME: make more robust to errors and
// Decorate Graph More Heavily.
// FIXME: Refactor this and account subcommand to reduce code duplication.
bool GraphRenderer::accountRecord(const XRayRecord &Record) {
  if (CurrentMaxTSC == 0)
    CurrentMaxTSC = Record.TSC;

  if (Record.TSC < CurrentMaxTSC)
    return false;

  auto &ThreadStack = PerThreadFunctionStack[Record.TId];
  switch (Record.Type) {
  case RecordTypes::ENTER: {
    if (VertexAttrs.count(Record.FuncId) == 0)
      VertexAttrs[Record.FuncId].SymbolName =
          FuncIdHelper.SymbolOrNumber(Record.FuncId);
    ThreadStack.push_back({Record.FuncId, Record.TSC});
    break;
  }
  case RecordTypes::EXIT: {
    // FIXME: Refactor this and the account subcommand to reducr code
    // duplication
    if (ThreadStack.size() == 0 || ThreadStack.back().FuncId != Record.FuncId) {
      if (!DeduceSiblingCalls)
        return false;
      auto Parent = std::find_if(
          ThreadStack.rbegin(), ThreadStack.rend(),
          [&](const FunctionAttr &A) { return A.FuncId == Record.FuncId; });
      if (Parent == ThreadStack.rend())
        return false; // There is no matching Function for this exit.
      while (ThreadStack.back().FuncId != Record.FuncId) {
        uint64_t D = diff(ThreadStack.back().TSC, Record.TSC);
        int32_t TopFuncId = ThreadStack.back().FuncId;
        ThreadStack.pop_back();
        assert(ThreadStack.size() != 0);
        auto &EA = Graph[ThreadStack.back().FuncId][TopFuncId];
        EA.Timings.push_back(D);
        updateStat(EA.S, D);
        updateStat(VertexAttrs[TopFuncId].S, D);
      }
    }
    uint64_t D = diff(ThreadStack.back().TSC, Record.TSC);
    ThreadStack.pop_back();
    auto &V = Graph[ThreadStack.empty() ? 0 : ThreadStack.back().FuncId];
    auto &EA = V[Record.FuncId];
    EA.Timings.push_back(D);
    updateStat(EA.S, D);
    updateStat(VertexAttrs[Record.FuncId].S, D);
    break;
  }
  }

  return true;
}

template <typename U>
void GraphRenderer::getStats(U begin, U end, GraphRenderer::TimeStat &S) {
  assert(begin != end);
  std::ptrdiff_t MedianOff = S.Count / 2;
  std::nth_element(begin, begin + MedianOff, end);
  S.Median = *(begin + MedianOff);
  std::ptrdiff_t Pct90Off = (S.Count * 9) / 10;
  std::nth_element(begin, begin + Pct90Off, end);
  S.Pct90 = *(begin + Pct90Off);
  std::ptrdiff_t Pct99Off = (S.Count * 99) / 100;
  std::nth_element(begin, begin + Pct99Off, end);
  S.Pct99 = *(begin + Pct99Off);
}

void GraphRenderer::calculateEdgeStatistics() {
  for (auto &V : Graph) {
    for (auto &E : V.second) {
      auto &A = E.second;
      getStats(A.Timings.begin(), A.Timings.end(), A.S);
    }
  }
}

void GraphRenderer::calculateVertexStatistics() {
  DenseMap<int32_t, std::pair<uint64_t, SmallVector<EdgeAttribute *, 4>>>
      IncommingEdges;
  uint64_t MaxCount = 0;
  for (auto &V : Graph) {
    for (auto &E : V.second) {
      auto &IEV = IncommingEdges[E.first];
      IEV.second.push_back(&E.second);
      IEV.first += E.second.S.Count;
      if (IEV.first > MaxCount)
        MaxCount = IEV.first;
    }
  }
  std::vector<uint64_t> TempTimings;
  TempTimings.reserve(MaxCount);
  for (auto &V : IncommingEdges) {
    for (auto &P : V.second.second) {
      TempTimings.insert(TempTimings.end(), P->Timings.begin(),
                         P->Timings.end());
    }
    getStats(TempTimings.begin(), TempTimings.end(), VertexAttrs[V.first].S);
    TempTimings.clear();
  }
}

void GraphRenderer::normaliseStatistics(double CycleFrequency) {
  for (auto &V : Graph) {
    for (auto &E : V.second) {
      auto &S = E.second.S;
      S.Min /= CycleFrequency;
      S.Median /= CycleFrequency;
      S.Max /= CycleFrequency;
      S.Sum /= CycleFrequency;
      S.Pct90 /= CycleFrequency;
      S.Pct99 /= CycleFrequency;
    }
  }
  for (auto &V : VertexAttrs) {
    auto &S = V.second.S;
    S.Min /= CycleFrequency;
    S.Median /= CycleFrequency;
    S.Max /= CycleFrequency;
    S.Sum /= CycleFrequency;
    S.Pct90 /= CycleFrequency;
    S.Pct99 /= CycleFrequency;
  }
}

namespace {
void outputEdgeInfo(const GraphRenderer::TimeStat &S, GraphRenderer::StatType T,
                    raw_ostream &OS) {
  switch (T) {
  case GraphRenderer::StatType::COUNT:
    OS << S.Count;
    break;
  case GraphRenderer::StatType::MIN:
    OS << S.Min;
    break;
  case GraphRenderer::StatType::MED:
    OS << S.Median;
    break;
  case GraphRenderer::StatType::PCT90:
    OS << S.Pct90;
    break;
  case GraphRenderer::StatType::PCT99:
    OS << S.Pct99;
    break;
  case GraphRenderer::StatType::MAX:
    OS << S.Max;
    break;
  case GraphRenderer::StatType::SUM:
    OS << S.Sum;
    break;
  }
}
}

// Outputs a DOT format version of the Graph embedded in the GraphRenderer
// object on OS. It does this in the expected way by itterating
// through all edges then vertices and then outputting them and their
// annotations.
//
// FIXME: output more information, better presented.
void GraphRenderer::exportGraphAsDOT(raw_ostream &OS, const XRayFileHeader &H,
                                     StatType T) {
  calculateEdgeStatistics();
  calculateVertexStatistics();
  if (H.CycleFrequency)
    normaliseStatistics(H.CycleFrequency);

  OS << "digraph xray {\n";

  for (const auto &V : Graph)
    for (const auto &E : V.second) {
      OS << "F" << V.first << " -> "
         << "F" << E.first << " [label=\"";
      outputEdgeInfo(E.second.S, T, OS);
      OS << "\"];\n";
    }

  for (const auto &V : VertexAttrs)
    OS << "F" << V.first << " [label=\""
       << (V.second.SymbolName.size() > 40
               ? V.second.SymbolName.substr(0, 40) + "..."
               : V.second.SymbolName)
       << "\"];\n";

  OS << "}\n";
}

// Here we register and implement the llvm-xray graph subcommand.
// The bulk of this code reads in the options, opens the required files, uses
// those files to create a context for analysing the xray trace, then there is a
// short loop which actually analyses the trace, generates the graph and then
// outputs it as a DOT.
//
// FIXME: include additional filtering and annalysis passes to provide more
// specific useful information.
static CommandRegistration Unused(&Graph, []() -> Error {
  int Fd;
  auto EC = sys::fs::openFileForRead(GraphInput, Fd);
  if (EC)
    return make_error<StringError>(
        Twine("Cannot open file '") + GraphInput + "'", EC);

  Error Err = Error::success();
  xray::InstrumentationMapExtractor Extractor(GraphInstrMap, InstrMapFormat,
                                              Err);
  handleAllErrors(std::move(Err),
                  [&](const ErrorInfoBase &E) { E.log(errs()); });

  const auto &FunctionAddresses = Extractor.getFunctionAddresses();

  symbolize::LLVMSymbolizer::Options Opts(
      symbolize::FunctionNameKind::LinkageName, true, true, false, "");

  symbolize::LLVMSymbolizer Symbolizer(Opts);

  llvm::xray::FuncIdConversionHelper FuncIdHelper(GraphInstrMap, Symbolizer,
                                                  FunctionAddresses);

  xray::GraphRenderer GR(FuncIdHelper, GraphDeduceSiblingCalls);

  raw_fd_ostream OS(GraphOutput, EC, sys::fs::OpenFlags::F_Text);

  if (EC)
    return make_error<StringError>(
        Twine("Cannot open file '") + GraphOutput + "' for writing.", EC);

  auto TraceOrErr = loadTraceFile(GraphInput, true);

  if (!TraceOrErr) {
    return joinErrors(
        make_error<StringError>(Twine("Failed loading input file '") +
                                    GraphInput + "'",
                                make_error_code(llvm::errc::invalid_argument)),
        std::move(Err));
  }

  auto &Trace = *TraceOrErr;
  const auto &Header = Trace.getFileHeader();
  for (const auto &Record : Trace) {
    // Generate graph, FIXME: better error recovery.
    if (!GR.accountRecord(Record)) {
      return make_error<StringError>(
          Twine("Failed accounting function calls in file '") + GraphInput +
              "'.",
          make_error_code(llvm::errc::invalid_argument));
    }
  }

  GR.exportGraphAsDOT(OS, Header, GraphEdgeLabel);
  return Error::success();
});
