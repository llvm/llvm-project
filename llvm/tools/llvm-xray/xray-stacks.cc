//===- xray-stacks.cc - XRay Function Call Stack Accounting ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements stack-based accounting. It takes XRay traces, and
// collates statistics across these traces to show a breakdown of time spent
// at various points of the stack to provide insight into which functions
// spend the most time in terms of a call stack. We provide a few
// sorting/filtering options for zero'ing in on the useful stacks.
//
//===----------------------------------------------------------------------===//

#include <forward_list>
#include <numeric>

#include "func-id-helper.h"
#include "xray-registry.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/XRay/Graph.h"
#include "llvm/XRay/InstrumentationMap.h"
#include "llvm/XRay/Trace.h"

using namespace llvm;
using namespace llvm::xray;

static cl::SubCommand Stack("stack", "Call stack accounting");
static cl::list<std::string> StackInputs(cl::Positional,
                                         cl::desc("<xray trace>"), cl::Required,
                                         cl::sub(Stack), cl::OneOrMore);

static cl::opt<bool>
    StackKeepGoing("keep-going", cl::desc("Keep going on errors encountered"),
                   cl::sub(Stack), cl::init(false));
static cl::alias StackKeepGoing2("k", cl::aliasopt(StackKeepGoing),
                                 cl::desc("Alias for -keep-going"),
                                 cl::sub(Stack));

// TODO: Does there need to be an option to deduce tail or sibling calls?

static cl::opt<std::string> StacksInstrMap(
    "instr_map",
    cl::desc("instrumentation map used to identify function ids. "
             "Currently supports elf file instrumentation maps."),
    cl::sub(Stack), cl::init(""));
static cl::alias StacksInstrMap2("m", cl::aliasopt(StacksInstrMap),
                                 cl::desc("Alias for -instr_map"),
                                 cl::sub(Stack));

static cl::opt<bool>
    SeparateThreadStacks("per-thread-stacks",
                         cl::desc("Report top stacks within each thread id"),
                         cl::sub(Stack), cl::init(false));

static cl::opt<bool>
    AggregateThreads("aggregate-threads",
                     cl::desc("Aggregate stack times across threads"),
                     cl::sub(Stack), cl::init(false));

/// A helper struct to work with formatv and XRayRecords. Makes it easier to use
/// instrumentation map names or addresses in formatted output.
struct format_xray_record : public FormatAdapter<XRayRecord> {
  explicit format_xray_record(XRayRecord record,
                              const FuncIdConversionHelper &conv)
      : FormatAdapter<XRayRecord>(std::move(record)), Converter(&conv) {}
  void format(raw_ostream &Stream, StringRef Style) override {
    Stream << formatv(
        "{FuncId: \"{0}\", ThreadId: \"{1}\", RecordType: \"{2}\"}",
        Converter->SymbolOrNumber(Item.FuncId), Item.TId,
        DecodeRecordType(Item.RecordType));
  }

private:
  Twine DecodeRecordType(uint16_t recordType) {
    switch (recordType) {
    case 0:
      return Twine("Fn Entry");
    case 1:
      return Twine("Fn Exit");
    default:
      // TODO: Add Tail exit when it is added to llvm/XRay/XRayRecord.h
      return Twine("Unknown");
    }
  }

  const FuncIdConversionHelper *Converter;
};

/// The stack command will take a set of XRay traces as arguments, and collects
/// information about the stacks of instrumented functions that appear in the
/// traces. We track the following pieces of information:
///
///   - Total time: amount of time/cycles accounted for in the traces.
///   - Stack count: number of times a specific stack appears in the
///     traces. Only instrumented functions show up in stacks.
///   - Cumulative stack time: amount of time spent in a stack accumulated
///     across the invocations in the traces.
///   - Cumulative local time: amount of time spent in each instrumented
///     function showing up in a specific stack, accumulated across the traces.
///
/// Example output for the kind of data we'd like to provide looks like the
/// following:
///
///   Total time: 3.33234 s
///   Stack ID: ...
///   Stack Count: 2093
///   #     Function                  Local Time     (%)      Stack Time     (%)
///   0     main                         2.34 ms   0.07%      3.33234  s    100%
///   1     foo()                     3.30000  s  99.02%         3.33  s  99.92%
///   2     bar()                          30 ms   0.90%           30 ms   0.90%
///
/// We can also show distributions of the function call durations with
/// statistics at each level of the stack. This works by doing the following
/// algorithm:
///
///   1. When unwinding, record the duration of each unwound function associated
///   with the path up to which the unwinding stops. For example:
///
///        Step                         Duration (? means has start time)
///
///        push a <start time>           a = ?
///        push b <start time>           a = ?, a->b = ?
///        push c <start time>           a = ?, a->b = ?, a->b->c = ?
///        pop  c <end time>             a = ?, a->b = ?, emit duration(a->b->c)
///        pop  b <end time>             a = ?, emit duration(a->b)
///        push c <start time>           a = ?, a->c = ?
///        pop  c <end time>             a = ?, emit duration(a->c)
///        pop  a <end time>             emit duration(a)
///
///   2. We then account for the various stacks we've collected, and for each of
///      them will have measurements that look like the following (continuing
///      with the above simple example):
///
///        c : [<id("a->b->c"), [durations]>, <id("a->c"), [durations]>]
///        b : [<id("a->b"), [durations]>]
///        a : [<id("a"), [durations]>]
///
///      This allows us to compute, for each stack id, and each function that
///      shows up in the stack,  some important statistics like:
///
///        - median
///        - 99th percentile
///        - mean + stddev
///        - count
///
///   3. For cases where we don't have durations for some of the higher levels
///   of the stack (perhaps instrumentation wasn't activated when the stack was
///   entered), we can mark them appropriately.
///
///  Computing this data also allows us to implement lookup by call stack nodes,
///  so that we can find functions that show up in multiple stack traces and
///  show the statistical properties of that function in various contexts. We
///  can compute information similar to the following:
///
///    Function: 'c'
///    Stacks: 2 / 2
///    Stack ID: ...
///    Stack Count: ...
///    #     Function  ...
///    0     a         ...
///    1     b         ...
///    2     c         ...
///
///    Stack ID: ...
///    Stack Count: ...
///    #     Function  ...
///    0     a         ...
///    1     c         ...
///    ----------------...
///
///    Function: 'b'
///    Stacks:  1 / 2
///    Stack ID: ...
///    Stack Count: ...
///    #     Function  ...
///    0     a         ...
///    1     b         ...
///    2     c         ...
///
///
/// To do this we require a Trie data structure that will allow us to represent
/// all the call stacks of instrumented functions in an easily traversible
/// manner when we do the aggregations and lookups. For instrumented call
/// sequences like the following:
///
///   a()
///    b()
///     c()
///     d()
///    c()
///
/// We will have a representation like so:
///
///   a -> b -> c
///   |    |
///   |    +--> d
///   |
///   +--> c
///
/// We maintain a sequence of durations on the leaves and in the internal nodes
/// as we go through and process every record from the XRay trace. We also
/// maintain an index of unique functions, and provide a means of iterating
/// through all the instrumented call stacks which we know about.

struct TrieNode {
  int32_t FuncId;
  TrieNode *Parent;
  SmallVector<TrieNode *, 4> Callees;
  // Separate durations depending on whether the node is the deepest node in the
  // stack.
  SmallVector<int64_t, 4> TerminalDurations;
  SmallVector<int64_t, 4> IntermediateDurations;
};

/// Merges together two TrieNodes with like function ids, aggregating their
/// callee lists and durations. The caller must provide storage where new merged
/// nodes can be allocated in the form of a linked list.
TrieNode *mergeTrieNodes(const TrieNode &Left, const TrieNode &Right,
                         TrieNode *NewParent,
                         std::forward_list<TrieNode> &NodeStore) {
  assert(Left.FuncId == Right.FuncId);
  NodeStore.push_front(TrieNode{Left.FuncId, NewParent, {}, {}, {}});
  auto I = NodeStore.begin();
  auto *Node = &*I;

  // Build a map of callees from the left side.
  DenseMap<int32_t, TrieNode *> LeftCalleesByFuncId;
  for (auto *Callee : Left.Callees) {
    LeftCalleesByFuncId[Callee->FuncId] = Callee;
  }

  // Iterate through the right side, either merging with the map values or
  // directly adding to the Callees vector. The iteration also removes any
  // merged values from the left side map.
  for (auto *Callee : Right.Callees) {
    auto iter = LeftCalleesByFuncId.find(Callee->FuncId);
    if (iter != LeftCalleesByFuncId.end()) {
      Node->Callees.push_back(
          mergeTrieNodes(*(iter->second), *Callee, Node, NodeStore));
      LeftCalleesByFuncId.erase(iter);
    } else {
      Node->Callees.push_back(Callee);
    }
  }

  // Add any callees that weren't found in the right side.
  for (auto MapPairIter : LeftCalleesByFuncId) {
    Node->Callees.push_back(MapPairIter.second);
  }

  // Aggregate the durations.
  for (auto duration : Left.TerminalDurations) {
    Node->TerminalDurations.push_back(duration);
  }
  for (auto duration : Right.TerminalDurations) {
    Node->TerminalDurations.push_back(duration);
  }
  for (auto duration : Left.IntermediateDurations) {
    Node->IntermediateDurations.push_back(duration);
  }
  for (auto duration : Right.IntermediateDurations) {
    Node->IntermediateDurations.push_back(duration);
  }

  return Node;
}

class StackTrie {

  // We maintain pointers to the roots of the tries we see.
  DenseMap<uint32_t, SmallVector<TrieNode *, 4>> Roots;

  // We make sure all the nodes are accounted for in this list.
  std::forward_list<TrieNode> NodeStore;

  // A map of thread ids to pairs call stack trie nodes and their start times.
  DenseMap<uint32_t, SmallVector<std::pair<TrieNode *, uint64_t>, 8>>
      ThreadStackMap;

  TrieNode *createTrieNode(uint32_t ThreadId, int32_t FuncId,
                           TrieNode *Parent) {
    NodeStore.push_front(TrieNode{FuncId, Parent, {}, {}, {}});
    auto I = NodeStore.begin();
    auto *Node = &*I;
    if (!Parent)
      Roots[ThreadId].push_back(Node);
    return Node;
  }

  TrieNode *findRootNode(uint32_t ThreadId, int32_t FuncId) {
    const auto &RootsByThread = Roots[ThreadId];
    auto I = find_if(RootsByThread,
                     [&](TrieNode *N) { return N->FuncId == FuncId; });
    return (I == RootsByThread.end()) ? nullptr : *I;
  }

public:
  enum class AccountRecordStatus {
    OK,              // Successfully processed
    ENTRY_NOT_FOUND, // An exit record had no matching call stack entry
    UNKNOWN_RECORD_TYPE
  };

  struct AccountRecordState {
    // We keep track of whether the call stack is currently unwinding.
    bool wasLastRecordExit;

    static AccountRecordState CreateInitialState() { return {false}; }
  };

  AccountRecordStatus accountRecord(const XRayRecord &R,
                                    AccountRecordState *state) {
    auto &TS = ThreadStackMap[R.TId];
    switch (R.Type) {
    case RecordTypes::ENTER: {
      state->wasLastRecordExit = false;
      // When we encounter a new function entry, we want to record the TSC for
      // that entry, and the function id. Before doing so we check the top of
      // the stack to see if there are callees that already represent this
      // function.
      if (TS.empty()) {
        auto *Root = findRootNode(R.TId, R.FuncId);
        TS.emplace_back(Root ? Root : createTrieNode(R.TId, R.FuncId, nullptr),
                        R.TSC);
        return AccountRecordStatus::OK;
      }

      auto &Top = TS.back();
      auto I = find_if(Top.first->Callees,
                       [&](TrieNode *N) { return N->FuncId == R.FuncId; });
      if (I == Top.first->Callees.end()) {
        // We didn't find the callee in the stack trie, so we're going to
        // add to the stack then set up the pointers properly.
        auto N = createTrieNode(R.TId, R.FuncId, Top.first);
        Top.first->Callees.emplace_back(N);

        // Top may be invalidated after this statement.
        TS.emplace_back(N, R.TSC);
      } else {
        // We found the callee in the stack trie, so we'll use that pointer
        // instead, add it to the stack associated with the TSC.
        TS.emplace_back(*I, R.TSC);
      }
      return AccountRecordStatus::OK;
    }
    case RecordTypes::EXIT:
    case RecordTypes::TAIL_EXIT: {
      bool wasLastRecordExit = state->wasLastRecordExit;
      state->wasLastRecordExit = true;
      // The exit case is more interesting, since we want to be able to deduce
      // missing exit records. To do that properly, we need to look up the stack
      // and see whether the exit record matches any of the entry records. If it
      // does match, we attempt to record the durations as we pop the stack to
      // where we see the parent.
      if (TS.empty()) {
        // Short circuit, and say we can't find it.

        return AccountRecordStatus::ENTRY_NOT_FOUND;
      }

      auto FunctionEntryMatch =
          find_if(reverse(TS), [&](const std::pair<TrieNode *, uint64_t> &E) {
            return E.first->FuncId == R.FuncId;
          });
      auto status = AccountRecordStatus::OK;
      if (FunctionEntryMatch == TS.rend()) {
        status = AccountRecordStatus::ENTRY_NOT_FOUND;
      } else {
        // Account for offset of 1 between reverse and forward iterators. We
        // want the forward iterator to include the function that is exited.
        ++FunctionEntryMatch;
      }
      auto I = FunctionEntryMatch.base();
      for (auto &E : make_range(I, TS.end() - 1))
        E.first->IntermediateDurations.push_back(std::max(E.second, R.TSC) -
                                                 std::min(E.second, R.TSC));
      auto &Deepest = TS.back();
      if (wasLastRecordExit)
        Deepest.first->IntermediateDurations.push_back(
            std::max(Deepest.second, R.TSC) - std::min(Deepest.second, R.TSC));
      else
        Deepest.first->TerminalDurations.push_back(
            std::max(Deepest.second, R.TSC) - std::min(Deepest.second, R.TSC));
      TS.erase(I, TS.end());
      return status;
    }
    }
    return AccountRecordStatus::UNKNOWN_RECORD_TYPE;
  }

  bool isEmpty() const { return Roots.empty(); }

  void printStack(raw_ostream &OS, const TrieNode *Top,
                  FuncIdConversionHelper &FN) {
    // Traverse the pointers up to the parent, noting the sums, then print
    // in reverse order (callers at top, callees down bottom).
    SmallVector<const TrieNode *, 8> CurrentStack;
    for (auto *F = Top; F != nullptr; F = F->Parent)
      CurrentStack.push_back(F);
    int Level = 0;
    OS << formatv("{0,-5} {1,-60} {2,+12} {3,+16}\n", "lvl", "function",
                  "count", "sum");
    for (auto *F :
         reverse(make_range(CurrentStack.begin() + 1, CurrentStack.end()))) {
      auto Sum = std::accumulate(F->IntermediateDurations.begin(),
                                 F->IntermediateDurations.end(), 0LL);
      auto FuncId = FN.SymbolOrNumber(F->FuncId);
      OS << formatv("#{0,-4} {1,-60} {2,+12} {3,+16}\n", Level++,
                    FuncId.size() > 60 ? FuncId.substr(0, 57) + "..." : FuncId,
                    F->IntermediateDurations.size(), Sum);
    }
    auto *Leaf = *CurrentStack.begin();
    auto LeafSum = std::accumulate(Leaf->TerminalDurations.begin(),
                                   Leaf->TerminalDurations.end(), 0LL);
    auto LeafFuncId = FN.SymbolOrNumber(Leaf->FuncId);
    OS << formatv("#{0,-4} {1,-60} {2,+12} {3,+16}\n", Level++,
                  LeafFuncId.size() > 60 ? LeafFuncId.substr(0, 57) + "..."
                                         : LeafFuncId,
                  Leaf->TerminalDurations.size(), LeafSum);
    OS << "\n";
  }

  /// Prints top stacks for each thread.
  void printPerThread(raw_ostream &OS, FuncIdConversionHelper &FN) {
    for (auto iter : Roots) {
      OS << "Thread " << iter.first << ":\n";
      print(OS, FN, iter.second);
      OS << "\n";
    }
  }

  /// Prints top stacks from looking at all the leaves and ignoring thread IDs.
  /// Stacks that consist of the same function IDs but were called in different
  /// thread IDs are not considered unique in this printout.
  void printIgnoringThreads(raw_ostream &OS, FuncIdConversionHelper &FN) {
    SmallVector<TrieNode *, 4> RootValues;

    // Function to pull the values out of a map iterator.
    using RootsType = decltype(Roots.begin())::value_type;
    auto MapValueFn = [](const RootsType &Value) { return Value.second; };

    for (const auto &RootNodeRange :
         make_range(map_iterator(Roots.begin(), MapValueFn),
                    map_iterator(Roots.end(), MapValueFn))) {
      for (auto *RootNode : RootNodeRange)
        RootValues.push_back(RootNode);
    }

    print(OS, FN, RootValues);
  }

  /// Merges the trie by thread id before printing top stacks.
  void printAggregatingThreads(raw_ostream &OS, FuncIdConversionHelper &FN) {
    std::forward_list<TrieNode> AggregatedNodeStore;
    SmallVector<TrieNode *, 4> RootValues;
    for (auto MapIter : Roots) {
      const auto &RootNodeVector = MapIter.second;
      for (auto *Node : RootNodeVector) {
        auto MaybeFoundIter = find_if(RootValues, [Node](TrieNode *elem) {
          return Node->FuncId == elem->FuncId;
        });
        if (MaybeFoundIter == RootValues.end()) {
          RootValues.push_back(Node);
        } else {
          RootValues.push_back(mergeTrieNodes(**MaybeFoundIter, *Node, nullptr,
                                              AggregatedNodeStore));
          RootValues.erase(MaybeFoundIter);
        }
      }
    }
    print(OS, FN, RootValues);
  }

  void print(raw_ostream &OS, FuncIdConversionHelper &FN,
             SmallVector<TrieNode *, 4> RootValues) {
    // Go through each of the roots, and traverse the call stack, producing the
    // aggregates as you go along. Remember these aggregates and stacks, and
    // show summary statistics about:
    //
    //   - Total number of unique stacks
    //   - Top 10 stacks by count
    //   - Top 10 stacks by aggregate duration
    SmallVector<std::pair<const TrieNode *, uint64_t>, 11> TopStacksByCount;
    SmallVector<std::pair<const TrieNode *, uint64_t>, 11> TopStacksBySum;
    auto greater_second = [](const std::pair<const TrieNode *, uint64_t> &A,
                             const std::pair<const TrieNode *, uint64_t> &B) {
      return A.second > B.second;
    };
    uint64_t UniqueStacks = 0;
    for (const auto *N : RootValues) {
      SmallVector<const TrieNode *, 16> S;
      S.emplace_back(N);

      while (!S.empty()) {
        auto Top = S.pop_back_val();

        // We only start printing the stack (by walking up the parent pointers)
        // when we get to a leaf function.
        if (!Top->TerminalDurations.empty()) {
          ++UniqueStacks;
          auto TopSum = std::accumulate(Top->TerminalDurations.begin(),
                                        Top->TerminalDurations.end(), 0uLL);
          {
            auto E = std::make_pair(Top, TopSum);
            TopStacksBySum.insert(std::lower_bound(TopStacksBySum.begin(),
                                                   TopStacksBySum.end(), E,
                                                   greater_second),
                                  E);
            if (TopStacksBySum.size() == 11)
              TopStacksBySum.pop_back();
          }
          {
            auto E = std::make_pair(Top, Top->TerminalDurations.size());
            TopStacksByCount.insert(std::lower_bound(TopStacksByCount.begin(),
                                                     TopStacksByCount.end(), E,
                                                     greater_second),
                                    E);
            if (TopStacksByCount.size() == 11)
              TopStacksByCount.pop_back();
          }
        }
        for (const auto *C : Top->Callees)
          S.push_back(C);
      }
    }

    // Now print the statistics in the end.
    OS << "\n";
    OS << "Unique Stacks: " << UniqueStacks << "\n";
    OS << "Top 10 Stacks by leaf sum:\n\n";
    for (const auto &P : TopStacksBySum) {
      OS << "Sum: " << P.second << "\n";
      printStack(OS, P.first, FN);
    }
    OS << "\n";
    OS << "Top 10 Stacks by leaf count:\n\n";
    for (const auto &P : TopStacksByCount) {
      OS << "Count: " << P.second << "\n";
      printStack(OS, P.first, FN);
    }
    OS << "\n";
  }
};

std::string CreateErrorMessage(StackTrie::AccountRecordStatus Error,
                               const XRayRecord &Record,
                               const FuncIdConversionHelper &Converter) {
  switch (Error) {
  case StackTrie::AccountRecordStatus::ENTRY_NOT_FOUND:
    return formatv("Found record {0} with no matching function entry\n",
                   format_xray_record(Record, Converter));
  default:
    return formatv("Unknown error type for record {0}\n",
                   format_xray_record(Record, Converter));
  }
}

static CommandRegistration Unused(&Stack, []() -> Error {
  // Load each file provided as a command-line argument. For each one of them
  // account to a single StackTrie, and just print the whole trie for now.
  StackTrie ST;
  InstrumentationMap Map;
  if (!StacksInstrMap.empty()) {
    auto InstrumentationMapOrError = loadInstrumentationMap(StacksInstrMap);
    if (!InstrumentationMapOrError)
      return joinErrors(
          make_error<StringError>(
              Twine("Cannot open instrumentation map: ") + StacksInstrMap,
              std::make_error_code(std::errc::invalid_argument)),
          InstrumentationMapOrError.takeError());
    Map = std::move(*InstrumentationMapOrError);
  }

  if (SeparateThreadStacks && AggregateThreads)
    return make_error<StringError>(
        Twine("Can't specify options for per thread reporting and reporting "
              "that aggregates threads."),
        std::make_error_code(std::errc::invalid_argument));

  symbolize::LLVMSymbolizer::Options Opts(
      symbolize::FunctionNameKind::LinkageName, true, true, false, "");
  symbolize::LLVMSymbolizer Symbolizer(Opts);
  FuncIdConversionHelper FuncIdHelper(StacksInstrMap, Symbolizer,
                                      Map.getFunctionAddresses());
  // TODO: Someday, support output to files instead of just directly to
  // standard output.
  for (const auto &Filename : StackInputs) {
    auto TraceOrErr = loadTraceFile(Filename);
    if (!TraceOrErr) {
      if (!StackKeepGoing)
        return joinErrors(
            make_error<StringError>(
                Twine("Failed loading input file '") + Filename + "'",
                std::make_error_code(std::errc::invalid_argument)),
            TraceOrErr.takeError());
      logAllUnhandledErrors(TraceOrErr.takeError(), errs(), "");
      continue;
    }
    auto &T = *TraceOrErr;
    StackTrie::AccountRecordState AccountRecordState =
        StackTrie::AccountRecordState::CreateInitialState();
    for (const auto &Record : T) {
      auto error = ST.accountRecord(Record, &AccountRecordState);
      if (error != StackTrie::AccountRecordStatus::OK) {
        if (!StackKeepGoing)
          return make_error<StringError>(
              CreateErrorMessage(error, Record, FuncIdHelper),
              make_error_code(errc::illegal_byte_sequence));
        errs() << CreateErrorMessage(error, Record, FuncIdHelper);
      }
    }
  }
  if (ST.isEmpty()) {
    return make_error<StringError>(
        "No instrumented calls were accounted in the input file.",
        make_error_code(errc::result_out_of_range));
  }
  if (AggregateThreads) {
    ST.printAggregatingThreads(outs(), FuncIdHelper);
  } else if (SeparateThreadStacks) {
    ST.printPerThread(outs(), FuncIdHelper);
  } else {
    ST.printIgnoringThreads(outs(), FuncIdHelper);
  }
  return Error::success();
});
