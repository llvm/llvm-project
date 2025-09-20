#include "llvm/Support/DebugCounter.h"

#include "DebugOptions.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"

using namespace llvm;

namespace llvm {

void DebugCounter::printChunks(raw_ostream &OS, ArrayRef<Range> Ranges) {
  RangeUtils::printRanges(OS, Ranges, ':');
}

} // namespace llvm

namespace {
// This class overrides the default list implementation of printing so we
// can pretty print the list of debug counter options.  This type of
// dynamic option is pretty rare (basically this and pass lists).
class DebugCounterList : public cl::list<std::string, DebugCounter> {
private:
  using Base = cl::list<std::string, DebugCounter>;

public:
  template <class... Mods>
  explicit DebugCounterList(Mods &&... Ms) : Base(std::forward<Mods>(Ms)...) {}

private:
  void printOptionInfo(size_t GlobalWidth) const override {
    // This is a variant of from generic_parser_base::printOptionInfo.  Sadly,
    // it's not easy to make it more usable.  We could get it to print these as
    // options if we were a cl::opt and registered them, but lists don't have
    // options, nor does the parser for std::string.  The other mechanisms for
    // options are global and would pollute the global namespace with our
    // counters.  Rather than go that route, we have just overridden the
    // printing, which only a few things call anyway.
    outs() << "  -" << ArgStr;
    // All of the other options in CommandLine.cpp use ArgStr.size() + 6 for
    // width, so we do the same.
    Option::printHelpStr(HelpStr, GlobalWidth, ArgStr.size() + 6);
    const auto &CounterInstance = DebugCounter::instance();
    for (const auto &Name : CounterInstance) {
      const auto Info =
          CounterInstance.getCounterInfo(CounterInstance.getCounterId(Name));
      size_t NumSpaces = GlobalWidth - Info.first.size() - 8;
      outs() << "    =" << Info.first;
      outs().indent(NumSpaces) << " -   " << Info.second << '\n';
    }
  }
};

// All global objects associated to the DebugCounter, including the DebugCounter
// itself, are owned by a single global instance of the DebugCounterOwner
// struct. This makes it easier to control the order in which constructors and
// destructors are run.
struct DebugCounterOwner : DebugCounter {
  DebugCounterList DebugCounterOption{
      "debug-counter", cl::Hidden,
      cl::desc("Comma separated list of debug counter skip and count"),
      cl::CommaSeparated, cl::location<DebugCounter>(*this)};
  cl::opt<bool, true> PrintDebugCounter{
      "print-debug-counter",
      cl::Hidden,
      cl::Optional,
      cl::location(this->ShouldPrintCounter),
      cl::init(false),
      cl::desc("Print out debug counter info after all counters accumulated")};
  cl::opt<bool, true> BreakOnLastCount{
      "debug-counter-break-on-last",
      cl::Hidden,
      cl::Optional,
      cl::location(this->BreakOnLast),
      cl::init(false),
      cl::desc("Insert a break point on the last enabled count of a "
               "chunks list")};

  DebugCounterOwner() {
    // Our destructor uses the debug stream. By referencing it here, we
    // ensure that its destructor runs after our destructor.
    (void)dbgs();
  }

  // Print information when destroyed, iff command line option is specified.
  ~DebugCounterOwner() {
    if (ShouldPrintCounter)
      print(dbgs());
  }
};

} // anonymous namespace

void llvm::initDebugCounterOptions() { (void)DebugCounter::instance(); }

DebugCounter &DebugCounter::instance() {
  static DebugCounterOwner O;
  return O;
}

// This is called by the command line parser when it sees a value for the
// debug-counter option defined above.
void DebugCounter::push_back(const std::string &Val) {
  if (Val.empty())
    return;
#ifdef NDEBUG
  // isCountingEnabled is hardcoded to false in NDEBUG.
  errs() << "Requested --debug-counter in LLVM build without assertions. This "
            "is a no-op.\n";
#endif

  // The strings should come in as counter=chunk_list
  auto CounterPair = StringRef(Val).split('=');
  if (CounterPair.second.empty()) {
    errs() << "DebugCounter Error: " << Val << " does not have an = in it\n";
    exit(1);
  }
  StringRef CounterName = CounterPair.first;

  auto ExpectedChunks = RangeUtils::parseRanges(CounterPair.second, ':');
  if (!ExpectedChunks) {
    handleAllErrors(ExpectedChunks.takeError(), [&](const StringError &E) {
      errs() << "DebugCounter Error: " << E.getMessage() << "\n";
    });
    exit(1);
  }
  RangeUtils::RangeList Chunks = std::move(*ExpectedChunks);

  unsigned CounterID = getCounterId(std::string(CounterName));
  if (!CounterID) {
    errs() << "DebugCounter Error: " << CounterName
           << " is not a registered counter\n";
    exit(1);
  }
  enableAllCounters();

  CounterInfo &Counter = Counters[CounterID];
  Counter.IsSet = true;
  Counter.Chunks = std::move(Chunks);
}

void DebugCounter::print(raw_ostream &OS) const {
  SmallVector<StringRef, 16> CounterNames(RegisteredCounters.begin(),
                                          RegisteredCounters.end());
  sort(CounterNames);

  auto &Us = instance();
  OS << "Counters and values:\n";
  for (auto &CounterName : CounterNames) {
    unsigned CounterID = getCounterId(std::string(CounterName));
    const CounterInfo &C = Us.Counters[CounterID];
    OS << left_justify(RegisteredCounters[CounterID], 32) << ": {" << C.Count
       << ",";
    printChunks(OS, C.Chunks);
    OS << "}\n";
  }
}

bool DebugCounter::shouldExecuteImpl(unsigned CounterName) {
  auto &Us = instance();
  auto Result = Us.Counters.find(CounterName);
  if (Result != Us.Counters.end()) {
    auto &CounterInfo = Result->second;
    int64_t CurrCount = CounterInfo.Count++;
    uint64_t CurrIdx = CounterInfo.CurrChunkIdx;

    if (CounterInfo.Chunks.empty())
      return true;
    if (CurrIdx >= CounterInfo.Chunks.size())
      return false;

    bool Res = CounterInfo.Chunks[CurrIdx].contains(CurrCount);
    if (Us.BreakOnLast && CurrIdx == (CounterInfo.Chunks.size() - 1) &&
        CurrCount == CounterInfo.Chunks[CurrIdx].getEnd()) {
      LLVM_BUILTIN_DEBUGTRAP;
    }
    if (CurrCount > CounterInfo.Chunks[CurrIdx].getEnd()) {
      CounterInfo.CurrChunkIdx++;

      /// Handle consecutive blocks.
      if (CounterInfo.CurrChunkIdx < CounterInfo.Chunks.size() &&
          CurrCount == CounterInfo.Chunks[CounterInfo.CurrChunkIdx].getBegin())
        return true;
    }
    return Res;
  }
  // Didn't find the counter, should we warn?
  return true;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void DebugCounter::dump() const {
  print(dbgs());
}
#endif
