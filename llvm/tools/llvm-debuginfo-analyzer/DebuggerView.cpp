#include "DebuggerView.h"
#include "Options.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/DebugInfo/LogicalView/LVReaderHandler.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;
using namespace debuggerview;

cl::OptionCategory llvm::debuggerview::Category(
    "Debugger View",
    "Special printing mode that emulate how debugger uses debug info.");

cl::opt<bool> llvm::debuggerview::Enable(
    "debugger-view",
    cl::desc("Enables debugger view. Normal debug-info printing is disabled "
             "and options are ignored."),
    cl::init(false), cl::cat(Category));
static cl::opt<bool>
    IncludeVars("debugger-view-vars",
                cl::desc("Include live variables at each statement line."),
                cl::init(false), cl::cat(Category));
static cl::opt<bool>
    IncludeCode("debugger-view-code",
                cl::desc("Include disassembly at each statement line"),
                cl::init(false), cl::cat(Category));
static cl::opt<bool> IncludeRanges("debugger-view-ranges",
                                   cl::desc("Include variable ranges"),
                                   cl::init(false), cl::cat(Category));
static cl::opt<bool> Help(
    "debugger-view-help",
    cl::desc(
        "Print a detailed help screen about what kind of output to expect"),
    cl::init(false), cl::cat(Category));

constexpr const char *HelpText =
    R"(Prints debug info in a way that is easy to verify correctness of debug info.
FUNCTION: main
  LINE: my_source_file.c:1 [main]       <---- New statement lines, inlined callstack
    VAR: argc : int     : {expression}  <---- Variables live at this point
    VAR: argv : char ** : {expression}
  LINE: my_source_file.c:2 [main]
    VAR: argc : int
    VAR: argv : char **
  LINE: my_source_file.c:3 [main]
    VAR: argc : int
    VAR: argv : char **
  LINE: my_source_file.c:4 [main]
)";

using namespace llvm;
using namespace logicalview;

template <typename T>
static T Take(Expected<T> ExpectedResult, const Twine &Msg) {
  if (!ExpectedResult) {
    auto Err = ExpectedResult.takeError();
    errs() << Msg << " " << toStringWithoutConsuming(Err) << '\n';
    exit(2);
  }
  T ret = std::move(*ExpectedResult);
  return ret;
}

namespace {

struct ScopePrinter {
  std::vector<const LVLine *> Lines;
  std::unordered_map<LVAddress, std::vector<const LVLocation *>> LivetimeBegins;
  std::unordered_map<LVAddress, std::vector<const LVLocation *>>
      LivetimeEndsExclusive;
  raw_ostream &OS;

  void Walk(raw_ostream &OS, const LVScope *Scope) {
    if (Scope->scopeCount()) {
      for (const LVScope *ChildScope : *Scope->getScopes())
        Walk(OS, ChildScope);
    }
    if (Scope->lineCount()) {
      for (const LVLine *Line : *Scope->getLines()) {
        Lines.push_back(Line);
      }
    }
    if (Scope->symbolCount()) {
      for (const LVSymbol *Symbol : *Scope->getSymbols()) {
        LVLocations SymbolLocations;
        Symbol->getLocations(SymbolLocations);
        if (SymbolLocations.empty())
          continue;

        if (IncludeRanges) {
          OS << "RANGES: " << Symbol->getName() << " (line "
             << Symbol->getLineNumber() << ")" << ": ";
        }

        for (const LVLocation *Loc : SymbolLocations) {
          if (Loc->getIsGapEntry())
            continue;

          LVAddress Begin = Loc->getLowerAddress();
          LVAddress End = Loc->getUpperAddress();
          LivetimeBegins[Begin].push_back(Loc);
          LivetimeEndsExclusive[End].push_back(Loc);
          if (IncludeRanges) {
            OS << "[" << hexValue(Begin) << ":" << hexValue(End) << "] ";
          }
        }

        if (IncludeRanges)
          OS << "\n";
      }
    }
  }

  ScopePrinter(raw_ostream &OS, const LVScopeFunction *Fn) : OS(OS) {
    Walk(OS, Fn);
    std::sort(Lines.begin(), Lines.end(),
              [](const LVLine *a, const LVLine *b) -> bool {
                if (a->getAddress() != b->getAddress())
                  return a->getAddress() < b->getAddress();
                if (a->getIsLineDebug() != b->getIsLineDebug())
                  return a->getIsLineDebug();
                return a->getID() < b->getID();
              });
  }

  static void PrintIndent(raw_ostream &OS, int Indent) {
    for (int i = 0; i < Indent; i++)
      OS << "  ";
  }

  static void PrintCallstack(raw_ostream &OS, const LVScope *Scope) {
    bool First = true;
    const LVScope *PrevScope = nullptr;
    while (Scope) {
      if (Scope->getIsFunction() || Scope->getIsInlinedFunction()) {
        OS << "[" << Scope->getName();
        if (PrevScope && PrevScope->getIsInlinedFunction()) {
          OS << ":"
             << cast<LVScopeFunctionInlined>(PrevScope)->getCallLineNumber();
        }
        OS << "]";
        First = false;
        PrevScope = Scope;
      }
      Scope = Scope->getParentScope();
    }
  }

  static bool IsChildScopeOf(const LVScope *A, const LVScope *B) {
    while (A) {
      A = A->getParentScope();
      if (A == B)
        return true;
    }
    return false;
  }

  void Print() {
    SetVector<const LVLocation *>
        LiveSymbols; // This needs to be ordered since we're iterating over it.
    for (const LVLine *Line : Lines) {

      const LVScope *Scope = Line->getParentScope();

      // Update live list: Add lives
      for (auto Loc : LivetimeBegins[Line->getAddress()])
        LiveSymbols.insert(Loc);
      // Update live list: remove dead
      for (auto Loc : LivetimeEndsExclusive[Line->getAddress()])
        LiveSymbols.remove(Loc);

      if (Line->getIsNewStatement() && Line->getIsLineDebug() &&
          Line->getLineNumber() != 0) {
        auto LineDebug = cast<LVLineDebug>(Line);

        OS << "LINE: " << " [" << hexValue(LineDebug->getAddress()) << "] "
           << LineDebug->getPathname() << ":" << LineDebug->getLineNumber()
           << " ";
        PrintCallstack(OS, Scope);
        OS << "\n";
        if (IncludeVars) {
          for (auto SymLoc : LiveSymbols) {
            const LVSymbol *Sym = SymLoc->getParentSymbol();
            auto SymScope = Sym->getParentScope();
            auto LineScope = LineDebug->getParentScope();
            if (SymScope != LineScope && !IsChildScopeOf(LineScope, SymScope))
              continue;
            PrintIndent(OS, 1);
            OS << "VAR: " << Sym->getName() << ": " << Sym->getType()->getName()
               << " : ";
            SymLoc->printLocations(OS);
            OS << " (line " << Sym->getLineNumber() << ")";
            OS << "\n";
          }
        }

      } else if (IncludeCode && Line->getIsLineAssembler()) {
        OS << "  CODE: " << " [" << hexValue(Line->getAddress()) << "]  "
           << Line->getName() << "\n";
      }
    }
  }
};

} // namespace

int llvm::debuggerview::printDebuggerView(std::vector<std::string> &Objects,
                                          raw_ostream &OS) {
  if (Help) {
    OS << HelpText;
    return EXIT_SUCCESS;
  }

  LVOptions Options;
  Options.setAttributeAll();
  Options.setAttributeAnyLocation();
  Options.setPrintAll();
  Options.setPrintAnyLine();
  Options.resolveDependencies();

  ScopedPrinter W(nulls());
  LVReaderHandler Handler(Objects, W, Options);
  std::vector<std::unique_ptr<LVReader>> Readers;
  for (auto &Object : Objects) {
    auto ExpectedReader = Handler.createReader(Object);
    if (!ExpectedReader) {
      auto Err = ExpectedReader.takeError();
      errs() << "Failed to create reader: " << toStringWithoutConsuming(Err)
             << '\n';
      return 2;
    }
    Readers.emplace_back(std::move(*ExpectedReader));
  }

  for (auto &Reader : Readers) {
    auto *CU = Reader->getCompileUnit();
    if (!CU) {
      errs() << "No compute unit found.\n";
      return 2;
    }

    for (LVElement *Child : *CU->getChildren()) {
      auto *Fn = dyn_cast<LVScopeFunction>(Child);
      if (Fn) {
        const LVLines *Lines = Fn->getLines();
        // If there's no lines, this function has no body.
        if (!Lines)
          continue;
        outs() << "FUNCTION: " << Child->getName() << "\n";

        ScopePrinter P(OS, Fn);
        P.Print();
      }
    }
  }

  return EXIT_SUCCESS;
}
