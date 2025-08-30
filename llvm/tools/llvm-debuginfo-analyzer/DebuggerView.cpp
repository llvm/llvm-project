#include "Options.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/DebugInfo/LogicalView/LVReaderHandler.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;

cl::opt<bool> EnableDebuggerView("debugger-view");
static cl::opt<bool> DebuggerViewVars("debugger-view-vars");
static cl::opt<bool> DebuggerViewCode("debugger-view-code");
static cl::opt<bool> DebuggerViewHelp("debugger-view-help");

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

static cl::opt<std::string>
    InputFilename(cl::Positional, "<input-file>",
                  cl::desc("Input file, an object file with DWARF."),
                  cl::Required);

static cl::opt<bool> IncludeCode("code", cl::desc("Include asm"));
static cl::opt<bool>
    IncludeRanges("ranges", cl::desc("Include variable ranges"), cl::Hidden);
static cl::opt<bool> IncludeVars("vars", cl::desc("Include live variables"));

template <typename T> T Take(Expected<T> ExpectedResult, const Twine &Msg) {
  if (!ExpectedResult) {
    auto Err = ExpectedResult.takeError();
    errs() << Msg << " " << toStringWithoutConsuming(Err) << '\n';
    exit(2);
  }
  T ret = std::move(*ExpectedResult);
  return ret;
}

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
#if 0
int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  cl::ParseCommandLineOptions(argc, argv, HelpText);

  ScopedPrinter W(llvm::outs());
  LVOptions Options;
  Options.setAttributeAll();
  Options.setAttributeAnyLocation();
  Options.setPrintAll();
  Options.setPrintAnyLine();
  Options.resolveDependencies();
  std::vector<std::string> Objects;
  LVReaderHandler Handler(Objects, W, Options);
  auto Readers = Take(Handler.createReader(InputFilename),
                      Twine("Failed to create LV reader from '") +
                          Twine(InputFilename) + Twine("'"));

  auto *CU = Readers->getCompileUnit();
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

      ScopePrinter P(outs(), Fn);
      P.Print();
    }
  }

  return EXIT_SUCCESS;
}
#endif

int llvm::debuggerview::printDebuggerView(std::vector<std::string> &Objects, raw_ostream &OS) {

  LVOptions Options;
  Options.setAttributeAll();
  Options.setAttributeAnyLocation();
  Options.setPrintAll();
  Options.setPrintAnyLine();
  Options.resolveDependencies();

  ScopedPrinter W(nulls());
  LVReaderHandler Handler(Objects, W, Options);
  auto Readers = Take(Handler.createReader(InputFilename),
                      Twine("Failed to create LV reader from '") +
                          Twine(InputFilename) + Twine("'"));

  auto *CU = Readers->getCompileUnit();
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

  return EXIT_SUCCESS;
}
