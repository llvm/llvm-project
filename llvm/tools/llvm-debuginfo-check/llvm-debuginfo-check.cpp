//===-- llvm-debuginfo-analyzer.cpp - LLVM Debug info analysis utility ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool prints out the logical view of debug info in a way that is easy
// to verify correctness.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/DebugInfo/LogicalView/LVReaderHandler.h"
#include "llvm/ADT/SetVector.h"

#include <string>
#include <unordered_set>
#include <unordered_map>

using namespace llvm;
using namespace logicalview;

static cl::opt<std::string>
    InputFilename(cl::Positional, "<input-file>",
                  cl::desc("Input file, an object file with DWARF."),
                  cl::Required);

static cl::opt<bool> IncludeCode("code", cl::desc("Include asm"));
static cl::opt<bool> IncludeRanges("ranges", cl::desc("Include variable ranges"));
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

struct ScopePrinterMk2 {

  std::vector<const LVLine *> Lines;
  std::unordered_map<LVAddress, std::vector<const LVLocation *>> LivetimeBegins;
  std::unordered_map<LVAddress, std::vector<const LVLocation *>>
      LivetimeEndsExclusive;
  raw_ostream &OS;

  void Walk(raw_ostream &OS, const LVScope* Scope) {
    if (Scope->scopeCount()) {
      for (const LVScope *ChildScope : *Scope->getScopes())
        Walk(OS, ChildScope);
    }
    if (Scope->lineCount()) {
      for (const LVLine *Line : *Scope->getLines()) {
        Lines.push_back(Line);
        if (Line->getParentScope() != Scope)
          OS << "";
      }
    }
    if (Scope->symbolCount()) {
      for (const LVSymbol *Symbol : *Scope->getSymbols()) {
        LVLocations SymbolLocations;
        Symbol->getLocations(SymbolLocations);
        if (SymbolLocations.empty())
          continue;

        if (IncludeRanges) {
          OS << "RANGES: " << Symbol->getName() << " (line " << Symbol->getLineNumber() << ")" << ": ";
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

  ScopePrinterMk2(raw_ostream &OS, const LVScopeFunction *Fn) : OS(OS) {
    Walk(OS, Fn);
    std::sort(Lines.begin(), Lines.end(),
              [](const LVLine *a, const LVLine *b) -> bool {
                if (a->getAddress() != b->getAddress())
                  return a->getAddress() < b->getAddress();
                if (a->getIsLineDebug() != b->getIsLineDebug())
                  return a->getIsLineDebug();
                return a->getID() < b->getID();
              });

    //DumpAllInlineFunctionLevels(Fn);
  }


  static void PrintIndent(int Indent) {
    for (int i = 0; i < Indent; i++)
      outs() << "  ";
  }

  static void PrintCallstack(raw_ostream& OS, const LVScope* Scope) {
    bool First = true;
    const LVScope *PrevScope = nullptr;
    while (Scope) {
      if (Scope->getIsFunction() || Scope->getIsInlinedFunction()) {
        OS << "[" << Scope->getName();
        if (PrevScope && PrevScope->getIsInlinedFunction()) {
          OS << ":" << cast<LVScopeFunctionInlined>(PrevScope)->getCallLineNumber();
        }
        OS << "]";
        First = false;
        PrevScope = Scope;
      }
      Scope = Scope->getParentScope();
    }
  }

  void DumpAllInlineFunctionLevels(const LVScope *Scope) {
    if (Scope->getIsInlinedFunction()) {
      outs() << Scope->getName() << " ID:" << Scope->getID();
      if (Scope->rangeCount()) {
        for (auto R : *Scope->getRanges()) {
          auto Lo = Scope->getRanges()->front()->getLowerAddress();
          auto Hi = Scope->getRanges()->front()->getUpperAddress();
          outs() << " [" << hexValue(Lo) << ":"
                 << hexValue(Hi) << "]";
        }
      }

      outs() << ", level: " << Scope->getLevel() << "\n";
    }
    if (Scope->scopeCount())
      for (auto S : *Scope->getScopes())
        DumpAllInlineFunctionLevels(S);
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
    SetVector<const LVLocation *> LiveSymbols; // This needs to be ordered since we're iterating over it.
    int LastLine = -1;
    StringRef LastFilename;
    for (const LVLine *Line : Lines) {

      const LVScope *Scope = Line->getParentScope();

      // Update live list: Add lives
      for (auto Loc : LivetimeBegins[Line->getAddress()])
        LiveSymbols.insert(Loc);
      // Update live list: remove dead
      for (auto Loc : LivetimeEndsExclusive[Line->getAddress()])
        LiveSymbols.remove(Loc);

      if (Line->getIsLineDebug() && Line->getLineNumber() != 0) {
        auto LineDebug = cast<LVLineDebug>(Line);

        if (LastLine != LineDebug->getLineNumber() ||
            LineDebug->getPathname() != LastFilename) {
          OS << "LINE: " << " [" << hexValue(LineDebug->getAddress()) << "] "
             << LineDebug->getPathname() << ":" << LineDebug->getLineNumber()
             << " ";
          PrintCallstack(OS, Scope);
          OS << "\n";
          if (IncludeVars) {
            for (auto SymLoc : LiveSymbols) {
              const LVSymbol *Sym = SymLoc->getParentSymbol();
              if (Sym->getName() == "InMRT1")
                outs() << "";
              auto SymScope = Sym->getParentScope();
              auto LineScope = LineDebug->getParentScope();
              if (SymScope != LineScope && !IsChildScopeOf(LineScope, SymScope))
                continue;
              PrintIndent(1);
              OS << "VAR: " << Sym->getName() << ": "
                     << Sym->getType()->getName()
                     << " : ";
              SymLoc->printLocations(OS);
              OS << " (line " << Sym->getLineNumber() << ")";
              OS << "\n";
            }
          }
        }

        LastLine = Line->getLineNumber();
        LastFilename = Line->getPathname();
      } else if (IncludeCode && Line->getIsLineAssembler()) {
        OS << "  CODE: " << " [" << hexValue(Line->getAddress()) << "]  "
           << Line->getName() << "\n";
      }
    }
  }
};

struct ScopePrinter {
  unsigned NumScopesPrinted = 0;
  std::unordered_map<LVAddress, std::vector<const LVLocation *>> LivetimeBegins;
  std::unordered_map<LVAddress, std::vector<const LVLocation *>>
      LivetimeEndsInclusive;
  SetVector<const LVLocation *> LiveSymbols; // This needs to be ordered since we're iterating over it.
  std::unordered_set<const LVScope *> SeenScopes;

  static void PrintIndent(int Indent) {
    for (int i = 0; i < Indent; i++)
      outs() << "  ";
  };

  void PrintScope(const LVScope *Scope, int Indent) {
    NumScopesPrinted++;
    SeenScopes.insert(Scope);

    bool IsInlined = false;
    if (Scope->getIsInlinedFunction()) {
      Indent++;
      auto InlinedFn = cast<LVScopeFunctionInlined>(Scope);
      IsInlined = true;
      PrintIndent(Indent);
      outs() << "INLINED_FUNCTION: " << InlinedFn->getName() << "\n";
    }

    if (const LVSymbols *Symbols = Scope->getSymbols()) {
      for (const LVSymbol *Symbol : *Symbols) {
        LVLocations SymbolLocations;
        Symbol->getLocations(SymbolLocations);
        for (const LVLocation *Loc : SymbolLocations) {
          if (Loc->getIsGapEntry())
            continue;

          LVAddress Begin = Loc->getLowerAddress();
          LVAddress End = Loc->getUpperAddress();
          LivetimeBegins[Begin].push_back(Loc);
          LivetimeEndsInclusive[End].push_back(Loc);
          if (IncludeRanges) {
            outs() << "RANGE: " << Symbol->getName() << ": ["
                   << hexValue(Begin) << ":" << hexValue(End) << "]\n";
          }
        }
      }
    }

    auto Lines = Scope->getLines();
    if (Lines) {
      int LastLine = -1;
      StringRef LastFilename;
      for (const LVLine *Line : *Lines) {

        if (auto Scopes = Scope->getScopes()) {
          for (const LVScope *SubScope : *Scopes) {
            if (SeenScopes.count(SubScope))
              continue;
#if 0
            if (SubScope->getBaseAddress() < Line->getAddress()) {
              PrintScope(SubScope, Indent);
            }
#else


            struct Local {
              static LVAddress GetBaseAddress(const LVScope *S) {
                if (S->rangeCount()) {
                  return S->getRanges()->front()->getLowerAddress();
                }
                if (S->lineCount()) {
                  return S->getLines()->front()->getAddress();
                }
                if (S->scopeCount()) {
                  for (LVScope *SubS : *S->getScopes()) {
                    LVAddress A = GetBaseAddress(SubS);
                    if (A != -1)
                      return A;
                  }
                }
                return -1;
              };
            };
#if 0
            auto ScopeLines = SubScope->getLines();
            if (ScopeLines && ScopeLines->size() &&
                ScopeLines->front()->getAddress() < Line->getAddress()) {
              PrintScope(SubScope, Indent);
            } else {
              outs() << "";
            }
#else
            unsigned BaseAdress = Local::GetBaseAddress(SubScope);
            if (BaseAdress < Line->getAddress()) {
              PrintScope(SubScope, Indent);
            }
#endif
#endif
          }
        }

        // Update live list: Add lives
        for (auto Loc : LivetimeBegins[Line->getAddress()])
          LiveSymbols.insert(Loc);

        if (Line->getIsLineDebug() && Line->getLineNumber() != 0) {
          auto LineDebug = cast<LVLineDebug>(Line);
          if (LineDebug->getLineNumber() == 605)
            outs() << "";
          if (LastLine != LineDebug->getLineNumber() ||
              LineDebug->getPathname() != LastFilename) {
            PrintIndent(Indent+1);
            outs() << "LINE: " << " ["
                   << hexValue(LineDebug->getAddress()) << "] "
                   << LineDebug->getPathname() << ":"
                   << LineDebug->getLineNumber() << "\n";

            if (IncludeVars) {
              for (auto SymLoc : LiveSymbols) {
                PrintIndent(Indent+2);
                const LVSymbol *Sym = SymLoc->getParentSymbol();
                outs() << "VAR: " << Sym->getName() << ": "
                       << Sym->getType()->getName()
                       << " : ";
                SymLoc->printLocations(outs());
                outs() << "\n";
              }
            }
          }
          LastLine = LineDebug->getLineNumber();
          LastFilename = LineDebug->getPathname();
        } else if (Line->getIsLineAssembler()) {
          if (IncludeCode)
            outs() << "  CODE: " << " [" << hexValue(Line->getAddress())
                   << "]  " << Line->getName() << "\n";
        }

        // Update live list: remove dead
        for (auto Loc : LivetimeEndsInclusive[Line->getAddress()])
          LiveSymbols.remove(Loc);
      }
    } else if (Scope->scopeCount()) {
      for (auto SubS : *Scope->getScopes()) {
        PrintScope(SubS, Indent);
      }
    }

    if (IsInlined) {
      PrintIndent(Indent);
      outs() << "END_INLINED_FUNCTION\n";
      Indent--;
    }
  }

  static unsigned CountScopes(const LVScope *Scope) {
    unsigned Count = 0;
    auto Scopes = Scope->getScopes();
    if (Scopes) {
      for (auto S : *Scopes)
        Count += CountScopes(S);
    }
    return Count + 1;
  }

  std::vector<const LVScope *> UnseenScopes;
  void CollectUnsceenScopes(const LVScope *Scope) {
    if (!SeenScopes.count(Scope)) {
      UnseenScopes.push_back(Scope);
      Scope->dump();
    }
    if (Scope->scopeCount()) {
      for (auto S : *Scope->getScopes())
        CollectUnsceenScopes(S);
    }
  }

};

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  cl::ParseCommandLineOptions(argc, argv,
                              "Check debug info correctness via annotations.\n");

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
       Twine("Failed to create LV reader from '") + Twine(InputFilename) +
           Twine("'"));

  auto *CU = Readers->getCompileUnit();
  if (!CU)
    return 2;

  for (LVElement *Child : *CU->getChildren()) {
    auto *Fn = dyn_cast<LVScopeFunction>(Child);
    if (Fn) {
      const LVLines *Lines = Fn->getLines();
      // If there's no lines, this function has no body.
      if (!Lines)
        continue;
      outs() << "FUNCTION: " << Child->getName() << "\n";
 
#if 0
      ScopePrinter P;
      P.PrintScope(Fn, 0);

      P.CollectUnsceenScopes(Fn);

      for (auto S : P.UnseenScopes) {
        if (S->symbolCount()) {
          for (auto Sym : *S->getSymbols())
            outs() << Sym->getName() << ": " << Sym->getTypeName() << " (line"
                   << Sym->getLineNumber() << ")" << "\n";
        }
      }

      //unsigned NumScopes = ScopePrinter::CountScopes(Fn);

      outs() << "";
#else
      ScopePrinterMk2 P(outs(), Fn);
      P.Print();
#endif

    }
  }

  return EXIT_SUCCESS;
}
