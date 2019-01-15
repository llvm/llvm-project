//===--- Core.cpp - Core ORC APIs (MaterializationUnit, JITDylib, etc.) ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/CoreTypes.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "orc"

using namespace llvm;

namespace {

#ifndef NDEBUG

cl::opt<bool> PrintHidden("debug-orc-print-hidden", cl::init(false),
                          cl::desc("debug print hidden symbols defined by "
                                   "materialization units"),
                          cl::Hidden);

cl::opt<bool> PrintCallable("debug-orc-print-callable", cl::init(false),
                            cl::desc("debug print callable symbols defined by "
                                     "materialization units"),
                            cl::Hidden);

cl::opt<bool> PrintData("debug-orc-print-data", cl::init(false),
                        cl::desc("debug print data symbols defined by "
                                 "materialization units"),
                        cl::Hidden);

#endif // NDEBUG

// SetPrinter predicate that prints every element.
template <typename T> struct PrintAll {
  bool operator()(const T &E) { return true; }
};

bool anyPrintSymbolOptionSet() {
#ifndef NDEBUG
  return PrintHidden || PrintCallable || PrintData;
#else
  return false;
#endif // NDEBUG
}

bool flagsMatchCLOpts(const JITSymbolFlags &Flags) {
#ifndef NDEBUG
  // Bail out early if this is a hidden symbol and we're not printing hiddens.
  if (!PrintHidden && !Flags.isExported())
    return false;

  // Return true if this is callable and we're printing callables.
  if (PrintCallable && Flags.isCallable())
    return true;

  // Return true if this is data and we're printing data.
  if (PrintData && !Flags.isCallable())
    return true;

  // otherwise return false.
  return false;
#else
  return false;
#endif // NDEBUG
}

// Prints a set of items, filtered by an user-supplied predicate.
template <typename Set, typename Pred = PrintAll<typename Set::value_type>>
class SetPrinter {
public:
  SetPrinter(const Set &S, Pred ShouldPrint = Pred())
      : S(S), ShouldPrint(std::move(ShouldPrint)) {}

  void printTo(llvm::raw_ostream &OS) const {
    bool PrintComma = false;
    OS << "{";
    for (auto &E : S) {
      if (ShouldPrint(E)) {
        if (PrintComma)
          OS << ',';
        OS << ' ' << E;
        PrintComma = true;
      }
    }
    OS << " }";
  }

private:
  const Set &S;
  mutable Pred ShouldPrint;
};

template <typename Set, typename Pred>
SetPrinter<Set, Pred> printSet(const Set &S, Pred P = Pred()) {
  return SetPrinter<Set, Pred>(S, std::move(P));
}

// Render a SetPrinter by delegating to its printTo method.
template <typename Set, typename Pred>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const SetPrinter<Set, Pred> &Printer) {
  Printer.printTo(OS);
  return OS;
}

struct PrintSymbolFlagsMapElemsMatchingCLOpts {
  bool operator()(const orc::SymbolFlagsMap::value_type &KV) {
    return flagsMatchCLOpts(KV.second);
  }
};

struct PrintSymbolMapElemsMatchingCLOpts {
  bool operator()(const orc::SymbolMap::value_type &KV) {
    return flagsMatchCLOpts(KV.second.getFlags());
  }
};

} // end anonymous namespace

namespace llvm {
namespace orc {

char FailedToMaterialize::ID = 0;
char SymbolsNotFound::ID = 0;
char SymbolsCouldNotBeRemoved::ID = 0;

raw_ostream &operator<<(raw_ostream &OS, const SymbolStringPtr &Sym) {
  return OS << *Sym;
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolNameSet &Symbols) {
  return OS << printSet(Symbols, PrintAll<SymbolStringPtr>());
}

raw_ostream &operator<<(raw_ostream &OS, const JITSymbolFlags &Flags) {
  if (Flags.isCallable())
    OS << "[Callable]";
  else
    OS << "[Data]";
  if (Flags.isWeak())
    OS << "[Weak]";
  else if (Flags.isCommon())
    OS << "[Common]";

  if (!Flags.isExported())
    OS << "[Hidden]";

  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const JITEvaluatedSymbol &Sym) {
  return OS << format("0x%016" PRIx64, Sym.getAddress()) << " "
            << Sym.getFlags();
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolFlagsMap::value_type &KV) {
  return OS << "(\"" << KV.first << "\", " << KV.second << ")";
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolMap::value_type &KV) {
  return OS << "(\"" << KV.first << "\": " << KV.second << ")";
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolFlagsMap &SymbolFlags) {
  return OS << printSet(SymbolFlags, PrintSymbolFlagsMapElemsMatchingCLOpts());
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolMap &Symbols) {
  return OS << printSet(Symbols, PrintSymbolMapElemsMatchingCLOpts());
}

raw_ostream &operator<<(raw_ostream &OS,
                        const SymbolDependenceMap::value_type &KV) {
  return OS << "(" << KV.first << ", " << KV.second << ")";
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolDependenceMap &Deps) {
  return OS << printSet(Deps, PrintAll<SymbolDependenceMap::value_type>());
}

raw_ostream &operator<<(raw_ostream &OS, const MaterializationUnit &MU) {
  OS << "MU@" << &MU << " (\"" << MU.getName() << "\"";
  if (anyPrintSymbolOptionSet())
    OS << ", " << MU.getSymbols();
  return OS << ")";
}

raw_ostream &operator<<(raw_ostream &OS, const JITDylibSearchList &JDs) {
  OS << "[";
  if (!JDs.empty()) {
    assert(JDs.front().first && "JITDylibList entries must not be null");
    OS << " (\"" << JDs.front().first->getName() << "\", "
       << (JDs.front().second ? "true" : "false") << ")";
    for (auto &KV : make_range(std::next(JDs.begin()), JDs.end())) {
      assert(KV.first && "JITDylibList entries must not be null");
      OS << ", (\"" << KV.first->getName() << "\", "
         << (KV.second ? "true" : "false") << ")";
    }
  }
  OS << " ]";
  return OS;
}

FailedToMaterialize::FailedToMaterialize(SymbolNameSet Symbols)
    : Symbols(std::move(Symbols)) {
  assert(!this->Symbols.empty() && "Can not fail to resolve an empty set");
}

std::error_code FailedToMaterialize::convertToErrorCode() const {
  return orcError(OrcErrorCode::UnknownORCError);
}

void FailedToMaterialize::log(raw_ostream &OS) const {
  OS << "Failed to materialize symbols: " << Symbols;
}

SymbolsNotFound::SymbolsNotFound(SymbolNameSet Symbols)
    : Symbols(std::move(Symbols)) {
  assert(!this->Symbols.empty() && "Can not fail to resolve an empty set");
}

std::error_code SymbolsNotFound::convertToErrorCode() const {
  return orcError(OrcErrorCode::UnknownORCError);
}

void SymbolsNotFound::log(raw_ostream &OS) const {
  OS << "Symbols not found: " << Symbols;
}

SymbolsCouldNotBeRemoved::SymbolsCouldNotBeRemoved(SymbolNameSet Symbols)
    : Symbols(std::move(Symbols)) {
  assert(!this->Symbols.empty() && "Can not fail to resolve an empty set");
}

std::error_code SymbolsCouldNotBeRemoved::convertToErrorCode() const {
  return orcError(OrcErrorCode::UnknownORCError);
}

void SymbolsCouldNotBeRemoved::log(raw_ostream &OS) const {
  OS << "Symbols could not be removed: " << Symbols;
}

} // End namespace orc.
} // End namespace llvm.
