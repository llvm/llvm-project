//===--------- Core.cpp - Core ORC APIs (SymbolSource, VSO, etc.) ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/OrcError.h"

namespace llvm {
namespace orc {

AsynchronousSymbolQuery::AsynchronousSymbolQuery(
                                  const SymbolNameSet &Symbols,
                                  SymbolsResolvedCallback NotifySymbolsResolved,
                                  SymbolsReadyCallback NotifySymbolsReady)
    : NotifySymbolsResolved(std::move(NotifySymbolsResolved)),
      NotifySymbolsReady(std::move(NotifySymbolsReady)) {
  assert(this->NotifySymbolsResolved &&
         "Symbols resolved callback must be set");
  assert(this->NotifySymbolsReady && "Symbols ready callback must be set");
  OutstandingResolutions = OutstandingFinalizations = Symbols.size();
}

void AsynchronousSymbolQuery::setFailed(Error Err) {
  OutstandingResolutions = OutstandingFinalizations = 0;
  if (NotifySymbolsResolved)
    NotifySymbolsResolved(std::move(Err));
  else
    NotifySymbolsReady(std::move(Err));
}

void AsynchronousSymbolQuery::setDefinition(SymbolStringPtr Name,
                                            JITSymbol Sym) {
  // If OutstandingResolutions is zero we must have errored out already. Just
  // ignore this.
  if (OutstandingResolutions == 0)
    return;

  assert(NotifySymbolsResolved && "Notify callback not set");

  errs()
    << "OutstandingResolutions = " << OutstandingResolutions << "\n"
    << "OutstandingFinalizations = " << OutstandingFinalizations << "\n"
    << "Symbols.size() = " << Symbols.size() << "\n"
    << "Symbols.count(Name) = " << Symbols.count(Name) << "\n";

  assert(!Symbols.count(Name) &&
         "Symbol has already been assigned an address");
  errs() << "Past assert\n";
  Symbols.insert(std::make_pair(std::move(Name), std::move(Sym)));
  errs() << "Past insert\n";
  --OutstandingResolutions;
  errs() << "Past subtract\n";
  if (OutstandingResolutions == 0) {
    errs() << "Past test\n";
    NotifySymbolsResolved(std::move(Symbols));
    // Null out NotifySymbolsResolved to indicate that we've already called it.
    errs() << "Past callback\n";
    NotifySymbolsResolved = {};
    errs() << "Past callback-reset\n";
  }
}

void AsynchronousSymbolQuery::notifySymbolFinalized() {
  // If OutstandingFinalizations is zero we must have errored out already. Just
  // ignore this.
  if (OutstandingFinalizations == 0)
    return;

  assert(OutstandingFinalizations > 0 && "All symbols already finalized");
  --OutstandingFinalizations;
  if (OutstandingFinalizations == 0)
    NotifySymbolsReady(Error::success());
}

} // End namespace orc.
} // End namespace llvm.
