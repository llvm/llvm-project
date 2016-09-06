//===- Linker.h - Module Linker Interface -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKER_LINKER_H
#define LLVM_LINKER_LINKER_H

#include "llvm/Linker/IRMover.h"

namespace llvm {
class Module;
class StructType;
class Type;

/// This class provides the core functionality of linking in LLVM. It keeps a
/// pointer to the merged module so far. It doesn't take ownership of the
/// module since it is assumed that the user of this class will want to do
/// something with it after the linking.
class Linker {
  IRMover Mover;

public:
  enum Flags {
    None = 0,
    OverrideFromSrc = (1 << 0),
    LinkOnlyNeeded = (1 << 1),
    InternalizeLinkedSymbols = (1 << 2),
    /// Don't force link referenced linkonce definitions, import declaration.
    DontForceLinkLinkonceODR = (1 << 3)

  };

  Linker(Module &M);

  /// \brief Link \p Src into the composite.
  ///
  /// Passing OverrideSymbols as true will have symbols from Src
  /// shadow those in the Dest.
  /// For ThinLTO function importing/exporting the \p ModuleSummaryIndex
  /// is passed. If \p GlobalsToImport is provided, only the globals that
  /// are part of the set will be imported from the source module.
  ///
  /// Returns true on error.
  bool linkInModule(std::unique_ptr<Module> Src, unsigned Flags = Flags::None,
                    DenseSet<const GlobalValue *> *GlobalsToImport = nullptr);

  static bool linkModules(Module &Dest, std::unique_ptr<Module> Src,
                          unsigned Flags = Flags::None);
};

} // End llvm namespace

#endif
