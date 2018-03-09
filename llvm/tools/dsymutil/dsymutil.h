//===- tools/dsymutil/dsymutil.h - dsymutil high-level functionality ------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// This file contains the class declaration for the code that parses STABS
/// debug maps that are embedded in the binaries symbol tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_DSYMUTIL_DSYMUTIL_H
#define LLVM_TOOLS_DSYMUTIL_DSYMUTIL_H

#include "DebugMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorOr.h"
#include <memory>
#include <string>
#include <vector>

namespace llvm {
namespace dsymutil {

struct LinkOptions {
  /// Verbosity
  bool Verbose = false;

  /// Skip emitting output
  bool NoOutput = false;

  /// Do not unique types according to ODR
  bool NoODR = false;

  /// Update
  bool Update = false;

  /// Minimize
  bool Minimize = false;

  /// Do not check swiftmodule timestamp
  bool NoTimestamp = false;

  /// -oso-prepend-path
  std::string PrependPath;

  LinkOptions() = default;
};

/// Extract the DebugMaps from the given file.
/// The file has to be a MachO object file. Multiple debug maps can be
/// returned when the file is universal (aka fat) binary.
ErrorOr<std::vector<std::unique_ptr<DebugMap>>>
parseDebugMap(StringRef InputFile, ArrayRef<std::string> Archs,
              StringRef PrependPath, bool Verbose, bool InputIsYAML);

/// Dump the symbol table
bool dumpStab(StringRef InputFile, ArrayRef<std::string> Archs,
              StringRef PrependPath = "");

/// Link the Dwarf debug info as directed by the passed DebugMap \p DM into a
/// DwarfFile named \p OutputFilename. \returns false if the link failed.
bool linkDwarf(raw_fd_ostream &OutFile, const DebugMap &DM,
               const LinkOptions &Options);

void warn(Twine Warning, Twine Context = {});
bool error(Twine Error, Twine Context = {});

} // end namespace dsymutil
} // end namespace llvm

#endif // LLVM_TOOLS_DSYMUTIL_DSYMUTIL_H
