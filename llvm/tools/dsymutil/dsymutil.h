//===- tools/dsymutil/dsymutil.h - dsymutil high-level functionality ------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// This file contains the class declaration for the code that parses STABS
/// debug maps that are embedded in the binaries symbol tables.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_DSYMUTIL_DSYMUTIL_H
#define LLVM_TOOLS_DSYMUTIL_DSYMUTIL_H

#include "DebugMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include <memory>

namespace llvm {
namespace dsymutil {

struct LinkOptions {
  bool Verbose;            ///< Verbosity
  bool NoOutput;           ///< Skip emitting output
  bool NoODR;              ///< Do not unique types according to ODR
  bool NoTimestamp;        ///< Do not check swiftmodule timestamp
  std::string PrependPath; ///< -oso-prepend-path

  LinkOptions() : Verbose(false), NoOutput(false), NoTimestamp(false) {}
};

/// \brief Extract the DebugMaps from the given file.
/// The file has to be a MachO object file. Multiple debug maps can be
/// returned when the file is universal (aka fat) binary.
llvm::ErrorOr<std::vector<std::unique_ptr<DebugMap>>>
parseDebugMap(StringRef InputFile, ArrayRef<std::string> Archs,
              StringRef PrependPath, bool Verbose, bool InputIsYAML);

/// \brief Dump the symbol table
bool dumpStab(StringRef InputFile, ArrayRef<std::string> Archs,
              StringRef PrependPath = "");

/// \brief Link the Dwarf debuginfo as directed by the passed DebugMap
/// \p DM into a DwarfFile named \p OutputFilename.
/// \returns false if the link failed.
bool linkDwarf(StringRef OutputFilename, const DebugMap &DM,
               const LinkOptions &Options);

/// \brief Exit the dsymutil process, cleaning up every temporary
/// files that we created.
LLVM_ATTRIBUTE_NORETURN void exitDsymutil(int ExitStatus);

void warn(const Twine &Warning, const Twine &Context);
bool error(const Twine &Error, const Twine &Context);
}
}
#endif // LLVM_TOOLS_DSYMUTIL_DSYMUTIL_H
