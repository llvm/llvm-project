//===--- BinaryAnalysisUtils.h - LLVM Advisor ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for binary analyzers that open object files or DWARF contexts.
// Follows the same composable pattern as IRAnalysisUtils.h.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Object/ObjectFile.h"

namespace llvm::advisor {

/// Open an object file from a path.  Returns an error if the file cannot be
/// read or is not a recognized object format.
Expected<object::OwningBinary<object::ObjectFile>>
openObjectFile(StringRef Path);

/// Run Analyze over the object file described by Context.  If the object
/// artifact is missing, returns an unavailable result.  Parse errors are
/// forwarded as-is.  The temporary OwningBinary is kept alive for the
/// duration of the call.
template <typename Func>
Expected<std::unique_ptr<CapabilityResult>>
withObjectFile(const CapabilityContext &Context, StringRef CapabilityID,
               StringRef UnitID, Func &&Analyze) {
  if (Context.ObjectPath.empty())
    return makeUnavailableResult(CapabilityID, UnitID, "missing object artifact");
  Expected<object::OwningBinary<object::ObjectFile>> Obj =
      object::ObjectFile::createObjectFile(Context.ObjectPath);
  if (!Obj)
    return Obj.takeError();
  return std::forward<Func>(Analyze)(*Obj->getBinary());
}

/// Same as withObjectFile, but also creates a DWARFContext and passes it
/// to Analyze.  Returns an unavailable result if the object path is missing,
/// and forwards any object-file or DWARF creation errors.
template <typename Func>
Expected<std::unique_ptr<CapabilityResult>>
withDWARFContext(const CapabilityContext &Context, StringRef CapabilityID,
                 StringRef UnitID, Func &&Analyze) {
  return withObjectFile(
      Context, CapabilityID, UnitID,
      [&](const object::ObjectFile &Obj) -> Expected<std::unique_ptr<CapabilityResult>> {
        std::unique_ptr<DWARFContext> DW = DWARFContext::create(Obj);
        if (!DW)
          return createStringError(inconvertibleErrorCode(), "no DWARF data");
        return std::forward<Func>(Analyze)(*DW);
      });
}

} // namespace llvm::advisor
