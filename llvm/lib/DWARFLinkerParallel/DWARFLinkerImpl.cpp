//=== DWARFLinkerImpl.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFLinkerImpl.h"

namespace llvm {
namespace dwarflinker_parallel {

/// Similar to DWARFUnitSection::getUnitForOffset(), but returning our
/// CompileUnit object instead.
CompileUnit *
DWARFLinkerImpl::LinkContext::getUnitForOffset(CompileUnit &CurrentCU,
                                               uint64_t Offset) const {
  if (CurrentCU.isClangModule())
    return &CurrentCU;

  auto CU = llvm::upper_bound(
      CompileUnits, Offset,
      [](uint64_t LHS, const std::unique_ptr<CompileUnit> &RHS) {
        return LHS < RHS->getOrigUnit().getNextUnitOffset();
      });

  return CU != CompileUnits.end() ? CU->get() : nullptr;
}

Error DWARFLinkerImpl::createEmitter(const Triple &TheTriple,
                                     OutputFileType FileType,
                                     raw_pwrite_stream &OutFile) {

  TheDwarfEmitter = std::make_unique<DwarfEmitterImpl>(
      FileType, OutFile, OutputStrings.getTranslator(), WarningHandler);

  return TheDwarfEmitter->init(TheTriple, "__DWARF");
}

ExtraDwarfEmitter *DWARFLinkerImpl::getEmitter() {
  return TheDwarfEmitter.get();
}

} // end of namespace dwarflinker_parallel
} // namespace llvm
