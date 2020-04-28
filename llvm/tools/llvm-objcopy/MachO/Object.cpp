//===- Object.cpp - Mach-O object file model --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Object.h"
#include "../llvm-objcopy.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <unordered_set>

namespace llvm {
namespace objcopy {
namespace macho {

const SymbolEntry *SymbolTable::getSymbolByIndex(uint32_t Index) const {
  assert(Index < Symbols.size() && "invalid symbol index");
  return Symbols[Index].get();
}

SymbolEntry *SymbolTable::getSymbolByIndex(uint32_t Index) {
  return const_cast<SymbolEntry *>(
      static_cast<const SymbolTable *>(this)->getSymbolByIndex(Index));
}

void SymbolTable::removeSymbols(
    function_ref<bool(const std::unique_ptr<SymbolEntry> &)> ToRemove) {
  Symbols.erase(
      std::remove_if(std::begin(Symbols), std::end(Symbols), ToRemove),
      std::end(Symbols));
}

Error Object::removeSections(
    function_ref<bool(const std::unique_ptr<Section> &)> ToRemove) {
  DenseMap<uint32_t, const Section *> OldIndexToSection;
  uint32_t NextSectionIndex = 1;
  for (LoadCommand &LC : LoadCommands) {
    auto It = std::stable_partition(
        std::begin(LC.Sections), std::end(LC.Sections),
        [&](const std::unique_ptr<Section> &Sec) { return !ToRemove(Sec); });
    for (auto I = LC.Sections.begin(), End = It; I != End; ++I) {
      OldIndexToSection[(*I)->Index] = I->get();
      (*I)->Index = NextSectionIndex++;
    }
    LC.Sections.erase(It, LC.Sections.end());
  }

  auto IsDead = [&](const std::unique_ptr<SymbolEntry> &S) -> bool {
    Optional<uint32_t> Section = S->section();
    return (Section && !OldIndexToSection.count(*Section));
  };

  SmallPtrSet<const SymbolEntry *, 2> DeadSymbols;
  for (const std::unique_ptr<SymbolEntry> &Sym : SymTable.Symbols)
    if (IsDead(Sym))
      DeadSymbols.insert(Sym.get());

  for (const LoadCommand &LC : LoadCommands)
    for (const std::unique_ptr<Section> &Sec : LC.Sections)
      for (const RelocationInfo &R : Sec->Relocations)
        if (R.Symbol && *R.Symbol && DeadSymbols.count(*R.Symbol))
          return createStringError(std::errc::invalid_argument,
                                   "symbol '%s' defined in section with index "
                                   "'%u' cannot be removed because it is "
                                   "referenced by a relocation in section '%s'",
                                   (*R.Symbol)->Name.c_str(),
                                   *((*R.Symbol)->section()),
                                   Sec->CanonicalName.c_str());
  SymTable.removeSymbols(IsDead);
  for (std::unique_ptr<SymbolEntry> &S : SymTable.Symbols)
    if (S->section())
      S->n_sect = OldIndexToSection[S->n_sect]->Index;
  return Error::success();
}

void Object::addLoadCommand(LoadCommand LC) {
  LoadCommands.push_back(std::move(LC));
}

template <typename SegmentType>
static void constructSegment(SegmentType &Seg,
                             llvm::MachO::LoadCommandType CmdType,
                             StringRef SegName) {
  assert(SegName.size() <= sizeof(Seg.segname) && "too long segment name");
  memset(&Seg, 0, sizeof(SegmentType));
  Seg.cmd = CmdType;
  strncpy(Seg.segname, SegName.data(), SegName.size());
}

LoadCommand &Object::addSegment(StringRef SegName) {
  LoadCommand LC;
  if (is64Bit())
    constructSegment(LC.MachOLoadCommand.segment_command_64_data,
                     MachO::LC_SEGMENT_64, SegName);
  else
    constructSegment(LC.MachOLoadCommand.segment_command_data,
                     MachO::LC_SEGMENT, SegName);

  LoadCommands.push_back(std::move(LC));
  return LoadCommands.back();
}

/// Extracts a segment name from a string which is possibly non-null-terminated.
static StringRef extractSegmentName(const char *SegName) {
  return StringRef(SegName,
                   strnlen(SegName, sizeof(MachO::segment_command::segname)));
}

Optional<StringRef> LoadCommand::getSegmentName() const {
  const MachO::macho_load_command &MLC = MachOLoadCommand;
  switch (MLC.load_command_data.cmd) {
  case MachO::LC_SEGMENT:
    return extractSegmentName(MLC.segment_command_data.segname);
  case MachO::LC_SEGMENT_64:
    return extractSegmentName(MLC.segment_command_64_data.segname);
  default:
    return None;
  }
}

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
