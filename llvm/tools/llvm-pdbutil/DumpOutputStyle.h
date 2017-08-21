//===- DumpOutputStyle.h -------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_DUMPOUTPUTSTYLE_H
#define LLVM_TOOLS_LLVMPDBDUMP_DUMPOUTPUTSTYLE_H

#include "LinePrinter.h"
#include "OutputStyle.h"
#include "StreamUtil.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"

#include <string>

namespace llvm {
class BitVector;

namespace codeview {
class LazyRandomTypeCollection;
}

namespace pdb {
class GSIHashTable;

struct StatCollection {
  struct Stat {
    Stat() {}
    Stat(uint32_t Count, uint32_t Size) : Count(Count), Size(Size) {}
    uint32_t Count = 0;
    uint32_t Size = 0;

    void update(uint32_t RecordSize) {
      ++Count;
      Size += RecordSize;
    }
  };

  void update(uint32_t Kind, uint32_t RecordSize) {
    Totals.update(RecordSize);
    auto Iter = Individual.try_emplace(Kind, 1, RecordSize);
    if (!Iter.second)
      Iter.first->second.update(RecordSize);
  }
  Stat Totals;
  DenseMap<uint32_t, Stat> Individual;
};

class DumpOutputStyle : public OutputStyle {

public:
  DumpOutputStyle(PDBFile &File);

  Error dump() override;

private:
  Expected<codeview::LazyRandomTypeCollection &> initializeTypes(uint32_t SN);

  Error dumpFileSummary();
  Error dumpStreamSummary();
  Error dumpModuleStats();
  Error dumpStringTable();
  Error dumpLines();
  Error dumpInlineeLines();
  Error dumpXmi();
  Error dumpXme();
  Error dumpTpiStream(uint32_t StreamIdx);
  Error dumpModules();
  Error dumpModuleFiles();
  Error dumpModuleSyms();
  Error dumpGlobals();
  Error dumpPublics();
  Error dumpSymbolsFromGSI(const GSIHashTable &Table, bool HashExtras);
  Error dumpSectionHeaders();
  Error dumpSectionContribs();
  Error dumpSectionMap();

  void dumpSectionHeaders(StringRef Label, DbgHeaderType Type);

  PDBFile &File;
  LinePrinter P;
  std::unique_ptr<codeview::LazyRandomTypeCollection> TpiTypes;
  std::unique_ptr<codeview::LazyRandomTypeCollection> IpiTypes;
  SmallVector<StreamInfo, 32> StreamPurposes;
};
} // namespace pdb
} // namespace llvm

#endif
