//===- Diff.cpp - PDB diff utility ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Diff.h"

#include "StreamUtil.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/StringTable.h"

#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::pdb;

template<typename R>
using ValueOfRange = llvm::detail::ValueOfRange<R>;

template<typename Range, typename Comp>
static void set_differences(Range &&R1, Range &&R2,
    SmallVectorImpl<ValueOfRange<Range>> *OnlyLeft,
    SmallVectorImpl<ValueOfRange<Range>> *OnlyRight,
    SmallVectorImpl<ValueOfRange<Range>> *Intersection, Comp Comparator) {

  std::sort(R1.begin(), R1.end(), Comparator);
  std::sort(R2.begin(), R2.end(), Comparator);

  if (OnlyLeft) {
    OnlyLeft->reserve(R1.size());
    auto End = std::set_difference(R1.begin(), R1.end(), R2.begin(), R2.end(),
      OnlyLeft->begin(), Comparator);
    OnlyLeft->set_size(std::distance(OnlyLeft->begin(), End));
  }
  if (OnlyRight) {
    OnlyLeft->reserve(R2.size());
    auto End = std::set_difference(R2.begin(), R2.end(), R1.begin(), R1.end(),
      OnlyRight->begin(), Comparator);
    OnlyRight->set_size(std::distance(OnlyRight->begin(), End));
  }
  if (Intersection) {
    Intersection->reserve(std::min(R1.size(), R2.size()));
    auto End = std::set_intersection(R1.begin(), R1.end(), R2.begin(),
      R2.end(), Intersection->begin(), Comparator);
    Intersection->set_size(std::distance(Intersection->begin(), End));
  }
}

template <typename Range>
static void
set_differences(Range &&R1, Range &&R2,
                SmallVectorImpl<ValueOfRange<Range>> *OnlyLeft,
                SmallVectorImpl<ValueOfRange<Range>> *OnlyRight,
                SmallVectorImpl<ValueOfRange<Range>> *Intersection = nullptr) {
  std::less<ValueOfRange<Range>> Comp;
  set_differences(std::forward<Range>(R1), std::forward<Range>(R2), OnlyLeft, OnlyRight, Intersection, Comp);
}


DiffStyle::DiffStyle(PDBFile &File1, PDBFile &File2)
    : File1(File1), File2(File2) {}

Error DiffStyle::dump() {
  if (auto EC = diffSuperBlock())
    return EC;

  if (auto EC = diffFreePageMap())
    return EC;

  if (auto EC = diffStreamDirectory())
    return EC;

  if (auto EC = diffStringTable())
    return EC;

  if (auto EC = diffInfoStream())
    return EC;

  if (auto EC = diffDbiStream())
    return EC;

  if (auto EC = diffSectionContribs())
    return EC;

  if (auto EC = diffSectionMap())
    return EC;

  if (auto EC = diffFpoStream())
    return EC;

  if (auto EC = diffTpiStream(StreamTPI))
    return EC;

  if (auto EC = diffTpiStream(StreamIPI))
    return EC;

  if (auto EC = diffPublics())
    return EC;

  if (auto EC = diffGlobals())
    return EC;

  return Error::success();
}

template <typename T>
static bool diffAndPrint(StringRef Label, PDBFile &File1, PDBFile &File2, T V1,
                         T V2) {
  if (V1 != V2) {
    outs().indent(2) << Label << "\n";
    outs().indent(4) << formatv("{0}: {1}\n", File1.getFilePath(), V1);
    outs().indent(4) << formatv("{0}: {1}\n", File2.getFilePath(), V2);
  }
  return (V1 != V2);
}

Error DiffStyle::diffSuperBlock() {
  outs() << "MSF Super Block: Searching for differences...\n";
  bool Diffs = false;

  Diffs |= diffAndPrint("Block Size", File1, File2, File1.getBlockSize(),
                        File2.getBlockSize());
  Diffs |= diffAndPrint("Block Count", File1, File2, File1.getBlockCount(),
                        File2.getBlockCount());
  Diffs |= diffAndPrint("Unknown 1", File1, File2, File1.getUnknown1(),
                        File2.getUnknown1());

  if (opts::diff::Pedantic) {
    Diffs |= diffAndPrint("Free Block Map", File1, File2,
                          File1.getFreeBlockMapBlock(),
                          File2.getFreeBlockMapBlock());
    Diffs |= diffAndPrint("Directory Size", File1, File2,
                          File1.getNumDirectoryBytes(),
                          File2.getNumDirectoryBytes());
    Diffs |= diffAndPrint("Block Map Addr", File1, File2,
                          File1.getBlockMapOffset(), File2.getBlockMapOffset());
  }
  if (!Diffs)
    outs() << "MSF Super Block: No differences detected...\n";
  return Error::success();
}

Error DiffStyle::diffStreamDirectory() {
  SmallVector<std::string, 32> P;
  SmallVector<std::string, 32> Q;
  discoverStreamPurposes(File1, P);
  discoverStreamPurposes(File2, Q);
  outs() << "Stream Directory: Searching for differences...\n";

  bool HasDifferences = false;
  if (opts::diff::Pedantic) {
    size_t Min = std::min(P.size(), Q.size());
    for (size_t I = 0; I < Min; ++I) {
      StringRef Names[] = {P[I], Q[I]};
      uint32_t Sizes[] = {File1.getStreamByteSize(I),
                          File2.getStreamByteSize(I)};
      bool NamesDiffer = Names[0] != Names[1];
      bool SizesDiffer = Sizes[0] != Sizes[1];
      if (NamesDiffer) {
        HasDifferences = true;
        outs().indent(2) << formatv("Stream {0} - {1}: {2}, {3}: {4}\n", I,
                                    File1.getFilePath(), Names[0],
                                    File2.getFilePath(), Names[1]);
        continue;
      }
      if (SizesDiffer) {
        HasDifferences = true;
        outs().indent(2) << formatv(
            "Stream {0} ({1}): {2}: {3} bytes, {4}: {5} bytes\n", I, Names[0],
            File1.getFilePath(), Sizes[0], File2.getFilePath(), Sizes[1]);
        continue;
      }
    }

    ArrayRef<std::string> MaxNames = (P.size() > Q.size() ? P : Q);
    size_t Max = std::max(P.size(), Q.size());
    PDBFile &MaxFile = (P.size() > Q.size() ? File1 : File2);
    StringRef MinFileName =
        (P.size() < Q.size() ? File1.getFilePath() : File2.getFilePath());
    for (size_t I = Min; I < Max; ++I) {
      HasDifferences = true;
      StringRef StreamName = MaxNames[I];

      outs().indent(2) << formatv(
          "Stream {0} - {1}: <not present>, {2}: Index {3}, {4} bytes\n",
          StreamName, MinFileName, MaxFile.getFilePath(), I,
          MaxFile.getStreamByteSize(I));
    }
    if (!HasDifferences)
      outs() << "Stream Directory: No differences detected...\n";
  } else {
    auto PI = to_vector<32>(enumerate(P));
    auto QI = to_vector<32>(enumerate(Q));

    typedef decltype(PI) ContainerType;
    typedef typename ContainerType::value_type value_type;

    auto Comparator = [](const value_type &I1, const value_type &I2) {
      return I1.value() < I2.value();
    };

    decltype(PI) OnlyP;
    decltype(QI) OnlyQ;
    decltype(PI) Common;

    set_differences(PI, QI, &OnlyP, &OnlyQ, &Common, Comparator);

    if (!OnlyP.empty()) {
      HasDifferences = true;
      outs().indent(2) << formatv("{0} Stream(s) only in ({1})\n", OnlyP.size(),
                                  File1.getFilePath());
      for (auto &Item : OnlyP) {
        outs().indent(4) << formatv("Stream {0} - {1}\n", Item.index(),
                                    Item.value());
      }
    }

    if (!OnlyQ.empty()) {
      HasDifferences = true;
      outs().indent(2) << formatv("{0} Streams(s) only in ({1})\n",
                                  OnlyQ.size(), File2.getFilePath());
      for (auto &Item : OnlyQ) {
        outs().indent(4) << formatv("Stream {0} - {1}\n", Item.index(),
                                    Item.value());
      }
    }
    if (!Common.empty()) {
      outs().indent(2) << formatv("Found {0} common streams.  Searching for "
                                  "intra-stream differences.\n",
                                  Common.size());
      bool HasCommonDifferences = false;
      for (const auto &Left : Common) {
        // Left was copied from the first range so its index refers to a stream
        // index in the first file.  Find the corresponding stream index in the
        // second file.
        auto Range =
            std::equal_range(QI.begin(), QI.end(), Left,
                             [](const value_type &L, const value_type &R) {
                               return L.value() < R.value();
                             });
        const auto &Right = *Range.first;
        assert(Left.value() == Right.value());
        uint32_t LeftSize = File1.getStreamByteSize(Left.index());
        uint32_t RightSize = File2.getStreamByteSize(Right.index());
        if (LeftSize != RightSize) {
          HasDifferences = true;
          HasCommonDifferences = true;
          outs().indent(4) << formatv("{0} ({1}: {2} bytes, {3}: {4} bytes)\n",
                                      Left.value(), File1.getFilePath(),
                                      LeftSize, File2.getFilePath(), RightSize);
        }
      }
      if (!HasCommonDifferences)
        outs().indent(2) << "Common Streams:  No differences detected!\n";
    }
    if (!HasDifferences)
      outs() << "Stream Directory: No differences detected!\n";
  }

  return Error::success();
}

Error DiffStyle::diffStringTable() {
  auto ExpectedST1 = File1.getStringTable();
  auto ExpectedST2 = File2.getStringTable();
  outs() << "String Table: Searching for differences...\n";
  bool Has1 = !!ExpectedST1;
  bool Has2 = !!ExpectedST2;
  if (!(Has1 && Has2)) {
    // If one has a string table and the other doesn't, we can print less output.
    if (Has1 != Has2) {
      if (Has1) {
        outs() << formatv("  {0}: ({1} strings)\n", File1.getFilePath(), ExpectedST1->getNameCount());
        outs() << formatv("  {0}: (string table not present)\n", File2.getFilePath());
      } else {
        outs() << formatv("  {0}: (string table not present)\n", File1.getFilePath());
        outs() << formatv("  {0}: ({1})\n", File2.getFilePath(), ExpectedST2->getNameCount());
      }
    }
    consumeError(ExpectedST1.takeError());
    consumeError(ExpectedST2.takeError());
    return Error::success();
  }

  bool HasDiff = false;
  auto &ST1 = *ExpectedST1;
  auto &ST2 = *ExpectedST2;

  HasDiff |= diffAndPrint("Stream Size", File1, File2, ST1.getByteSize(), ST2.getByteSize());
  HasDiff |= diffAndPrint("Hash Version", File1, File2, ST1.getHashVersion(), ST1.getHashVersion());
  HasDiff |= diffAndPrint("Signature", File1, File2, ST1.getSignature(), ST1.getSignature());

  // Both have a valid string table, dive in and compare individual strings.

  auto IdList1 = ST1.name_ids();
  auto IdList2 = ST2.name_ids();
  if (opts::diff::Pedantic) {
    // In pedantic mode, we compare index by index (i.e. the strings are in the same order
    // in both tables.
    uint32_t Max = std::max(IdList1.size(), IdList2.size());
    for (uint32_t I = 0; I < Max; ++I) {
      Optional<uint32_t> Id1, Id2;
      StringRef S1, S2;
      if (I < IdList1.size()) {
        Id1 = IdList1[I];
        S1 = ST1.getStringForID(*Id1);
      }
      if (I < IdList2.size()) {
        Id2 = IdList2[I];
        S2 = ST2.getStringForID(*Id2);
      }
      if (Id1 == Id2 && S1 == S2)
        continue;

      std::string OutId1 = Id1 ? formatv("{0}", *Id1).str() : "(index not present)";
      std::string OutId2 = Id2 ? formatv("{0}", *Id2).str() : "(index not present)";
      outs() << formatv("  String {0}\n", I);
      outs() << formatv("    {0}: Hash - {1}, Value - {2}\n", File1.getFilePath(), OutId1, S1);
      outs() << formatv("    {0}: Hash - {1}, Value - {2}\n", File2.getFilePath(), OutId2, S2);
      HasDiff = true;
    }
  } else {
    std::vector<StringRef> Strings1, Strings2;
    Strings1.reserve(IdList1.size());
    Strings2.reserve(IdList2.size());
    for (auto ID : IdList1)
      Strings1.push_back(ST1.getStringForID(ID));
    for (auto ID : IdList2)
      Strings2.push_back(ST2.getStringForID(ID));

    SmallVector<StringRef, 64> OnlyP;
    SmallVector<StringRef, 64> OnlyQ;

    set_differences(Strings1, Strings2, &OnlyP, &OnlyQ);

    if (!OnlyP.empty()) {
      HasDiff = true;
      outs() << formatv("  {0} String(s) only in ({1})\n",
        OnlyP.size(), File1.getFilePath());
      for (auto Item : OnlyP)
        outs() << formatv("    {2}\n", Item);
    }
    if (!OnlyQ.empty()) {
      HasDiff = true;
      outs() << formatv("  {0} String(s) only in ({1})\n",
        OnlyQ.size(), File2.getFilePath());
      for (auto Item : OnlyQ)
        outs() << formatv("    {2}\n", Item);
    }
  }
  if (!HasDiff)
    outs() << "String Table: No differences detected!\n";
  return Error::success();
}

Error DiffStyle::diffFreePageMap() { return Error::success(); }

Error DiffStyle::diffInfoStream() { return Error::success(); }

Error DiffStyle::diffDbiStream() { return Error::success(); }

Error DiffStyle::diffSectionContribs() { return Error::success(); }

Error DiffStyle::diffSectionMap() { return Error::success(); }

Error DiffStyle::diffFpoStream() { return Error::success(); }

Error DiffStyle::diffTpiStream(int Index) { return Error::success(); }

Error DiffStyle::diffModuleInfoStream(int Index) { return Error::success(); }

Error DiffStyle::diffPublics() { return Error::success(); }

Error DiffStyle::diffGlobals() { return Error::success(); }
