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

#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::pdb;

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
    std::sort(PI.begin(), PI.end(), Comparator);
    std::sort(QI.begin(), QI.end(), Comparator);

    decltype(PI) OnlyP;
    decltype(QI) OnlyQ;
    decltype(PI) Common;

    OnlyP.reserve(P.size());
    OnlyQ.reserve(Q.size());
    Common.reserve(Q.size());

    auto PEnd = std::set_difference(PI.begin(), PI.end(), QI.begin(), QI.end(),
                                    OnlyP.begin(), Comparator);
    auto QEnd = std::set_difference(QI.begin(), QI.end(), PI.begin(), PI.end(),
                                    OnlyQ.begin(), Comparator);
    auto ComEnd = std::set_intersection(PI.begin(), PI.end(), QI.begin(),
                                        QI.end(), Common.begin(), Comparator);
    OnlyP.set_size(std::distance(OnlyP.begin(), PEnd));
    OnlyQ.set_size(std::distance(OnlyQ.begin(), QEnd));
    Common.set_size(std::distance(Common.begin(), ComEnd));

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

Error DiffStyle::diffStringTable() { return Error::success(); }

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
