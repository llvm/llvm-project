//===- Diff.cpp - PDB diff utility ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Diff.h"

#include "DiffPrinter.h"
#include "FormatUtil.h"
#include "StreamUtil.h"
#include "llvm-pdbutil.h"

#include "llvm/ADT/StringSet.h"

#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/Formatters.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/PDBStringTable.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"

#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatProviders.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::pdb;

namespace {
// Compare and format two stream numbers.  Stream numbers are considered
// identical if they contain the same value, equivalent if they are both
// the invalid stream or neither is the invalid stream, and different if
// one is the invalid stream and another isn't.
struct StreamNumberProvider {
  static DiffResult compare(uint16_t L, uint16_t R) {
    if (L == R)
      return DiffResult::IDENTICAL;
    bool LP = L != kInvalidStreamIndex;
    bool RP = R != kInvalidStreamIndex;
    if (LP != RP)
      return DiffResult::DIFFERENT;
    return DiffResult::EQUIVALENT;
  }

  static std::string format(uint16_t SN, bool Right) {
    if (SN == kInvalidStreamIndex)
      return "(not present)";
    return formatv("{0}", SN).str();
  }
};

// Compares and formats two module indices.  Modis are considered identical
// if they are identical, equivalent if they either both contain a value or
// both don't contain a value, and different if one contains a value and the
// other doesn't.
struct ModiProvider {
  DiffResult compare(Optional<uint32_t> L, Optional<uint32_t> R) {
    if (L == R)
      return DiffResult::IDENTICAL;
    if (L.hasValue() != R.hasValue())
      return DiffResult::DIFFERENT;
    return DiffResult::EQUIVALENT;
  }

  std::string format(Optional<uint32_t> Modi, bool Right) {
    if (!Modi.hasValue())
      return "(not present)";
    return formatv("{0}", *Modi).str();
  }
};

// Compares and formats two paths embedded in the PDB, ignoring the beginning
// of the path if the user specified it as a "root path" on the command line.
struct BinaryPathProvider {
  explicit BinaryPathProvider(uint32_t MaxLen) : MaxLen(MaxLen) {}

  DiffResult compare(StringRef L, StringRef R) {
    if (L == R)
      return DiffResult::IDENTICAL;

    SmallString<64> LN = removeRoot(L, false);
    SmallString<64> RN = removeRoot(R, true);

    return (LN.equals_lower(RN)) ? DiffResult::EQUIVALENT
                                 : DiffResult::DIFFERENT;
  }

  std::string format(StringRef S, bool Right) {
    if (S.empty())
      return "(empty)";

    SmallString<64> Native = removeRoot(S, Right);
    return truncateStringFront(Native.str(), MaxLen);
  }

  SmallString<64> removeRoot(StringRef Path, bool IsRight) const {
    SmallString<64> Native(Path);
    auto &RootOpt = IsRight ? opts::diff::RightRoot : opts::diff::LeftRoot;
    SmallString<64> Root(static_cast<std::string>(RootOpt));
    // pdb paths always use windows syntax, convert slashes to backslashes.
    sys::path::native(Root, sys::path::Style::windows);
    if (sys::path::has_stem(Root, sys::path::Style::windows))
      sys::path::append(Root, sys::path::Style::windows,
                        sys::path::get_separator(sys::path::Style::windows));

    sys::path::replace_path_prefix(Native, Root, "", sys::path::Style::windows);
    return Native;
  }
  uint32_t MaxLen;
};

// Compare and format two stream purposes.  For general streams, this just
// compares the description.  For module streams it uses the path comparison
// algorithm taking into consideration the binary root, described above.
// Formatting stream purposes just prints the stream purpose, except for
// module streams and named streams, where it prefixes the name / module
// with an identifier.  Example:
//
//   Named Stream "\names"
//   Module Stream "foo.obj"
//
// If a named stream is too long to fit in a column, it is truncated at the
// end, and if a module is too long to fit in a column, it is truncated at the
// beginning.  Example:
//
//  Named Stream "\Really Long Str..."
//  Module Stream "...puts\foo.obj"
//
struct StreamPurposeProvider {
  explicit StreamPurposeProvider(uint32_t MaxLen) : MaxLen(MaxLen) {}

  DiffResult compare(const StreamInfo &L, const StreamInfo &R) {
    if (L.getPurpose() != R.getPurpose())
      return DiffResult::DIFFERENT;
    if (L.getPurpose() == StreamPurpose::ModuleStream) {
      BinaryPathProvider PathProvider(MaxLen);
      return PathProvider.compare(L.getShortName(), R.getShortName());
    }
    return (L.getShortName() == R.getShortName()) ? DiffResult::IDENTICAL
                                                  : DiffResult::DIFFERENT;
  }

  std::string format(const StreamInfo &P, bool Right) {
    if (P.getPurpose() == StreamPurpose::Other ||
        P.getPurpose() == StreamPurpose::Symbols)
      return truncateStringBack(P.getShortName(), MaxLen);
    if (P.getPurpose() == StreamPurpose::NamedStream)
      return truncateQuotedNameBack("Named Stream", P.getShortName(), MaxLen);

    assert(P.getPurpose() == StreamPurpose::ModuleStream);
    uint32_t ExtraChars = strlen("Module \"\"");
    BinaryPathProvider PathProvider(MaxLen - ExtraChars);
    std::string Result = PathProvider.format(P.getShortName(), Right);
    return formatv("Module \"{0}\"", Result);
  }

  uint32_t MaxLen;
};
} // namespace

namespace llvm {
template <> struct format_provider<PdbRaw_FeatureSig> {
  static void format(const PdbRaw_FeatureSig &Sig, raw_ostream &Stream,
                     StringRef Style) {
    switch (Sig) {
    case PdbRaw_FeatureSig::MinimalDebugInfo:
      Stream << "MinimalDebugInfo";
      break;
    case PdbRaw_FeatureSig::NoTypeMerge:
      Stream << "NoTypeMerge";
      break;
    case PdbRaw_FeatureSig::VC110:
      Stream << "VC110";
      break;
    case PdbRaw_FeatureSig::VC140:
      Stream << "VC140";
      break;
    }
  }
};
}

template <typename R> using ValueOfRange = llvm::detail::ValueOfRange<R>;

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

Error DiffStyle::diffSuperBlock() {
  DiffPrinter D(2, "MSF Super Block", 16, 20, opts::diff::PrintResultColumn,
                opts::diff::PrintValueColumns, outs());
  D.printExplicit("File", DiffResult::UNSPECIFIED,
                  truncateStringFront(File1.getFilePath(), 18),
                  truncateStringFront(File2.getFilePath(), 18));
  D.print("Block Size", File1.getBlockSize(), File2.getBlockSize());
  D.print("Block Count", File1.getBlockCount(), File2.getBlockCount());
  D.print("Unknown 1", File1.getUnknown1(), File2.getUnknown1());
  D.print("Directory Size", File1.getNumDirectoryBytes(),
          File2.getNumDirectoryBytes());
  return Error::success();
}

Error DiffStyle::diffStreamDirectory() {
  DiffPrinter D(2, "Stream Directory", 30, 20, opts::diff::PrintResultColumn,
                opts::diff::PrintValueColumns, outs());
  D.printExplicit("File", DiffResult::UNSPECIFIED,
                  truncateStringFront(File1.getFilePath(), 18),
                  truncateStringFront(File2.getFilePath(), 18));

  SmallVector<StreamInfo, 32> P;
  SmallVector<StreamInfo, 32> Q;
  discoverStreamPurposes(File1, P);
  discoverStreamPurposes(File2, Q);
  D.print("Stream Count", File1.getNumStreams(), File2.getNumStreams());
  auto PI = to_vector<32>(enumerate(P));
  auto QI = to_vector<32>(enumerate(Q));

  // Scan all streams in the left hand side, looking for ones that are also
  // in the right.  Each time we find one, remove it.  When we're done, Q
  // should contain all the streams that are in the right but not in the left.
  StreamPurposeProvider StreamProvider(28);
  for (const auto &P : PI) {
    typedef decltype(PI) ContainerType;
    typedef typename ContainerType::value_type value_type;

    auto Iter = llvm::find_if(QI, [P, &StreamProvider](const value_type &V) {
      DiffResult Result = StreamProvider.compare(P.value(), V.value());
      return Result == DiffResult::EQUIVALENT ||
             Result == DiffResult::IDENTICAL;
    });

    if (Iter == QI.end()) {
      D.printExplicit(StreamProvider.format(P.value(), false),
                      DiffResult::DIFFERENT, P.index(), "(not present)");
      continue;
    }

    D.print<EquivalentDiffProvider>(StreamProvider.format(P.value(), false),
                                    P.index(), Iter->index());
    QI.erase(Iter);
  }

  for (const auto &Q : QI) {
    D.printExplicit(StreamProvider.format(Q.value(), true),
                    DiffResult::DIFFERENT, "(not present)", Q.index());
  }

  return Error::success();
}

Error DiffStyle::diffStringTable() {
  DiffPrinter D(2, "String Table", 30, 20, opts::diff::PrintResultColumn,
                opts::diff::PrintValueColumns, outs());
  D.printExplicit("File", DiffResult::UNSPECIFIED,
                  truncateStringFront(File1.getFilePath(), 18),
                  truncateStringFront(File2.getFilePath(), 18));

  auto ExpectedST1 = File1.getStringTable();
  auto ExpectedST2 = File2.getStringTable();
  bool Has1 = !!ExpectedST1;
  bool Has2 = !!ExpectedST2;
  std::string Count1 = Has1 ? llvm::utostr(ExpectedST1->getNameCount())
                            : "(string table not present)";
  std::string Count2 = Has2 ? llvm::utostr(ExpectedST2->getNameCount())
                            : "(string table not present)";
  D.print("Number of Strings", Count1, Count2);

  if (!Has1 || !Has2) {
    consumeError(ExpectedST1.takeError());
    consumeError(ExpectedST2.takeError());
    return Error::success();
  }

  auto &ST1 = *ExpectedST1;
  auto &ST2 = *ExpectedST2;

  D.print("Hash Version", ST1.getHashVersion(), ST2.getHashVersion());
  D.print("Byte Size", ST1.getByteSize(), ST2.getByteSize());
  D.print("Signature", ST1.getSignature(), ST2.getSignature());

  // Both have a valid string table, dive in and compare individual strings.

  auto IdList1 = ST1.name_ids();
  auto IdList2 = ST2.name_ids();
  StringSet<> LS;
  StringSet<> RS;
  uint32_t Empty1 = 0;
  uint32_t Empty2 = 0;
  for (auto ID : IdList1) {
    auto S = ST1.getStringForID(ID);
    if (!S)
      return S.takeError();
    if (S->empty())
      ++Empty1;
    else
      LS.insert(*S);
  }
  for (auto ID : IdList2) {
    auto S = ST2.getStringForID(ID);
    if (!S)
      return S.takeError();
    if (S->empty())
      ++Empty2;
    else
      RS.insert(*S);
  }
  D.print("Empty Strings", Empty1, Empty2);

  for (const auto &S : LS) {
    auto R = RS.find(S.getKey());
    std::string Truncated = truncateStringMiddle(S.getKey(), 28);
    uint32_t I = cantFail(ST1.getIDForString(S.getKey()));
    if (R == RS.end()) {
      D.printExplicit(Truncated, DiffResult::DIFFERENT, I, "(not present)");
      continue;
    }

    uint32_t J = cantFail(ST2.getIDForString(R->getKey()));
    D.print<EquivalentDiffProvider>(Truncated, I, J);
    RS.erase(R);
  }

  for (const auto &S : RS) {
    auto L = LS.find(S.getKey());
    std::string Truncated = truncateStringMiddle(S.getKey(), 28);
    uint32_t J = cantFail(ST2.getIDForString(S.getKey()));
    if (L == LS.end()) {
      D.printExplicit(Truncated, DiffResult::DIFFERENT, "(not present)", J);
      continue;
    }

    uint32_t I = cantFail(ST1.getIDForString(L->getKey()));
    D.print<EquivalentDiffProvider>(Truncated, I, J);
  }
  return Error::success();
}

Error DiffStyle::diffFreePageMap() { return Error::success(); }

Error DiffStyle::diffInfoStream() {
  DiffPrinter D(2, "PDB Stream", 22, 40, opts::diff::PrintResultColumn,
                opts::diff::PrintValueColumns, outs());
  D.printExplicit("File", DiffResult::UNSPECIFIED,
                  truncateStringFront(File1.getFilePath(), 38),
                  truncateStringFront(File2.getFilePath(), 38));

  auto ExpectedInfo1 = File1.getPDBInfoStream();
  auto ExpectedInfo2 = File2.getPDBInfoStream();

  bool Has1 = !!ExpectedInfo1;
  bool Has2 = !!ExpectedInfo2;
  if (!(Has1 && Has2)) {
    std::string L = Has1 ? "(present)" : "(not present)";
    std::string R = Has2 ? "(present)" : "(not present)";
    D.print("Stream", L, R);

    consumeError(ExpectedInfo1.takeError());
    consumeError(ExpectedInfo2.takeError());
    return Error::success();
  }

  auto &IS1 = *ExpectedInfo1;
  auto &IS2 = *ExpectedInfo2;
  D.print("Stream Size", IS1.getStreamSize(), IS2.getStreamSize());
  D.print("Age", IS1.getAge(), IS2.getAge());
  D.print("Guid", IS1.getGuid(), IS2.getGuid());
  D.print("Signature", IS1.getSignature(), IS2.getSignature());
  D.print("Version", IS1.getVersion(), IS2.getVersion());
  D.diffUnorderedArray("Feature", IS1.getFeatureSignatures(),
                       IS2.getFeatureSignatures());
  D.print("Named Stream Size", IS1.getNamedStreamMapByteSize(),
          IS2.getNamedStreamMapByteSize());
  StringMap<uint32_t> NSL = IS1.getNamedStreams().getStringMap();
  StringMap<uint32_t> NSR = IS2.getNamedStreams().getStringMap();
  D.diffUnorderedMap<EquivalentDiffProvider>("Named Stream", NSL, NSR);
  return Error::success();
}

typedef std::pair<uint32_t, DbiModuleDescriptor> IndexedModuleDescriptor;
typedef std::vector<IndexedModuleDescriptor> IndexedModuleDescriptorList;

static IndexedModuleDescriptorList
getModuleDescriptors(const DbiModuleList &ML) {
  IndexedModuleDescriptorList List;
  List.reserve(ML.getModuleCount());
  for (uint32_t I = 0; I < ML.getModuleCount(); ++I)
    List.emplace_back(I, ML.getModuleDescriptor(I));
  return List;
}

static IndexedModuleDescriptorList::iterator
findOverrideEquivalentModule(uint32_t Modi,
                             IndexedModuleDescriptorList &OtherList) {
  auto &EqMap = opts::diff::Equivalences;

  auto Iter = EqMap.find(Modi);
  if (Iter == EqMap.end())
    return OtherList.end();

  uint32_t EqValue = Iter->second;

  return llvm::find_if(OtherList,
                       [EqValue](const IndexedModuleDescriptor &Desc) {
                         return Desc.first == EqValue;
                       });
}

static IndexedModuleDescriptorList::iterator
findEquivalentModule(const IndexedModuleDescriptor &Item,
                     IndexedModuleDescriptorList &OtherList, bool ItemIsRight) {

  if (!ItemIsRight) {
    uint32_t Modi = Item.first;
    auto OverrideIter = findOverrideEquivalentModule(Modi, OtherList);
    if (OverrideIter != OtherList.end())
      return OverrideIter;
  }

  BinaryPathProvider PathProvider(28);

  auto Iter = OtherList.begin();
  auto End = OtherList.end();
  for (; Iter != End; ++Iter) {
    const IndexedModuleDescriptor *Left = &Item;
    const IndexedModuleDescriptor *Right = &*Iter;
    if (ItemIsRight)
      std::swap(Left, Right);
    DiffResult Result = PathProvider.compare(Left->second.getModuleName(),
                                             Right->second.getModuleName());
    if (Result == DiffResult::EQUIVALENT || Result == DiffResult::IDENTICAL)
      return Iter;
  }
  return OtherList.end();
}

static void diffOneModule(DiffPrinter &D, const IndexedModuleDescriptor &Item,
                          IndexedModuleDescriptorList &Other,
                          bool ItemIsRight) {
  StreamPurposeProvider HeaderProvider(70);
  StreamInfo Info = StreamInfo::createModuleStream(
      Item.second.getModuleName(), Item.second.getModuleStreamIndex(),
      Item.first);
  D.printFullRow(HeaderProvider.format(Info, ItemIsRight));

  const auto *L = &Item;

  auto Iter = findEquivalentModule(Item, Other, ItemIsRight);
  if (Iter == Other.end()) {
    // We didn't find this module at all on the other side.  Just print one row
    // and continue.
    if (ItemIsRight)
      D.print<ModiProvider>("- Modi", None, Item.first);
    else
      D.print<ModiProvider>("- Modi", Item.first, None);
    return;
  }

  // We did find this module.  Go through and compare each field.
  const auto *R = &*Iter;
  if (ItemIsRight)
    std::swap(L, R);

  BinaryPathProvider PathProvider(28);
  D.print<ModiProvider>("- Modi", L->first, R->first);
  D.print<BinaryPathProvider>("- Obj File Name", L->second.getObjFileName(),
                              R->second.getObjFileName(), PathProvider);
  D.print<StreamNumberProvider>("- Debug Stream",
                                L->second.getModuleStreamIndex(),
                                R->second.getModuleStreamIndex());
  D.print("- C11 Byte Size", L->second.getC11LineInfoByteSize(),
          R->second.getC11LineInfoByteSize());
  D.print("- C13 Byte Size", L->second.getC13LineInfoByteSize(),
          R->second.getC13LineInfoByteSize());
  D.print("- # of files", L->second.getNumberOfFiles(),
          R->second.getNumberOfFiles());
  D.print("- Pdb File Path Index", L->second.getPdbFilePathNameIndex(),
          R->second.getPdbFilePathNameIndex());
  D.print("- Source File Name Index", L->second.getSourceFileNameIndex(),
          R->second.getSourceFileNameIndex());
  D.print("- Symbol Byte Size", L->second.getSymbolDebugInfoByteSize(),
          R->second.getSymbolDebugInfoByteSize());
  Other.erase(Iter);
}

Error DiffStyle::diffDbiStream() {
  DiffPrinter D(2, "DBI Stream", 40, 30, opts::diff::PrintResultColumn,
                opts::diff::PrintValueColumns, outs());
  D.printExplicit("File", DiffResult::UNSPECIFIED,
                  truncateStringFront(File1.getFilePath(), 28),
                  truncateStringFront(File2.getFilePath(), 28));

  auto ExpectedDbi1 = File1.getPDBDbiStream();
  auto ExpectedDbi2 = File2.getPDBDbiStream();

  bool Has1 = !!ExpectedDbi1;
  bool Has2 = !!ExpectedDbi2;
  if (!(Has1 && Has2)) {
    std::string L = Has1 ? "(present)" : "(not present)";
    std::string R = Has2 ? "(present)" : "(not present)";
    D.print("Stream", L, R);

    consumeError(ExpectedDbi1.takeError());
    consumeError(ExpectedDbi2.takeError());
    return Error::success();
  }

  auto &DL = *ExpectedDbi1;
  auto &DR = *ExpectedDbi2;

  D.print("Dbi Version", (uint32_t)DL.getDbiVersion(),
          (uint32_t)DR.getDbiVersion());
  D.print("Age", DL.getAge(), DR.getAge());
  D.print("Machine", (uint16_t)DL.getMachineType(),
          (uint16_t)DR.getMachineType());
  D.print("Flags", DL.getFlags(), DR.getFlags());
  D.print("Build Major", DL.getBuildMajorVersion(), DR.getBuildMajorVersion());
  D.print("Build Minor", DL.getBuildMinorVersion(), DR.getBuildMinorVersion());
  D.print("Build Number", DL.getBuildNumber(), DR.getBuildNumber());
  D.print("PDB DLL Version", DL.getPdbDllVersion(), DR.getPdbDllVersion());
  D.print("PDB DLL RBLD", DL.getPdbDllRbld(), DR.getPdbDllRbld());
  D.print<StreamNumberProvider>("DBG (FPO)",
                                DL.getDebugStreamIndex(DbgHeaderType::FPO),
                                DR.getDebugStreamIndex(DbgHeaderType::FPO));
  D.print<StreamNumberProvider>(
      "DBG (Exception)", DL.getDebugStreamIndex(DbgHeaderType::Exception),
      DR.getDebugStreamIndex(DbgHeaderType::Exception));
  D.print<StreamNumberProvider>("DBG (Fixup)",
                                DL.getDebugStreamIndex(DbgHeaderType::Fixup),
                                DR.getDebugStreamIndex(DbgHeaderType::Fixup));
  D.print<StreamNumberProvider>(
      "DBG (OmapToSrc)", DL.getDebugStreamIndex(DbgHeaderType::OmapToSrc),
      DR.getDebugStreamIndex(DbgHeaderType::OmapToSrc));
  D.print<StreamNumberProvider>(
      "DBG (OmapFromSrc)", DL.getDebugStreamIndex(DbgHeaderType::OmapFromSrc),
      DR.getDebugStreamIndex(DbgHeaderType::OmapFromSrc));
  D.print<StreamNumberProvider>(
      "DBG (SectionHdr)", DL.getDebugStreamIndex(DbgHeaderType::SectionHdr),
      DR.getDebugStreamIndex(DbgHeaderType::SectionHdr));
  D.print<StreamNumberProvider>(
      "DBG (TokenRidMap)", DL.getDebugStreamIndex(DbgHeaderType::TokenRidMap),
      DR.getDebugStreamIndex(DbgHeaderType::TokenRidMap));
  D.print<StreamNumberProvider>("DBG (Xdata)",
                                DL.getDebugStreamIndex(DbgHeaderType::Xdata),
                                DR.getDebugStreamIndex(DbgHeaderType::Xdata));
  D.print<StreamNumberProvider>("DBG (Pdata)",
                                DL.getDebugStreamIndex(DbgHeaderType::Pdata),
                                DR.getDebugStreamIndex(DbgHeaderType::Pdata));
  D.print<StreamNumberProvider>("DBG (NewFPO)",
                                DL.getDebugStreamIndex(DbgHeaderType::NewFPO),
                                DR.getDebugStreamIndex(DbgHeaderType::NewFPO));
  D.print<StreamNumberProvider>(
      "DBG (SectionHdrOrig)",
      DL.getDebugStreamIndex(DbgHeaderType::SectionHdrOrig),
      DR.getDebugStreamIndex(DbgHeaderType::SectionHdrOrig));
  D.print<StreamNumberProvider>("Globals Stream",
                                DL.getGlobalSymbolStreamIndex(),
                                DR.getGlobalSymbolStreamIndex());
  D.print<StreamNumberProvider>("Publics Stream",
                                DL.getPublicSymbolStreamIndex(),
                                DR.getPublicSymbolStreamIndex());
  D.print<StreamNumberProvider>("Symbol Records", DL.getSymRecordStreamIndex(),
                                DR.getSymRecordStreamIndex());
  D.print("Has CTypes", DL.hasCTypes(), DR.hasCTypes());
  D.print("Is Incrementally Linked", DL.isIncrementallyLinked(),
          DR.isIncrementallyLinked());
  D.print("Is Stripped", DL.isStripped(), DR.isStripped());
  const DbiModuleList &ML = DL.modules();
  const DbiModuleList &MR = DR.modules();
  D.print("Module Count", ML.getModuleCount(), MR.getModuleCount());
  D.print("Source File Count", ML.getSourceFileCount(),
          MR.getSourceFileCount());
  auto MDL = getModuleDescriptors(ML);
  auto MDR = getModuleDescriptors(MR);
  // Scan all module descriptors from the left, and look for corresponding
  // module descriptors on the right.
  for (const auto &L : MDL)
    diffOneModule(D, L, MDR, false);

  for (const auto &R : MDR)
    diffOneModule(D, R, MDL, true);

  return Error::success();
}

Error DiffStyle::diffSectionContribs() { return Error::success(); }

Error DiffStyle::diffSectionMap() { return Error::success(); }

Error DiffStyle::diffFpoStream() { return Error::success(); }

Error DiffStyle::diffTpiStream(int Index) { return Error::success(); }

Error DiffStyle::diffModuleInfoStream(int Index) { return Error::success(); }

Error DiffStyle::diffPublics() { return Error::success(); }

Error DiffStyle::diffGlobals() { return Error::success(); }
