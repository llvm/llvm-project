//=-- InstrProf.cpp - Instrumented profiling format support -----------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for clang's instrumentation based PGO and
// coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;

namespace {
class InstrProfErrorCategoryType : public std::error_category {
  const char *name() const LLVM_NOEXCEPT override { return "llvm.instrprof"; }
  std::string message(int IE) const override {
    instrprof_error E = static_cast<instrprof_error>(IE);
    switch (E) {
    case instrprof_error::success:
      return "Success";
    case instrprof_error::eof:
      return "End of File";
    case instrprof_error::unrecognized_format:
      return "Unrecognized instrumentation profile encoding format";
    case instrprof_error::bad_magic:
      return "Invalid instrumentation profile data (bad magic)";
    case instrprof_error::bad_header:
      return "Invalid instrumentation profile data (file header is corrupt)";
    case instrprof_error::unsupported_version:
      return "Unsupported instrumentation profile format version";
    case instrprof_error::unsupported_hash_type:
      return "Unsupported instrumentation profile hash type";
    case instrprof_error::too_large:
      return "Too much profile data";
    case instrprof_error::truncated:
      return "Truncated profile data";
    case instrprof_error::malformed:
      return "Malformed instrumentation profile data";
    case instrprof_error::unknown_function:
      return "No profile data available for function";
    case instrprof_error::hash_mismatch:
      return "Function control flow change detected (hash mismatch)";
    case instrprof_error::count_mismatch:
      return "Function basic block count change detected (counter mismatch)";
    case instrprof_error::counter_overflow:
      return "Counter overflow";
    case instrprof_error::value_site_count_mismatch:
      return "Function value site count change detected (counter mismatch)";
    }
    llvm_unreachable("A value of instrprof_error has no message.");
  }
};
}

static ManagedStatic<InstrProfErrorCategoryType> ErrorCategory;

const std::error_category &llvm::instrprof_category() {
  return *ErrorCategory;
}

namespace llvm {

std::string getPGOFuncName(StringRef RawFuncName,
                           GlobalValue::LinkageTypes Linkage,
                           StringRef FileName,
                           uint64_t Version LLVM_ATTRIBUTE_UNUSED) {

  // Function names may be prefixed with a binary '1' to indicate
  // that the backend should not modify the symbols due to any platform
  // naming convention. Do not include that '1' in the PGO profile name.
  if (RawFuncName[0] == '\1')
    RawFuncName = RawFuncName.substr(1);

  std::string FuncName = RawFuncName;
  if (llvm::GlobalValue::isLocalLinkage(Linkage)) {
    // For local symbols, prepend the main file name to distinguish them.
    // Do not include the full path in the file name since there's no guarantee
    // that it will stay the same, e.g., if the files are checked out from
    // version control in different locations.
    if (FileName.empty())
      FuncName = FuncName.insert(0, "<unknown>:");
    else
      FuncName = FuncName.insert(0, FileName.str() + ":");
  }
  return FuncName;
}

std::string getPGOFuncName(const Function &F, uint64_t Version) {
  return getPGOFuncName(F.getName(), F.getLinkage(), F.getParent()->getName(),
                        Version);
}

StringRef getFuncNameWithoutPrefix(StringRef PGOFuncName, StringRef FileName) {
  if (FileName.empty())
    return PGOFuncName;
  // Drop the file name including ':'. See also getPGOFuncName.
  if (PGOFuncName.startswith(FileName))
    PGOFuncName = PGOFuncName.drop_front(FileName.size() + 1);
  return PGOFuncName;
}

// \p FuncName is the string used as profile lookup key for the function. A
// symbol is created to hold the name. Return the legalized symbol name.
static std::string getPGOFuncNameVarName(StringRef FuncName,
                                         GlobalValue::LinkageTypes Linkage) {
  std::string VarName = getInstrProfNameVarPrefix();
  VarName += FuncName;

  if (!GlobalValue::isLocalLinkage(Linkage))
    return VarName;

  // Now fix up illegal chars in local VarName that may upset the assembler.
  const char *InvalidChars = "-:<>\"'";
  size_t found = VarName.find_first_of(InvalidChars);
  while (found != std::string::npos) {
    VarName[found] = '_';
    found = VarName.find_first_of(InvalidChars, found + 1);
  }
  return VarName;
}

GlobalVariable *createPGOFuncNameVar(Module &M,
                                     GlobalValue::LinkageTypes Linkage,
                                     StringRef FuncName) {

  // We generally want to match the function's linkage, but available_externally
  // and extern_weak both have the wrong semantics, and anything that doesn't
  // need to link across compilation units doesn't need to be visible at all.
  if (Linkage == GlobalValue::ExternalWeakLinkage)
    Linkage = GlobalValue::LinkOnceAnyLinkage;
  else if (Linkage == GlobalValue::AvailableExternallyLinkage)
    Linkage = GlobalValue::LinkOnceODRLinkage;
  else if (Linkage == GlobalValue::InternalLinkage ||
           Linkage == GlobalValue::ExternalLinkage)
    Linkage = GlobalValue::PrivateLinkage;

  auto *Value = ConstantDataArray::getString(M.getContext(), FuncName, false);
  auto FuncNameVar =
      new GlobalVariable(M, Value->getType(), true, Linkage, Value,
                         getPGOFuncNameVarName(FuncName, Linkage));

  // Hide the symbol so that we correctly get a copy for each executable.
  if (!GlobalValue::isLocalLinkage(FuncNameVar->getLinkage()))
    FuncNameVar->setVisibility(GlobalValue::HiddenVisibility);

  return FuncNameVar;
}

GlobalVariable *createPGOFuncNameVar(Function &F, StringRef FuncName) {
  return createPGOFuncNameVar(*F.getParent(), F.getLinkage(), FuncName);
}

void InstrProfSymtab::create(const Module &M) {
  for (const Function &F : M)
    addFuncName(getPGOFuncName(F));

  finalizeSymtab();
}

int collectPGOFuncNameStrings(const std::vector<std::string> &NameStrs,
                              bool doCompression, std::string &Result) {
  uint8_t Header[16], *P = Header;
  std::string UncompressedNameStrings =
      join(NameStrs.begin(), NameStrs.end(), StringRef(" "));

  unsigned EncLen = encodeULEB128(UncompressedNameStrings.length(), P);
  P += EncLen;

  auto WriteStringToResult = [&](size_t CompressedLen,
                                 const std::string &InputStr) {
    EncLen = encodeULEB128(CompressedLen, P);
    P += EncLen;
    char *HeaderStr = reinterpret_cast<char *>(&Header[0]);
    unsigned HeaderLen = P - &Header[0];
    Result.append(HeaderStr, HeaderLen);
    Result += InputStr;
    return 0;
  };

  if (!doCompression)
    return WriteStringToResult(0, UncompressedNameStrings);

  SmallVector<char, 128> CompressedNameStrings;
  zlib::Status Success =
      zlib::compress(StringRef(UncompressedNameStrings), CompressedNameStrings,
                     zlib::BestSizeCompression);

  if (Success != zlib::StatusOK)
    return 1;

  return WriteStringToResult(
      CompressedNameStrings.size(),
      std::string(CompressedNameStrings.data(), CompressedNameStrings.size()));
}

StringRef getPGOFuncNameInitializer(GlobalVariable *NameVar) {
  auto *Arr = cast<ConstantDataArray>(NameVar->getInitializer());
  StringRef NameStr =
      Arr->isCString() ? Arr->getAsCString() : Arr->getAsString();
  return NameStr;
}

int collectPGOFuncNameStrings(const std::vector<GlobalVariable *> &NameVars,
                              std::string &Result) {
  std::vector<std::string> NameStrs;
  for (auto *NameVar : NameVars) {
    NameStrs.push_back(getPGOFuncNameInitializer(NameVar));
  }
  return collectPGOFuncNameStrings(NameStrs, zlib::isAvailable(), Result);
}

int readPGOFuncNameStrings(StringRef NameStrings, InstrProfSymtab &Symtab) {
  const uint8_t *P = reinterpret_cast<const uint8_t *>(NameStrings.data());
  const uint8_t *EndP = reinterpret_cast<const uint8_t *>(NameStrings.data() +
                                                          NameStrings.size());
  while (P < EndP) {
    uint32_t N;
    uint64_t UncompressedSize = decodeULEB128(P, &N);
    P += N;
    uint64_t CompressedSize = decodeULEB128(P, &N);
    P += N;
    bool isCompressed = (CompressedSize != 0);
    SmallString<128> UncompressedNameStrings;
    StringRef NameStrings;
    if (isCompressed) {
      StringRef CompressedNameStrings(reinterpret_cast<const char *>(P),
                                      CompressedSize);
      if (zlib::uncompress(CompressedNameStrings, UncompressedNameStrings,
                           UncompressedSize) != zlib::StatusOK)
        return 1;
      P += CompressedSize;
      NameStrings = StringRef(UncompressedNameStrings.data(),
                              UncompressedNameStrings.size());
    } else {
      NameStrings =
          StringRef(reinterpret_cast<const char *>(P), UncompressedSize);
      P += UncompressedSize;
    }
    // Now parse the name strings.
    SmallVector<StringRef, 0> Names;
    NameStrings.split(Names, ' ');
    for (StringRef &Name : Names)
      Symtab.addFuncName(Name);

    while (P < EndP && *P == 0)
      P++;
  }
  Symtab.finalizeSymtab();
  return 0;
}

instrprof_error InstrProfValueSiteRecord::merge(InstrProfValueSiteRecord &Input,
                                                uint64_t Weight) {
  this->sortByTargetValues();
  Input.sortByTargetValues();
  auto I = ValueData.begin();
  auto IE = ValueData.end();
  instrprof_error Result = instrprof_error::success;
  for (auto J = Input.ValueData.begin(), JE = Input.ValueData.end(); J != JE;
       ++J) {
    while (I != IE && I->Value < J->Value)
      ++I;
    if (I != IE && I->Value == J->Value) {
      bool Overflowed;
      I->Count = SaturatingMultiplyAdd(J->Count, Weight, I->Count, &Overflowed);
      if (Overflowed)
        Result = instrprof_error::counter_overflow;
      ++I;
      continue;
    }
    ValueData.insert(I, *J);
  }
  return Result;
}

instrprof_error InstrProfValueSiteRecord::scale(uint64_t Weight) {
  instrprof_error Result = instrprof_error::success;
  for (auto I = ValueData.begin(), IE = ValueData.end(); I != IE; ++I) {
    bool Overflowed;
    I->Count = SaturatingMultiply(I->Count, Weight, &Overflowed);
    if (Overflowed)
      Result = instrprof_error::counter_overflow;
  }
  return Result;
}

// Merge Value Profile data from Src record to this record for ValueKind.
// Scale merged value counts by \p Weight.
instrprof_error InstrProfRecord::mergeValueProfData(uint32_t ValueKind,
                                                    InstrProfRecord &Src,
                                                    uint64_t Weight) {
  uint32_t ThisNumValueSites = getNumValueSites(ValueKind);
  uint32_t OtherNumValueSites = Src.getNumValueSites(ValueKind);
  if (ThisNumValueSites != OtherNumValueSites)
    return instrprof_error::value_site_count_mismatch;
  std::vector<InstrProfValueSiteRecord> &ThisSiteRecords =
      getValueSitesForKind(ValueKind);
  std::vector<InstrProfValueSiteRecord> &OtherSiteRecords =
      Src.getValueSitesForKind(ValueKind);
  instrprof_error Result = instrprof_error::success;
  for (uint32_t I = 0; I < ThisNumValueSites; I++)
    MergeResult(Result, ThisSiteRecords[I].merge(OtherSiteRecords[I], Weight));
  return Result;
}

instrprof_error InstrProfRecord::merge(InstrProfRecord &Other,
                                       uint64_t Weight) {
  // If the number of counters doesn't match we either have bad data
  // or a hash collision.
  if (Counts.size() != Other.Counts.size())
    return instrprof_error::count_mismatch;

  instrprof_error Result = instrprof_error::success;

  for (size_t I = 0, E = Other.Counts.size(); I < E; ++I) {
    bool Overflowed;
    Counts[I] =
        SaturatingMultiplyAdd(Other.Counts[I], Weight, Counts[I], &Overflowed);
    if (Overflowed)
      Result = instrprof_error::counter_overflow;
  }

  for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind)
    MergeResult(Result, mergeValueProfData(Kind, Other, Weight));

  return Result;
}

instrprof_error InstrProfRecord::scaleValueProfData(uint32_t ValueKind,
                                                    uint64_t Weight) {
  uint32_t ThisNumValueSites = getNumValueSites(ValueKind);
  std::vector<InstrProfValueSiteRecord> &ThisSiteRecords =
      getValueSitesForKind(ValueKind);
  instrprof_error Result = instrprof_error::success;
  for (uint32_t I = 0; I < ThisNumValueSites; I++)
    MergeResult(Result, ThisSiteRecords[I].scale(Weight));
  return Result;
}

instrprof_error InstrProfRecord::scale(uint64_t Weight) {
  instrprof_error Result = instrprof_error::success;
  for (auto &Count : this->Counts) {
    bool Overflowed;
    Count = SaturatingMultiply(Count, Weight, &Overflowed);
    if (Overflowed && Result == instrprof_error::success) {
      Result = instrprof_error::counter_overflow;
    }
  }
  for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind)
    MergeResult(Result, scaleValueProfData(Kind, Weight));

  return Result;
}

// Map indirect call target name hash to name string.
uint64_t InstrProfRecord::remapValue(uint64_t Value, uint32_t ValueKind,
                                     ValueMapType *ValueMap) {
  if (!ValueMap)
    return Value;
  switch (ValueKind) {
  case IPVK_IndirectCallTarget: {
    auto Result =
        std::lower_bound(ValueMap->begin(), ValueMap->end(), Value,
                         [](const std::pair<uint64_t, uint64_t> &LHS,
                            uint64_t RHS) { return LHS.first < RHS; });
    if (Result != ValueMap->end())
      Value = (uint64_t)Result->second;
    break;
  }
  }
  return Value;
}

void InstrProfRecord::addValueData(uint32_t ValueKind, uint32_t Site,
                                   InstrProfValueData *VData, uint32_t N,
                                   ValueMapType *ValueMap) {
  for (uint32_t I = 0; I < N; I++) {
    VData[I].Value = remapValue(VData[I].Value, ValueKind, ValueMap);
  }
  std::vector<InstrProfValueSiteRecord> &ValueSites =
      getValueSitesForKind(ValueKind);
  if (N == 0)
    ValueSites.push_back(InstrProfValueSiteRecord());
  else
    ValueSites.emplace_back(VData, VData + N);
}

#define INSTR_PROF_COMMON_API_IMPL
#include "llvm/ProfileData/InstrProfData.inc"

/*!
 * \brief ValueProfRecordClosure Interface implementation for  InstrProfRecord
 *  class. These C wrappers are used as adaptors so that C++ code can be
 *  invoked as callbacks.
 */
uint32_t getNumValueKindsInstrProf(const void *Record) {
  return reinterpret_cast<const InstrProfRecord *>(Record)->getNumValueKinds();
}

uint32_t getNumValueSitesInstrProf(const void *Record, uint32_t VKind) {
  return reinterpret_cast<const InstrProfRecord *>(Record)
      ->getNumValueSites(VKind);
}

uint32_t getNumValueDataInstrProf(const void *Record, uint32_t VKind) {
  return reinterpret_cast<const InstrProfRecord *>(Record)
      ->getNumValueData(VKind);
}

uint32_t getNumValueDataForSiteInstrProf(const void *R, uint32_t VK,
                                         uint32_t S) {
  return reinterpret_cast<const InstrProfRecord *>(R)
      ->getNumValueDataForSite(VK, S);
}

void getValueForSiteInstrProf(const void *R, InstrProfValueData *Dst,
                              uint32_t K, uint32_t S,
                              uint64_t (*Mapper)(uint32_t, uint64_t)) {
  return reinterpret_cast<const InstrProfRecord *>(R)->getValueForSite(
      Dst, K, S, Mapper);
}

ValueProfData *allocValueProfDataInstrProf(size_t TotalSizeInBytes) {
  ValueProfData *VD =
      (ValueProfData *)(new (::operator new(TotalSizeInBytes)) ValueProfData());
  memset(VD, 0, TotalSizeInBytes);
  return VD;
}

static ValueProfRecordClosure InstrProfRecordClosure = {
    0,
    getNumValueKindsInstrProf,
    getNumValueSitesInstrProf,
    getNumValueDataInstrProf,
    getNumValueDataForSiteInstrProf,
    0,
    getValueForSiteInstrProf,
    allocValueProfDataInstrProf};

// Wrapper implementation using the closure mechanism.
uint32_t ValueProfData::getSize(const InstrProfRecord &Record) {
  InstrProfRecordClosure.Record = &Record;
  return getValueProfDataSize(&InstrProfRecordClosure);
}

// Wrapper implementation using the closure mechanism.
std::unique_ptr<ValueProfData>
ValueProfData::serializeFrom(const InstrProfRecord &Record) {
  InstrProfRecordClosure.Record = &Record;

  std::unique_ptr<ValueProfData> VPD(
      serializeValueProfDataFrom(&InstrProfRecordClosure, nullptr));
  return VPD;
}

void ValueProfRecord::deserializeTo(InstrProfRecord &Record,
                                    InstrProfRecord::ValueMapType *VMap) {
  Record.reserveSites(Kind, NumValueSites);

  InstrProfValueData *ValueData = getValueProfRecordValueData(this);
  for (uint64_t VSite = 0; VSite < NumValueSites; ++VSite) {
    uint8_t ValueDataCount = this->SiteCountArray[VSite];
    Record.addValueData(Kind, VSite, ValueData, ValueDataCount, VMap);
    ValueData += ValueDataCount;
  }
}

// For writing/serializing,  Old is the host endianness, and  New is
// byte order intended on disk. For Reading/deserialization, Old
// is the on-disk source endianness, and New is the host endianness.
void ValueProfRecord::swapBytes(support::endianness Old,
                                support::endianness New) {
  using namespace support;
  if (Old == New)
    return;

  if (getHostEndianness() != Old) {
    sys::swapByteOrder<uint32_t>(NumValueSites);
    sys::swapByteOrder<uint32_t>(Kind);
  }
  uint32_t ND = getValueProfRecordNumValueData(this);
  InstrProfValueData *VD = getValueProfRecordValueData(this);

  // No need to swap byte array: SiteCountArrray.
  for (uint32_t I = 0; I < ND; I++) {
    sys::swapByteOrder<uint64_t>(VD[I].Value);
    sys::swapByteOrder<uint64_t>(VD[I].Count);
  }
  if (getHostEndianness() == Old) {
    sys::swapByteOrder<uint32_t>(NumValueSites);
    sys::swapByteOrder<uint32_t>(Kind);
  }
}

void ValueProfData::deserializeTo(InstrProfRecord &Record,
                                  InstrProfRecord::ValueMapType *VMap) {
  if (NumValueKinds == 0)
    return;

  ValueProfRecord *VR = getFirstValueProfRecord(this);
  for (uint32_t K = 0; K < NumValueKinds; K++) {
    VR->deserializeTo(Record, VMap);
    VR = getValueProfRecordNext(VR);
  }
}

template <class T>
static T swapToHostOrder(const unsigned char *&D, support::endianness Orig) {
  using namespace support;
  if (Orig == little)
    return endian::readNext<T, little, unaligned>(D);
  else
    return endian::readNext<T, big, unaligned>(D);
}

static std::unique_ptr<ValueProfData> allocValueProfData(uint32_t TotalSize) {
  return std::unique_ptr<ValueProfData>(new (::operator new(TotalSize))
                                            ValueProfData());
}

instrprof_error ValueProfData::checkIntegrity() {
  if (NumValueKinds > IPVK_Last + 1)
    return instrprof_error::malformed;
  // Total size needs to be mulltiple of quadword size.
  if (TotalSize % sizeof(uint64_t))
    return instrprof_error::malformed;

  ValueProfRecord *VR = getFirstValueProfRecord(this);
  for (uint32_t K = 0; K < this->NumValueKinds; K++) {
    if (VR->Kind > IPVK_Last)
      return instrprof_error::malformed;
    VR = getValueProfRecordNext(VR);
    if ((char *)VR - (char *)this > (ptrdiff_t)TotalSize)
      return instrprof_error::malformed;
  }
  return instrprof_error::success;
}

ErrorOr<std::unique_ptr<ValueProfData>>
ValueProfData::getValueProfData(const unsigned char *D,
                                const unsigned char *const BufferEnd,
                                support::endianness Endianness) {
  using namespace support;
  if (D + sizeof(ValueProfData) > BufferEnd)
    return instrprof_error::truncated;

  const unsigned char *Header = D;
  uint32_t TotalSize = swapToHostOrder<uint32_t>(Header, Endianness);
  if (D + TotalSize > BufferEnd)
    return instrprof_error::too_large;

  std::unique_ptr<ValueProfData> VPD = allocValueProfData(TotalSize);
  memcpy(VPD.get(), D, TotalSize);
  // Byte swap.
  VPD->swapBytesToHost(Endianness);

  instrprof_error EC = VPD->checkIntegrity();
  if (EC != instrprof_error::success)
    return EC;

  return std::move(VPD);
}

void ValueProfData::swapBytesToHost(support::endianness Endianness) {
  using namespace support;
  if (Endianness == getHostEndianness())
    return;

  sys::swapByteOrder<uint32_t>(TotalSize);
  sys::swapByteOrder<uint32_t>(NumValueKinds);

  ValueProfRecord *VR = getFirstValueProfRecord(this);
  for (uint32_t K = 0; K < NumValueKinds; K++) {
    VR->swapBytes(Endianness, getHostEndianness());
    VR = getValueProfRecordNext(VR);
  }
}

void ValueProfData::swapBytesFromHost(support::endianness Endianness) {
  using namespace support;
  if (Endianness == getHostEndianness())
    return;

  ValueProfRecord *VR = getFirstValueProfRecord(this);
  for (uint32_t K = 0; K < NumValueKinds; K++) {
    ValueProfRecord *NVR = getValueProfRecordNext(VR);
    VR->swapBytes(getHostEndianness(), Endianness);
    VR = NVR;
  }
  sys::swapByteOrder<uint32_t>(TotalSize);
  sys::swapByteOrder<uint32_t>(NumValueKinds);
}

// The argument to this method is a vector of cutoff percentages and the return
// value is a vector of (Cutoff, MinBlockCount, NumBlocks) triplets.
void ProfileSummary::computeDetailedSummary() {
  if (DetailedSummaryCutoffs.empty())
    return;
  auto Iter = CountFrequencies.begin();
  auto End = CountFrequencies.end();
  std::sort(DetailedSummaryCutoffs.begin(), DetailedSummaryCutoffs.end());

  uint32_t BlocksSeen = 0;
  uint64_t CurrSum = 0, Count;

  for (uint32_t Cutoff : DetailedSummaryCutoffs) {
    assert(Cutoff <= 999999);
    APInt Temp(128, TotalCount);
    APInt N(128, Cutoff);
    APInt D(128, ProfileSummary::Scale);
    Temp *= N;
    Temp = Temp.sdiv(D);
    uint64_t DesiredCount = Temp.getZExtValue();
    assert(DesiredCount <= TotalCount);
    while (CurrSum < DesiredCount && Iter != End) {
      Count = Iter->first;
      uint32_t Freq = Iter->second;
      CurrSum += (Count * Freq);
      BlocksSeen += Freq;
      Iter++;
    }
    assert(CurrSum >= DesiredCount);
    ProfileSummaryEntry PSE = {Cutoff, Count, BlocksSeen};
    DetailedSummary.push_back(PSE);
  }
  return;
}

}
