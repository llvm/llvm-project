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

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/ErrorHandling.h"
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
                           StringRef FileName) {

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

std::string getPGOFuncName(const Function &F) {
  return getPGOFuncName(F.getName(), F.getLinkage(), F.getParent()->getName());
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
                         Twine(getInstrProfNameVarPrefix()) + FuncName);

  // Hide the symbol so that we correctly get a copy for each executable.
  if (!GlobalValue::isLocalLinkage(FuncNameVar->getLinkage()))
    FuncNameVar->setVisibility(GlobalValue::HiddenVisibility);

  return FuncNameVar;
}

GlobalVariable *createPGOFuncNameVar(Function &F, StringRef FuncName) {
  return createPGOFuncNameVar(*F.getParent(), F.getLinkage(), FuncName);
}

uint32_t ValueProfRecord::getHeaderSize(uint32_t NumValueSites) {
  uint32_t Size = offsetof(ValueProfRecord, SiteCountArray) +
                  sizeof(uint8_t) * NumValueSites;
  // Round the size to multiple of 8 bytes.
  Size = (Size + 7) & ~7;
  return Size;
}

uint32_t ValueProfRecord::getSize(uint32_t NumValueSites,
                                  uint32_t NumValueData) {
  return getHeaderSize(NumValueSites) +
         sizeof(InstrProfValueData) * NumValueData;
}

void ValueProfRecord::deserializeTo(InstrProfRecord &Record,
                                    InstrProfRecord::ValueMapType *VMap) {
  Record.reserveSites(Kind, NumValueSites);

  InstrProfValueData *ValueData = this->getValueData();
  for (uint64_t VSite = 0; VSite < NumValueSites; ++VSite) {
    uint8_t ValueDataCount = this->SiteCountArray[VSite];
    Record.addValueData(Kind, VSite, ValueData, ValueDataCount, VMap);
    ValueData += ValueDataCount;
  }
}

void ValueProfRecord::serializeFrom(const InstrProfRecord &Record,
                                    uint32_t ValueKind,
                                    uint32_t NumValueSites) {
  Kind = ValueKind;
  this->NumValueSites = NumValueSites;
  InstrProfValueData *DstVD = getValueData();
  for (uint32_t S = 0; S < NumValueSites; S++) {
    uint32_t ND = Record.getNumValueDataForSite(ValueKind, S);
    SiteCountArray[S] = ND;
    std::unique_ptr<InstrProfValueData[]> SrcVD =
        Record.getValueForSite(ValueKind, S);
    for (uint32_t I = 0; I < ND; I++) {
      DstVD[I] = SrcVD[I];
      switch (ValueKind) {
      case IPVK_IndirectCallTarget:
        DstVD[I].Value = IndexedInstrProf::ComputeHash(
            IndexedInstrProf::HashType, (const char *)DstVD[I].Value);
        break;
      default:
        llvm_unreachable("value kind not handled !");
      }
    }
    DstVD += ND;
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
  uint32_t ND = getNumValueData();
  InstrProfValueData *VD = getValueData();

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

uint32_t ValueProfData::getSize(const InstrProfRecord &Record) {
  uint32_t TotalSize = sizeof(ValueProfData);
  uint32_t NumValueKinds = Record.getNumValueKinds();
  if (NumValueKinds == 0)
    return TotalSize;

  for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; Kind++) {
    uint32_t NumValueSites = Record.getNumValueSites(Kind);
    if (!NumValueSites)
      continue;
    TotalSize +=
        ValueProfRecord::getSize(NumValueSites, Record.getNumValueData(Kind));
  }
  return TotalSize;
}

void ValueProfData::deserializeTo(InstrProfRecord &Record,
                                  InstrProfRecord::ValueMapType *VMap) {
  if (NumValueKinds == 0)
    return;

  ValueProfRecord *VR = getFirstValueProfRecord();
  for (uint32_t K = 0; K < NumValueKinds; K++) {
    VR->deserializeTo(Record, VMap);
    VR = VR->getNext();
  }
}

static std::unique_ptr<ValueProfData> AllocValueProfData(uint32_t TotalSize) {
  return std::unique_ptr<ValueProfData>(new (::operator new(TotalSize))
                                            ValueProfData());
}

std::unique_ptr<ValueProfData>
ValueProfData::serializeFrom(const InstrProfRecord &Record) {
  uint32_t TotalSize = getSize(Record);

  std::unique_ptr<ValueProfData> VPD = AllocValueProfData(TotalSize);

  VPD->TotalSize = TotalSize;
  VPD->NumValueKinds = Record.getNumValueKinds();
  ValueProfRecord *VR = VPD->getFirstValueProfRecord();
  for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; Kind++) {
    uint32_t NumValueSites = Record.getNumValueSites(Kind);
    if (!NumValueSites)
      continue;
    VR->serializeFrom(Record, Kind, NumValueSites);
    VR = VR->getNext();
  }
  return VPD;
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
  uint32_t NumValueKinds = swapToHostOrder<uint32_t>(Header, Endianness);

  if (D + TotalSize > BufferEnd)
    return instrprof_error::too_large;
  if (NumValueKinds > IPVK_Last + 1)
    return instrprof_error::malformed;
  // Total size needs to be mulltiple of quadword size.
  if (TotalSize % sizeof(uint64_t))
    return instrprof_error::malformed;

  std::unique_ptr<ValueProfData> VPD = AllocValueProfData(TotalSize);

  memcpy(VPD.get(), D, TotalSize);
  // Byte swap.
  VPD->swapBytesToHost(Endianness);

  // Data integrety check:
  ValueProfRecord *VR = VPD->getFirstValueProfRecord();
  for (uint32_t K = 0; K < VPD->NumValueKinds; K++) {
    if (VR->Kind > IPVK_Last)
      return instrprof_error::malformed;
    VR = VR->getNext();
    if ((char *)VR - (char *)VPD.get() > (ptrdiff_t)TotalSize)
      return instrprof_error::malformed;
  }

  return std::move(VPD);
}

void ValueProfData::swapBytesToHost(support::endianness Endianness) {
  using namespace support;
  if (Endianness == getHostEndianness())
    return;

  sys::swapByteOrder<uint32_t>(TotalSize);
  sys::swapByteOrder<uint32_t>(NumValueKinds);

  ValueProfRecord *VR = getFirstValueProfRecord();
  for (uint32_t K = 0; K < NumValueKinds; K++) {
    VR->swapBytes(Endianness, getHostEndianness());
    VR = VR->getNext();
  }
}

void ValueProfData::swapBytesFromHost(support::endianness Endianness) {
  using namespace support;
  if (Endianness == getHostEndianness())
    return;

  ValueProfRecord *VR = getFirstValueProfRecord();
  for (uint32_t K = 0; K < NumValueKinds; K++) {
    ValueProfRecord *NVR = VR->getNext();
    VR->swapBytes(getHostEndianness(), Endianness);
    VR = NVR;
  }
  sys::swapByteOrder<uint32_t>(TotalSize);
  sys::swapByteOrder<uint32_t>(NumValueKinds);
}

ValueProfRecord *ValueProfData::getFirstValueProfRecord() {
  return reinterpret_cast<ValueProfRecord *>((char *)this +
                                             sizeof(ValueProfData));
}

uint32_t ValueProfRecord::getNumValueData() const {
  uint32_t NumValueData = 0;
  for (uint32_t I = 0; I < NumValueSites; I++)
    NumValueData += SiteCountArray[I];
  return NumValueData;
}

ValueProfRecord *ValueProfRecord::getNext() {
  return reinterpret_cast<ValueProfRecord *>((char *)this + getSize());
}

InstrProfValueData *ValueProfRecord::getValueData() {
  return reinterpret_cast<InstrProfValueData *>((char *)this +
                                                getHeaderSize(NumValueSites));
}
}
