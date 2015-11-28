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

/// Return the total size in bytes of the on-disk value profile data
/// given the data stored in Record.
uint32_t getValueProfDataSize(ValueProfRecordClosure *Closure) {
  uint32_t Kind;
  uint32_t TotalSize = sizeof(ValueProfData);
  const void *Record = Closure->Record;
  uint32_t NumValueKinds = Closure->GetNumValueKinds(Record);
  if (NumValueKinds == 0)
    return TotalSize;

  for (Kind = IPVK_First; Kind <= IPVK_Last; Kind++) {
    uint32_t NumValueSites = Closure->GetNumValueSites(Record, Kind);
    if (!NumValueSites)
      continue;
    TotalSize += getValueProfRecordSize(NumValueSites,
                                        Closure->GetNumValueData(Record, Kind));
  }
  return TotalSize;
}

// Extract data from \c Closure and serialize into \c This instance.
void serializeValueProfRecordFrom(ValueProfRecord *This,
                                  ValueProfRecordClosure *Closure,
                                  uint32_t ValueKind, uint32_t NumValueSites) {
  uint32_t S;
  const void *Record = Closure->Record;
  This->Kind = ValueKind;
  This->NumValueSites = NumValueSites;
  InstrProfValueData *DstVD = getValueProfRecordValueData(This);

  for (S = 0; S < NumValueSites; S++) {
    uint32_t ND = Closure->GetNumValueDataForSite(Record, ValueKind, S);
    This->SiteCountArray[S] = ND;
    Closure->GetValueForSite(Record, DstVD, ValueKind, S,
                             Closure->RemapValueData);
    DstVD += ND;
  }
}

ValueProfData *serializeValueProfDataFrom(ValueProfRecordClosure *Closure,
                                          ValueProfData *DstData) {
  uint32_t TotalSize = getValueProfDataSize(Closure);

  ValueProfData *VPD =
      DstData ? DstData : Closure->AllocValueProfData(TotalSize);

  VPD->TotalSize = TotalSize;
  VPD->NumValueKinds = Closure->GetNumValueKinds(Closure->Record);
  ValueProfRecord *VR = getFirstValueProfRecord(VPD);
  for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; Kind++) {
    uint32_t NumValueSites = Closure->GetNumValueSites(Closure->Record, Kind);
    if (!NumValueSites)
      continue;
    serializeValueProfRecordFrom(VR, Closure, Kind, NumValueSites);
    VR = getValueProfRecordNext(VR);
  }
  return VPD;
}

/*! \brief ValueProfRecordClosure Interface implementation for  InstrProfRecord
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
  return reinterpret_cast<const InstrProfRecord *>(R)
      ->getValueForSite(Dst, K, S, Mapper);
}

uint64_t stringToHash(uint32_t ValueKind, uint64_t Value) {
  switch (ValueKind) {
  case IPVK_IndirectCallTarget:
    return IndexedInstrProf::ComputeHash(IndexedInstrProf::HashType,
                                         (const char *)Value);
    break;
  default:
    llvm_unreachable("value kind not handled !");
  }
  return Value;
}

ValueProfData *allocValueProfDataInstrProf(size_t TotalSizeInBytes) {
  return (ValueProfData *)(new (::operator new(TotalSizeInBytes))
                               ValueProfData());
}

static ValueProfRecordClosure InstrProfRecordClosure = {
    0,
    getNumValueKindsInstrProf,
    getNumValueSitesInstrProf,
    getNumValueDataInstrProf,
    getNumValueDataForSiteInstrProf,
    stringToHash,
    getValueForSiteInstrProf,
    allocValueProfDataInstrProf
};

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
      serializeValueProfDataFrom(&InstrProfRecordClosure, 0));
  return VPD;
}

/* The value profiler runtime library stores the value profile data
 * for a given function in NumValueSites and Nodes. This is the
 * method to initialize the RuntimeRecord with the runtime data to
 * pre-compute the information needed to efficiently implement
 * ValueProfRecordClosure's callback interfaces.
 */
int initializeValueProfRuntimeRecord(ValueProfRuntimeRecord *RuntimeRecord,
                                     uint16_t *NumValueSites,
                                     ValueProfNode **Nodes) {
  unsigned I, J, S = 0, NumValueKinds = 0;
  RuntimeRecord->NumValueSites = NumValueSites;
  RuntimeRecord->Nodes = Nodes;
  for (I = 0; I <= IPVK_Last; I++) {
    uint16_t N = NumValueSites[I];
    if (!N) {
      RuntimeRecord->SiteCountArray[I] = 0;
      continue;
    }
    NumValueKinds++;
    RuntimeRecord->SiteCountArray[I] = (uint8_t *)calloc(N, 1);
    RuntimeRecord->NodesKind[I] = &RuntimeRecord->Nodes[S];
    if (!RuntimeRecord->NodesKind[I])
      return 1;
    for (J = 0; J < N; J++) {
      uint8_t C = 0;
      ValueProfNode *Site = RuntimeRecord->Nodes[S + J];
      while (Site) {
        C++;
        Site = Site->Next;
      }
      if (C > UCHAR_MAX)
        C = UCHAR_MAX;
      RuntimeRecord->SiteCountArray[I][J] = C;
    }
    S += N;
  }
  RuntimeRecord->NumValueKinds = NumValueKinds;
  return 0;
}

void finalizeValueProfRuntimeRecord(ValueProfRuntimeRecord *RuntimeRecord) {
  unsigned I;
  for (I = 0; I <= IPVK_Last; I++) {
    if (RuntimeRecord->SiteCountArray[I])
      free(RuntimeRecord->SiteCountArray[I]);
  }
}

/* ValueProfRecordClosure Interface implementation for
 * ValueProfDataRuntimeRecord.  */
uint32_t getNumValueKindsRT(const void *R) {
  return ((const ValueProfRuntimeRecord *)R)->NumValueKinds;
}

uint32_t getNumValueSitesRT(const void *R, uint32_t VK) {
  return ((const ValueProfRuntimeRecord *)R)->NumValueSites[VK];
}

uint32_t getNumValueDataForSiteRT(const void *R, uint32_t VK, uint32_t S) {
  const ValueProfRuntimeRecord *Record = (const ValueProfRuntimeRecord *)R;
  return Record->SiteCountArray[VK][S];
}

uint32_t getNumValueDataRT(const void *R, uint32_t VK) {
  unsigned I, S = 0;
  const ValueProfRuntimeRecord *Record = (const ValueProfRuntimeRecord *)R;
  if (Record->SiteCountArray[VK] == 0)
    return 0;
  for (I = 0; I < Record->NumValueSites[VK]; I++)
    S += Record->SiteCountArray[VK][I];
  return S;
}

void getValueForSiteRT(const void *R, InstrProfValueData *Dst, uint32_t VK,
                       uint32_t S, uint64_t (*Mapper)(uint32_t, uint64_t)) {
  unsigned I, N = 0;
  const ValueProfRuntimeRecord *Record = (const ValueProfRuntimeRecord *)R;
  N = getNumValueDataForSiteRT(R, VK, S);
  ValueProfNode *VNode = Record->NodesKind[VK][S];
  for (I = 0; I < N; I++) {
    Dst[I] = VNode->VData;
    VNode = VNode->Next;
  }
}

ValueProfData *allocValueProfDataRT(size_t TotalSizeInBytes) {
  return (ValueProfData *)calloc(TotalSizeInBytes, 1);
}

static ValueProfRecordClosure RTRecordClosure = {0,
                                                 getNumValueKindsRT,
                                                 getNumValueSitesRT,
                                                 getNumValueDataRT,
                                                 getNumValueDataForSiteRT,
                                                 0,
                                                 getValueForSiteRT,
                                                 allocValueProfDataRT};

/* Return the size of ValueProfData structure to store data
 * recorded in the runtime record.
 */
uint32_t getValueProfDataSizeRT(const ValueProfRuntimeRecord *Record) {
  RTRecordClosure.Record = Record;
  return getValueProfDataSize(&RTRecordClosure);
}

/* Return a ValueProfData instance that stores the data collected
 * from runtime. If \c DstData is provided by the caller, the value
 * profile data will be store in *DstData and DstData is returned,
 * otherwise the method will allocate space for the value data and
 * return pointer to the newly allocated space.
 */
ValueProfData *
serializeValueProfDataFromRT(const ValueProfRuntimeRecord *Record,
                             ValueProfData *DstData) {
  RTRecordClosure.Record = Record;
  return serializeValueProfDataFrom(&RTRecordClosure, DstData);
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

}
