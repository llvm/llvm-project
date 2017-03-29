//===- NativeRawSymbol.cpp - Native implementation of IPDBRawSymbol -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeRawSymbol.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::pdb;

NativeRawSymbol::NativeRawSymbol(NativeSession &PDBSession)
  : Session(PDBSession) {}

void NativeRawSymbol::dump(raw_ostream &OS, int Indent) const {}

std::unique_ptr<IPDBEnumSymbols>
NativeRawSymbol::findChildren(PDB_SymType Type) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumSymbols>
NativeRawSymbol::findChildren(PDB_SymType Type, StringRef Name,
    PDB_NameSearchFlags Flags) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumSymbols>
NativeRawSymbol::findChildrenByRVA(PDB_SymType Type, StringRef Name,
    PDB_NameSearchFlags Flags, uint32_t RVA) const {
  return nullptr;
}

std::unique_ptr<IPDBEnumSymbols>
NativeRawSymbol::findInlineFramesByRVA(uint32_t RVA) const {
  return nullptr;
}

void NativeRawSymbol::getDataBytes(llvm::SmallVector<uint8_t, 32> &bytes) const {
  bytes.clear();
}

PDB_MemberAccess NativeRawSymbol::getAccess() const {
  return PDB_MemberAccess::Private;
}

uint32_t NativeRawSymbol::getAddressOffset() const {
  return 0;
}

uint32_t NativeRawSymbol::getAddressSection() const {
  return 0;
}

uint32_t NativeRawSymbol::getAge() const {
  return 0;
}

uint32_t NativeRawSymbol::getArrayIndexTypeId() const {
  return 0;
}

void NativeRawSymbol::getBackEndVersion(VersionInfo &Version) const {
  Version.Major = 0;
  Version.Minor = 0;
  Version.Build = 0;
  Version.QFE = 0;
}

uint32_t NativeRawSymbol::getBaseDataOffset() const {
  return 0;
}

uint32_t NativeRawSymbol::getBaseDataSlot() const {
  return 0;
}

uint32_t NativeRawSymbol::getBaseSymbolId() const {
  return 0;
}

PDB_BuiltinType NativeRawSymbol::getBuiltinType() const {
  return PDB_BuiltinType::None;
}

uint32_t NativeRawSymbol::getBitPosition() const {
  return 0;
}

PDB_CallingConv NativeRawSymbol::getCallingConvention() const {
  return PDB_CallingConv::FarStdCall;
}

uint32_t NativeRawSymbol::getClassParentId() const {
  return 0;
}

std::string NativeRawSymbol::getCompilerName() const {
  return 0;
}

uint32_t NativeRawSymbol::getCount() const {
  return 0;
}

uint32_t NativeRawSymbol::getCountLiveRanges() const {
  return 0;
}

void NativeRawSymbol::getFrontEndVersion(VersionInfo &Version) const {
  Version.Major = 0;
  Version.Minor = 0;
  Version.Build = 0;
  Version.QFE = 0;
}

PDB_Lang NativeRawSymbol::getLanguage() const {
  return PDB_Lang::Cobol;
}

uint32_t NativeRawSymbol::getLexicalParentId() const {
  return 0;
}

std::string NativeRawSymbol::getLibraryName() const {
  return "";
}

uint32_t NativeRawSymbol::getLiveRangeStartAddressOffset() const {
  return 0;
}

uint32_t NativeRawSymbol::getLiveRangeStartAddressSection() const {
  return 0;
}

uint32_t NativeRawSymbol::getLiveRangeStartRelativeVirtualAddress() const {
  return 0;
}

codeview::RegisterId NativeRawSymbol::getLocalBasePointerRegisterId() const {
  return codeview::RegisterId::EAX;
}

uint32_t NativeRawSymbol::getLowerBoundId() const {
  return 0;
}

uint32_t NativeRawSymbol::getMemorySpaceKind() const {
  return 0;
}

std::string NativeRawSymbol::getName() const {
  return 0;
}

uint32_t NativeRawSymbol::getNumberOfAcceleratorPointerTags() const {
  return 0;
}

uint32_t NativeRawSymbol::getNumberOfColumns() const {
  return 0;
}

uint32_t NativeRawSymbol::getNumberOfModifiers() const {
  return 0;
}

uint32_t NativeRawSymbol::getNumberOfRegisterIndices() const {
  return 0;
}

uint32_t NativeRawSymbol::getNumberOfRows() const {
  return 0;
}

std::string NativeRawSymbol::getObjectFileName() const {
  return "";
}

uint32_t NativeRawSymbol::getOemId() const {
  return 0;
}

uint32_t NativeRawSymbol::getOemSymbolId() const {
  return 0;
}

uint32_t NativeRawSymbol::getOffsetInUdt() const {
  return 0;
}

PDB_Cpu NativeRawSymbol::getPlatform() const {
  return PDB_Cpu::Intel8080;
}

uint32_t NativeRawSymbol::getRank() const {
  return 0;
}

codeview::RegisterId NativeRawSymbol::getRegisterId() const {
  return codeview::RegisterId::EAX;
}

uint32_t NativeRawSymbol::getRegisterType() const {
  return 0;
}

uint32_t NativeRawSymbol::getRelativeVirtualAddress() const {
  return 0;
}

uint32_t NativeRawSymbol::getSamplerSlot() const {
  return 0;
}

uint32_t NativeRawSymbol::getSignature() const {
  return 0;
}

uint32_t NativeRawSymbol::getSizeInUdt() const {
  return 0;
}

uint32_t NativeRawSymbol::getSlot() const {
  return 0;
}

std::string NativeRawSymbol::getSourceFileName() const {
  return 0;
}

uint32_t NativeRawSymbol::getStride() const {
  return 0;
}

uint32_t NativeRawSymbol::getSubTypeId() const {
  return 0;
}

std::string NativeRawSymbol::getSymbolsFileName() const { return ""; }

uint32_t NativeRawSymbol::getSymIndexId() const {
  return 0;
}

uint32_t NativeRawSymbol::getTargetOffset() const {
  return 0;
}

uint32_t NativeRawSymbol::getTargetRelativeVirtualAddress() const {
  return 0;
}

uint64_t NativeRawSymbol::getTargetVirtualAddress() const {
  return 0;
}

uint32_t NativeRawSymbol::getTargetSection() const {
  return 0;
}

uint32_t NativeRawSymbol::getTextureSlot() const {
  return 0;
}

uint32_t NativeRawSymbol::getTimeStamp() const {
  return 0;
}

uint32_t NativeRawSymbol::getToken() const {
  return 0;
}

uint32_t NativeRawSymbol::getTypeId() const {
  return 0;
}

uint32_t NativeRawSymbol::getUavSlot() const {
  return 0;
}

std::string NativeRawSymbol::getUndecoratedName() const {
  return 0;
}

uint32_t NativeRawSymbol::getUnmodifiedTypeId() const {
  return 0;
}

uint32_t NativeRawSymbol::getUpperBoundId() const {
  return 0;
}

Variant NativeRawSymbol::getValue() const {
  return Variant();
}

uint32_t NativeRawSymbol::getVirtualBaseDispIndex() const {
  return 0;
}

uint32_t NativeRawSymbol::getVirtualBaseOffset() const {
  return 0;
}

uint32_t NativeRawSymbol::getVirtualTableShapeId() const {
  return 0;
}

PDB_DataKind NativeRawSymbol::getDataKind() const {
  return PDB_DataKind::Unknown;
}

PDB_SymType NativeRawSymbol::getSymTag() const {
  return PDB_SymType::None;
}

PDB_UniqueId NativeRawSymbol::getGuid() const {
  return PDB_UniqueId{{0}};
}

int32_t NativeRawSymbol::getOffset() const {
  return 0;
}

int32_t NativeRawSymbol::getThisAdjust() const {
  return 0;
}

int32_t NativeRawSymbol::getVirtualBasePointerOffset() const {
  return 0;
}

PDB_LocType NativeRawSymbol::getLocationType() const {
  return PDB_LocType::Null;
}

PDB_Machine NativeRawSymbol::getMachineType() const {
  return PDB_Machine::Invalid;
}

codeview::ThunkOrdinal NativeRawSymbol::getThunkOrdinal() const {
  return codeview::ThunkOrdinal::Standard;
}

uint64_t NativeRawSymbol::getLength() const {
  return 0;
}

uint64_t NativeRawSymbol::getLiveRangeLength() const {
  return 0;
}

uint64_t NativeRawSymbol::getVirtualAddress() const {
  return 0;
}

PDB_UdtType NativeRawSymbol::getUdtKind() const {
  return PDB_UdtType::Struct;
}

bool NativeRawSymbol::hasConstructor() const {
  return false;
}

bool NativeRawSymbol::hasCustomCallingConvention() const {
  return false;
}

bool NativeRawSymbol::hasFarReturn() const {
  return false;
}

bool NativeRawSymbol::isCode() const {
  return false;
}

bool NativeRawSymbol::isCompilerGenerated() const {
  return false;
}

bool NativeRawSymbol::isConstType() const {
  return false;
}

bool NativeRawSymbol::isEditAndContinueEnabled() const {
  return false;
}

bool NativeRawSymbol::isFunction() const {
  return false;
}

bool NativeRawSymbol::getAddressTaken() const {
  return false;
}

bool NativeRawSymbol::getNoStackOrdering() const {
  return false;
}

bool NativeRawSymbol::hasAlloca() const {
  return false;
}

bool NativeRawSymbol::hasAssignmentOperator() const {
  return false;
}

bool NativeRawSymbol::hasCTypes() const {
  return false;
}

bool NativeRawSymbol::hasCastOperator() const {
  return false;
}

bool NativeRawSymbol::hasDebugInfo() const {
  return false;
}

bool NativeRawSymbol::hasEH() const {
  return false;
}

bool NativeRawSymbol::hasEHa() const {
  return false;
}

bool NativeRawSymbol::hasInlAsm() const {
  return false;
}

bool NativeRawSymbol::hasInlineAttribute() const {
  return false;
}

bool NativeRawSymbol::hasInterruptReturn() const {
  return false;
}

bool NativeRawSymbol::hasFramePointer() const {
  return false;
}

bool NativeRawSymbol::hasLongJump() const {
  return false;
}

bool NativeRawSymbol::hasManagedCode() const {
  return false;
}

bool NativeRawSymbol::hasNestedTypes() const {
  return false;
}

bool NativeRawSymbol::hasNoInlineAttribute() const {
  return false;
}

bool NativeRawSymbol::hasNoReturnAttribute() const {
  return false;
}

bool NativeRawSymbol::hasOptimizedCodeDebugInfo() const {
  return false;
}

bool NativeRawSymbol::hasOverloadedOperator() const {
  return false;
}

bool NativeRawSymbol::hasSEH() const {
  return false;
}

bool NativeRawSymbol::hasSecurityChecks() const {
  return false;
}

bool NativeRawSymbol::hasSetJump() const {
  return false;
}

bool NativeRawSymbol::hasStrictGSCheck() const {
  return false;
}

bool NativeRawSymbol::isAcceleratorGroupSharedLocal() const {
  return false;
}

bool NativeRawSymbol::isAcceleratorPointerTagLiveRange() const {
  return false;
}

bool NativeRawSymbol::isAcceleratorStubFunction() const {
  return false;
}

bool NativeRawSymbol::isAggregated() const {
  return false;
}

bool NativeRawSymbol::isIntroVirtualFunction() const {
  return false;
}

bool NativeRawSymbol::isCVTCIL() const {
  return false;
}

bool NativeRawSymbol::isConstructorVirtualBase() const {
  return false;
}

bool NativeRawSymbol::isCxxReturnUdt() const {
  return false;
}

bool NativeRawSymbol::isDataAligned() const {
  return false;
}

bool NativeRawSymbol::isHLSLData() const {
  return false;
}

bool NativeRawSymbol::isHotpatchable() const {
  return false;
}

bool NativeRawSymbol::isIndirectVirtualBaseClass() const {
  return false;
}

bool NativeRawSymbol::isInterfaceUdt() const {
  return false;
}

bool NativeRawSymbol::isIntrinsic() const {
  return false;
}

bool NativeRawSymbol::isLTCG() const {
  return false;
}

bool NativeRawSymbol::isLocationControlFlowDependent() const {
  return false;
}

bool NativeRawSymbol::isMSILNetmodule() const {
  return false;
}

bool NativeRawSymbol::isMatrixRowMajor() const {
  return false;
}

bool NativeRawSymbol::isManagedCode() const {
  return false;
}

bool NativeRawSymbol::isMSILCode() const {
  return false;
}

bool NativeRawSymbol::isMultipleInheritance() const {
  return false;
}

bool NativeRawSymbol::isNaked() const {
  return false;
}

bool NativeRawSymbol::isNested() const {
  return false;
}

bool NativeRawSymbol::isOptimizedAway() const {
  return false;
}

bool NativeRawSymbol::isPacked() const {
  return false;
}

bool NativeRawSymbol::isPointerBasedOnSymbolValue() const {
  return false;
}

bool NativeRawSymbol::isPointerToDataMember() const {
  return false;
}

bool NativeRawSymbol::isPointerToMemberFunction() const {
  return false;
}

bool NativeRawSymbol::isPureVirtual() const {
  return false;
}

bool NativeRawSymbol::isRValueReference() const {
  return false;
}

bool NativeRawSymbol::isRefUdt() const {
  return false;
}

bool NativeRawSymbol::isReference() const {
  return false;
}

bool NativeRawSymbol::isRestrictedType() const {
  return false;
}

bool NativeRawSymbol::isReturnValue() const {
  return false;
}

bool NativeRawSymbol::isSafeBuffers() const {
  return false;
}

bool NativeRawSymbol::isScoped() const {
  return false;
}

bool NativeRawSymbol::isSdl() const {
  return false;
}

bool NativeRawSymbol::isSingleInheritance() const {
  return false;
}

bool NativeRawSymbol::isSplitted() const {
  return false;
}

bool NativeRawSymbol::isStatic() const {
  return false;
}

bool NativeRawSymbol::hasPrivateSymbols() const {
  return false;
}

bool NativeRawSymbol::isUnalignedType() const {
  return false;
}

bool NativeRawSymbol::isUnreached() const {
  return false;
}

bool NativeRawSymbol::isValueUdt() const {
  return false;
}

bool NativeRawSymbol::isVirtual() const {
  return false;
}

bool NativeRawSymbol::isVirtualBaseClass() const {
  return false;
}

bool NativeRawSymbol::isVirtualInheritance() const {
  return false;
}

bool NativeRawSymbol::isVolatileType() const {
  return false;
}

bool NativeRawSymbol::wasInlined() const {
  return false;
}

std::string NativeRawSymbol::getUnused() const {
  return "";
}
