//===- llvm/unittest/DebugInfo/PDB/PDBApiTest.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <unordered_map>

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBInjectedSource.h"
#include "llvm/DebugInfo/PDB/IPDBLineNumber.h"
#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/IPDBSectionContrib.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/IPDBSourceFile.h"
#include "llvm/DebugInfo/PDB/IPDBTable.h"

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolAnnotation.h"
#include "llvm/DebugInfo/PDB/PDBSymbolBlock.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandDetails.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandEnv.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCustom.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugEnd.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugStart.h"
#include "llvm/DebugInfo/PDB/PDBSymbolLabel.h"
#include "llvm/DebugInfo/PDB/PDBSymbolPublicSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolThunk.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeArray.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBaseClass.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeCustom.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeDimension.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFriend.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionArg.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeManaged.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTable.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTableShape.h"
#include "llvm/DebugInfo/PDB/PDBSymbolUnknown.h"
#include "llvm/DebugInfo/PDB/PDBSymbolUsingNamespace.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "gtest/gtest.h"
using namespace llvm;
using namespace llvm::pdb;

namespace {

#define MOCK_SYMBOL_ACCESSOR(Func)                                             \
  decltype(std::declval<IPDBRawSymbol>().Func()) Func() const override {       \
    typedef decltype(IPDBRawSymbol::Func()) ReturnType;                        \
    return ReturnType();                                                       \
  }

class MockSession : public IPDBSession {
  uint64_t getLoadAddress() const override { return 0; }
  bool setLoadAddress(uint64_t Address) override { return false; }
  std::unique_ptr<PDBSymbolExe> getGlobalScope() override { return nullptr; }
  std::unique_ptr<PDBSymbol> getSymbolById(SymIndexId SymbolId) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBSourceFile>
  getSourceFileById(uint32_t SymbolId) const override {
    return nullptr;
  }
  bool addressForVA(uint64_t VA, uint32_t &Section,
                    uint32_t &Offset) const override {
    return false;
  }
  bool addressForRVA(uint32_t RVA, uint32_t &Section,
                     uint32_t &Offset) const override {
    return false;
  }
  std::unique_ptr<PDBSymbol>
  findSymbolByAddress(uint64_t Address, PDB_SymType Type) const override {
    return nullptr;
  }
  std::unique_ptr<PDBSymbol> findSymbolByRVA(uint32_t RVA,
                                             PDB_SymType Type) const override {
    return nullptr;
  }
  std::unique_ptr<PDBSymbol>
  findSymbolBySectOffset(uint32_t Sect, uint32_t Offset,
                         PDB_SymType Type) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumLineNumbers>
  findLineNumbers(const PDBSymbolCompiland &Compiland,
                  const IPDBSourceFile &File) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumLineNumbers>
  findLineNumbersByAddress(uint64_t Address, uint32_t Length) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumLineNumbers>
  findLineNumbersByRVA(uint32_t RVA, uint32_t Length) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumLineNumbers>
  findLineNumbersBySectOffset(uint32_t Section, uint32_t Offset,
                              uint32_t Length) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumSourceFiles>
  findSourceFiles(const PDBSymbolCompiland *Compiland, llvm::StringRef Pattern,
                  PDB_NameSearchFlags Flags) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBSourceFile>
  findOneSourceFile(const PDBSymbolCompiland *Compiland,
                    llvm::StringRef Pattern,
                    PDB_NameSearchFlags Flags) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumChildren<PDBSymbolCompiland>>
  findCompilandsForSourceFile(llvm::StringRef Pattern,
                              PDB_NameSearchFlags Flags) const override {
    return nullptr;
  }
  std::unique_ptr<PDBSymbolCompiland>
  findOneCompilandForSourceFile(llvm::StringRef Pattern,
                                PDB_NameSearchFlags Flags) const override {
    return nullptr;
  }

  std::unique_ptr<IPDBEnumSourceFiles> getAllSourceFiles() const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumSourceFiles> getSourceFilesForCompiland(
      const PDBSymbolCompiland &Compiland) const override {
    return nullptr;
  }

  std::unique_ptr<IPDBEnumDataStreams> getDebugStreams() const override {
    return nullptr;
  }

  std::unique_ptr<IPDBEnumTables> getEnumTables() const override {
    return nullptr;
  }

  std::unique_ptr<IPDBEnumInjectedSources> getInjectedSources() const override {
    return nullptr;
  }

  std::unique_ptr<IPDBEnumSectionContribs> getSectionContribs() const override {
    return nullptr;
  }

  std::unique_ptr<IPDBEnumFrameData> getFrameData() const override {
    return nullptr;
  }
};

class MockRawSymbol : public IPDBRawSymbol {
public:
  MockRawSymbol(PDB_SymType SymType)
      : Type(SymType) {}

  void dump(raw_ostream &OS, int Indent, PdbSymbolIdField ShowIdFields,
    PdbSymbolIdField RecurseIdFields) const override {}

  std::unique_ptr<IPDBEnumSymbols>
  findChildren(PDB_SymType Type) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumSymbols>
  findChildren(PDB_SymType Type, StringRef Name,
               PDB_NameSearchFlags Flags) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumSymbols>
  findChildrenByAddr(PDB_SymType Type, StringRef Name, PDB_NameSearchFlags Flags,
                     uint32_t Section, uint32_t Offset) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumSymbols>
  findChildrenByVA(PDB_SymType Type, StringRef Name, PDB_NameSearchFlags Flags,
                   uint64_t VA) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumSymbols>
  findChildrenByRVA(PDB_SymType Type, StringRef Name, PDB_NameSearchFlags Flags,
                    uint32_t RVA) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumSymbols>
  findInlineFramesByAddr(uint32_t Section, uint32_t Offset) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumSymbols>
  findInlineFramesByRVA(uint32_t RVA) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumSymbols>
  findInlineFramesByVA(uint64_t VA) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumLineNumbers> findInlineeLines() const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumLineNumbers>
  findInlineeLinesByAddr(uint32_t Section, uint32_t Offset,
                         uint32_t Length) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumLineNumbers>
  findInlineeLinesByRVA(uint32_t RVA, uint32_t Length) const override {
    return nullptr;
  }
  std::unique_ptr<IPDBEnumLineNumbers>
  findInlineeLinesByVA(uint64_t VA, uint32_t Length) const override {
    return nullptr;
  }

  void getDataBytes(llvm::SmallVector<uint8_t, 32> &bytes) const override {}
  void getFrontEndVersion(VersionInfo &Version) const override {}
  void getBackEndVersion(VersionInfo &Version) const override {}

  PDB_SymType getSymTag() const override { return Type; }

  std::string getUndecoratedNameEx(PDB_UndnameFlags Flags) const override {
    return {};
  }

  std::unique_ptr<IPDBLineNumber> getSrcLineOnTypeDefn() const override {
    return nullptr;
  }

  MOCK_SYMBOL_ACCESSOR(getAccess)
  MOCK_SYMBOL_ACCESSOR(getAddressOffset)
  MOCK_SYMBOL_ACCESSOR(getAddressSection)
  MOCK_SYMBOL_ACCESSOR(getAge)
  MOCK_SYMBOL_ACCESSOR(getArrayIndexTypeId)
  MOCK_SYMBOL_ACCESSOR(getBaseDataOffset)
  MOCK_SYMBOL_ACCESSOR(getBaseDataSlot)
  MOCK_SYMBOL_ACCESSOR(getBaseSymbolId)
  MOCK_SYMBOL_ACCESSOR(getBuiltinType)
  MOCK_SYMBOL_ACCESSOR(getBitPosition)
  MOCK_SYMBOL_ACCESSOR(getCallingConvention)
  MOCK_SYMBOL_ACCESSOR(getClassParentId)
  MOCK_SYMBOL_ACCESSOR(getCompilerName)
  MOCK_SYMBOL_ACCESSOR(getCount)
  MOCK_SYMBOL_ACCESSOR(getCountLiveRanges)
  MOCK_SYMBOL_ACCESSOR(getLanguage)
  MOCK_SYMBOL_ACCESSOR(getLexicalParentId)
  MOCK_SYMBOL_ACCESSOR(getLibraryName)
  MOCK_SYMBOL_ACCESSOR(getLiveRangeStartAddressOffset)
  MOCK_SYMBOL_ACCESSOR(getLiveRangeStartAddressSection)
  MOCK_SYMBOL_ACCESSOR(getLiveRangeStartRelativeVirtualAddress)
  MOCK_SYMBOL_ACCESSOR(getLocalBasePointerRegisterId)
  MOCK_SYMBOL_ACCESSOR(getLowerBoundId)
  MOCK_SYMBOL_ACCESSOR(getMemorySpaceKind)
  MOCK_SYMBOL_ACCESSOR(getName)
  MOCK_SYMBOL_ACCESSOR(getNumberOfAcceleratorPointerTags)
  MOCK_SYMBOL_ACCESSOR(getNumberOfColumns)
  MOCK_SYMBOL_ACCESSOR(getNumberOfModifiers)
  MOCK_SYMBOL_ACCESSOR(getNumberOfRegisterIndices)
  MOCK_SYMBOL_ACCESSOR(getNumberOfRows)
  MOCK_SYMBOL_ACCESSOR(getObjectFileName)
  MOCK_SYMBOL_ACCESSOR(getOemId)
  MOCK_SYMBOL_ACCESSOR(getOemSymbolId)
  MOCK_SYMBOL_ACCESSOR(getOffsetInUdt)
  MOCK_SYMBOL_ACCESSOR(getPlatform)
  MOCK_SYMBOL_ACCESSOR(getRank)
  MOCK_SYMBOL_ACCESSOR(getRegisterId)
  MOCK_SYMBOL_ACCESSOR(getRegisterType)
  MOCK_SYMBOL_ACCESSOR(getRelativeVirtualAddress)
  MOCK_SYMBOL_ACCESSOR(getSamplerSlot)
  MOCK_SYMBOL_ACCESSOR(getSignature)
  MOCK_SYMBOL_ACCESSOR(getSizeInUdt)
  MOCK_SYMBOL_ACCESSOR(getSlot)
  MOCK_SYMBOL_ACCESSOR(getSourceFileName)
  MOCK_SYMBOL_ACCESSOR(getStride)
  MOCK_SYMBOL_ACCESSOR(getSubTypeId)
  MOCK_SYMBOL_ACCESSOR(getSymbolsFileName)
  MOCK_SYMBOL_ACCESSOR(getSymIndexId)
  MOCK_SYMBOL_ACCESSOR(getTargetOffset)
  MOCK_SYMBOL_ACCESSOR(getTargetRelativeVirtualAddress)
  MOCK_SYMBOL_ACCESSOR(getTargetVirtualAddress)
  MOCK_SYMBOL_ACCESSOR(getTargetSection)
  MOCK_SYMBOL_ACCESSOR(getTextureSlot)
  MOCK_SYMBOL_ACCESSOR(getTimeStamp)
  MOCK_SYMBOL_ACCESSOR(getToken)
  MOCK_SYMBOL_ACCESSOR(getTypeId)
  MOCK_SYMBOL_ACCESSOR(getUavSlot)
  MOCK_SYMBOL_ACCESSOR(getUndecoratedName)
  MOCK_SYMBOL_ACCESSOR(getUnmodifiedTypeId)
  MOCK_SYMBOL_ACCESSOR(getUpperBoundId)
  MOCK_SYMBOL_ACCESSOR(getVirtualBaseDispIndex)
  MOCK_SYMBOL_ACCESSOR(getVirtualBaseOffset)
  MOCK_SYMBOL_ACCESSOR(getVirtualTableShapeId)
  MOCK_SYMBOL_ACCESSOR(getDataKind)
  MOCK_SYMBOL_ACCESSOR(getGuid)
  MOCK_SYMBOL_ACCESSOR(getOffset)
  MOCK_SYMBOL_ACCESSOR(getThisAdjust)
  MOCK_SYMBOL_ACCESSOR(getVirtualBasePointerOffset)
  MOCK_SYMBOL_ACCESSOR(getLocationType)
  MOCK_SYMBOL_ACCESSOR(getMachineType)
  MOCK_SYMBOL_ACCESSOR(getThunkOrdinal)
  MOCK_SYMBOL_ACCESSOR(getLength)
  MOCK_SYMBOL_ACCESSOR(getVirtualBaseTableType)
  MOCK_SYMBOL_ACCESSOR(getLiveRangeLength)
  MOCK_SYMBOL_ACCESSOR(getVirtualAddress)
  MOCK_SYMBOL_ACCESSOR(getUdtKind)
  MOCK_SYMBOL_ACCESSOR(hasConstructor)
  MOCK_SYMBOL_ACCESSOR(hasCustomCallingConvention)
  MOCK_SYMBOL_ACCESSOR(hasFarReturn)
  MOCK_SYMBOL_ACCESSOR(isCode)
  MOCK_SYMBOL_ACCESSOR(isCompilerGenerated)
  MOCK_SYMBOL_ACCESSOR(isConstType)
  MOCK_SYMBOL_ACCESSOR(isEditAndContinueEnabled)
  MOCK_SYMBOL_ACCESSOR(isFunction)
  MOCK_SYMBOL_ACCESSOR(getAddressTaken)
  MOCK_SYMBOL_ACCESSOR(getNoStackOrdering)
  MOCK_SYMBOL_ACCESSOR(hasAlloca)
  MOCK_SYMBOL_ACCESSOR(hasAssignmentOperator)
  MOCK_SYMBOL_ACCESSOR(hasCTypes)
  MOCK_SYMBOL_ACCESSOR(hasCastOperator)
  MOCK_SYMBOL_ACCESSOR(hasDebugInfo)
  MOCK_SYMBOL_ACCESSOR(hasEH)
  MOCK_SYMBOL_ACCESSOR(hasEHa)
  MOCK_SYMBOL_ACCESSOR(hasFramePointer)
  MOCK_SYMBOL_ACCESSOR(hasInlAsm)
  MOCK_SYMBOL_ACCESSOR(hasInlineAttribute)
  MOCK_SYMBOL_ACCESSOR(hasInterruptReturn)
  MOCK_SYMBOL_ACCESSOR(hasLongJump)
  MOCK_SYMBOL_ACCESSOR(hasManagedCode)
  MOCK_SYMBOL_ACCESSOR(hasNestedTypes)
  MOCK_SYMBOL_ACCESSOR(hasNoInlineAttribute)
  MOCK_SYMBOL_ACCESSOR(hasNoReturnAttribute)
  MOCK_SYMBOL_ACCESSOR(hasOptimizedCodeDebugInfo)
  MOCK_SYMBOL_ACCESSOR(hasOverloadedOperator)
  MOCK_SYMBOL_ACCESSOR(hasSEH)
  MOCK_SYMBOL_ACCESSOR(hasSecurityChecks)
  MOCK_SYMBOL_ACCESSOR(hasSetJump)
  MOCK_SYMBOL_ACCESSOR(hasStrictGSCheck)
  MOCK_SYMBOL_ACCESSOR(isAcceleratorGroupSharedLocal)
  MOCK_SYMBOL_ACCESSOR(isAcceleratorPointerTagLiveRange)
  MOCK_SYMBOL_ACCESSOR(isAcceleratorStubFunction)
  MOCK_SYMBOL_ACCESSOR(isAggregated)
  MOCK_SYMBOL_ACCESSOR(isIntroVirtualFunction)
  MOCK_SYMBOL_ACCESSOR(isCVTCIL)
  MOCK_SYMBOL_ACCESSOR(isConstructorVirtualBase)
  MOCK_SYMBOL_ACCESSOR(isCxxReturnUdt)
  MOCK_SYMBOL_ACCESSOR(isDataAligned)
  MOCK_SYMBOL_ACCESSOR(isHLSLData)
  MOCK_SYMBOL_ACCESSOR(isHotpatchable)
  MOCK_SYMBOL_ACCESSOR(isIndirectVirtualBaseClass)
  MOCK_SYMBOL_ACCESSOR(isInterfaceUdt)
  MOCK_SYMBOL_ACCESSOR(isIntrinsic)
  MOCK_SYMBOL_ACCESSOR(isLTCG)
  MOCK_SYMBOL_ACCESSOR(isLocationControlFlowDependent)
  MOCK_SYMBOL_ACCESSOR(isMSILNetmodule)
  MOCK_SYMBOL_ACCESSOR(isMatrixRowMajor)
  MOCK_SYMBOL_ACCESSOR(isManagedCode)
  MOCK_SYMBOL_ACCESSOR(isMSILCode)
  MOCK_SYMBOL_ACCESSOR(isMultipleInheritance)
  MOCK_SYMBOL_ACCESSOR(isNaked)
  MOCK_SYMBOL_ACCESSOR(isNested)
  MOCK_SYMBOL_ACCESSOR(isOptimizedAway)
  MOCK_SYMBOL_ACCESSOR(isPacked)
  MOCK_SYMBOL_ACCESSOR(isPointerBasedOnSymbolValue)
  MOCK_SYMBOL_ACCESSOR(isPointerToDataMember)
  MOCK_SYMBOL_ACCESSOR(isPointerToMemberFunction)
  MOCK_SYMBOL_ACCESSOR(isPureVirtual)
  MOCK_SYMBOL_ACCESSOR(isRValueReference)
  MOCK_SYMBOL_ACCESSOR(isRefUdt)
  MOCK_SYMBOL_ACCESSOR(isReference)
  MOCK_SYMBOL_ACCESSOR(isRestrictedType)
  MOCK_SYMBOL_ACCESSOR(isReturnValue)
  MOCK_SYMBOL_ACCESSOR(isSafeBuffers)
  MOCK_SYMBOL_ACCESSOR(isScoped)
  MOCK_SYMBOL_ACCESSOR(isSdl)
  MOCK_SYMBOL_ACCESSOR(isSingleInheritance)
  MOCK_SYMBOL_ACCESSOR(isSplitted)
  MOCK_SYMBOL_ACCESSOR(isStatic)
  MOCK_SYMBOL_ACCESSOR(hasPrivateSymbols)
  MOCK_SYMBOL_ACCESSOR(isUnalignedType)
  MOCK_SYMBOL_ACCESSOR(isUnreached)
  MOCK_SYMBOL_ACCESSOR(isValueUdt)
  MOCK_SYMBOL_ACCESSOR(isVirtual)
  MOCK_SYMBOL_ACCESSOR(isVirtualBaseClass)
  MOCK_SYMBOL_ACCESSOR(isVirtualInheritance)
  MOCK_SYMBOL_ACCESSOR(isVolatileType)
  MOCK_SYMBOL_ACCESSOR(getValue)
  MOCK_SYMBOL_ACCESSOR(wasInlined)
  MOCK_SYMBOL_ACCESSOR(getUnused)

private:
  PDB_SymType Type;
};

class PDBApiTest : public testing::Test {
public:
  std::unordered_map<PDB_SymType, std::unique_ptr<PDBSymbol>> SymbolMap;

  void SetUp() override {
    Session.reset(new MockSession());

    InsertItemWithTag(PDB_SymType::None);
    InsertItemWithTag(PDB_SymType::Exe);
    InsertItemWithTag(PDB_SymType::Compiland);
    InsertItemWithTag(PDB_SymType::CompilandDetails);
    InsertItemWithTag(PDB_SymType::CompilandEnv);
    InsertItemWithTag(PDB_SymType::Function);
    InsertItemWithTag(PDB_SymType::Block);
    InsertItemWithTag(PDB_SymType::Data);
    InsertItemWithTag(PDB_SymType::Annotation);
    InsertItemWithTag(PDB_SymType::Label);
    InsertItemWithTag(PDB_SymType::PublicSymbol);
    InsertItemWithTag(PDB_SymType::UDT);
    InsertItemWithTag(PDB_SymType::Enum);
    InsertItemWithTag(PDB_SymType::FunctionSig);
    InsertItemWithTag(PDB_SymType::PointerType);
    InsertItemWithTag(PDB_SymType::ArrayType);
    InsertItemWithTag(PDB_SymType::BuiltinType);
    InsertItemWithTag(PDB_SymType::Typedef);
    InsertItemWithTag(PDB_SymType::BaseClass);
    InsertItemWithTag(PDB_SymType::Friend);
    InsertItemWithTag(PDB_SymType::FunctionArg);
    InsertItemWithTag(PDB_SymType::FuncDebugStart);
    InsertItemWithTag(PDB_SymType::FuncDebugEnd);
    InsertItemWithTag(PDB_SymType::UsingNamespace);
    InsertItemWithTag(PDB_SymType::VTableShape);
    InsertItemWithTag(PDB_SymType::VTable);
    InsertItemWithTag(PDB_SymType::Custom);
    InsertItemWithTag(PDB_SymType::Thunk);
    InsertItemWithTag(PDB_SymType::CustomType);
    InsertItemWithTag(PDB_SymType::ManagedType);
    InsertItemWithTag(PDB_SymType::Dimension);
    InsertItemWithTag(PDB_SymType::Max);
  }

  template <class ExpectedType> void VerifyDyncast(PDB_SymType Tag) {
    for (auto item = SymbolMap.begin(); item != SymbolMap.end(); ++item) {
      EXPECT_EQ(item->first == Tag, llvm::isa<ExpectedType>(*item->second));
    }
  }

  void VerifyUnknownDyncasts() {
    for (auto item = SymbolMap.begin(); item != SymbolMap.end(); ++item) {
      bool should_match = false;
      if (item->first == PDB_SymType::None || item->first >= PDB_SymType::Max)
        should_match = true;

      EXPECT_EQ(should_match, llvm::isa<PDBSymbolUnknown>(*item->second));
    }
  }

private:
  std::unique_ptr<IPDBSession> Session;

  void InsertItemWithTag(PDB_SymType Tag) {
    auto RawSymbol = llvm::make_unique<MockRawSymbol>(Tag);
    auto Symbol = PDBSymbol::create(*Session, std::move(RawSymbol));
    SymbolMap.insert(std::make_pair(Tag, std::move(Symbol)));
  }
};

TEST_F(PDBApiTest, Dyncast) {

  // Most of the types have a one-to-one mapping between Tag and concrete type.
  VerifyDyncast<PDBSymbolExe>(PDB_SymType::Exe);
  VerifyDyncast<PDBSymbolCompiland>(PDB_SymType::Compiland);
  VerifyDyncast<PDBSymbolCompilandDetails>(PDB_SymType::CompilandDetails);
  VerifyDyncast<PDBSymbolCompilandEnv>(PDB_SymType::CompilandEnv);
  VerifyDyncast<PDBSymbolFunc>(PDB_SymType::Function);
  VerifyDyncast<PDBSymbolBlock>(PDB_SymType::Block);
  VerifyDyncast<PDBSymbolData>(PDB_SymType::Data);
  VerifyDyncast<PDBSymbolAnnotation>(PDB_SymType::Annotation);
  VerifyDyncast<PDBSymbolLabel>(PDB_SymType::Label);
  VerifyDyncast<PDBSymbolPublicSymbol>(PDB_SymType::PublicSymbol);
  VerifyDyncast<PDBSymbolTypeUDT>(PDB_SymType::UDT);
  VerifyDyncast<PDBSymbolTypeEnum>(PDB_SymType::Enum);
  VerifyDyncast<PDBSymbolTypeFunctionSig>(PDB_SymType::FunctionSig);
  VerifyDyncast<PDBSymbolTypePointer>(PDB_SymType::PointerType);
  VerifyDyncast<PDBSymbolTypeArray>(PDB_SymType::ArrayType);
  VerifyDyncast<PDBSymbolTypeBuiltin>(PDB_SymType::BuiltinType);
  VerifyDyncast<PDBSymbolTypeTypedef>(PDB_SymType::Typedef);
  VerifyDyncast<PDBSymbolTypeBaseClass>(PDB_SymType::BaseClass);
  VerifyDyncast<PDBSymbolTypeFriend>(PDB_SymType::Friend);
  VerifyDyncast<PDBSymbolTypeFunctionArg>(PDB_SymType::FunctionArg);
  VerifyDyncast<PDBSymbolFuncDebugStart>(PDB_SymType::FuncDebugStart);
  VerifyDyncast<PDBSymbolFuncDebugEnd>(PDB_SymType::FuncDebugEnd);
  VerifyDyncast<PDBSymbolUsingNamespace>(PDB_SymType::UsingNamespace);
  VerifyDyncast<PDBSymbolTypeVTableShape>(PDB_SymType::VTableShape);
  VerifyDyncast<PDBSymbolTypeVTable>(PDB_SymType::VTable);
  VerifyDyncast<PDBSymbolCustom>(PDB_SymType::Custom);
  VerifyDyncast<PDBSymbolThunk>(PDB_SymType::Thunk);
  VerifyDyncast<PDBSymbolTypeCustom>(PDB_SymType::CustomType);
  VerifyDyncast<PDBSymbolTypeManaged>(PDB_SymType::ManagedType);
  VerifyDyncast<PDBSymbolTypeDimension>(PDB_SymType::Dimension);

  VerifyUnknownDyncasts();
}
} // end anonymous namespace
