//===- COFFStaticLibraryDefinitionGeneratorTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/COFF.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

#if defined(__x86_64__) || defined(_M_X64)

static int ImportedData = 42;
static const int ImportedConstData = 43;
static int ControllerData = 45;
static int importedFunction() { return 44; }
static int controllerFunction() { return 46; }

class SuspendingDefinitionGenerator : public DefinitionGenerator {
public:
  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) override {
    PendingLookup = std::move(LS);
    return Error::success();
  }

  bool hasPendingLookup() const { return PendingLookup.has_value(); }

  void resume() {
    assert(PendingLookup && "no lookup to resume");
    auto LS = std::move(*PendingLookup);
    PendingLookup.reset();
    LS.continueLookup(Error::success());
  }

private:
  std::optional<LookupState> PendingLookup;
};

class COFFStaticLibraryDefinitionGeneratorTest : public testing::Test {
public:
  COFFStaticLibraryDefinitionGeneratorTest()
      : ES(cantFail(SelfExecutorProcessControl::Create())),
        SourceJD(ES.createBareJITDylib("source")),
        TargetJD(ES.createBareJITDylib("target")),
        ObjLinkingLayer(
            ES, std::make_unique<jitlink::InProcessMemoryManager>(4096)) {}

  ~COFFStaticLibraryDefinitionGeneratorTest() override {
    if (auto Err = ES.endSession())
      ES.reportError(std::move(Err));
  }

protected:
  Error addObject(StringRef YAML) {
    SmallString<0> Storage;
    std::string YAMLFailure;
    auto Obj = yaml::yaml2ObjectFile(
        Storage, YAML, [&](const Twine &Msg) { YAMLFailure = Msg.str(); });
    if (!Obj)
      return make_error<StringError>(YAMLFailure, inconvertibleErrorCode());
    auto ObjBuffer = MemoryBuffer::getMemBufferCopy(
        Obj->getMemoryBufferRef().getBuffer(), "test.obj");
    return ObjLinkingLayer.add(TargetJD, std::move(ObjBuffer));
  }

  Expected<std::unique_ptr<COFFStaticLibraryDefinitionGenerator>>
  createImportLibrary(ArrayRef<object::COFFShortExport> Exports,
                      std::set<std::string> &ImportedLibraries) {
    unittest::TempDir Tmp("coff-import-library", /*Unique=*/true);
    SmallString<128> Path(Tmp.path().begin(), Tmp.path().end());
    sys::path::append(Path, "test.lib");

    if (auto Err = object::writeImportLibrary("test.dll", Path, Exports,
                                              COFF::IMAGE_FILE_MACHINE_AMD64,
                                              /*MinGW=*/false))
      return std::move(Err);
    return COFFStaticLibraryDefinitionGenerator::Load(
        ObjLinkingLayer, Path.c_str(), ImportedLibraries);
  }

  Expected<std::unique_ptr<COFFStaticLibraryDefinitionGenerator>>
  createObjectArchive(std::set<std::string> &ImportedLibraries) {
    SmallString<0> Storage;
    std::string YAML;
    raw_string_ostream OS(YAML);
    OS << R"(
--- !COFF
header:
  Machine: IMAGE_FILE_MACHINE_AMD64
  Characteristics: [ ]
sections:
  - Name: .text
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    Alignment: 16
    SectionData: '4883EC28FF15000000004883C428488B0D000000000301C3'
    Relocations:
      - VirtualAddress: 6
        SymbolName: __imp_controller_function
        Type: IMAGE_REL_AMD64_REL32
      - VirtualAddress: 17
        SymbolName: __imp_controller_data
        Type: IMAGE_REL_AMD64_REL32
symbols:
  - Name: selected_member
    Value: 0
    SectionNumber: 1
    SimpleType: IMAGE_SYM_TYPE_NULL
    ComplexType: IMAGE_SYM_DTYPE_FUNCTION
    StorageClass: IMAGE_SYM_CLASS_EXTERNAL
  - Name: __imp_controller_data
    Value: 0
    SectionNumber: 0
    SimpleType: IMAGE_SYM_TYPE_NULL
    ComplexType: IMAGE_SYM_DTYPE_NULL
    StorageClass: IMAGE_SYM_CLASS_EXTERNAL
  - Name: __imp_controller_function
    Value: 0
    SectionNumber: 0
    SimpleType: IMAGE_SYM_TYPE_NULL
    ComplexType: IMAGE_SYM_DTYPE_FUNCTION
    StorageClass: IMAGE_SYM_CLASS_EXTERNAL
)";

    std::string YAMLFailure;
    auto Obj = yaml::yaml2ObjectFile(
        Storage, OS.str(), [&](const Twine &Msg) { YAMLFailure = Msg.str(); });
    if (!Obj)
      return make_error<StringError>(YAMLFailure, inconvertibleErrorCode());

    NewArchiveMember Member(Obj->getMemoryBufferRef());
    Member.MemberName = "controller.obj";
    SmallVector<NewArchiveMember, 1> Members;
    Members.push_back(std::move(Member));
    auto ArchiveBuffer = writeArchiveToBuffer(
        Members, SymtabWritingMode::NormalSymtab, object::Archive::K_COFF,
        /*Deterministic=*/true, /*Thin=*/false);
    if (!ArchiveBuffer)
      return ArchiveBuffer.takeError();
    auto Archive = object::Archive::create((*ArchiveBuffer)->getMemBufferRef());
    if (!Archive)
      return Archive.takeError();
    return COFFStaticLibraryDefinitionGenerator::Create(
        ObjLinkingLayer, std::move(*ArchiveBuffer), std::move(*Archive),
        ImportedLibraries);
  }

  ExecutionSession ES;
  JITDylib &SourceJD;
  JITDylib &TargetJD;
  ObjectLinkingLayer ObjLinkingLayer;
};

TEST_F(COFFStaticLibraryDefinitionGeneratorTest,
       CodeImportProvidesPointerAndThunk) {
  auto Function = ES.intern("imported_function");
  cantFail(SourceJD.define(absoluteSymbols(
      {{Function,
        {ExecutorAddr::fromPtr(&importedFunction),
         JITSymbolFlags::Exported | JITSymbolFlags::Callable}}})));
  TargetJD.addToLinkOrder(SourceJD);

  object::COFFShortExport Export;
  Export.Name = "imported_function";
  std::set<std::string> ImportedLibraries;
  auto G = createImportLibrary({Export}, ImportedLibraries);
  ASSERT_THAT_EXPECTED(G, Succeeded());
  EXPECT_EQ(ImportedLibraries.count("test.dll"), 1U);
  TargetJD.addGenerator(std::move(*G));

  auto ImpFunction = ES.lookup(&TargetJD, "__imp_imported_function");
  ASSERT_THAT_EXPECTED(ImpFunction, Succeeded());
  EXPECT_EQ(*ImpFunction->getAddress().toPtr<void **>(), &importedFunction);

  TargetJD.setLinkOrder({});
  auto FunctionThunk = ES.lookup(&TargetJD, "imported_function");
  ASSERT_THAT_EXPECTED(FunctionThunk, Succeeded());
  EXPECT_EQ(FunctionThunk->getAddress().toPtr<int (*)()>()(), 44);
}

TEST_F(COFFStaticLibraryDefinitionGeneratorTest,
       DataImportProvidesOnlyPointer) {
  auto Data = ES.intern("imported_data");
  cantFail(SourceJD.define(absoluteSymbols(
      {{Data,
        {ExecutorAddr::fromPtr(&ImportedData), JITSymbolFlags::Exported}}})));
  TargetJD.addToLinkOrder(SourceJD);

  object::COFFShortExport Export;
  Export.Name = "imported_data";
  Export.Data = true;
  std::set<std::string> ImportedLibraries;
  auto G = createImportLibrary({Export}, ImportedLibraries);
  ASSERT_THAT_EXPECTED(G, Succeeded());
  TargetJD.addGenerator(std::move(*G));

  auto ImpData = ES.lookup(&TargetJD, "__imp_imported_data");
  ASSERT_THAT_EXPECTED(ImpData, Succeeded());
  EXPECT_EQ(*ImpData->getAddress().toPtr<void **>(), &ImportedData);

  TargetJD.setLinkOrder({});
  EXPECT_THAT_EXPECTED(ES.lookup(&TargetJD, "imported_data"), Failed());
}

TEST_F(COFFStaticLibraryDefinitionGeneratorTest,
       ConstImportNamesShareImportAddress) {
  auto ConstData = ES.intern("imported_const_data");
  cantFail(SourceJD.define(
      absoluteSymbols({{ConstData,
                        {ExecutorAddr::fromPtr(&ImportedConstData),
                         JITSymbolFlags::Exported}}})));
  TargetJD.addToLinkOrder(SourceJD);

  object::COFFShortExport Export;
  Export.Name = "imported_const_data";
  Export.Constant = true;
  std::set<std::string> ImportedLibraries;
  auto G = createImportLibrary({Export}, ImportedLibraries);
  ASSERT_THAT_EXPECTED(G, Succeeded());
  TargetJD.addGenerator(std::move(*G));

  auto ImpConstData = ES.lookup(&TargetJD, "__imp_imported_const_data");
  ASSERT_THAT_EXPECTED(ImpConstData, Succeeded());
  EXPECT_EQ(*ImpConstData->getAddress().toPtr<const void **>(),
            &ImportedConstData);

  TargetJD.setLinkOrder({});
  auto ConstDataImport = ES.lookup(&TargetJD, "imported_const_data");
  ASSERT_THAT_EXPECTED(ConstDataImport, Succeeded());
  EXPECT_EQ(ConstDataImport->getAddress(), ImpConstData->getAddress());
}

TEST_F(COFFStaticLibraryDefinitionGeneratorTest, ExportAsResolvesDLLName) {
  auto ExportedFunction = ES.intern("exported_function");
  cantFail(SourceJD.define(absoluteSymbols(
      {{ExportedFunction,
        {ExecutorAddr::fromPtr(&importedFunction),
         JITSymbolFlags::Exported | JITSymbolFlags::Callable}}})));
  TargetJD.addToLinkOrder(SourceJD);

  object::COFFShortExport Export;
  Export.Name = "local_function";
  Export.ExportAs = "exported_function";
  std::set<std::string> ImportedLibraries;
  auto G = createImportLibrary({Export}, ImportedLibraries);
  ASSERT_THAT_EXPECTED(G, Succeeded());
  TargetJD.addGenerator(std::move(*G));

  auto ImpFunction = ES.lookup(&TargetJD, "__imp_local_function");
  ASSERT_THAT_EXPECTED(ImpFunction, Succeeded());
  EXPECT_EQ(*ImpFunction->getAddress().toPtr<void **>(), &importedFunction);

  TargetJD.setLinkOrder({});
  auto FunctionThunk = ES.lookup(&TargetJD, "local_function");
  ASSERT_THAT_EXPECTED(FunctionThunk, Succeeded());
  EXPECT_EQ(FunctionThunk->getAddress().toPtr<int (*)()>()(), 44);
}

TEST_F(COFFStaticLibraryDefinitionGeneratorTest,
       OrdinalImportReportsUnsupported) {
  object::COFFShortExport Export;
  Export.Name = "ordinal_function";
  Export.Ordinal = 1;
  Export.Noname = true;
  std::set<std::string> ImportedLibraries;
  auto G = createImportLibrary({Export}, ImportedLibraries);
  ASSERT_THAT_EXPECTED(G, Succeeded());
  TargetJD.addGenerator(std::move(*G));

  EXPECT_THAT_EXPECTED(
      ES.lookup(&TargetJD, "__imp_ordinal_function"),
      FailedWithMessage(
          "ordinal COFF imports are not supported for ordinal_function"));
}

TEST_F(COFFStaticLibraryDefinitionGeneratorTest,
       SuspendsWhileResolvingImportTarget) {
  auto ExportedFunction = ES.intern("async_exported_function");
  auto ImpFunction = ES.intern("__imp_async_local_function");
  auto &SourceGenerator =
      SourceJD.addGenerator(std::make_unique<SuspendingDefinitionGenerator>());
  TargetJD.addToLinkOrder(SourceJD);

  object::COFFShortExport Export;
  Export.Name = "async_local_function";
  Export.ExportAs = "async_exported_function";
  std::set<std::string> ImportedLibraries;
  auto G = createImportLibrary({Export}, ImportedLibraries);
  ASSERT_THAT_EXPECTED(G, Succeeded());
  TargetJD.addGenerator(std::move(*G));

  std::optional<Expected<SymbolMap>> LookupResult;
  ES.lookup(
      LookupKind::Static,
      {{&TargetJD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
      SymbolLookupSet(ImpFunction), SymbolState::Ready,
      [&](Expected<SymbolMap> Result) {
        LookupResult.emplace(std::move(Result));
      },
      NoDependenciesToRegister);

  EXPECT_FALSE(LookupResult.has_value());
  ASSERT_TRUE(SourceGenerator.hasPendingLookup());

  cantFail(SourceJD.define(absoluteSymbols(
      {{ExportedFunction,
        {ExecutorAddr::fromPtr(&importedFunction),
         JITSymbolFlags::Exported | JITSymbolFlags::Callable}}})));
  SourceGenerator.resume();

  ASSERT_TRUE(LookupResult.has_value());
  ASSERT_THAT_EXPECTED(*LookupResult, Succeeded());
  auto I = (*LookupResult)->find(ImpFunction);
  ASSERT_NE(I, (*LookupResult)->end());
  EXPECT_EQ(*I->second.getAddress().toPtr<void **>(), &importedFunction);
}

TEST_F(COFFStaticLibraryDefinitionGeneratorTest,
       ObjectFunctionImportExportAsPreservesIATAddress) {
  auto ExportedFunction = ES.intern("exported_function");
  cantFail(SourceJD.define(absoluteSymbols(
      {{ExportedFunction,
        {ExecutorAddr::fromPtr(&importedFunction),
         JITSymbolFlags::Exported | JITSymbolFlags::Callable}}})));
  TargetJD.addToLinkOrder(SourceJD);

  object::COFFShortExport Export;
  Export.Name = "local_function";
  Export.ExportAs = "exported_function";
  std::set<std::string> ImportedLibraries;
  auto G = createImportLibrary({Export}, ImportedLibraries);
  ASSERT_THAT_EXPECTED(G, Succeeded());
  TargetJD.addGenerator(std::move(*G));

  ASSERT_THAT_ERROR(addObject(R"(
--- !COFF
header:
  Machine: IMAGE_FILE_MACHINE_AMD64
sections:
  - Name: .text
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    Alignment: 16
    SectionData: '488B0500000000C3'
    Relocations:
      - VirtualAddress: 3
        SymbolName: __imp_local_function
        Type: IMAGE_REL_AMD64_REL32
symbols:
  - Name: import_address
    Value: 0
    SectionNumber: 1
    SimpleType: IMAGE_SYM_TYPE_NULL
    ComplexType: IMAGE_SYM_DTYPE_FUNCTION
    StorageClass: IMAGE_SYM_CLASS_EXTERNAL
  - Name: __imp_local_function
    Value: 0
    SectionNumber: 0
    SimpleType: IMAGE_SYM_TYPE_NULL
    ComplexType: IMAGE_SYM_DTYPE_NULL
    StorageClass: IMAGE_SYM_CLASS_EXTERNAL
)"),
                    Succeeded());

  auto ImportAddress = ES.lookup(&TargetJD, "import_address");
  ASSERT_THAT_EXPECTED(ImportAddress, Succeeded());
  EXPECT_EQ(ImportAddress->getAddress().toPtr<void *(*)()>()(),
            reinterpret_cast<void *>(&importedFunction));
}

TEST_F(COFFStaticLibraryDefinitionGeneratorTest,
       ObjectDataImportExportAsResolvesDLLName) {
  auto ExportedData = ES.intern("exported_data");
  cantFail(SourceJD.define(absoluteSymbols(
      {{ExportedData,
        {ExecutorAddr::fromPtr(&ImportedData), JITSymbolFlags::Exported}}})));
  TargetJD.addToLinkOrder(SourceJD);

  object::COFFShortExport Export;
  Export.Name = "local_data";
  Export.ExportAs = "exported_data";
  Export.Data = true;
  std::set<std::string> ImportedLibraries;
  auto G = createImportLibrary({Export}, ImportedLibraries);
  ASSERT_THAT_EXPECTED(G, Succeeded());
  TargetJD.addGenerator(std::move(*G));

  ASSERT_THAT_ERROR(addObject(R"(
--- !COFF
header:
  Machine: IMAGE_FILE_MACHINE_AMD64
sections:
  - Name: .text
    Characteristics: [ IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ ]
    Alignment: 16
    SectionData: '488B05000000008B00C3'
    Relocations:
      - VirtualAddress: 3
        SymbolName: __imp_local_data
        Type: IMAGE_REL_AMD64_REL32
symbols:
  - Name: read_imported_data
    Value: 0
    SectionNumber: 1
    SimpleType: IMAGE_SYM_TYPE_NULL
    ComplexType: IMAGE_SYM_DTYPE_FUNCTION
    StorageClass: IMAGE_SYM_CLASS_EXTERNAL
  - Name: __imp_local_data
    Value: 0
    SectionNumber: 0
    SimpleType: IMAGE_SYM_TYPE_NULL
    ComplexType: IMAGE_SYM_DTYPE_NULL
    StorageClass: IMAGE_SYM_CLASS_EXTERNAL
)"),
                    Succeeded());

  auto ReadImportedData = ES.lookup(&TargetJD, "read_imported_data");
  ASSERT_THAT_EXPECTED(ReadImportedData, Succeeded());
  EXPECT_EQ(ReadImportedData->getAddress().toPtr<int (*)()>()(), ImportedData);
}

TEST_F(COFFStaticLibraryDefinitionGeneratorTest,
       OrdinaryObjectUsesJITLinkIATEntries) {
  auto Data = ES.intern("controller_data");
  auto Function = ES.intern("controller_function");
  cantFail(SourceJD.define(absoluteSymbols(
      {{Data,
        {ExecutorAddr::fromPtr(&ControllerData), JITSymbolFlags::Exported}},
       {Function,
        {ExecutorAddr::fromPtr(&controllerFunction),
         JITSymbolFlags::Exported | JITSymbolFlags::Callable}}})));
  TargetJD.addToLinkOrder(SourceJD);

  std::set<std::string> ImportedLibraries;
  auto G = createObjectArchive(ImportedLibraries);
  ASSERT_THAT_EXPECTED(G, Succeeded());
  EXPECT_TRUE(ImportedLibraries.empty());
  TargetJD.addGenerator(std::move(*G));

  auto SelectedMember = ES.lookup(&TargetJD, "selected_member");
  ASSERT_THAT_EXPECTED(SelectedMember, Succeeded());
  EXPECT_EQ(SelectedMember->getAddress().toPtr<int (*)()>()(), 91);

  TargetJD.setLinkOrder({});
  EXPECT_THAT_EXPECTED(ES.lookup(&TargetJD, "controller_function"), Failed());
}

#endif // x86_64

} // namespace
