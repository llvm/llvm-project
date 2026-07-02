//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/ClangASTMetadata.h"
#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/Platform/Windows/PlatformWindows.h"
#include "Plugins/SymbolFile/NativePDB/SymbolFileNativePDB.h"
#include "Plugins/SymbolFile/NativePDB/UdtRecordCompleter.h"
#include "Plugins/SymbolFile/PDB/SymbolFilePDB.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "llvm/DebugInfo/CodeView/AppendingTypeTableBuilder.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/PDB/Native/DbiStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/GSIStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/InfoStreamBuilder.h"
#include "llvm/DebugInfo/PDB/Native/PDBFileBuilder.h"
#include "llvm/DebugInfo/PDB/Native/TpiHashing.h"
#include "llvm/DebugInfo/PDB/Native/TpiStreamBuilder.h"
#include "llvm/ObjectYAML/COFFYAML.h"
#include "llvm/ObjectYAML/CodeViewYAMLTypes.h"
#include "llvm/ObjectYAML/yaml2obj.h"

#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Utility/ArchSpec.h"

#include "gtest/gtest.h"

#include <optional>

using namespace lldb_private;
using namespace lldb_private::npdb;
using namespace llvm;
using namespace llvm::codeview;

namespace {

struct TpiStreamYaml {
  std::vector<CodeViewYAML::LeafRecord> records;
};

struct MinimalPdbYaml {
  TpiStreamYaml tpi;
  // FIXME: Take DBI section headers from YAML too.
};

} // namespace

template <> struct llvm::yaml::MappingTraits<TpiStreamYaml> {
  static void mapping(IO &io, TpiStreamYaml &tpi) {
    io.mapRequired("Records", tpi.records);
  }
};

template <> struct llvm::yaml::MappingTraits<MinimalPdbYaml> {
  static void mapping(IO &io, MinimalPdbYaml &pdb) {
    io.mapRequired("TpiStream", pdb.tpi);
  }
};

static object::coff_section g_fixed_coff_sections[]{
    {
        /*.Name=*/".text",
        /*.VirtualSize=*/ulittle32_t(542),
        /*.VirtualAddress=*/ulittle32_t(4096),
        /*.SizeOfRawData=*/ulittle32_t(1024),
        /*.PointerToRawData=*/ulittle32_t(1024),
        /*.PointerToRelocations=*/ulittle32_t(0),
        /*.PointerToLinenumbers=*/ulittle32_t(0),
        /*.NumberOfRelocations=*/ulittle16_t(0),
        /*.NumberOfLinenumbers=*/ulittle16_t(0),
        /*.Characteristics=*/
        ulittle32_t(COFF::IMAGE_SCN_CNT_CODE | COFF::IMAGE_SCN_MEM_EXECUTE |
                    COFF::IMAGE_SCN_MEM_READ),
    },
    {
        /*.Name=*/".rdata",
        /*.VirtualSize=*/ulittle32_t(244),
        /*.VirtualAddress=*/ulittle32_t(8192),
        /*.SizeOfRawData=*/ulittle32_t(512),
        /*.PointerToRawData=*/ulittle32_t(2048),
        /*.PointerToRelocations=*/ulittle32_t(0),
        /*.PointerToLinenumbers=*/ulittle32_t(0),
        /*.NumberOfRelocations=*/ulittle16_t(0),
        /*.NumberOfLinenumbers=*/ulittle16_t(0),
        /*.Characteristics=*/
        ulittle32_t(COFF::IMAGE_SCN_CNT_CODE | COFF::IMAGE_SCN_MEM_READ),
    },
    {
        /*.Name=*/".pdata",
        /*.VirtualSize=*/ulittle32_t(12),
        /*.VirtualAddress=*/ulittle32_t(16384),
        /*.SizeOfRawData=*/ulittle32_t(512),
        /*.PointerToRawData=*/ulittle32_t(2048),
        /*.PointerToRelocations=*/ulittle32_t(0),
        /*.PointerToLinenumbers=*/ulittle32_t(0),
        /*.NumberOfRelocations=*/ulittle16_t(0),
        /*.NumberOfLinenumbers=*/ulittle16_t(0),
        /*.Characteristics=*/
        ulittle32_t(COFF::IMAGE_SCN_CNT_CODE | COFF::IMAGE_SCN_MEM_READ),
    },
};

/// Create a PDB from the YAML input.
///
/// This is essentially a trimmed down version of llvm-pdbutil's yamlToPdb which
/// we can't use directly here.
static Expected<std::string> createPdb() {
  std::string input_path = GetInputFilePath("dynamic-types.yaml");
  ErrorOr<std::unique_ptr<MemoryBuffer>> in_buffer =
      llvm::MemoryBuffer::getFile(input_path, /*IsText=*/true);
  if (std::error_code ec = in_buffer.getError())
    return errorCodeToError(ec);

  MinimalPdbYaml minimal_pdb;
  yaml::Input yin(*in_buffer.get());
  // FIXME: Remove this once we can build a PDB with ObjectYAML or similar.
  yin.setAllowUnknownKeys(true);
  yin >> minimal_pdb;
  if (yin.error())
    return errorCodeToError(yin.error());

  BumpPtrAllocator allocator;
  pdb::PDBFileBuilder builder(allocator);
  Error err = builder.initialize(/*BlockSize=*/4096);
  if (err)
    return std::move(err);

  for (uint32_t I = 0; I < pdb::kSpecialStreamCount; ++I) {
    Expected<uint32_t> res = builder.getMsfBuilder().addStream(0);
    if (!res)
      return res.takeError();
  }

  pdb::InfoStreamBuilder &info = builder.getInfoBuilder();
  info.setAge(1);
  // GUID from dynamic-types-exe.yaml: {863E815D-0880-FDB2-4C4C-44205044422E}
  info.setGuid({0x5D, 0x81, 0x3E, 0x86, 0x80, 0x08, 0xB2, 0xFD, 0x4C, 0x4C,
                0x44, 0x20, 0x50, 0x44, 0x42, 0x2E});
  info.setSignature(0);
  info.setVersion(pdb::PdbImplVC70);
  info.addFeature(pdb::PdbRaw_FeatureSig::VC140);

  pdb::TpiStreamBuilder &tpi = builder.getTpiBuilder();
  codeview::AppendingTypeTableBuilder type_builder(allocator);
  tpi.setVersionHeader(pdb::PdbTpiV80);
  for (const auto &type : minimal_pdb.tpi.records) {
    codeview::CVType Type = type.toCodeViewRecord(type_builder);
    Expected<uint32_t> Hash = pdb::hashTypeRecord(Type);
    if (!Hash)
      return Hash.takeError();
    tpi.addTypeRecord(Type.RecordData, *Hash);
  }

  pdb::TpiStreamBuilder &ipi = builder.getIpiBuilder();
  ipi.setVersionHeader(pdb::PdbTpiV80);

  pdb::GSIStreamBuilder &gsi = builder.getGsiBuilder();
  // Add a fake public symbol to make sure the PDB has a publics stream.
  std::vector<pdb::BulkPublic> publics{pdb::BulkPublic()};
  gsi.addPublicSymbols(std::move(publics));

  pdb::DbiStreamBuilder &dbi = builder.getDbiBuilder();
  dbi.setAge(1);
  dbi.setBuildNumber(0);
  dbi.setFlags(1);
  dbi.setMachineType(pdb::PDB_Machine::Amd64);
  dbi.setPdbDllRbld(0);
  dbi.setPdbDllVersion(0);
  dbi.setVersionHeader(pdb::PdbDbiV70);

  ArrayRef<object::coff_section> coff_sections(g_fixed_coff_sections);
  dbi.createSectionMap(coff_sections);
  err = dbi.addDbgStream(
      pdb::DbgHeaderType::SectionHdr,
      ArrayRef<uint8_t>{(const uint8_t *)coff_sections.data(),
                        coff_sections.size() * sizeof(object::coff_section)});
  if (err)
    return std::move(err);

  std::string output_path = GetInputFilePath("dynamic-types.pdb");
  GUID out_guid;
  err = builder.commit(output_path, &out_guid);
  if (err)
    return std::move(err);
  return output_path;
}

static Expected<std::string> createExe() {
  std::string input_path = GetInputFilePath("dynamic-types-exe.yaml");
  ErrorOr<std::unique_ptr<MemoryBuffer>> in_buffer =
      llvm::MemoryBuffer::getFile(input_path, /*IsText=*/true);
  if (std::error_code ec = in_buffer.getError())
    return errorCodeToError(ec);

  COFFYAML::Object obj;
  yaml::Input yin(*in_buffer.get());
  yin >> obj;
  if (yin.error())
    return errorCodeToError(yin.error());

  std::string output_path = GetInputFilePath("dynamic-types.exe");
  std::error_code ec;
  raw_fd_ostream os(output_path, ec);
  if (ec)
    return errorCodeToError(ec);
  std::string err;
  bool ok = yaml::yaml2coff(obj, os,
                            [&](const Twine &msg) { err.append(msg.str()); });
  if (!ok)
    return createStringError(err);
  return output_path;
}

class SymbolFilePDBTests : public testing::Test {
public:
  void SetUp() override {
    ArchSpec arch("x86_64-pc-windows-msvc");
    Platform::SetHostPlatform(PlatformWindows::CreateInstance(true, &arch));
    m_debugger_sp = Debugger::CreateInstance();
    m_debugger_sp->SetPropertyValue(nullptr,
                                    lldb_private::eVarSetOperationAssign,
                                    "plugin.symbol-file.pdb.reader", "native");
  }

  std::optional<ClangASTMetadata> GetMetadataFor(SymbolFile *symfile,
                                                 llvm::StringRef query,
                                                 TypeSystemClang *clang) {
    TypeResults results;
    symfile->FindTypes(TypeQuery(query), results);
    lldb::TypeSP type_sp = results.GetFirstType();
    if (!type_sp)
      return std::nullopt;

    CompilerType ct = type_sp->GetFullCompilerType();
    ct.GetCompleteType();
    if (!ct.IsValid())
      return std::nullopt;

    ct.IsPossibleDynamicType(nullptr, true, false);

    clang::TagDecl *tag_decl = clang->GetAsTagDecl(ct);
    if (!tag_decl)
      return std::nullopt;

    return clang->GetMetadata(tag_decl);
  }

protected:
  SubsystemRAII<FileSystem, HostInfo, ObjectFilePECOFF, SymbolFileNativePDB,
                SymbolFilePDB, TypeSystemClang>
      m_subsystems;
  lldb::DebuggerSP m_debugger_sp;
};

TEST_F(SymbolFilePDBTests, TestDynamicCxxType) {
  Expected<std::string> pdb_path = createPdb();
  EXPECT_THAT_EXPECTED(pdb_path, Succeeded());
  Expected<std::string> exe_path = createExe();
  EXPECT_THAT_EXPECTED(exe_path, Succeeded());

  FileSpec fspec(*exe_path);
  ArchSpec aspec("x86_64-pc-windows");
  lldb::ModuleSP module = std::make_shared<Module>(fspec, aspec);

  SymbolFile *symfile = module->GetSymbolFile();
  ASSERT_TRUE(symfile);
  ASSERT_TRUE(llvm::isa<SymbolFileNativePDB>(symfile));

  auto ts = symfile->GetTypeSystemForLanguage(lldb::eLanguageTypeC_plus_plus);
  ASSERT_TRUE(bool(ts));
  TypeSystemClang *clang = llvm::dyn_cast_or_null<TypeSystemClang>(ts->get());
  ASSERT_NE(clang, nullptr);

  auto using_base_meta = GetMetadataFor(symfile, "UsingBase", clang);
  ASSERT_TRUE(using_base_meta.has_value());
  ASSERT_TRUE(using_base_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(using_base_meta->GetIsDynamicCXXType(), true); // has vtable

  auto base_meta = GetMetadataFor(symfile, "Base", clang);
  ASSERT_TRUE(base_meta.has_value());
  ASSERT_TRUE(base_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(base_meta->GetIsDynamicCXXType(), true); // has vtable

  auto vbase_meta = GetMetadataFor(symfile, "VBase", clang);
  ASSERT_TRUE(vbase_meta.has_value());
  ASSERT_TRUE(vbase_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(vbase_meta->GetIsDynamicCXXType(), false); // empty struct

  auto using_vbase_meta = GetMetadataFor(symfile, "UsingVBase", clang);
  ASSERT_TRUE(using_vbase_meta.has_value());
  ASSERT_TRUE(using_vbase_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(using_vbase_meta->GetIsDynamicCXXType(), true); // has virtual base

  auto uu_vbase_meta = GetMetadataFor(symfile, "UsingUsingVBase", clang);
  ASSERT_TRUE(uu_vbase_meta.has_value());
  ASSERT_TRUE(uu_vbase_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(uu_vbase_meta->GetIsDynamicCXXType(),
            true); // has 'UsingVBase' as non-virtual base

  auto not_dynamic_meta = GetMetadataFor(symfile, "NotDynamic", clang);
  ASSERT_TRUE(not_dynamic_meta.has_value());
  ASSERT_TRUE(not_dynamic_meta->GetIsDynamicCXXType().has_value());
  ASSERT_EQ(not_dynamic_meta->GetIsDynamicCXXType(),
            false); // has 'VBase' as non-virtual base
}
