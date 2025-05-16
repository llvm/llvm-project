//===-- LineTableTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolFile.h"
#include "gtest/gtest.h"
#include <memory>

using namespace lldb;
using namespace llvm;
using namespace lldb_private;

namespace {

// A fake symbol file class to allow us to create the line table "the right
// way". Pretty much all methods except for GetCompileUnitAtIndex and
// GetNumCompileUnits are stubbed out.
class FakeSymbolFile : public SymbolFile {
public:
  /// LLVM RTTI support.
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || SymbolFile::isA(ClassID);
  }
  static bool classof(const SymbolFile *obj) { return obj->isA(&ID); }
  /// \}

  static void Initialize() {
    PluginManager::RegisterPlugin("FakeSymbolFile", "", CreateInstance,
                                  DebuggerInitialize);
  }
  static void Terminate() { PluginManager::UnregisterPlugin(CreateInstance); }

  void InjectCompileUnit(std::unique_ptr<CompileUnit> cu_up) {
    m_cu_sp = std::move(cu_up);
  }

private:
  /// LLVM RTTI support.
  static char ID;

  static SymbolFile *CreateInstance(ObjectFileSP objfile_sp) {
    return new FakeSymbolFile(std::move(objfile_sp));
  }
  static void DebuggerInitialize(Debugger &) {}

  StringRef GetPluginName() override { return "FakeSymbolFile"; }
  uint32_t GetAbilities() override { return UINT32_MAX; }
  uint32_t CalculateAbilities() override { return UINT32_MAX; }
  uint32_t GetNumCompileUnits() override { return 1; }
  CompUnitSP GetCompileUnitAtIndex(uint32_t) override { return m_cu_sp; }
  Symtab *GetSymtab(bool can_create = true) override { return nullptr; }
  LanguageType ParseLanguage(CompileUnit &) override { return eLanguageTypeC; }
  size_t ParseFunctions(CompileUnit &) override { return 0; }
  bool ParseLineTable(CompileUnit &) override { return true; }
  bool ParseDebugMacros(CompileUnit &) override { return true; }
  bool ParseSupportFiles(CompileUnit &, SupportFileList &) override {
    return true;
  }
  size_t ParseTypes(CompileUnit &) override { return 0; }
  bool ParseImportedModules(const SymbolContext &,
                            std::vector<SourceModule> &) override {
    return false;
  }
  size_t ParseBlocksRecursive(Function &) override { return 0; }
  size_t ParseVariablesForContext(const SymbolContext &) override { return 0; }
  Type *ResolveTypeUID(user_id_t) override { return nullptr; }
  std::optional<ArrayInfo>
  GetDynamicArrayInfoForUID(user_id_t, const ExecutionContext *) override {
    return std::nullopt;
  }
  bool CompleteType(CompilerType &) override { return true; }
  uint32_t ResolveSymbolContext(const Address &, SymbolContextItem,
                                SymbolContext &) override {
    return 0;
  }
  void GetTypes(SymbolContextScope *, TypeClass, TypeList &) override {}
  Expected<TypeSystemSP> GetTypeSystemForLanguage(LanguageType) override {
    return createStringError(std::errc::not_supported, "");
  }
  const ObjectFile *GetObjectFile() const override {
    return m_objfile_sp.get();
  }
  ObjectFile *GetObjectFile() override { return m_objfile_sp.get(); }
  ObjectFile *GetMainObjectFile() override { return m_objfile_sp.get(); }
  void SectionFileAddressesChanged() override {}
  void Dump(Stream &) override {}
  uint64_t GetDebugInfoSize(bool) override { return 0; }
  bool GetDebugInfoIndexWasLoadedFromCache() const override { return false; }
  void SetDebugInfoIndexWasLoadedFromCache() override {}
  bool GetDebugInfoIndexWasSavedToCache() const override { return false; }
  void SetDebugInfoIndexWasSavedToCache() override {}
  bool GetDebugInfoHadFrameVariableErrors() const override { return false; }
  void SetDebugInfoHadFrameVariableErrors() override {}
  TypeSP MakeType(user_id_t, ConstString, std::optional<uint64_t>,
                  SymbolContextScope *, user_id_t, Type::EncodingDataType,
                  const Declaration &, const CompilerType &, Type::ResolveState,
                  uint32_t) override {
    return nullptr;
  }
  TypeSP CopyType(const TypeSP &) override { return nullptr; }

  FakeSymbolFile(ObjectFileSP objfile_sp)
      : m_objfile_sp(std::move(objfile_sp)) {}

  ObjectFileSP m_objfile_sp;
  CompUnitSP m_cu_sp;
};

struct FakeModuleFixture {
  TestFile file;
  ModuleSP module_sp;
  SectionSP text_sp;
  LineTable *line_table;
};

class LineTableTest : public testing::Test {
  SubsystemRAII<ObjectFileELF, FakeSymbolFile> subsystems;
};

class LineSequenceBuilder {
public:
  std::vector<LineTable::Sequence> Build() { return std::move(m_sequences); }
  enum Terminal : bool { Terminal = true };
  void Entry(addr_t addr, bool terminal = false) {
    LineTable::AppendLineEntryToSequence(
        m_sequence, addr, /*line=*/1, /*column=*/0,
        /*file_idx=*/0,
        /*is_start_of_statement=*/false, /*is_start_of_basic_block=*/false,
        /*is_prologue_end=*/false, /*is_epilogue_begin=*/false, terminal);
    if (terminal)
      m_sequences.push_back(std::move(m_sequence));
  }

private:
  std::vector<LineTable::Sequence> m_sequences;
  LineTable::Sequence m_sequence;
};

} // namespace

char FakeSymbolFile::ID;

static llvm::Expected<FakeModuleFixture>
CreateFakeModule(std::vector<LineTable::Sequence> line_sequences) {
  Expected<TestFile> file = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0010
    Address:         0x0000
    Size:            0x1000
)");
  if (!file)
    return file.takeError();

  auto module_sp = std::make_shared<Module>(file->moduleSpec());
  SectionSP text_sp =
      module_sp->GetSectionList()->FindSectionByName(ConstString(".text"));
  if (!text_sp)
    return createStringError("No .text");

  auto cu_up = std::make_unique<CompileUnit>(module_sp, /*user_data=*/nullptr,
                                             /*support_file_sp=*/nullptr,
                                             /*uid=*/0, eLanguageTypeC,
                                             /*is_optimized=*/eLazyBoolNo);
  LineTable *line_table = new LineTable(cu_up.get(), std::move(line_sequences));
  cu_up->SetLineTable(line_table);
  cast<FakeSymbolFile>(module_sp->GetSymbolFile())
      ->InjectCompileUnit(std::move(cu_up));

  return FakeModuleFixture{std::move(*file), std::move(module_sp),
                           std::move(text_sp), line_table};
}

TEST_F(LineTableTest, lower_bound) {
  LineSequenceBuilder builder;
  builder.Entry(0);
  builder.Entry(10);
  builder.Entry(20, LineSequenceBuilder::Terminal);
  builder.Entry(20); // Starts right after the previous sequence.
  builder.Entry(30, LineSequenceBuilder::Terminal);
  builder.Entry(40); // Gap after the previous sequence.
  builder.Entry(50, LineSequenceBuilder::Terminal);

  llvm::Expected<FakeModuleFixture> fixture = CreateFakeModule(builder.Build());
  ASSERT_THAT_EXPECTED(fixture, llvm::Succeeded());

  LineTable *table = fixture->line_table;

  auto make_addr = [&](addr_t addr) { return Address(fixture->text_sp, addr); };

  EXPECT_EQ(table->lower_bound(make_addr(0)), 0u);
  EXPECT_EQ(table->lower_bound(make_addr(9)), 0u);
  EXPECT_EQ(table->lower_bound(make_addr(10)), 1u);
  EXPECT_EQ(table->lower_bound(make_addr(19)), 1u);

  // Skips over the terminal entry.
  EXPECT_EQ(table->lower_bound(make_addr(20)), 3u);
  EXPECT_EQ(table->lower_bound(make_addr(29)), 3u);

  // In case there's no "real" entry at this address, the function returns the
  // first real entry.
  EXPECT_EQ(table->lower_bound(make_addr(30)), 5u);
  EXPECT_EQ(table->lower_bound(make_addr(40)), 5u);

  // In a gap, return the first entry after the gap.
  EXPECT_EQ(table->lower_bound(make_addr(39)), 5u);

  // And if there's no such entry, return the size of the list.
  EXPECT_EQ(table->lower_bound(make_addr(50)), table->GetSize());
  EXPECT_EQ(table->lower_bound(make_addr(59)), table->GetSize());
}

TEST_F(LineTableTest, GetLineEntryIndexRange) {
  LineSequenceBuilder builder;
  builder.Entry(0);
  builder.Entry(10);
  builder.Entry(20, LineSequenceBuilder::Terminal);

  llvm::Expected<FakeModuleFixture> fixture = CreateFakeModule(builder.Build());
  ASSERT_THAT_EXPECTED(fixture, llvm::Succeeded());

  LineTable *table = fixture->line_table;

  auto make_range = [&](addr_t addr, addr_t size) {
    return AddressRange(fixture->text_sp, addr, size);
  };

  EXPECT_THAT(table->GetLineEntryIndexRange(make_range(0, 10)),
              testing::Pair(0, 1));
  EXPECT_THAT(table->GetLineEntryIndexRange(make_range(0, 20)),
              testing::Pair(0, 3)); // Includes the terminal entry.
  // Partial overlap on one side.
  EXPECT_THAT(table->GetLineEntryIndexRange(make_range(3, 7)),
              testing::Pair(0, 1));
  // On the other side
  EXPECT_THAT(table->GetLineEntryIndexRange(make_range(0, 15)),
              testing::Pair(0, 2));
  // On both sides
  EXPECT_THAT(table->GetLineEntryIndexRange(make_range(2, 3)),
              testing::Pair(0, 1));
  // Empty ranges
  EXPECT_THAT(table->GetLineEntryIndexRange(make_range(0, 0)),
              testing::Pair(0, 0));
  EXPECT_THAT(table->GetLineEntryIndexRange(make_range(5, 0)),
              testing::Pair(0, 0));
  EXPECT_THAT(table->GetLineEntryIndexRange(make_range(10, 0)),
              testing::Pair(1, 1));
}

TEST_F(LineTableTest, FindLineEntryByAddress) {
  LineSequenceBuilder builder;
  builder.Entry(0);
  builder.Entry(10);
  builder.Entry(20, LineSequenceBuilder::Terminal);
  builder.Entry(20); // Starts right after the previous sequence.
  builder.Entry(30, LineSequenceBuilder::Terminal);
  builder.Entry(40); // Gap after the previous sequence.
  builder.Entry(50, LineSequenceBuilder::Terminal);

  llvm::Expected<FakeModuleFixture> fixture = CreateFakeModule(builder.Build());
  ASSERT_THAT_EXPECTED(fixture, llvm::Succeeded());

  LineTable *table = fixture->line_table;

  auto find = [&](addr_t addr) -> std::tuple<addr_t, addr_t, bool> {
    LineEntry entry;
    if (!table->FindLineEntryByAddress(Address(fixture->text_sp, addr), entry))
      return {LLDB_INVALID_ADDRESS, LLDB_INVALID_ADDRESS, false};
    return {entry.range.GetBaseAddress().GetFileAddress(),
            entry.range.GetByteSize(),
            static_cast<bool>(entry.is_terminal_entry)};
  };

  EXPECT_THAT(find(0), testing::FieldsAre(0, 10, false));
  EXPECT_THAT(find(9), testing::FieldsAre(0, 10, false));
  EXPECT_THAT(find(10), testing::FieldsAre(10, 10, false));
  EXPECT_THAT(find(19), testing::FieldsAre(10, 10, false));
  EXPECT_THAT(find(20), testing::FieldsAre(20, 10, false));
  EXPECT_THAT(find(30), testing::FieldsAre(LLDB_INVALID_ADDRESS,
                                           LLDB_INVALID_ADDRESS, false));
  EXPECT_THAT(find(40), testing::FieldsAre(40, 10, false));
  EXPECT_THAT(find(50), testing::FieldsAre(LLDB_INVALID_ADDRESS,
                                           LLDB_INVALID_ADDRESS, false));
}
