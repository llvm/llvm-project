//===-- DWARFExpressionTest.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/DWARFExpression.h"
#include "../../source/Plugins/SymbolFile/DWARF/DWARFUnit.h"
#include "../../source/Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

static llvm::Expected<Scalar> Evaluate(llvm::ArrayRef<uint8_t> expr,
                                       lldb::ModuleSP module_sp = {},
                                       DWARFUnit *unit = nullptr) {
  DataExtractor extractor(expr.data(), expr.size(), lldb::eByteOrderLittle,
                          /*addr_size*/ 4);
  Value result;
  Status status;
  if (!DWARFExpression::Evaluate(
          /*exe_ctx*/ nullptr, /*reg_ctx*/ nullptr, module_sp, extractor, unit,
          lldb::eRegisterKindLLDB,
          /*initial_value_ptr*/ nullptr,
          /*object_address_ptr*/ nullptr, result, &status))
    return status.ToError();

  return result.GetScalar();
}

/// A mock module holding an object file parsed from YAML.
class YAMLModule : public lldb_private::Module {
public:
  YAMLModule(ArchSpec &arch) : Module(FileSpec("test"), arch) {}
  void SetObjectFile(lldb::ObjectFileSP obj_file) { m_objfile_sp = obj_file; }
  ObjectFile *GetObjectFile() override { return m_objfile_sp.get(); }
};

/// A mock object file that can be parsed from YAML.
class YAMLObjectFile : public lldb_private::ObjectFile {
  const lldb::ModuleSP m_module_sp;
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> &m_section_map;
  /// Because there is only one DataExtractor in the ObjectFile
  /// interface, all sections are copied into a contiguous buffer.
  std::vector<char> m_buffer;

public:
  YAMLObjectFile(const lldb::ModuleSP &module_sp,
                 llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> &map)
      : ObjectFile(module_sp, &module_sp->GetFileSpec(), /*file_offset*/ 0,
                   /*length*/ 0, /*data_sp*/ nullptr, /*data_offset*/ 0),
        m_module_sp(module_sp), m_section_map(map) {}

  /// Callback for initializing the module's list of sections.
  void CreateSections(SectionList &unified_section_list) override {
    lldb::offset_t total_bytes = 0;
    for (auto &entry : m_section_map)
      total_bytes += entry.getValue()->getBufferSize();
    m_buffer.reserve(total_bytes);
    m_data =
        DataExtractor(m_buffer.data(), total_bytes, lldb::eByteOrderLittle, 4);

    lldb::user_id_t sect_id = 1;
    for (auto &entry : m_section_map) {
      llvm::StringRef name = entry.getKey();
      lldb::SectionType sect_type =
          llvm::StringSwitch<lldb::SectionType>(name)
              .Case("debug_info", lldb::eSectionTypeDWARFDebugInfo)
              .Case("debug_abbrev", lldb::eSectionTypeDWARFDebugAbbrev);
      auto &membuf = entry.getValue();
      lldb::addr_t file_vm_addr = 0;
      lldb::addr_t vm_size = 0;
      lldb::offset_t file_offset = m_buffer.size();
      lldb::offset_t file_size = membuf->getBufferSize();
      m_buffer.resize(file_offset + file_size);
      memcpy(m_buffer.data() + file_offset, membuf->getBufferStart(),
             file_size);
      uint32_t log2align = 0;
      uint32_t flags = 0;
      auto section_sp = std::make_shared<lldb_private::Section>(
          m_module_sp, this, sect_id++, ConstString(name), sect_type,
          file_vm_addr, vm_size, file_offset, file_size, log2align, flags);
      unified_section_list.AddSection(section_sp);
    }
  }

  /// \{
  /// Stub methods that aren't needed here.
  ConstString GetPluginName() override { return ConstString("YAMLObjectFile"); }
  uint32_t GetPluginVersion() override { return 0; }
  void Dump(Stream *s) override {}
  uint32_t GetAddressByteSize() const override { return 8; }
  uint32_t GetDependentModules(FileSpecList &file_list) override { return 0; }
  bool IsExecutable() const override { return 0; }
  ArchSpec GetArchitecture() override { return {}; }
  Symtab *GetSymtab() override { return nullptr; }
  bool IsStripped() override { return false; }
  UUID GetUUID() override { return {}; }
  lldb::ByteOrder GetByteOrder() const override {
    return lldb::eByteOrderLittle;
  }
  bool ParseHeader() override { return false; }
  Type CalculateType() override { return {}; }
  Strata CalculateStrata() override { return {}; }
  /// \}
};

/// Helper class that can construct a module from YAML and evaluate
/// DWARF expressions on it.
class YAMLModuleTester {
  SubsystemRAII<FileSystem> subsystems;
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> m_sections_map;
  lldb::ModuleSP m_module_sp;
  lldb::ObjectFileSP m_objfile_sp;
  DWARFUnitSP m_dwarf_unit;
  std::unique_ptr<SymbolFileDWARF> m_symfile_dwarf;

public:
  /// Parse the debug info sections from the YAML description.
  YAMLModuleTester(llvm::StringRef yaml_data, llvm::StringRef triple) {
    auto sections_map = llvm::DWARFYAML::EmitDebugSections(yaml_data, true);
    if (!sections_map)
      return;
    m_sections_map = std::move(*sections_map);
    ArchSpec arch(triple);
    m_module_sp = std::make_shared<YAMLModule>(arch);
    m_objfile_sp = std::make_shared<YAMLObjectFile>(m_module_sp, m_sections_map);
    static_cast<YAMLModule *>(m_module_sp.get())->SetObjectFile(m_objfile_sp);

    lldb::user_id_t uid = 0;
    llvm::StringRef raw_debug_info = m_sections_map["debug_info"]->getBuffer();
    lldb_private::DataExtractor debug_info(
        raw_debug_info.data(), raw_debug_info.size(),
        m_objfile_sp->GetByteOrder(), m_objfile_sp->GetAddressByteSize());
    lldb::offset_t offset_ptr = 0;
    m_symfile_dwarf = std::make_unique<SymbolFileDWARF>(m_objfile_sp, nullptr);
    llvm::Expected<DWARFUnitSP> dwarf_unit = DWARFUnit::extract(
        *m_symfile_dwarf, uid,
        *static_cast<lldb_private::DWARFDataExtractor *>(&debug_info),
        DIERef::DebugInfo, &offset_ptr);
    if (dwarf_unit)
      m_dwarf_unit = dwarf_unit.get();
  }
  DWARFUnitSP GetDwarfUnit() { return m_dwarf_unit; }

  // Evaluate a raw DWARF expression.
  llvm::Expected<Scalar> Eval(llvm::ArrayRef<uint8_t> expr) {
    return ::Evaluate(expr, m_module_sp, m_dwarf_unit.get());
  }
};

/// Unfortunately Scalar's operator==() is really picky.
static Scalar GetScalar(unsigned bits, uint64_t value, bool sign) {
  Scalar scalar;
  auto type = Scalar::GetBestTypeForBitSize(bits, sign);
  switch (type) {
  case Scalar::e_sint:
    scalar = Scalar((int)value);
    break;
  case Scalar::e_slong:
    scalar = Scalar((long)value);
    break;
  case Scalar::e_slonglong:
    scalar = Scalar((long long)value);
    break;
  case Scalar::e_uint:
    scalar = Scalar((unsigned int)value);
    break;
  case Scalar::e_ulong:
    scalar = Scalar((unsigned long)value);
    break;
  case Scalar::e_ulonglong:
    scalar = Scalar((unsigned long long)value);
    break;
  default:
    llvm_unreachable("not implemented");
  }
  scalar.TruncOrExtendTo(type, bits);
  if (sign)
    scalar.MakeSigned();
  else
    scalar.MakeUnsigned();
  return scalar;
}

TEST(DWARFExpression, DW_OP_pick) {
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 0}),
                       llvm::HasValue(0));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 1}),
                       llvm::HasValue(1));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 2}),
                       llvm::Failed());
}

TEST(DWARFExpression, DW_OP_convert) {
  /// Auxiliary debug info.
  const char *yamldata =
      "debug_abbrev:\n"
      "  - Code:            0x00000001\n"
      "    Tag:             DW_TAG_compile_unit\n"
      "    Children:        DW_CHILDREN_yes\n"
      "    Attributes:\n"
      "      - Attribute:       DW_AT_language\n"
      "        Form:            DW_FORM_data2\n"
      "  - Code:            0x00000002\n"
      "    Tag:             DW_TAG_base_type\n"
      "    Children:        DW_CHILDREN_no\n"
      "    Attributes:\n"
      "      - Attribute:       DW_AT_encoding\n"
      "        Form:            DW_FORM_data1\n"
      "      - Attribute:       DW_AT_byte_size\n"
      "        Form:            DW_FORM_data1\n"
      "debug_info:\n"
      "  - Length:\n"
      "      TotalLength:     0\n"
      "    Version:         4\n"
      "    AbbrOffset:      0\n"
      "    AddrSize:        8\n"
      "    Entries:\n"
      "      - AbbrCode:        0x00000001\n"
      "        Values:\n"
      "          - Value:           0x000000000000000C\n"
      // 0x0000000e:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000007\n" // DW_ATE_unsigned
      "          - Value:           0x0000000000000004\n"
      // 0x00000011:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000007\n" // DW_ATE_unsigned
      "          - Value:           0x0000000000000008\n"
      // 0x00000014:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000005\n" // DW_ATE_signed
      "          - Value:           0x0000000000000008\n"
      // 0x00000017:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000008\n" // DW_ATE_unsigned_char
      "          - Value:           0x0000000000000001\n"
      // 0x0000001a:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000006\n" // DW_ATE_signed_char
      "          - Value:           0x0000000000000001\n"
      // 0x0000001d:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x000000000000000b\n" // DW_ATE_numeric_string
      "          - Value:           0x0000000000000001\n"
      ""
      "      - AbbrCode:        0x00000000\n"
      "        Values:          []\n";
  uint8_t offs_uint32_t = 0x0000000e;
  uint8_t offs_uint64_t = 0x00000011;
  uint8_t offs_sint64_t = 0x00000014;
  uint8_t offs_uchar = 0x00000017;
  uint8_t offs_schar = 0x0000001a;

  YAMLModuleTester t(yamldata, "i386-unknown-linux");
  ASSERT_TRUE((bool)t.GetDwarfUnit());

  // Constant is given as little-endian.
  bool is_signed = true;
  bool not_signed = false;

  //
  // Positive tests.
  //

  // Truncate to default unspecified (pointer-sized) type.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const8u, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, //
              DW_OP_convert, 0x00}),
      llvm::HasValue(GetScalar(32, 0x44332211, not_signed)));
  // Truncate to 32 bits.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const8u, //
                               0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,//
                               DW_OP_convert, offs_uint32_t}),
                       llvm::HasValue(GetScalar(32, 0x44332211, not_signed)));

  // Leave as is.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const8u, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, //
              DW_OP_convert, offs_uint64_t}),
      llvm::HasValue(GetScalar(64, 0x8877665544332211, not_signed)));

  // Sign-extend to 64 bits.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4s, 0xcc, 0xdd, 0xee, 0xff, //
              DW_OP_convert, offs_sint64_t}),
      llvm::HasValue(GetScalar(64, 0xffffffffffeeddcc, is_signed)));

  // Truncate to 8 bits.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const4s, 'A', 'B', 'C', 'D', 0xee, 0xff, //
                               DW_OP_convert, offs_uchar}),
                       llvm::HasValue(GetScalar(8, 'A', not_signed)));

  // Also truncate to 8 bits.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const4s, 'A', 'B', 'C', 'D', 0xee, 0xff, //
                               DW_OP_convert, offs_schar}),
                       llvm::HasValue(GetScalar(8, 'A', is_signed)));

  //
  // Errors.
  //

  // No Module.
  EXPECT_THAT_ERROR(Evaluate({DW_OP_const1s, 'X', DW_OP_convert, 0x00}, nullptr,
                             t.GetDwarfUnit().get())
                        .takeError(),
                    llvm::Failed());

  // No DIE.
  EXPECT_THAT_ERROR(
      t.Eval({DW_OP_const1s, 'X', DW_OP_convert, 0x01}).takeError(),
      llvm::Failed());

  // Unsupported.
  EXPECT_THAT_ERROR(
      t.Eval({DW_OP_const1s, 'X', DW_OP_convert, 0x1d}).takeError(),
      llvm::Failed());
}
