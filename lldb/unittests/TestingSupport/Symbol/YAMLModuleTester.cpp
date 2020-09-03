//===-- YAMLModuleTester.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/Section.h"
#include "llvm/ObjectYAML/DWARFEmitter.h"

using namespace lldb_private;

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
              .Case("debug_abbrev", lldb::eSectionTypeDWARFDebugAbbrev)
              .Case("debug_str", lldb::eSectionTypeDWARFDebugStr);
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

YAMLModuleTester::YAMLModuleTester(llvm::StringRef yaml_data,
                                   llvm::StringRef triple) {
  auto sections_map = llvm::DWARFYAML::emitDebugSections(yaml_data, true);
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
      DIERef::DebugInfo, &offset_ptr, nullptr);
  if (dwarf_unit)
    m_dwarf_unit = dwarf_unit.get();
}
