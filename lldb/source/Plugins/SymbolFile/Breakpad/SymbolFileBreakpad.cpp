//===-- SymbolFileBreakpad.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/Breakpad/SymbolFileBreakpad.h"
#include "Plugins/ObjectFile/Breakpad/BreakpadRecords.h"
#include "Plugins/ObjectFile/Breakpad/ObjectFileBreakpad.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/StringExtras.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::breakpad;

namespace {
class LineIterator {
public:
  // begin iterator for sections of given type
  LineIterator(ObjectFile &obj, Record::Kind section_type)
      : m_obj(&obj), m_section_type(toString(section_type)),
        m_next_section_idx(0) {
    ++*this;
  }

  // end iterator
  explicit LineIterator(ObjectFile &obj)
      : m_obj(&obj),
        m_next_section_idx(m_obj->GetSectionList()->GetNumSections(0)) {}

  friend bool operator!=(const LineIterator &lhs, const LineIterator &rhs) {
    assert(lhs.m_obj == rhs.m_obj);
    if (lhs.m_next_section_idx != rhs.m_next_section_idx)
      return true;
    if (lhs.m_next_text.data() != rhs.m_next_text.data())
      return true;
    assert(lhs.m_current_text == rhs.m_current_text);
    assert(rhs.m_next_text == rhs.m_next_text);
    return false;
  }

  const LineIterator &operator++();
  llvm::StringRef operator*() const { return m_current_text; }

private:
  ObjectFile *m_obj;
  ConstString m_section_type;
  uint32_t m_next_section_idx;
  llvm::StringRef m_current_text;
  llvm::StringRef m_next_text;
};
} // namespace

const LineIterator &LineIterator::operator++() {
  const SectionList &list = *m_obj->GetSectionList();
  size_t num_sections = list.GetNumSections(0);
  while (m_next_text.empty() && m_next_section_idx < num_sections) {
    Section &sect = *list.GetSectionAtIndex(m_next_section_idx++);
    if (sect.GetName() != m_section_type)
      continue;
    DataExtractor data;
    m_obj->ReadSectionData(&sect, data);
    m_next_text =
        llvm::StringRef(reinterpret_cast<const char *>(data.GetDataStart()),
                        data.GetByteSize());
  }
  std::tie(m_current_text, m_next_text) = m_next_text.split('\n');
  return *this;
}

static llvm::iterator_range<LineIterator> lines(ObjectFile &obj,
                                                Record::Kind section_type) {
  return llvm::make_range(LineIterator(obj, section_type), LineIterator(obj));
}

void SymbolFileBreakpad::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                DebuggerInitialize);
}

void SymbolFileBreakpad::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ConstString SymbolFileBreakpad::GetPluginNameStatic() {
  static ConstString g_name("breakpad");
  return g_name;
}

uint32_t SymbolFileBreakpad::CalculateAbilities() {
  if (!m_obj_file)
    return 0;
  if (m_obj_file->GetPluginName() != ObjectFileBreakpad::GetPluginNameStatic())
    return 0;

  return CompileUnits | Functions;
}

uint32_t SymbolFileBreakpad::GetNumCompileUnits() {
  // TODO
  return 0;
}

CompUnitSP SymbolFileBreakpad::ParseCompileUnitAtIndex(uint32_t index) {
  // TODO
  return nullptr;
}

size_t SymbolFileBreakpad::ParseFunctions(CompileUnit &comp_unit) {
  // TODO
  return 0;
}

bool SymbolFileBreakpad::ParseLineTable(CompileUnit &comp_unit) {
  // TODO
  return 0;
}

uint32_t
SymbolFileBreakpad::ResolveSymbolContext(const Address &so_addr,
                                         SymbolContextItem resolve_scope,
                                         SymbolContext &sc) {
  // TODO
  return 0;
}

uint32_t SymbolFileBreakpad::FindFunctions(
    ConstString name, const CompilerDeclContext *parent_decl_ctx,
    FunctionNameType name_type_mask, bool include_inlines, bool append,
    SymbolContextList &sc_list) {
  // TODO
  if (!append)
    sc_list.Clear();
  return sc_list.GetSize();
}

uint32_t SymbolFileBreakpad::FindFunctions(const RegularExpression &regex,
                                           bool include_inlines, bool append,
                                           SymbolContextList &sc_list) {
  // TODO
  if (!append)
    sc_list.Clear();
  return sc_list.GetSize();
}

uint32_t SymbolFileBreakpad::FindTypes(
    ConstString name, const CompilerDeclContext *parent_decl_ctx,
    bool append, uint32_t max_matches,
    llvm::DenseSet<SymbolFile *> &searched_symbol_files, TypeMap &types) {
  if (!append)
    types.Clear();
  return types.GetSize();
}

size_t
SymbolFileBreakpad::FindTypes(const std::vector<CompilerContext> &context,
                              bool append, TypeMap &types) {
  if (!append)
    types.Clear();
  return types.GetSize();
}

void SymbolFileBreakpad::AddSymbols(Symtab &symtab) {
  Log *log = GetLogIfAllCategoriesSet(LIBLLDB_LOG_SYMBOLS);
  Module &module = *m_obj_file->GetModule();
  addr_t base = module.GetObjectFile()->GetBaseAddress().GetFileAddress();
  if (base == LLDB_INVALID_ADDRESS) {
    LLDB_LOG(log, "Unable to fetch the base address of object file. Skipping "
                  "symtab population.");
    return;
  }

  const SectionList &list = *module.GetSectionList();
  llvm::DenseMap<addr_t, Symbol> symbols;
  auto add_symbol = [&](addr_t address, llvm::Optional<addr_t> size,
                        llvm::StringRef name) {
    address += base;
    SectionSP section_sp = list.FindSectionContainingFileAddress(address);
    if (!section_sp) {
      LLDB_LOG(log,
               "Ignoring symbol {0}, whose address ({1}) is outside of the "
               "object file. Mismatched symbol file?",
               name, address);
      return;
    }
    symbols.try_emplace(
        address, /*symID*/ 0, Mangled(name, /*is_mangled*/ false),
        eSymbolTypeCode, /*is_global*/ true, /*is_debug*/ false,
        /*is_trampoline*/ false, /*is_artificial*/ false,
        AddressRange(section_sp, address - section_sp->GetFileAddress(),
                     size.getValueOr(0)),
        size.hasValue(), /*contains_linker_annotations*/ false, /*flags*/ 0);
  };

  for (llvm::StringRef line : lines(*m_obj_file, Record::Func)) {
    if (auto record = FuncRecord::parse(line))
      add_symbol(record->Address, record->Size, record->Name);
  }

  for (llvm::StringRef line : lines(*m_obj_file, Record::Public)) {
    if (auto record = PublicRecord::parse(line))
      add_symbol(record->Address, llvm::None, record->Name);
    else
      LLDB_LOG(log, "Failed to parse: {0}. Skipping record.", line);
  }

  for (auto &KV : symbols)
    symtab.AddSymbol(std::move(KV.second));
  symtab.CalculateSymbolSizes();
}
