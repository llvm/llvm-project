//===-- SymbolFileSymtab.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolFileSymtab.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Timer.h"

#include <memory>
#include <optional>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SymbolFileSymtab)

char SymbolFileSymtab::ID;

void SymbolFileSymtab::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void SymbolFileSymtab::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef SymbolFileSymtab::GetPluginDescriptionStatic() {
  return "Reads debug symbols from an object file's symbol table.";
}

SymbolFile *SymbolFileSymtab::CreateInstance(ObjectFileSP objfile_sp) {
  return new SymbolFileSymtab(std::move(objfile_sp));
}

void SymbolFileSymtab::GetTypes(SymbolContextScope *sc_scope,
                                TypeClass type_mask,
                                lldb_private::TypeList &type_list) {}

SymbolFileSymtab::SymbolFileSymtab(ObjectFileSP objfile_sp)
    : SymbolFileCommon(std::move(objfile_sp)), m_source_indexes(),
      m_code_indexes() {}

uint32_t SymbolFileSymtab::CalculateAbilities() {
  uint32_t abilities = 0;
  if (m_objfile_sp) {
    const Symtab *symtab = m_objfile_sp->GetSymtab();
    if (symtab) {
      if (symtab->GetNumSymbols() > 0)
        abilities |= Symbols;

      // Only source-file entries provide a genuine debug-info ability. Code
      // and data entries remain available as symbols but are not equivalent
      // to debug-info functions or global variables. The symbol indexes are
      // built later by InitializeSymbolIndexes() so we don't need to iterate
      // over all of the symbols to calculate our abilities.
      if (symtab->HasSourceFiles())
        abilities |= CompileUnits;
    }
  }
  return abilities;
}

void SymbolFileSymtab::InitializeSymbolIndexes() {
  // GetModuleMutex() dereferences the object file, so check it first.
  if (!m_objfile_sp)
    return;
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  if (m_symbol_indexes_initialized)
    return;
  const Symtab *symtab = m_objfile_sp->GetSymtab();
  if (!symtab)
    return;
  m_symbol_indexes_initialized = true;

  // Get the indexes of the source file and non-debug code entries in the
  // module symbol table.
  symtab->AppendSymbolIndexesWithType(eSymbolTypeSourceFile, m_source_indexes);

  if (symtab->AppendSymbolIndexesWithType(eSymbolTypeCode, Symtab::eDebugNo,
                                          Symtab::eVisibilityAny,
                                          m_code_indexes)) {
    symtab->SortSymbolIndexesByValue(m_code_indexes, true);
  }
}

uint32_t SymbolFileSymtab::CalculateNumCompileUnits() {
  // Compile units are only created from source file symbols, so don't build
  // the symbol indexes, which iterates over all of the symbols, if there
  // aren't any. Nothing else uses the indexes without a compile unit.
  if (!m_objfile_sp)
    return 0;
  const Symtab *symtab = m_objfile_sp->GetSymtab();
  if (!symtab || !symtab->HasSourceFiles())
    return 0;

  InitializeSymbolIndexes();

  // Each source file symbol is a compile unit, and the object file symbols
  // are logically organized using these.
  return m_source_indexes.size();
}

CompUnitSP SymbolFileSymtab::ParseCompileUnitAtIndex(uint32_t idx) {
  InitializeSymbolIndexes();

  CompUnitSP cu_sp;

  // Each source file symbol is a compile unit.
  if (idx < m_source_indexes.size()) {
    const Symbol *cu_symbol =
        m_objfile_sp->GetSymtab()->SymbolAtIndex(m_source_indexes[idx]);
    if (cu_symbol)
      cu_sp =
          std::make_shared<CompileUnit>(m_objfile_sp->GetModule(), nullptr,
                                        cu_symbol->GetName().AsCString(nullptr),
                                        0, eLanguageTypeUnknown, eLazyBoolNo);
  }
  return cu_sp;
}

lldb::LanguageType SymbolFileSymtab::ParseLanguage(CompileUnit &comp_unit) {
  return eLanguageTypeUnknown;
}

size_t SymbolFileSymtab::ParseFunctions(CompileUnit &comp_unit) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  InitializeSymbolIndexes();
  size_t num_added = 0;
  // We must at least have a valid compile unit
  const Symtab *symtab = m_objfile_sp->GetSymtab();
  const Symbol *curr_symbol = nullptr;
  const Symbol *next_symbol = nullptr;
  //  const char *prefix = m_objfile_sp->SymbolPrefix();
  //  if (prefix == NULL)
  //      prefix == "";
  //
  //  const uint32_t prefix_len = strlen(prefix);

  // If we don't have any source file symbols we will just have one compile
  // unit for the entire object file
  if (m_source_indexes.empty()) {
    // The only time we will have a user ID of zero is when we don't have and
    // source file symbols and we declare one compile unit for the entire
    // object file
    if (!m_code_indexes.empty()) {
      //          StreamFile s(stdout);
      //          symtab->Dump(&s, m_code_indexes);

      uint32_t idx = 0; // Index into the indexes
      const uint32_t num_indexes = m_code_indexes.size();
      for (idx = 0; idx < num_indexes; ++idx) {
        uint32_t symbol_idx = m_code_indexes[idx];
        curr_symbol = symtab->SymbolAtIndex(symbol_idx);
        if (curr_symbol) {
          // Union of all ranges in the function DIE (if the function is
          // discontiguous)
          AddressRange func_range(curr_symbol->GetAddress(), 0);
          if (func_range.GetBaseAddress().IsSectionOffset()) {
            uint32_t symbol_size = curr_symbol->GetByteSize();
            if (symbol_size != 0 && !curr_symbol->GetSizeIsSibling())
              func_range.SetByteSize(symbol_size);
            else if (idx + 1 < num_indexes) {
              next_symbol = symtab->SymbolAtIndex(m_code_indexes[idx + 1]);
              if (next_symbol) {
                func_range.SetByteSize(
                    next_symbol->GetAddressRef().GetOffset() -
                    curr_symbol->GetAddressRef().GetOffset());
              }
            }

            FunctionSP func_sp(new Function(
                &comp_unit,
                symbol_idx,                // UserID is the DIE offset
                LLDB_INVALID_UID,          // We don't have any type info
                                           // for this function
                curr_symbol->GetMangled(), // Linker/mangled name
                nullptr, // no return type for a code symbol...
                curr_symbol->GetAddress(), AddressRanges{func_range}));

            if (func_sp.get() != nullptr) {
              comp_unit.AddFunction(func_sp);
              ++num_added;
            }
          }
        }
      }
    }
  } else {
    // We assume we
  }
  return num_added;
}

size_t SymbolFileSymtab::ParseTypes(CompileUnit &comp_unit) { return 0; }

bool SymbolFileSymtab::ParseLineTable(CompileUnit &comp_unit) { return false; }

bool SymbolFileSymtab::ParseDebugMacros(CompileUnit &comp_unit) {
  return false;
}

bool SymbolFileSymtab::ParseSupportFiles(CompileUnit &comp_unit,
                                         SupportFileList &support_files) {
  return false;
}

bool SymbolFileSymtab::ParseImportedModules(
    const SymbolContext &sc, std::vector<SourceModule> &imported_modules) {
  return false;
}

size_t SymbolFileSymtab::ParseBlocksRecursive(Function &func) { return 0; }

size_t SymbolFileSymtab::ParseVariablesForContext(const SymbolContext &sc) {
  return 0;
}

Type *SymbolFileSymtab::ResolveTypeUID(lldb::user_id_t type_uid) {
  return nullptr;
}

std::optional<SymbolFile::ArrayInfo>
SymbolFileSymtab::GetDynamicArrayInfoForUID(
    lldb::user_id_t type_uid, const lldb_private::ExecutionContext *exe_ctx) {
  return std::nullopt;
}

bool SymbolFileSymtab::CompleteType(lldb_private::CompilerType &compiler_type) {
  return false;
}

uint32_t SymbolFileSymtab::ResolveSymbolContext(const Address &so_addr,
                                                SymbolContextItem resolve_scope,
                                                SymbolContext &sc) {
  std::lock_guard<std::recursive_mutex> guard(GetModuleMutex());
  if (m_objfile_sp->GetSymtab() == nullptr)
    return 0;

  uint32_t resolved_flags = 0;
  if (resolve_scope & eSymbolContextSymbol) {
    sc.symbol = m_objfile_sp->GetSymtab()->FindSymbolContainingFileAddress(
        so_addr.GetFileAddress());
    if (sc.symbol)
      resolved_flags |= eSymbolContextSymbol;
  }
  return resolved_flags;
}
