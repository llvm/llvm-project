//===-- ManualDWARFIndex.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/ManualDWARFIndex.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"
#include "Plugins/SymbolFile/DWARF/DWARFDeclContext.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/TaskPool.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/Timer.h"

using namespace lldb_private;
using namespace lldb;

void ManualDWARFIndex::Index() {
  if (!m_debug_info)
    return;

  DWARFDebugInfo &debug_info = *m_debug_info;
  m_debug_info = nullptr;

  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, "%p", static_cast<void *>(&debug_info));

  const uint32_t num_compile_units = debug_info.GetNumCompileUnits();
  if (num_compile_units == 0)
    return;

  std::vector<NameToDIE> function_basenames(num_compile_units);
  std::vector<NameToDIE> function_fullnames(num_compile_units);
  std::vector<NameToDIE> function_methods(num_compile_units);
  std::vector<NameToDIE> function_selectors(num_compile_units);
  std::vector<NameToDIE> objc_class_selectors(num_compile_units);
  std::vector<NameToDIE> globals(num_compile_units);
  std::vector<NameToDIE> types(num_compile_units);
  std::vector<NameToDIE> namespaces(num_compile_units);

  // std::vector<bool> might be implemented using bit test-and-set, so use
  // uint8_t instead.
  std::vector<uint8_t> clear_cu_dies(num_compile_units, false);
  auto parser_fn = [&](size_t cu_idx) {
    DWARFUnit *dwarf_cu = debug_info.GetCompileUnitAtIndex(cu_idx);
    if (dwarf_cu) {
      dwarf_cu->Index(function_basenames[cu_idx], function_fullnames[cu_idx],
                      function_methods[cu_idx], function_selectors[cu_idx],
                      objc_class_selectors[cu_idx], globals[cu_idx],
                      types[cu_idx], namespaces[cu_idx]);
    }
  };

  auto extract_fn = [&debug_info, &clear_cu_dies](size_t cu_idx) {
    DWARFUnit *dwarf_cu = debug_info.GetCompileUnitAtIndex(cu_idx);
    if (dwarf_cu) {
      // dwarf_cu->ExtractDIEsIfNeeded(false) will return zero if the DIEs
      // for a compile unit have already been parsed.
      if (dwarf_cu->ExtractDIEsIfNeeded(false) > 1)
        clear_cu_dies[cu_idx] = true;
    }
  };

  // Create a task runner that extracts dies for each DWARF compile unit in a
  // separate thread
  //----------------------------------------------------------------------
  // First figure out which compile units didn't have their DIEs already
  // parsed and remember this.  If no DIEs were parsed prior to this index
  // function call, we are going to want to clear the CU dies after we are
  // done indexing to make sure we don't pull in all DWARF dies, but we need
  // to wait until all compile units have been indexed in case a DIE in one
  // compile unit refers to another and the indexes accesses those DIEs.
  //----------------------------------------------------------------------
  TaskMapOverInt(0, num_compile_units, extract_fn);

  // Now create a task runner that can index each DWARF compile unit in a
  // separate thread so we can index quickly.

  TaskMapOverInt(0, num_compile_units, parser_fn);

  auto finalize_fn = [](NameToDIE &index, std::vector<NameToDIE> &srcs) {
    for (auto &src : srcs)
      index.Append(src);
    index.Finalize();
  };

  TaskPool::RunTasks(
      [&]() { finalize_fn(m_function_basenames, function_basenames); },
      [&]() { finalize_fn(m_function_fullnames, function_fullnames); },
      [&]() { finalize_fn(m_function_methods, function_methods); },
      [&]() { finalize_fn(m_function_selectors, function_selectors); },
      [&]() { finalize_fn(m_objc_class_selectors, objc_class_selectors); },
      [&]() { finalize_fn(m_globals, globals); },
      [&]() { finalize_fn(m_types, types); },
      [&]() { finalize_fn(m_namespaces, namespaces); });

  //----------------------------------------------------------------------
  // Keep memory down by clearing DIEs for any compile units if indexing
  // caused us to load the compile unit's DIEs.
  //----------------------------------------------------------------------
  for (uint32_t cu_idx = 0; cu_idx < num_compile_units; ++cu_idx) {
    if (clear_cu_dies[cu_idx])
      debug_info.GetCompileUnitAtIndex(cu_idx)->ClearDIEs(true);
  }
}

void ManualDWARFIndex::GetGlobalVariables(ConstString name, DIEArray &offsets) {
  Index();
  m_globals.Find(name, offsets);
}

void ManualDWARFIndex::GetGlobalVariables(const RegularExpression &regex,
                                          DIEArray &offsets) {
  Index();
  m_globals.Find(regex, offsets);
}

void ManualDWARFIndex::GetGlobalVariables(const DWARFUnit &cu,
                                          DIEArray &offsets) {
  Index();
  m_globals.FindAllEntriesForCompileUnit(cu.GetOffset(), offsets);
}

void ManualDWARFIndex::GetObjCMethods(ConstString class_name,
                                      DIEArray &offsets) {
  Index();
  m_objc_class_selectors.Find(class_name, offsets);
}

void ManualDWARFIndex::GetCompleteObjCClass(ConstString class_name,
                                            bool must_be_implementation,
                                            DIEArray &offsets) {
  Index();
  m_types.Find(class_name, offsets);
}

void ManualDWARFIndex::GetTypes(ConstString name, DIEArray &offsets) {
  Index();
  m_types.Find(name, offsets);
}

void ManualDWARFIndex::GetTypes(const DWARFDeclContext &context,
                                DIEArray &offsets) {
  Index();
  m_types.Find(ConstString(context[0].name), offsets);
}

void ManualDWARFIndex::GetNamespaces(ConstString name, DIEArray &offsets) {
  Index();
  m_namespaces.Find(name, offsets);
}

void ManualDWARFIndex::GetFunctions(
    ConstString name, DWARFDebugInfo &info,
    llvm::function_ref<bool(const DWARFDIE &die, bool include_inlines,
                            lldb_private::SymbolContextList &sc_list)>
        resolve_function,
    llvm::function_ref<CompilerDeclContext(lldb::user_id_t type_uid)>
        get_decl_context_containing_uid,
    const CompilerDeclContext *parent_decl_ctx, uint32_t name_type_mask,
    bool include_inlines, SymbolContextList &sc_list) {

  Index();

  std::set<const DWARFDebugInfoEntry *> resolved_dies;
  DIEArray offsets;
  if (name_type_mask & eFunctionNameTypeFull) {
    uint32_t num_matches = m_function_basenames.Find(name, offsets);
    num_matches += m_function_methods.Find(name, offsets);
    num_matches += m_function_fullnames.Find(name, offsets);
    for (uint32_t i = 0; i < num_matches; i++) {
      const DIERef &die_ref = offsets[i];
      DWARFDIE die = info.GetDIE(die_ref);
      if (die) {
        if (!SymbolFileDWARF::DIEInDeclContext(parent_decl_ctx, die))
          continue; // The containing decl contexts don't match

        if (resolved_dies.find(die.GetDIE()) == resolved_dies.end()) {
          if (resolve_function(die, include_inlines, sc_list))
            resolved_dies.insert(die.GetDIE());
        }
      }
    }
    offsets.clear();
  }
  if (name_type_mask & eFunctionNameTypeBase) {
    uint32_t num_base = m_function_basenames.Find(name, offsets);
    for (uint32_t i = 0; i < num_base; i++) {
      DWARFDIE die = info.GetDIE(offsets[i]);
      if (die) {
        if (!SymbolFileDWARF::DIEInDeclContext(parent_decl_ctx, die))
          continue; // The containing decl contexts don't match

        // If we get to here, the die is good, and we should add it:
        if (resolved_dies.find(die.GetDIE()) == resolved_dies.end()) {
          if (resolve_function(die, include_inlines, sc_list))
            resolved_dies.insert(die.GetDIE());
        }
      }
    }
    offsets.clear();
  }

  if (name_type_mask & eFunctionNameTypeMethod) {
    if (parent_decl_ctx && parent_decl_ctx->IsValid())
      return; // no methods in namespaces

    uint32_t num_base = m_function_methods.Find(name, offsets);
    {
      for (uint32_t i = 0; i < num_base; i++) {
        DWARFDIE die = info.GetDIE(offsets[i]);
        if (die) {
          // If we get to here, the die is good, and we should add it:
          if (resolved_dies.find(die.GetDIE()) == resolved_dies.end()) {
            if (resolve_function(die, include_inlines, sc_list))
              resolved_dies.insert(die.GetDIE());
          }
        }
      }
    }
    offsets.clear();
  }

  if ((name_type_mask & eFunctionNameTypeSelector) &&
      (!parent_decl_ctx || !parent_decl_ctx->IsValid())) {
    uint32_t num_selectors = m_function_selectors.Find(name, offsets);
    for (uint32_t i = 0; i < num_selectors; i++) {
      DWARFDIE die = info.GetDIE(offsets[i]);
      if (die) {
        // If we get to here, the die is good, and we should add it:
        if (resolved_dies.find(die.GetDIE()) == resolved_dies.end()) {
          if (resolve_function(die, include_inlines, sc_list))
            resolved_dies.insert(die.GetDIE());
        }
      }
    }
  }
}

void ManualDWARFIndex::GetFunctions(
    const RegularExpression &regex, DWARFDebugInfo &info,
    llvm::function_ref<bool(const DWARFDIE &die, bool include_inlines,
                            lldb_private::SymbolContextList &sc_list)>
        resolve_function,
    bool include_inlines, SymbolContextList &sc_list) {
  Index();

  DIEArray offsets;
  m_function_basenames.Find(regex, offsets);
  m_function_fullnames.Find(regex, offsets);
  ParseFunctions(offsets, info, resolve_function, include_inlines, sc_list);
}

void ManualDWARFIndex::Dump(Stream &s) {
  s.Format("DWARF index for ({0}) '{1:F}':",
           m_module.GetArchitecture().GetArchitectureName(),
           m_module.GetObjectFile()->GetFileSpec());
  s.Printf("\nFunction basenames:\n");
  m_function_basenames.Dump(&s);
  s.Printf("\nFunction fullnames:\n");
  m_function_fullnames.Dump(&s);
  s.Printf("\nFunction methods:\n");
  m_function_methods.Dump(&s);
  s.Printf("\nFunction selectors:\n");
  m_function_selectors.Dump(&s);
  s.Printf("\nObjective C class selectors:\n");
  m_objc_class_selectors.Dump(&s);
  s.Printf("\nGlobals and statics:\n");
  m_globals.Dump(&s);
  s.Printf("\nTypes:\n");
  m_types.Dump(&s);
  s.Printf("\nNamespaces:\n");
  m_namespaces.Dump(&s);
}
