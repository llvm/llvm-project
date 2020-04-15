//===-- ManualDWARFIndex.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/ManualDWARFIndex.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"
#include "Plugins/SymbolFile/DWARF/DWARFDeclContext.h"
#include "Plugins/SymbolFile/DWARF/LogChannelDWARF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDwo.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/TaskPool.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/Timer.h"

using namespace lldb_private;
using namespace lldb;

void ManualDWARFIndex::Index() {
  if (!m_dwarf)
    return;

  SymbolFileDWARF &main_dwarf = *m_dwarf;
  m_dwarf = nullptr;

  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(func_cat, "%p", static_cast<void *>(&main_dwarf));

  DWARFDebugInfo &main_info = main_dwarf.DebugInfo();
  SymbolFileDWARFDwo *dwp_dwarf = main_dwarf.GetDwpSymbolFile().get();
  DWARFDebugInfo *dwp_info = dwp_dwarf ? &dwp_dwarf->DebugInfo() : nullptr;

  std::vector<DWARFUnit *> units_to_index;
  units_to_index.reserve(main_info.GetNumUnits() +
                         (dwp_info ? dwp_info->GetNumUnits() : 0));

  // Process all units in the main file, as well as any type units in the dwp
  // file. Type units in dwo files are handled when we reach the dwo file in
  // IndexUnit.
  for (size_t U = 0; U < main_info.GetNumUnits(); ++U) {
    DWARFUnit *unit = main_info.GetUnitAtIndex(U);
    if (unit && m_units_to_avoid.count(unit->GetOffset()) == 0)
      units_to_index.push_back(unit);
  }
  if (dwp_info && dwp_info->ContainsTypeUnits()) {
    for (size_t U = 0; U < dwp_info->GetNumUnits(); ++U) {
      if (auto *tu = llvm::dyn_cast<DWARFTypeUnit>(dwp_info->GetUnitAtIndex(U)))
        units_to_index.push_back(tu);
    }
  }

  if (units_to_index.empty())
    return;

  std::vector<IndexSet> sets(units_to_index.size());

  // Keep memory down by clearing DIEs for any units if indexing
  // caused us to load the unit's DIEs.
  std::vector<llvm::Optional<DWARFUnit::ScopedExtractDIEs>> clear_cu_dies(
      units_to_index.size());
  auto parser_fn = [&](size_t cu_idx) {
    IndexUnit(*units_to_index[cu_idx], dwp_dwarf, sets[cu_idx]);
  };

  auto extract_fn = [&units_to_index, &clear_cu_dies](size_t cu_idx) {
    clear_cu_dies[cu_idx] = units_to_index[cu_idx]->ExtractDIEsScoped();
  };

  // Create a task runner that extracts dies for each DWARF unit in a
  // separate thread
  // First figure out which units didn't have their DIEs already
  // parsed and remember this.  If no DIEs were parsed prior to this index
  // function call, we are going to want to clear the CU dies after we are
  // done indexing to make sure we don't pull in all DWARF dies, but we need
  // to wait until all units have been indexed in case a DIE in one
  // unit refers to another and the indexes accesses those DIEs.
  TaskMapOverInt(0, units_to_index.size(), extract_fn);

  // Now create a task runner that can index each DWARF unit in a
  // separate thread so we can index quickly.

  TaskMapOverInt(0, units_to_index.size(), parser_fn);

  auto finalize_fn = [this, &sets](NameToDIE(IndexSet::*index)) {
    NameToDIE &result = m_set.*index;
    for (auto &set : sets)
      result.Append(set.*index);
    result.Finalize();
  };

  TaskPool::RunTasks([&]() { finalize_fn(&IndexSet::function_basenames); },
                     [&]() { finalize_fn(&IndexSet::function_fullnames); },
                     [&]() { finalize_fn(&IndexSet::function_methods); },
                     [&]() { finalize_fn(&IndexSet::function_selectors); },
                     [&]() { finalize_fn(&IndexSet::objc_class_selectors); },
                     [&]() { finalize_fn(&IndexSet::globals); },
                     [&]() { finalize_fn(&IndexSet::types); },
                     [&]() { finalize_fn(&IndexSet::namespaces); });
}

void ManualDWARFIndex::IndexUnit(DWARFUnit &unit, SymbolFileDWARFDwo *dwp,
                                 IndexSet &set) {
  Log *log = LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS);

  if (log) {
    m_module.LogMessage(
        log, "ManualDWARFIndex::IndexUnit for unit at .debug_info[0x%8.8x]",
        unit.GetOffset());
  }

  const LanguageType cu_language = SymbolFileDWARF::GetLanguage(unit);

  IndexUnitImpl(unit, cu_language, set);

  if (SymbolFileDWARFDwo *dwo_symbol_file = unit.GetDwoSymbolFile()) {
    // Type units in a dwp file are indexed separately, so we just need to
    // process the split unit here. However, if the split unit is in a dwo file,
    // then we need to process type units here.
    if (dwo_symbol_file == dwp) {
      IndexUnitImpl(unit.GetNonSkeletonUnit(), cu_language, set);
    } else {
      DWARFDebugInfo &dwo_info = dwo_symbol_file->DebugInfo();
      for (size_t i = 0; i < dwo_info.GetNumUnits(); ++i)
        IndexUnitImpl(*dwo_info.GetUnitAtIndex(i), cu_language, set);
    }
  }
}

void ManualDWARFIndex::IndexUnitImpl(DWARFUnit &unit,
                                     const LanguageType cu_language,
                                     IndexSet &set) {
  for (const DWARFDebugInfoEntry &die : unit.dies()) {
    const dw_tag_t tag = die.Tag();

    switch (tag) {
    case DW_TAG_array_type:
    case DW_TAG_base_type:
    case DW_TAG_class_type:
    case DW_TAG_constant:
    case DW_TAG_enumeration_type:
    case DW_TAG_inlined_subroutine:
    case DW_TAG_namespace:
    case DW_TAG_string_type:
    case DW_TAG_structure_type:
    case DW_TAG_subprogram:
    case DW_TAG_subroutine_type:
    case DW_TAG_typedef:
    case DW_TAG_union_type:
    case DW_TAG_unspecified_type:
    case DW_TAG_variable:
      break;

    default:
      continue;
    }

    DWARFAttributes attributes;
    const char *name = nullptr;
    const char *mangled_cstr = nullptr;
    bool is_declaration = false;
    // bool is_artificial = false;
    bool has_address = false;
    bool has_location_or_const_value = false;
    bool is_global_or_static_variable = false;

    DWARFFormValue specification_die_form;
    const size_t num_attributes = die.GetAttributes(&unit, attributes);
    if (num_attributes > 0) {
      for (uint32_t i = 0; i < num_attributes; ++i) {
        dw_attr_t attr = attributes.AttributeAtIndex(i);
        DWARFFormValue form_value;
        switch (attr) {
        case DW_AT_name:
          if (attributes.ExtractFormValueAtIndex(i, form_value))
            name = form_value.AsCString();
          break;

        case DW_AT_declaration:
          if (attributes.ExtractFormValueAtIndex(i, form_value))
            is_declaration = form_value.Unsigned() != 0;
          break;

        case DW_AT_MIPS_linkage_name:
        case DW_AT_linkage_name:
          if (attributes.ExtractFormValueAtIndex(i, form_value))
            mangled_cstr = form_value.AsCString();
          break;

        case DW_AT_low_pc:
        case DW_AT_high_pc:
        case DW_AT_ranges:
          has_address = true;
          break;

        case DW_AT_entry_pc:
          has_address = true;
          break;

        case DW_AT_location:
        case DW_AT_const_value:
          has_location_or_const_value = true;
          is_global_or_static_variable = die.IsGlobalOrStaticVariable();

          break;

        case DW_AT_specification:
          if (attributes.ExtractFormValueAtIndex(i, form_value))
            specification_die_form = form_value;
          break;
        }
      }
    }

    DIERef ref = *DWARFDIE(&unit, &die).GetDIERef();
    switch (tag) {
    case DW_TAG_inlined_subroutine:
    case DW_TAG_subprogram:
      if (has_address) {
        if (name) {
          bool is_objc_method = false;
          if (cu_language == eLanguageTypeObjC ||
              cu_language == eLanguageTypeObjC_plus_plus) {
            ObjCLanguage::MethodName objc_method(name, true);
            if (objc_method.IsValid(true)) {
              is_objc_method = true;
              ConstString class_name_with_category(
                  objc_method.GetClassNameWithCategory());
              ConstString objc_selector_name(objc_method.GetSelector());
              ConstString objc_fullname_no_category_name(
                  objc_method.GetFullNameWithoutCategory(true));
              ConstString class_name_no_category(objc_method.GetClassName());
              set.function_fullnames.Insert(ConstString(name), ref);
              if (class_name_with_category)
                set.objc_class_selectors.Insert(class_name_with_category, ref);
              if (class_name_no_category &&
                  class_name_no_category != class_name_with_category)
                set.objc_class_selectors.Insert(class_name_no_category, ref);
              if (objc_selector_name)
                set.function_selectors.Insert(objc_selector_name, ref);
              if (objc_fullname_no_category_name)
                set.function_fullnames.Insert(objc_fullname_no_category_name,
                                              ref);
            }
          }
          // If we have a mangled name, then the DW_AT_name attribute is
          // usually the method name without the class or any parameters
          bool is_method = DWARFDIE(&unit, &die).IsMethod();

          if (is_method)
            set.function_methods.Insert(ConstString(name), ref);
          else
            set.function_basenames.Insert(ConstString(name), ref);

          if (!is_method && !mangled_cstr && !is_objc_method)
            set.function_fullnames.Insert(ConstString(name), ref);
        }
        if (mangled_cstr) {
          // Make sure our mangled name isn't the same string table entry as
          // our name. If it starts with '_', then it is ok, else compare the
          // string to make sure it isn't the same and we don't end up with
          // duplicate entries
          if (name && name != mangled_cstr &&
              ((mangled_cstr[0] == '_') ||
               (::strcmp(name, mangled_cstr) != 0))) {
            set.function_fullnames.Insert(ConstString(mangled_cstr), ref);
          }
        }
      }
      break;

    case DW_TAG_array_type:
    case DW_TAG_base_type:
    case DW_TAG_class_type:
    case DW_TAG_constant:
    case DW_TAG_enumeration_type:
    case DW_TAG_string_type:
    case DW_TAG_structure_type:
    case DW_TAG_subroutine_type:
    case DW_TAG_typedef:
    case DW_TAG_union_type:
    case DW_TAG_unspecified_type:
      if (name && !is_declaration)
        set.types.Insert(ConstString(name), ref);
      if (mangled_cstr && !is_declaration)
        set.types.Insert(ConstString(mangled_cstr), ref);
      break;

    case DW_TAG_namespace:
      if (name)
        set.namespaces.Insert(ConstString(name), ref);
      break;

    case DW_TAG_variable:
      if (name && has_location_or_const_value && is_global_or_static_variable) {
        set.globals.Insert(ConstString(name), ref);
        // Be sure to include variables by their mangled and demangled names if
        // they have any since a variable can have a basename "i", a mangled
        // named "_ZN12_GLOBAL__N_11iE" and a demangled mangled name
        // "(anonymous namespace)::i"...

        // Make sure our mangled name isn't the same string table entry as our
        // name. If it starts with '_', then it is ok, else compare the string
        // to make sure it isn't the same and we don't end up with duplicate
        // entries
        if (mangled_cstr && name != mangled_cstr &&
            ((mangled_cstr[0] == '_') || (::strcmp(name, mangled_cstr) != 0))) {
          set.globals.Insert(ConstString(mangled_cstr), ref);
        }
      }
      break;

    default:
      continue;
    }
  }
}

void ManualDWARFIndex::GetGlobalVariables(ConstString basename, DIEArray &offsets) {
  Index();
  m_set.globals.Find(basename, offsets);
}

void ManualDWARFIndex::GetGlobalVariables(const RegularExpression &regex,
                                          DIEArray &offsets) {
  Index();
  m_set.globals.Find(regex, offsets);
}

void ManualDWARFIndex::GetGlobalVariables(const DWARFUnit &unit,
                                          DIEArray &offsets) {
  Index();
  m_set.globals.FindAllEntriesForUnit(unit, offsets);
}

void ManualDWARFIndex::GetObjCMethods(ConstString class_name,
                                      DIEArray &offsets) {
  Index();
  m_set.objc_class_selectors.Find(class_name, offsets);
}

void ManualDWARFIndex::GetCompleteObjCClass(ConstString class_name,
                                            bool must_be_implementation,
                                            DIEArray &offsets) {
  Index();
  m_set.types.Find(class_name, offsets);
}

void ManualDWARFIndex::GetTypes(ConstString name, DIEArray &offsets) {
  Index();
  m_set.types.Find(name, offsets);
}

void ManualDWARFIndex::GetTypes(const DWARFDeclContext &context,
                                DIEArray &offsets) {
  Index();
  m_set.types.Find(ConstString(context[0].name), offsets);
}

void ManualDWARFIndex::GetNamespaces(ConstString name, DIEArray &offsets) {
  Index();
  m_set.namespaces.Find(name, offsets);
}

void ManualDWARFIndex::GetFunctions(ConstString name, SymbolFileDWARF &dwarf,
                                    const CompilerDeclContext &parent_decl_ctx,
                                    uint32_t name_type_mask,
                                    std::vector<DWARFDIE> &dies) {
  Index();

  if (name_type_mask & eFunctionNameTypeFull) {
    DIEArray offsets;
    m_set.function_fullnames.Find(name, offsets);
    for (const DIERef &die_ref: offsets) {
      DWARFDIE die = dwarf.GetDIE(die_ref);
      if (!die)
        continue;
      if (!SymbolFileDWARF::DIEInDeclContext(parent_decl_ctx, die))
        continue;
      dies.push_back(die);
    }
  }
  if (name_type_mask & eFunctionNameTypeBase) {
    DIEArray offsets;
    m_set.function_basenames.Find(name, offsets);
    for (const DIERef &die_ref: offsets) {
      DWARFDIE die = dwarf.GetDIE(die_ref);
      if (!die)
        continue;
      if (!SymbolFileDWARF::DIEInDeclContext(parent_decl_ctx, die))
        continue;
      dies.push_back(die);
    }
  }

  if (name_type_mask & eFunctionNameTypeMethod && !parent_decl_ctx.IsValid()) {
    DIEArray offsets;
    m_set.function_methods.Find(name, offsets);
    for (const DIERef &die_ref: offsets) {
      DWARFDIE die = dwarf.GetDIE(die_ref);
      if (!die)
        continue;
      dies.push_back(die);
    }
  }

  if (name_type_mask & eFunctionNameTypeSelector &&
      !parent_decl_ctx.IsValid()) {
    DIEArray offsets;
    m_set.function_selectors.Find(name, offsets);
    for (const DIERef &die_ref: offsets) {
      DWARFDIE die = dwarf.GetDIE(die_ref);
      if (!die)
        continue;
      dies.push_back(die);
    }
  }
}

void ManualDWARFIndex::GetFunctions(const RegularExpression &regex,
                                    DIEArray &offsets) {
  Index();

  m_set.function_basenames.Find(regex, offsets);
  m_set.function_fullnames.Find(regex, offsets);
}

void ManualDWARFIndex::Dump(Stream &s) {
  s.Format("Manual DWARF index for ({0}) '{1:F}':",
           m_module.GetArchitecture().GetArchitectureName(),
           m_module.GetObjectFile()->GetFileSpec());
  s.Printf("\nFunction basenames:\n");
  m_set.function_basenames.Dump(&s);
  s.Printf("\nFunction fullnames:\n");
  m_set.function_fullnames.Dump(&s);
  s.Printf("\nFunction methods:\n");
  m_set.function_methods.Dump(&s);
  s.Printf("\nFunction selectors:\n");
  m_set.function_selectors.Dump(&s);
  s.Printf("\nObjective-C class selectors:\n");
  m_set.objc_class_selectors.Dump(&s);
  s.Printf("\nGlobals and statics:\n");
  m_set.globals.Dump(&s);
  s.Printf("\nTypes:\n");
  m_set.types.Dump(&s);
  s.Printf("\nNamespaces:\n");
  m_set.namespaces.Dump(&s);
}
