//===-- ManualDWARFIndex.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
      IndexUnit(*dwarf_cu, function_basenames[cu_idx],
                function_fullnames[cu_idx], function_methods[cu_idx],
                function_selectors[cu_idx], objc_class_selectors[cu_idx],
                globals[cu_idx], types[cu_idx], namespaces[cu_idx]);
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

void ManualDWARFIndex::IndexUnit(DWARFUnit &unit, NameToDIE &func_basenames,
                                 NameToDIE &func_fullnames,
                                 NameToDIE &func_methods,
                                 NameToDIE &func_selectors,
                                 NameToDIE &objc_class_selectors,
                                 NameToDIE &globals, NameToDIE &types,
                                 NameToDIE &namespaces) {
  Log *log = LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS);

  if (log) {
    m_module.LogMessage(
        log, "ManualDWARFIndex::IndexUnit for compile unit at .debug_info[0x%8.8x]",
        unit.GetOffset());
  }

  const LanguageType cu_language = unit.GetLanguageType();
  DWARFFormValue::FixedFormSizes fixed_form_sizes = unit.GetFixedFormSizes();

  IndexUnitImpl(unit, cu_language, fixed_form_sizes, unit.GetOffset(),
                func_basenames, func_fullnames, func_methods, func_selectors,
                objc_class_selectors, globals, types, namespaces);

  SymbolFileDWARFDwo *dwo_symbol_file = unit.GetDwoSymbolFile();
  if (dwo_symbol_file && dwo_symbol_file->GetCompileUnit()) {
    IndexUnitImpl(*dwo_symbol_file->GetCompileUnit(), cu_language,
                  fixed_form_sizes, unit.GetOffset(), func_basenames,
                  func_fullnames, func_methods, func_selectors,
                  objc_class_selectors, globals, types, namespaces);
  }
}

void ManualDWARFIndex::IndexUnitImpl(
    DWARFUnit &unit, const LanguageType cu_language,
    const DWARFFormValue::FixedFormSizes &fixed_form_sizes,
    const dw_offset_t cu_offset, NameToDIE &func_basenames,
    NameToDIE &func_fullnames, NameToDIE &func_methods,
    NameToDIE &func_selectors, NameToDIE &objc_class_selectors,
    NameToDIE &globals, NameToDIE &types, NameToDIE &namespaces) {
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
    const char *name = NULL;
    const char *mangled_cstr = NULL;
    bool is_declaration = false;
    // bool is_artificial = false;
    bool has_address = false;
    bool has_location_or_const_value = false;
    bool is_global_or_static_variable = false;

    DWARFFormValue specification_die_form;
    const size_t num_attributes =
        die.GetAttributes(&unit, fixed_form_sizes, attributes);
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

        //                case DW_AT_artificial:
        //                    if (attributes.ExtractFormValueAtIndex(i,
        //                    form_value))
        //                        is_artificial = form_value.Unsigned() != 0;
        //                    break;

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
          if (tag == DW_TAG_variable) {
            const DWARFDebugInfoEntry *parent_die = die.GetParent();
            while (parent_die != NULL) {
              switch (parent_die->Tag()) {
              case DW_TAG_subprogram:
              case DW_TAG_lexical_block:
              case DW_TAG_inlined_subroutine:
                // Even if this is a function level static, we don't add it. We
                // could theoretically add these if we wanted to by
                // introspecting into the DW_AT_location and seeing if the
                // location describes a hard coded address, but we don't want
                // the performance penalty of that right now.
                is_global_or_static_variable = false;
                // if (attributes.ExtractFormValueAtIndex(dwarf, i,
                //                                        form_value)) {
                //   // If we have valid block data, then we have location
                //   // expression bytesthat are fixed (not a location list).
                //   const uint8_t *block_data = form_value.BlockData();
                //   if (block_data) {
                //     uint32_t block_length = form_value.Unsigned();
                //     if (block_length == 1 +
                //     attributes.CompileUnitAtIndex(i)->GetAddressByteSize()) {
                //       if (block_data[0] == DW_OP_addr)
                //         add_die = true;
                //     }
                //   }
                // }
                parent_die = NULL; // Terminate the while loop.
                break;

              case DW_TAG_compile_unit:
              case DW_TAG_partial_unit:
                is_global_or_static_variable = true;
                parent_die = NULL; // Terminate the while loop.
                break;

              default:
                parent_die =
                    parent_die->GetParent(); // Keep going in the while loop.
                break;
              }
            }
          }
          break;

        case DW_AT_specification:
          if (attributes.ExtractFormValueAtIndex(i, form_value))
            specification_die_form = form_value;
          break;
        }
      }
    }

    switch (tag) {
    case DW_TAG_subprogram:
      if (has_address) {
        if (name) {
          ObjCLanguage::MethodName objc_method(name, true);
          if (objc_method.IsValid(true)) {
            ConstString objc_class_name_with_category(
                objc_method.GetClassNameWithCategory());
            ConstString objc_selector_name(objc_method.GetSelector());
            ConstString objc_fullname_no_category_name(
                objc_method.GetFullNameWithoutCategory(true));
            ConstString objc_class_name_no_category(objc_method.GetClassName());
            func_fullnames.Insert(ConstString(name),
                                  DIERef(cu_offset, die.GetOffset()));
            if (objc_class_name_with_category)
              objc_class_selectors.Insert(objc_class_name_with_category,
                                          DIERef(cu_offset, die.GetOffset()));
            if (objc_class_name_no_category &&
                objc_class_name_no_category != objc_class_name_with_category)
              objc_class_selectors.Insert(objc_class_name_no_category,
                                          DIERef(cu_offset, die.GetOffset()));
            if (objc_selector_name)
              func_selectors.Insert(objc_selector_name,
                                    DIERef(cu_offset, die.GetOffset()));
            if (objc_fullname_no_category_name)
              func_fullnames.Insert(objc_fullname_no_category_name,
                                    DIERef(cu_offset, die.GetOffset()));
          }
          // If we have a mangled name, then the DW_AT_name attribute is
          // usually the method name without the class or any parameters
          const DWARFDebugInfoEntry *parent = die.GetParent();
          bool is_method = false;
          if (parent) {
            dw_tag_t parent_tag = parent->Tag();
            if (parent_tag == DW_TAG_class_type ||
                parent_tag == DW_TAG_structure_type) {
              is_method = true;
            } else {
              if (specification_die_form.IsValid()) {
                DWARFDIE specification_die =
                    unit.GetSymbolFileDWARF()->DebugInfo()->GetDIE(
                        DIERef(specification_die_form));
                if (specification_die.GetParent().IsStructOrClass())
                  is_method = true;
              }
            }
          }

          if (is_method)
            func_methods.Insert(ConstString(name),
                                DIERef(cu_offset, die.GetOffset()));
          else
            func_basenames.Insert(ConstString(name),
                                  DIERef(cu_offset, die.GetOffset()));

          if (!is_method && !mangled_cstr && !objc_method.IsValid(true))
            func_fullnames.Insert(ConstString(name),
                                  DIERef(cu_offset, die.GetOffset()));
        }
        if (mangled_cstr) {
          // Make sure our mangled name isn't the same string table entry as
          // our name. If it starts with '_', then it is ok, else compare the
          // string to make sure it isn't the same and we don't end up with
          // duplicate entries
          if (name && name != mangled_cstr &&
              ((mangled_cstr[0] == '_') ||
               (::strcmp(name, mangled_cstr) != 0))) {
            func_fullnames.Insert(ConstString(mangled_cstr),
                                  DIERef(cu_offset, die.GetOffset()));
          }
        }
      }
      break;

    case DW_TAG_inlined_subroutine:
      if (has_address) {
        if (name)
          func_basenames.Insert(ConstString(name),
                                DIERef(cu_offset, die.GetOffset()));
        if (mangled_cstr) {
          // Make sure our mangled name isn't the same string table entry as
          // our name. If it starts with '_', then it is ok, else compare the
          // string to make sure it isn't the same and we don't end up with
          // duplicate entries
          if (name && name != mangled_cstr &&
              ((mangled_cstr[0] == '_') ||
               (::strcmp(name, mangled_cstr) != 0))) {
            func_fullnames.Insert(ConstString(mangled_cstr),
                                  DIERef(cu_offset, die.GetOffset()));
          }
        } else
          func_fullnames.Insert(ConstString(name),
                                DIERef(cu_offset, die.GetOffset()));
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
        types.Insert(ConstString(name), DIERef(cu_offset, die.GetOffset()));
      if (mangled_cstr && !is_declaration)
        types.Insert(ConstString(mangled_cstr),
                     DIERef(cu_offset, die.GetOffset()));
      break;

    case DW_TAG_namespace:
      if (name)
        namespaces.Insert(ConstString(name),
                          DIERef(cu_offset, die.GetOffset()));
      break;

    case DW_TAG_variable:
      if (name && has_location_or_const_value && is_global_or_static_variable) {
        globals.Insert(ConstString(name), DIERef(cu_offset, die.GetOffset()));
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
          Mangled mangled(ConstString(mangled_cstr), true);
          globals.Insert(mangled.GetMangledName(),
                         DIERef(cu_offset, die.GetOffset()));
          ConstString demangled = mangled.GetDemangledName(cu_language);
          if (demangled)
            globals.Insert(demangled, DIERef(cu_offset, die.GetOffset()));
        }
      }
      break;

    default:
      continue;
    }
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
