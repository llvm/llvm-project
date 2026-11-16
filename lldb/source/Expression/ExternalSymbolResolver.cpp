//===-- ExternalSymbolResolver.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ExternalSymbolResolver.h"
#include "lldb/Core/Module.h"
#include "lldb/Expression/Expression.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-defines.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <optional>

using namespace lldb_private;

ExternalSymbolResolver::ExternalSymbolResolver(const SymbolContext &sym_ctx,
                                               bool strip_underscore)
    : m_sym_ctx(sym_ctx), m_strip_underscore(strip_underscore),
      m_preferred_modules() {}

void ExternalSymbolResolver::CollectCandidateCNames(
    std::vector<ConstString> &C_names, ConstString name) {
  if (m_strip_underscore && name.GetStringRef().starts_with('_'))
    C_names.insert(C_names.begin(), ConstString(&name.GetCString()[1]));
  C_names.push_back(name);
}

void ExternalSymbolResolver::CollectCandidateCPlusPlusNames(
    std::vector<ConstString> &CPP_names,
    const std::vector<ConstString> &C_names, const SymbolContext &sc) {
  if (auto *cpp_lang = Language::FindPlugin(lldb::eLanguageTypeC_plus_plus)) {
    for (const ConstString &name : C_names) {
      Mangled mangled(name);
      if (cpp_lang->SymbolNameFitsToLanguage(mangled)) {
        if (ConstString best_alternate =
                cpp_lang->FindBestAlternateFunctionMangledName(mangled, sc)) {
          CPP_names.push_back(best_alternate);
        }
      }

      std::vector<ConstString> alternates =
          cpp_lang->GenerateAlternateFunctionManglings(name);
      CPP_names.insert(CPP_names.end(), alternates.begin(), alternates.end());

      // As a last-ditch fallback, try the base name for C++ names.  It's
      // terrible, but the DWARF doesn't always encode "extern C" correctly.
      ConstString basename =
          cpp_lang->GetDemangledFunctionNameWithoutArguments(mangled);
      CPP_names.push_back(basename);
    }
  }
}

class LoadAddressResolver {
public:
  LoadAddressResolver(Target &target, bool &symbol_was_missing_weak)
      : m_target(target), m_symbol_was_missing_weak(symbol_was_missing_weak) {}

  std::optional<lldb::addr_t> Resolve(SymbolContextList &sc_list) {
    if (sc_list.IsEmpty())
      return std::nullopt;

    lldb::addr_t load_address = LLDB_INVALID_ADDRESS;

    // Missing_weak_symbol will be true only if we found only weak undefined
    // references to this symbol.
    m_symbol_was_missing_weak = true;

    for (auto candidate_sc : sc_list.SymbolContexts()) {
      // Only symbols can be weak undefined.
      if (!candidate_sc.symbol ||
          candidate_sc.symbol->GetType() != lldb::eSymbolTypeUndefined ||
          !candidate_sc.symbol->IsWeak())
        m_symbol_was_missing_weak = false;

      // First try the symbol.
      if (candidate_sc.symbol) {
        load_address = candidate_sc.symbol->ResolveCallableAddress(
            m_target, candidate_sc.module_sp);
        if (load_address == LLDB_INVALID_ADDRESS) {
          Address addr = candidate_sc.symbol->GetAddress();
          load_address = m_target.GetProcessSP()
                             ? addr.GetLoadAddress(&m_target)
                             : addr.GetFileAddress();
        }
      }

      // If that didn't work, try the function.
      if (load_address == LLDB_INVALID_ADDRESS && candidate_sc.function) {
        Address addr = candidate_sc.function->GetAddress();
        load_address = m_target.GetProcessSP()
                           ? addr.GetCallableLoadAddress(&m_target)
                           : addr.GetFileAddress();
      }

      // We found a load address.
      if (load_address != LLDB_INVALID_ADDRESS) {
        // If the load address is external, we're done.
        const bool is_external =
            (candidate_sc.function) ||
            (candidate_sc.symbol && candidate_sc.symbol->IsExternal());
        if (is_external)
          return load_address;

        // Otherwise, remember the best internal load address.
        if (m_best_internal_load_address == LLDB_INVALID_ADDRESS)
          m_best_internal_load_address = load_address;
      }
    }

    // You test the address of a weak symbol against NULL to see if it is
    // present. So we should return 0 for a missing weak symbol.
    if (m_symbol_was_missing_weak)
      return 0;

    return std::nullopt;
  }

  lldb::addr_t GetBestInternalLoadAddress() const {
    return m_best_internal_load_address;
  }

private:
  Target &m_target;
  bool &m_symbol_was_missing_weak;
  lldb::addr_t m_best_internal_load_address = LLDB_INVALID_ADDRESS;
};

/// Returns address of the function referred to by the special function call
/// label \c label.
static llvm::Expected<lldb::addr_t>
ResolveFunctionCallLabel(FunctionCallLabel &label,
                         const lldb_private::SymbolContext &sc,
                         bool &symbol_was_missing_weak) {
  symbol_was_missing_weak = false;

  if (!sc.target_sp)
    return llvm::createStringError("target not available");

  auto module_sp = sc.target_sp->GetImages().FindModule(label.module_id);
  if (!module_sp)
    return llvm::createStringError(
        llvm::formatv("failed to find module by UID {0}", label.module_id));

  auto *symbol_file = module_sp->GetSymbolFile();
  if (!symbol_file)
    return llvm::createStringError(
        llvm::formatv("no SymbolFile found on module {0:x}.", module_sp.get()));

  auto sc_or_err = symbol_file->ResolveFunctionCallLabel(label);
  if (!sc_or_err)
    return llvm::joinErrors(
        llvm::createStringError("failed to resolve function by UID:"),
        sc_or_err.takeError());

  SymbolContextList sc_list;
  sc_list.Append(*sc_or_err);

  LoadAddressResolver resolver(*sc.target_sp, symbol_was_missing_weak);
  lldb::addr_t resolved_addr =
      resolver.Resolve(sc_list).value_or(LLDB_INVALID_ADDRESS);
  if (resolved_addr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("couldn't resolve address for function");

  return resolved_addr;
}

lldb::addr_t
ExternalSymbolResolver::FindInSymbols(const std::vector<ConstString> &names,
                                      const lldb_private::SymbolContext &sc,
                                      bool &symbol_was_missing_weak) {
  symbol_was_missing_weak = false;

  Target *target = sc.target_sp.get();
  if (!target) {
    // We shouldn't be doing any symbol lookup at all without a target.
    return LLDB_INVALID_ADDRESS;
  }

  ModuleList non_local_images = target->GetImages();
  // We'll process module_sp and any preferred modules separately, before the
  // other modules.
  non_local_images.Remove(sc.module_sp);
  for (size_t i = 0; i < m_preferred_modules.GetSize(); ++i)
    non_local_images.Remove(m_preferred_modules.GetModuleAtIndex(i));

  // Drop modules the platform considers off-limits to unconstrained symbol
  // searches.
  for (size_t i = non_local_images.GetSize(); i > 0; --i) {
    lldb::ModuleSP module_sp = non_local_images.GetModuleAtIndex(i - 1);
    if (target->ModuleIsExcludedForUnconstrainedSearches(module_sp))
      non_local_images.Remove(module_sp);
  }

  LoadAddressResolver resolver(*target, symbol_was_missing_weak);

  ModuleFunctionSearchOptions function_options;
  function_options.include_symbols = true;
  function_options.include_inlines = false;

  for (const ConstString &name : names) {
    // The lookup order here is as follows:
    // 1) Functions in `sc.module_sp`
    // 2) Functions in the preferred modules list
    // 3) Functions in the other modules
    // 4) Symbols in `sc.module_sp`
    // 5) Symbols in the preferred modules list
    // 6) Symbols in the other modules
    if (sc.module_sp) {
      SymbolContextList sc_list;
      sc.module_sp->FindFunctions(name, CompilerDeclContext(),
                                  lldb::eFunctionNameTypeFull, function_options,
                                  sc_list);
      if (auto load_addr = resolver.Resolve(sc_list))
        return *load_addr;
    }

    {
      SymbolContextList sc_list;
      m_preferred_modules.FindFunctions(name, lldb::eFunctionNameTypeFull,
                                        function_options, sc_list);
      if (auto load_addr = resolver.Resolve(sc_list))
        return *load_addr;
    }

    {
      SymbolContextList sc_list;
      non_local_images.FindFunctions(name, lldb::eFunctionNameTypeFull,
                                     function_options, sc_list);
      if (auto load_addr = resolver.Resolve(sc_list))
        return *load_addr;
    }

    if (sc.module_sp) {
      SymbolContextList sc_list;
      sc.module_sp->FindSymbolsWithNameAndType(name, lldb::eSymbolTypeAny,
                                               sc_list);
      if (auto load_addr = resolver.Resolve(sc_list))
        return *load_addr;
    }

    {
      SymbolContextList sc_list;
      m_preferred_modules.FindSymbolsWithNameAndType(name, lldb::eSymbolTypeAny,
                                                     sc_list);
      if (auto load_addr = resolver.Resolve(sc_list))
        return *load_addr;
    }

    {
      SymbolContextList sc_list;
      non_local_images.FindSymbolsWithNameAndType(name, lldb::eSymbolTypeAny,
                                                  sc_list);
      if (auto load_addr = resolver.Resolve(sc_list))
        return *load_addr;
    }

    lldb::addr_t best_internal_load_address =
        resolver.GetBestInternalLoadAddress();
    if (best_internal_load_address != LLDB_INVALID_ADDRESS)
      return best_internal_load_address;
  }

  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t
ExternalSymbolResolver::FindInRuntimes(const std::vector<ConstString> &names,
                                       const lldb_private::SymbolContext &sc) {
  lldb::TargetSP target_sp = sc.target_sp;

  if (!target_sp) {
    return LLDB_INVALID_ADDRESS;
  }

  lldb::ProcessSP process_sp = sc.target_sp->GetProcessSP();

  if (!process_sp) {
    return LLDB_INVALID_ADDRESS;
  }

  for (const ConstString &name : names) {
    for (LanguageRuntime *runtime : process_sp->GetLanguageRuntimes()) {
      lldb::addr_t symbol_load_addr = runtime->LookupRuntimeSymbol(name);

      if (symbol_load_addr != LLDB_INVALID_ADDRESS)
        return symbol_load_addr;
    }
  }

  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t ExternalSymbolResolver::FindInUserDefinedSymbols(
    const std::vector<ConstString> &names,
    const lldb_private::SymbolContext &sc) {
  lldb::TargetSP target_sp = sc.target_sp;

  for (const ConstString &name : names) {
    lldb::addr_t symbol_load_addr = target_sp->GetPersistentSymbol(name);

    if (symbol_load_addr != LLDB_INVALID_ADDRESS)
      return symbol_load_addr;
  }

  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t ExternalSymbolResolver::FindSymbol(lldb_private::ConstString name,
                                                bool &missing_weak) {
  if (name.GetStringRef().starts_with(FunctionCallLabelPrefix)) {
    auto label_or_err = FunctionCallLabel::fromString(name);
    if (!label_or_err) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Expressions), label_or_err.takeError(),
                     "failed to create FunctionCallLabel from '{1}': {0}",
                     name.GetStringRef());
      return LLDB_INVALID_ADDRESS;
    }

    if (auto addr_or_err =
            ResolveFunctionCallLabel(*label_or_err, m_sym_ctx, missing_weak)) {
      return *addr_or_err;
    } else {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Expressions), addr_or_err.takeError(),
                     "Failed to resolve function call label '{1}': {0}",
                     name.GetStringRef());

      // Fall back to lookup by name despite error in resolving the label.
      // May happen in practice if the definition of a function lives in
      // a different lldb_private::Module than it's declaration. Meaning
      // we couldn't pin-point it using the information encoded in the label.
      name.SetString(label_or_err->lookup_name);
    }
  }

  // TODO: now with function call labels, do we still need to
  // generate alternate manglings?

  std::vector<ConstString> candidate_C_names;
  std::vector<ConstString> candidate_CPlusPlus_names;

  CollectCandidateCNames(candidate_C_names, name);

  lldb::addr_t ret = FindInSymbols(candidate_C_names, m_sym_ctx, missing_weak);
  if (ret != LLDB_INVALID_ADDRESS)
    return ret;

  // If we find the symbol in runtimes or user defined symbols it can't be
  // a missing weak symbol.
  missing_weak = false;
  ret = FindInRuntimes(candidate_C_names, m_sym_ctx);
  if (ret != LLDB_INVALID_ADDRESS)
    return ret;

  ret = FindInUserDefinedSymbols(candidate_C_names, m_sym_ctx);
  if (ret != LLDB_INVALID_ADDRESS)
    return ret;

  CollectCandidateCPlusPlusNames(candidate_CPlusPlus_names, candidate_C_names,
                                 m_sym_ctx);
  ret = FindInSymbols(candidate_CPlusPlus_names, m_sym_ctx, missing_weak);
  return ret;
}
