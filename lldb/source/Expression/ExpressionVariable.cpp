//===-- ExpressionVariable.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include <optional>

using namespace lldb_private;

char ExpressionVariable::ID;

ExpressionVariable::ExpressionVariable() : m_flags(0) {}

uint8_t *ExpressionVariable::GetValueBytes() {
  lldb::ValueObjectSP valobj_sp = GetValueObject();
  std::optional<uint64_t> byte_size =
      llvm::expectedToOptional(valobj_sp->GetByteSize());
  if (byte_size && *byte_size) {
    if (valobj_sp->GetDataExtractor().GetByteSize() < *byte_size) {
      valobj_sp->GetValue().ResizeData(*byte_size);
      valobj_sp->GetValue().GetData(valobj_sp->GetDataExtractor());
    }
    return const_cast<uint8_t *>(valobj_sp->GetDataExtractor().GetDataStart());
  }
  return nullptr;
}

char PersistentExpressionState::ID;

PersistentExpressionState::PersistentExpressionState() = default;

void ExpressionVariable::TransferAddress(bool force) {
  if (!m_live_sp)
    return;

  if (!m_frozen_sp)
    return;

  if (force || (m_frozen_sp->GetLiveAddress() == LLDB_INVALID_ADDRESS)) {
    lldb::addr_t live_addr = m_live_sp->GetLiveAddress();
    m_frozen_sp->SetLiveAddress(live_addr);
    // One more detail, if there's an offset_to_top in the frozen_sp, then we
    // need to appy that offset by hand.  The live_sp can't compute this
    // itself as its type is the type of the contained object which confuses
    // the dynamic type calculation.  So we have to update the contents of the
    // m_live_sp with the dynamic value.
    // Note: We could get this right when we originally write the address, but
    // that happens in different ways for the various flavors of
    // Entity*::Materialize, but everything comes through here, and it's just
    // one extra memory write.

    // You can only have an "offset_to_top" with pointers or references:
    if (!m_frozen_sp->GetCompilerType().IsPointerOrReferenceType())
      return;

    lldb::ProcessSP process_sp = m_frozen_sp->GetProcessSP();
    // If there's no dynamic value, then there can't be an offset_to_top:
    if (!process_sp ||
        !process_sp->IsPossibleDynamicValue(*(m_frozen_sp.get())))
      return;

    lldb::ValueObjectSP dyn_sp = m_frozen_sp->GetDynamicValue(m_dyn_option);
    if (!dyn_sp)
      return;
    ValueObject::AddrAndType static_addr = m_frozen_sp->GetPointerValue();
    if (static_addr.type != eAddressTypeLoad)
      return;

    ValueObject::AddrAndType dynamic_addr = dyn_sp->GetPointerValue();
    if (dynamic_addr.type != eAddressTypeLoad ||
        static_addr.address == dynamic_addr.address)
      return;

    Status error;
    Log *log = GetLog(LLDBLog::Expressions);
    lldb::addr_t cur_value =
        process_sp->ReadPointerFromMemory(live_addr, error);
    if (error.Fail())
      return;

    if (cur_value != static_addr.address) {
      LLDB_LOG(log,
               "Stored value: {0} read from {1} doesn't "
               "match static addr: {2}",
               cur_value, live_addr, static_addr.address);
      return;
    }

    if (!process_sp->WritePointerToMemory(live_addr, dynamic_addr.address,
                                          error)) {
      LLDB_LOG(log, "Got error: {0} writing dynamic value: {1} to {2}", error,
               dynamic_addr.address, live_addr);
      return;
    }
  }
}

PersistentExpressionState::~PersistentExpressionState() = default;

lldb::addr_t PersistentExpressionState::LookupSymbol(ConstString name) {
  SymbolMap::iterator si = m_symbol_map.find(name.GetCString());

  if (si != m_symbol_map.end())
    return si->second;
  else
    return LLDB_INVALID_ADDRESS;
}

void PersistentExpressionState::RegisterExecutionUnit(
    lldb::IRExecutionUnitSP &execution_unit_sp) {
  Log *log = GetLog(LLDBLog::Expressions);

  m_execution_units.insert(execution_unit_sp);

  LLDB_LOGF(log, "Registering JITted Functions:\n");

  for (const IRExecutionUnit::JittedFunction &jitted_function :
       execution_unit_sp->GetJittedFunctions()) {
    if (jitted_function.m_external &&
        jitted_function.m_name != execution_unit_sp->GetFunctionName() &&
        jitted_function.m_remote_addr != LLDB_INVALID_ADDRESS) {
      m_symbol_map[jitted_function.m_name.GetCString()] =
          jitted_function.m_remote_addr;
      LLDB_LOGF(log, "  Function: %s at 0x%" PRIx64 ".",
                jitted_function.m_name.GetCString(),
                jitted_function.m_remote_addr);
    }
  }

  LLDB_LOGF(log, "Registering JIIted Symbols:\n");

  for (const IRExecutionUnit::JittedGlobalVariable &global_var :
       execution_unit_sp->GetJittedGlobalVariables()) {
    if (global_var.m_remote_addr != LLDB_INVALID_ADDRESS) {
      // Demangle the name before inserting it, so that lookups by the ConstStr
      // of the demangled name will find the mangled one (needed for looking up
      // metadata pointers.)
      Mangled mangler(global_var.m_name);
      mangler.GetDemangledName();
      m_symbol_map[global_var.m_name.GetCString()] = global_var.m_remote_addr;
      LLDB_LOGF(log, "  Symbol: %s at 0x%" PRIx64 ".",
                global_var.m_name.GetCString(), global_var.m_remote_addr);
    }
  }
}
