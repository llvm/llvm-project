//===-- OperatingSystemGo.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
#include <unordered_map>

// Other libraries and framework includes
// Project includes
#include "OperatingSystemGo.h"

#include "Plugins/Process/Utility/DynamicRegisterInfo.h"
#include "Plugins/Process/Utility/RegisterContextMemory.h"
#include "Plugins/Process/Utility/ThreadMemory.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/OptionGroupBoolean.h"
#include "lldb/Interpreter/OptionGroupUInt64.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/Property.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadList.h"

using namespace lldb;
using namespace lldb_private;

namespace {

static PropertyDefinition g_properties[] = {
    {"enable", OptionValue::eTypeBoolean, true, true, nullptr, nullptr,
     "Specify whether goroutines should be treated as threads."},
    {NULL, OptionValue::eTypeInvalid, false, 0, NULL, NULL, NULL}};

enum {
  ePropertyEnableGoroutines,
};

class PluginProperties : public Properties {
public:
  PluginProperties() : Properties() {
    m_collection_sp.reset(new OptionValueProperties(GetSettingName()));
    m_collection_sp->Initialize(g_properties);
  }

  ~PluginProperties() override = default;

  static ConstString GetSettingName() {
    return OperatingSystemGo::GetPluginNameStatic();
  }

  bool GetEnableGoroutines() {
    const uint32_t idx = ePropertyEnableGoroutines;
    return m_collection_sp->GetPropertyAtIndexAsBoolean(
        NULL, idx, g_properties[idx].default_uint_value);
  }

  bool SetEnableGoroutines(bool enable) {
    const uint32_t idx = ePropertyEnableGoroutines;
    return m_collection_sp->SetPropertyAtIndexAsUInt64(NULL, idx, enable);
  }
};

typedef std::shared_ptr<PluginProperties> OperatingSystemGoPropertiesSP;

static const OperatingSystemGoPropertiesSP &GetGlobalPluginProperties() {
  static OperatingSystemGoPropertiesSP g_settings_sp;
  if (!g_settings_sp)
    g_settings_sp.reset(new PluginProperties());
  return g_settings_sp;
}

class RegisterContextGo : public RegisterContextMemory {
public:
  RegisterContextGo(lldb_private::Thread &thread, uint32_t concrete_frame_idx,
                    DynamicRegisterInfo &reg_info, lldb::addr_t reg_data_addr)
      : RegisterContextMemory(thread, concrete_frame_idx, reg_info,
                              reg_data_addr) {
    const RegisterInfo *sp = reg_info.GetRegisterInfoAtIndex(
        reg_info.ConvertRegisterKindToRegisterNumber(eRegisterKindGeneric,
                                                     LLDB_REGNUM_GENERIC_SP));
    const RegisterInfo *pc = reg_info.GetRegisterInfoAtIndex(
        reg_info.ConvertRegisterKindToRegisterNumber(eRegisterKindGeneric,
                                                     LLDB_REGNUM_GENERIC_PC));
    size_t byte_size = std::max(sp->byte_offset + sp->byte_size,
                                pc->byte_offset + pc->byte_size);

    DataBufferSP reg_data_sp(new DataBufferHeap(byte_size, 0));
    m_reg_data.SetData(reg_data_sp);
  }

  ~RegisterContextGo() override = default;

  bool ReadRegister(const lldb_private::RegisterInfo *reg_info,
                    lldb_private::RegisterValue &reg_value) override {
    switch (reg_info->kinds[eRegisterKindGeneric]) {
    case LLDB_REGNUM_GENERIC_SP:
    case LLDB_REGNUM_GENERIC_PC:
      return RegisterContextMemory::ReadRegister(reg_info, reg_value);
    default:
      reg_value.SetValueToInvalid();
      return true;
    }
  }

  bool WriteRegister(const lldb_private::RegisterInfo *reg_info,
                     const lldb_private::RegisterValue &reg_value) override {
    switch (reg_info->kinds[eRegisterKindGeneric]) {
    case LLDB_REGNUM_GENERIC_SP:
    case LLDB_REGNUM_GENERIC_PC:
      return RegisterContextMemory::WriteRegister(reg_info, reg_value);
    default:
      return false;
    }
  }

private:
  DISALLOW_COPY_AND_ASSIGN(RegisterContextGo);
};

} // anonymous namespace

struct OperatingSystemGo::Goroutine {
  uint64_t m_lostack;
  uint64_t m_histack;
  uint64_t m_goid;
  addr_t m_gobuf;
  uint32_t m_status;
};

void OperatingSystemGo::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                DebuggerInitialize);
}

void OperatingSystemGo::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForOperatingSystemPlugin(
          debugger, PluginProperties::GetSettingName())) {
    const bool is_global_setting = true;
    PluginManager::CreateSettingForOperatingSystemPlugin(
        debugger, GetGlobalPluginProperties()->GetValueProperties(),
        ConstString("Properties for the goroutine thread plug-in."),
        is_global_setting);
  }
}

void OperatingSystemGo::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

OperatingSystem *OperatingSystemGo::CreateInstance(Process *process,
                                                   bool force) {
  if (!force) {
    TargetSP target_sp = process->CalculateTarget();
    if (!target_sp)
      return nullptr;
    ModuleList &module_list = target_sp->GetImages();
    std::lock_guard<std::recursive_mutex> guard(module_list.GetMutex());
    const size_t num_modules = module_list.GetSize();
    bool found_go_runtime = false;
    for (size_t i = 0; i < num_modules; ++i) {
      Module *module = module_list.GetModulePointerAtIndexUnlocked(i);
      const SectionList *section_list = module->GetSectionList();
      if (section_list) {
        SectionSP section_sp(
            section_list->FindSectionByType(eSectionTypeGoSymtab, true));
        if (section_sp) {
          found_go_runtime = true;
          break;
        }
      }
    }
    if (!found_go_runtime)
      return nullptr;
  }
  return new OperatingSystemGo(process);
}

OperatingSystemGo::OperatingSystemGo(lldb_private::Process *process)
    : OperatingSystem(process), m_reginfo(new DynamicRegisterInfo) {}

OperatingSystemGo::~OperatingSystemGo() = default;

ConstString OperatingSystemGo::GetPluginNameStatic() {
  static ConstString g_name("goroutines");
  return g_name;
}

const char *OperatingSystemGo::GetPluginDescriptionStatic() {
  return "Operating system plug-in that reads runtime data-structures for "
         "goroutines.";
}

bool OperatingSystemGo::Init(ThreadList &threads) {
  if (threads.GetSize(false) < 1)
    return false;
  TargetSP target_sp = m_process->CalculateTarget();
  if (!target_sp)
    return false;
  // Go 1.6 stores goroutines in a slice called runtime.allgs
  ValueObjectSP allgs_sp = FindGlobal(target_sp, "runtime.allgs");
  if (allgs_sp) {
    m_allg_sp = allgs_sp->GetChildMemberWithName(ConstString("array"), true);
    m_allglen_sp = allgs_sp->GetChildMemberWithName(ConstString("len"), true);
  } else {
    // Go 1.4 stores goroutines in the variable runtime.allg.
    m_allg_sp = FindGlobal(target_sp, "runtime.allg");
    m_allglen_sp = FindGlobal(target_sp, "runtime.allglen");
  }

  if (m_allg_sp && !m_allglen_sp) {
    StreamSP error_sp = target_sp->GetDebugger().GetAsyncErrorStream();
    error_sp->Printf("Unsupported Go runtime version detected.");
    return false;
  }

  if (!m_allg_sp)
    return false;

  RegisterContextSP real_registers_sp =
      threads.GetThreadAtIndex(0, false)->GetRegisterContext();

  std::unordered_map<size_t, ConstString> register_sets;
  for (size_t set_idx = 0; set_idx < real_registers_sp->GetRegisterSetCount();
       ++set_idx) {
    const RegisterSet *set = real_registers_sp->GetRegisterSet(set_idx);
    ConstString name(set->name);
    for (size_t reg_idx = 0; reg_idx < set->num_registers; ++reg_idx) {
      register_sets[reg_idx] = name;
    }
  }
  TypeSP gobuf_sp = FindType(target_sp, "runtime.gobuf");
  if (!gobuf_sp) {
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_OS));

    if (log)
      log->Printf("OperatingSystemGo unable to find struct Gobuf");
    return false;
  }
  CompilerType gobuf_type(gobuf_sp->GetLayoutCompilerType());
  for (size_t idx = 0; idx < real_registers_sp->GetRegisterCount(); ++idx) {
    RegisterInfo reg = *real_registers_sp->GetRegisterInfoAtIndex(idx);
    int field_index = -1;
    if (reg.kinds[eRegisterKindGeneric] == LLDB_REGNUM_GENERIC_SP) {
      field_index = 0;
    } else if (reg.kinds[eRegisterKindGeneric] == LLDB_REGNUM_GENERIC_PC) {
      field_index = 1;
    }
    if (field_index == -1) {
      reg.byte_offset = ~0;
    } else {
      std::string field_name;
      uint64_t bit_offset = 0;
      CompilerType field_type = gobuf_type.GetFieldAtIndex(
          field_index, field_name, &bit_offset, nullptr, nullptr);
      reg.byte_size = field_type.GetByteSize(nullptr);
      reg.byte_offset = bit_offset / 8;
    }
    ConstString name(reg.name);
    ConstString alt_name(reg.alt_name);
    m_reginfo->AddRegister(reg, name, alt_name, register_sets[idx]);
  }
  return true;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
ConstString OperatingSystemGo::GetPluginName() { return GetPluginNameStatic(); }

uint32_t OperatingSystemGo::GetPluginVersion() { return 1; }

bool OperatingSystemGo::UpdateThreadList(ThreadList &old_thread_list,
                                         ThreadList &real_thread_list,
                                         ThreadList &new_thread_list) {
  new_thread_list = real_thread_list;
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_OS));

  if (!(m_allg_sp || Init(real_thread_list)) || (m_allg_sp && !m_allglen_sp) ||
      !GetGlobalPluginProperties()->GetEnableGoroutines()) {
    return new_thread_list.GetSize(false) > 0;
  }

  if (log)
    log->Printf("OperatingSystemGo::UpdateThreadList(%d, %d, %d) fetching "
                "thread data from Go for pid %" PRIu64,
                old_thread_list.GetSize(false), real_thread_list.GetSize(false),
                new_thread_list.GetSize(0), m_process->GetID());
  uint64_t allglen = m_allglen_sp->GetValueAsUnsigned(0);
  if (allglen == 0) {
    return new_thread_list.GetSize(false) > 0;
  }
  std::vector<Goroutine> goroutines;
  // The threads that are in "new_thread_list" upon entry are the threads from
  // the
  // lldb_private::Process subclass, no memory threads will be in this list.

  Error err;
  for (uint64_t i = 0; i < allglen; ++i) {
    goroutines.push_back(CreateGoroutineAtIndex(i, err));
    if (err.Fail()) {
      err.PutToLog(log, "OperatingSystemGo::UpdateThreadList");
      return new_thread_list.GetSize(false) > 0;
    }
  }
  // Make a map so we can match goroutines with backing threads.
  std::map<uint64_t, ThreadSP> stack_map;
  for (uint32_t i = 0; i < real_thread_list.GetSize(false); ++i) {
    ThreadSP thread = real_thread_list.GetThreadAtIndex(i, false);
    stack_map[thread->GetRegisterContext()->GetSP()] = thread;
  }
  for (const Goroutine &goroutine : goroutines) {
    if (0 /* Gidle */ == goroutine.m_status ||
        6 /* Gdead */ == goroutine.m_status) {
      continue;
    }
    ThreadSP memory_thread =
        old_thread_list.FindThreadByID(goroutine.m_goid, false);
    if (memory_thread && IsOperatingSystemPluginThread(memory_thread) &&
        memory_thread->IsValid()) {
      memory_thread->ClearBackingThread();
    } else {
      memory_thread.reset(new ThreadMemory(
          *m_process, goroutine.m_goid, nullptr, nullptr, goroutine.m_gobuf));
    }
    // Search for the backing thread if the goroutine is running.
    if (2 == (goroutine.m_status & 0xfff)) {
      auto backing_it = stack_map.lower_bound(goroutine.m_lostack);
      if (backing_it != stack_map.end()) {
        if (goroutine.m_histack >= backing_it->first) {
          if (log)
            log->Printf(
                "OperatingSystemGo::UpdateThreadList found backing thread "
                "%" PRIx64 " (%" PRIx64 ") for thread %" PRIx64 "",
                backing_it->second->GetID(),
                backing_it->second->GetProtocolID(), memory_thread->GetID());
          memory_thread->SetBackingThread(backing_it->second);
          new_thread_list.RemoveThreadByID(backing_it->second->GetID(), false);
        }
      }
    }
    new_thread_list.AddThread(memory_thread);
  }

  return new_thread_list.GetSize(false) > 0;
}

void OperatingSystemGo::ThreadWasSelected(Thread *thread) {}

RegisterContextSP
OperatingSystemGo::CreateRegisterContextForThread(Thread *thread,
                                                  addr_t reg_data_addr) {
  RegisterContextSP reg_ctx_sp;
  if (!thread)
    return reg_ctx_sp;

  if (!IsOperatingSystemPluginThread(thread->shared_from_this()))
    return reg_ctx_sp;

  reg_ctx_sp.reset(
      new RegisterContextGo(*thread, 0, *m_reginfo, reg_data_addr));
  return reg_ctx_sp;
}

StopInfoSP
OperatingSystemGo::CreateThreadStopReason(lldb_private::Thread *thread) {
  StopInfoSP stop_info_sp;
  return stop_info_sp;
}

lldb::ThreadSP OperatingSystemGo::CreateThread(lldb::tid_t tid,
                                               addr_t context) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_OS));

  if (log)
    log->Printf("OperatingSystemGo::CreateThread (tid = 0x%" PRIx64
                ", context = 0x%" PRIx64 ") not implemented",
                tid, context);

  return ThreadSP();
}

ValueObjectSP OperatingSystemGo::FindGlobal(TargetSP target, const char *name) {
  VariableList variable_list;
  const bool append = true;

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_OS));

  if (log) {
    log->Printf(
        "exe: %s",
        target->GetExecutableModule()->GetSpecificationDescription().c_str());
    log->Printf("modules: %zu", target->GetImages().GetSize());
  }

  uint32_t match_count = target->GetImages().FindGlobalVariables(
      ConstString(name), append, 1, variable_list);
  if (match_count > 0) {
    ExecutionContextScope *exe_scope = target->GetProcessSP().get();
    if (exe_scope == NULL)
      exe_scope = target.get();
    return ValueObjectVariable::Create(exe_scope,
                                       variable_list.GetVariableAtIndex(0));
  }
  return ValueObjectSP();
}

TypeSP OperatingSystemGo::FindType(TargetSP target_sp, const char *name) {
  ConstString const_typename(name);
  SymbolContext sc;
  const bool exact_match = false;

  const ModuleList &module_list = target_sp->GetImages();
  size_t count = module_list.GetSize();
  for (size_t idx = 0; idx < count; idx++) {
    ModuleSP module_sp(module_list.GetModuleAtIndex(idx));
    if (module_sp) {
      TypeSP type_sp(module_sp->FindFirstType(sc, const_typename, exact_match));
      if (type_sp)
        return type_sp;
    }
  }
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_OS));

  if (log)
    log->Printf("OperatingSystemGo::FindType(%s): not found", name);
  return TypeSP();
}

OperatingSystemGo::Goroutine
OperatingSystemGo::CreateGoroutineAtIndex(uint64_t idx, Error &err) {
  err.Clear();
  Goroutine result = {};
  ValueObjectSP g =
      m_allg_sp->GetSyntheticArrayMember(idx, true)->Dereference(err);
  if (err.Fail()) {
    return result;
  }

  ConstString name("goid");
  ValueObjectSP val = g->GetChildMemberWithName(name, true);
  bool success = false;
  result.m_goid = val->GetValueAsUnsigned(0, &success);
  if (!success) {
    err.SetErrorToGenericError();
    err.SetErrorString("unable to read goid");
    return result;
  }
  name.SetCString("atomicstatus");
  val = g->GetChildMemberWithName(name, true);
  result.m_status = (uint32_t)val->GetValueAsUnsigned(0, &success);
  if (!success) {
    err.SetErrorToGenericError();
    err.SetErrorString("unable to read atomicstatus");
    return result;
  }
  name.SetCString("sched");
  val = g->GetChildMemberWithName(name, true);
  result.m_gobuf = val->GetAddressOf(false);
  name.SetCString("stack");
  val = g->GetChildMemberWithName(name, true);
  name.SetCString("lo");
  ValueObjectSP child = val->GetChildMemberWithName(name, true);
  result.m_lostack = child->GetValueAsUnsigned(0, &success);
  if (!success) {
    err.SetErrorToGenericError();
    err.SetErrorString("unable to read stack.lo");
    return result;
  }
  name.SetCString("hi");
  child = val->GetChildMemberWithName(name, true);
  result.m_histack = child->GetValueAsUnsigned(0, &success);
  if (!success) {
    err.SetErrorToGenericError();
    err.SetErrorString("unable to read stack.hi");
    return result;
  }
  return result;
}
