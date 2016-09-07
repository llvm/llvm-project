//===-- DynamicLoaderPOSIX.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Main header include
#include "DynamicLoaderPOSIXDYLD.h"

// Project includes
#include "AuxVector.h"

// Other libraries and framework includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"

// C++ Includes
// C Includes

using namespace lldb;
using namespace lldb_private;

void DynamicLoaderPOSIXDYLD::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderPOSIXDYLD::Terminate() {}

lldb_private::ConstString DynamicLoaderPOSIXDYLD::GetPluginName() {
  return GetPluginNameStatic();
}

lldb_private::ConstString DynamicLoaderPOSIXDYLD::GetPluginNameStatic() {
  static ConstString g_name("linux-dyld");
  return g_name;
}

const char *DynamicLoaderPOSIXDYLD::GetPluginDescriptionStatic() {
  return "Dynamic loader plug-in that watches for shared library "
         "loads/unloads in POSIX processes.";
}

uint32_t DynamicLoaderPOSIXDYLD::GetPluginVersion() { return 1; }

DynamicLoader *DynamicLoaderPOSIXDYLD::CreateInstance(Process *process,
                                                      bool force) {
  bool create = force;
  if (!create) {
    const llvm::Triple &triple_ref =
        process->GetTarget().GetArchitecture().GetTriple();
    if (triple_ref.getOS() == llvm::Triple::Linux ||
        triple_ref.getOS() == llvm::Triple::FreeBSD)
      create = true;
  }

  if (create)
    return new DynamicLoaderPOSIXDYLD(process);
  return NULL;
}

DynamicLoaderPOSIXDYLD::DynamicLoaderPOSIXDYLD(Process *process)
    : DynamicLoader(process), m_rendezvous(process),
      m_load_offset(LLDB_INVALID_ADDRESS), m_entry_point(LLDB_INVALID_ADDRESS),
      m_auxv(), m_dyld_bid(LLDB_INVALID_BREAK_ID),
      m_vdso_base(LLDB_INVALID_ADDRESS) {}

DynamicLoaderPOSIXDYLD::~DynamicLoaderPOSIXDYLD() {
  if (m_dyld_bid != LLDB_INVALID_BREAK_ID) {
    m_process->GetTarget().RemoveBreakpointByID(m_dyld_bid);
    m_dyld_bid = LLDB_INVALID_BREAK_ID;
  }
}

void DynamicLoaderPOSIXDYLD::DidAttach() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  if (log)
    log->Printf("DynamicLoaderPOSIXDYLD::%s() pid %" PRIu64, __FUNCTION__,
                m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID);

  m_auxv.reset(new AuxVector(m_process));
  if (log)
    log->Printf("DynamicLoaderPOSIXDYLD::%s pid %" PRIu64 " reloaded auxv data",
                __FUNCTION__,
                m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID);

  // ask the process if it can load any of its own modules
  m_process->LoadModules();

  ModuleSP executable_sp = GetTargetExecutable();
  ResolveExecutableModule(executable_sp);

  // find the main process load offset
  addr_t load_offset = ComputeLoadOffset();
  if (log)
    log->Printf("DynamicLoaderPOSIXDYLD::%s pid %" PRIu64
                " executable '%s', load_offset 0x%" PRIx64,
                __FUNCTION__,
                m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID,
                executable_sp ? executable_sp->GetFileSpec().GetPath().c_str()
                              : "<null executable>",
                load_offset);

  EvalVdsoStatus();

  // if we dont have a load address we cant re-base
  bool rebase_exec = (load_offset == LLDB_INVALID_ADDRESS) ? false : true;

  // if we have a valid executable
  if (executable_sp.get()) {
    lldb_private::ObjectFile *obj = executable_sp->GetObjectFile();
    if (obj) {
      // don't rebase if the module already has a load address
      Target &target = m_process->GetTarget();
      Address addr = obj->GetImageInfoAddress(&target);
      if (addr.GetLoadAddress(&target) != LLDB_INVALID_ADDRESS)
        rebase_exec = false;
    }
  } else {
    // no executable, nothing to re-base
    rebase_exec = false;
  }

  // if the target executable should be re-based
  if (rebase_exec) {
    ModuleList module_list;

    module_list.Append(executable_sp);
    if (log)
      log->Printf("DynamicLoaderPOSIXDYLD::%s pid %" PRIu64
                  " added executable '%s' to module load list",
                  __FUNCTION__,
                  m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID,
                  executable_sp->GetFileSpec().GetPath().c_str());

    UpdateLoadedSections(executable_sp, LLDB_INVALID_ADDRESS, load_offset,
                         true);

    // When attaching to a target, there are two possible states:
    // (1) We already crossed the entry point and therefore the rendezvous
    //     structure is ready to be used and we can load the list of modules
    //     and place the rendezvous breakpoint.
    // (2) We didn't cross the entry point yet, so these structures are not
    //     ready; we should behave as if we just launched the target and
    //     call ProbeEntry(). This will place a breakpoint on the entry
    //     point which itself will be hit after the rendezvous structure is
    //     set up and will perform actions described in (1).
    if (m_rendezvous.Resolve()) {
      if (log)
        log->Printf("DynamicLoaderPOSIXDYLD::%s() pid %" PRIu64
                    " rendezvous could resolve: attach assuming dynamic loader "
                    "info is available now",
                    __FUNCTION__,
                    m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID);
      LoadAllCurrentModules();
      SetRendezvousBreakpoint();
    } else {
      if (log)
        log->Printf("DynamicLoaderPOSIXDYLD::%s() pid %" PRIu64
                    " rendezvous could not yet resolve: adding breakpoint to "
                    "catch future rendezvous setup",
                    __FUNCTION__,
                    m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID);
      ProbeEntry();
    }

    m_process->GetTarget().ModulesDidLoad(module_list);
    if (log) {
      log->Printf("DynamicLoaderPOSIXDYLD::%s told the target about the "
                  "modules that loaded:",
                  __FUNCTION__);
      for (auto module_sp : module_list.Modules()) {
        log->Printf("-- [module] %s (pid %" PRIu64 ")",
                    module_sp ? module_sp->GetFileSpec().GetPath().c_str()
                              : "<null>",
                    m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID);
      }
    }
  }
}

void DynamicLoaderPOSIXDYLD::DidLaunch() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  if (log)
    log->Printf("DynamicLoaderPOSIXDYLD::%s()", __FUNCTION__);

  ModuleSP executable;
  addr_t load_offset;

  m_auxv.reset(new AuxVector(m_process));

  executable = GetTargetExecutable();
  load_offset = ComputeLoadOffset();
  EvalVdsoStatus();

  if (executable.get() && load_offset != LLDB_INVALID_ADDRESS) {
    ModuleList module_list;
    module_list.Append(executable);
    UpdateLoadedSections(executable, LLDB_INVALID_ADDRESS, load_offset, true);

    if (log)
      log->Printf("DynamicLoaderPOSIXDYLD::%s about to call ProbeEntry()",
                  __FUNCTION__);
    ProbeEntry();

    m_process->GetTarget().ModulesDidLoad(module_list);
  }
}

Error DynamicLoaderPOSIXDYLD::CanLoadImage() { return Error(); }

void DynamicLoaderPOSIXDYLD::UpdateLoadedSections(ModuleSP module,
                                                  addr_t link_map_addr,
                                                  addr_t base_addr,
                                                  bool base_addr_is_offset) {
  m_loaded_modules[module] = link_map_addr;
  UpdateLoadedSectionsCommon(module, base_addr, base_addr_is_offset);
}

void DynamicLoaderPOSIXDYLD::UnloadSections(const ModuleSP module) {
  m_loaded_modules.erase(module);

  UnloadSectionsCommon(module);
}

void DynamicLoaderPOSIXDYLD::ProbeEntry() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));

  const addr_t entry = GetEntryPoint();
  if (entry == LLDB_INVALID_ADDRESS) {
    if (log)
      log->Printf(
          "DynamicLoaderPOSIXDYLD::%s pid %" PRIu64
          " GetEntryPoint() returned no address, not setting entry breakpoint",
          __FUNCTION__,
          m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID);
    return;
  }

  if (log)
    log->Printf("DynamicLoaderPOSIXDYLD::%s pid %" PRIu64
                " GetEntryPoint() returned address 0x%" PRIx64
                ", setting entry breakpoint",
                __FUNCTION__,
                m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID,
                entry);

  if (m_process) {
    Breakpoint *const entry_break =
        m_process->GetTarget().CreateBreakpoint(entry, true, false).get();
    entry_break->SetCallback(EntryBreakpointHit, this, true);
    entry_break->SetBreakpointKind("shared-library-event");

    // Shoudn't hit this more than once.
    entry_break->SetOneShot(true);
  }
}

// The runtime linker has run and initialized the rendezvous structure once the
// process has hit its entry point.  When we hit the corresponding breakpoint we
// interrogate the rendezvous structure to get the load addresses of all
// dependent modules for the process.  Similarly, we can discover the runtime
// linker function and setup a breakpoint to notify us of any dynamically loaded
// modules (via dlopen).
bool DynamicLoaderPOSIXDYLD::EntryBreakpointHit(
    void *baton, StoppointCallbackContext *context, user_id_t break_id,
    user_id_t break_loc_id) {
  assert(baton && "null baton");
  if (!baton)
    return false;

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  DynamicLoaderPOSIXDYLD *const dyld_instance =
      static_cast<DynamicLoaderPOSIXDYLD *>(baton);
  if (log)
    log->Printf("DynamicLoaderPOSIXDYLD::%s called for pid %" PRIu64,
                __FUNCTION__,
                dyld_instance->m_process ? dyld_instance->m_process->GetID()
                                         : LLDB_INVALID_PROCESS_ID);

  // Disable the breakpoint --- if a stop happens right after this, which we've
  // seen on occasion, we don't
  // want the breakpoint stepping thread-plan logic to show a breakpoint
  // instruction at the disassembled
  // entry point to the program.  Disabling it prevents it.  (One-shot is not
  // enough - one-shot removal logic
  // only happens after the breakpoint goes public, which wasn't happening in
  // our scenario).
  if (dyld_instance->m_process) {
    BreakpointSP breakpoint_sp =
        dyld_instance->m_process->GetTarget().GetBreakpointByID(break_id);
    if (breakpoint_sp) {
      if (log)
        log->Printf("DynamicLoaderPOSIXDYLD::%s pid %" PRIu64
                    " disabling breakpoint id %" PRIu64,
                    __FUNCTION__, dyld_instance->m_process->GetID(), break_id);
      breakpoint_sp->SetEnabled(false);
    } else {
      if (log)
        log->Printf("DynamicLoaderPOSIXDYLD::%s pid %" PRIu64
                    " failed to find breakpoint for breakpoint id %" PRIu64,
                    __FUNCTION__, dyld_instance->m_process->GetID(), break_id);
    }
  } else {
    if (log)
      log->Printf("DynamicLoaderPOSIXDYLD::%s breakpoint id %" PRIu64
                  " no Process instance!  Cannot disable breakpoint",
                  __FUNCTION__, break_id);
  }

  dyld_instance->LoadAllCurrentModules();
  dyld_instance->SetRendezvousBreakpoint();
  return false; // Continue running.
}

void DynamicLoaderPOSIXDYLD::SetRendezvousBreakpoint() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));

  addr_t break_addr = m_rendezvous.GetBreakAddress();
  Target &target = m_process->GetTarget();

  if (m_dyld_bid == LLDB_INVALID_BREAK_ID) {
    if (log)
      log->Printf("DynamicLoaderPOSIXDYLD::%s pid %" PRIu64
                  " setting rendezvous break address at 0x%" PRIx64,
                  __FUNCTION__,
                  m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID,
                  break_addr);
    Breakpoint *dyld_break =
        target.CreateBreakpoint(break_addr, true, false).get();
    dyld_break->SetCallback(RendezvousBreakpointHit, this, true);
    dyld_break->SetBreakpointKind("shared-library-event");
    m_dyld_bid = dyld_break->GetID();
  } else {
    if (log)
      log->Printf("DynamicLoaderPOSIXDYLD::%s pid %" PRIu64
                  " reusing break id %" PRIu32 ", address at 0x%" PRIx64,
                  __FUNCTION__,
                  m_process ? m_process->GetID() : LLDB_INVALID_PROCESS_ID,
                  m_dyld_bid, break_addr);
  }

  // Make sure our breakpoint is at the right address.
  assert(target.GetBreakpointByID(m_dyld_bid)
             ->FindLocationByAddress(break_addr)
             ->GetBreakpoint()
             .GetID() == m_dyld_bid);
}

bool DynamicLoaderPOSIXDYLD::RendezvousBreakpointHit(
    void *baton, StoppointCallbackContext *context, user_id_t break_id,
    user_id_t break_loc_id) {
  assert(baton && "null baton");
  if (!baton)
    return false;

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  DynamicLoaderPOSIXDYLD *const dyld_instance =
      static_cast<DynamicLoaderPOSIXDYLD *>(baton);
  if (log)
    log->Printf("DynamicLoaderPOSIXDYLD::%s called for pid %" PRIu64,
                __FUNCTION__,
                dyld_instance->m_process ? dyld_instance->m_process->GetID()
                                         : LLDB_INVALID_PROCESS_ID);

  dyld_instance->RefreshModules();

  // Return true to stop the target, false to just let the target run.
  const bool stop_when_images_change = dyld_instance->GetStopWhenImagesChange();
  if (log)
    log->Printf("DynamicLoaderPOSIXDYLD::%s pid %" PRIu64
                " stop_when_images_change=%s",
                __FUNCTION__,
                dyld_instance->m_process ? dyld_instance->m_process->GetID()
                                         : LLDB_INVALID_PROCESS_ID,
                stop_when_images_change ? "true" : "false");
  return stop_when_images_change;
}

void DynamicLoaderPOSIXDYLD::RefreshModules() {
  if (!m_rendezvous.Resolve())
    return;

  DYLDRendezvous::iterator I;
  DYLDRendezvous::iterator E;

  ModuleList &loaded_modules = m_process->GetTarget().GetImages();

  if (m_rendezvous.ModulesDidLoad()) {
    ModuleList new_modules;

    E = m_rendezvous.loaded_end();
    for (I = m_rendezvous.loaded_begin(); I != E; ++I) {
      ModuleSP module_sp =
          LoadModuleAtAddress(I->file_spec, I->link_addr, I->base_addr, true);
      if (module_sp.get()) {
        loaded_modules.AppendIfNeeded(module_sp);
        new_modules.Append(module_sp);
      }
    }
    m_process->GetTarget().ModulesDidLoad(new_modules);
  }

  if (m_rendezvous.ModulesDidUnload()) {
    ModuleList old_modules;

    E = m_rendezvous.unloaded_end();
    for (I = m_rendezvous.unloaded_begin(); I != E; ++I) {
      ModuleSpec module_spec{I->file_spec};
      ModuleSP module_sp = loaded_modules.FindFirstModule(module_spec);

      if (module_sp.get()) {
        old_modules.Append(module_sp);
        UnloadSections(module_sp);
      }
    }
    loaded_modules.Remove(old_modules);
    m_process->GetTarget().ModulesDidUnload(old_modules, false);
  }
}

ThreadPlanSP
DynamicLoaderPOSIXDYLD::GetStepThroughTrampolinePlan(Thread &thread,
                                                     bool stop) {
  ThreadPlanSP thread_plan_sp;

  StackFrame *frame = thread.GetStackFrameAtIndex(0).get();
  const SymbolContext &context = frame->GetSymbolContext(eSymbolContextSymbol);
  Symbol *sym = context.symbol;

  if (sym == NULL || !sym->IsTrampoline())
    return thread_plan_sp;

  ConstString sym_name = sym->GetName();
  if (!sym_name)
    return thread_plan_sp;

  SymbolContextList target_symbols;
  Target &target = thread.GetProcess()->GetTarget();
  const ModuleList &images = target.GetImages();

  images.FindSymbolsWithNameAndType(sym_name, eSymbolTypeCode, target_symbols);
  size_t num_targets = target_symbols.GetSize();
  if (!num_targets)
    return thread_plan_sp;

  typedef std::vector<lldb::addr_t> AddressVector;
  AddressVector addrs;
  for (size_t i = 0; i < num_targets; ++i) {
    SymbolContext context;
    AddressRange range;
    if (target_symbols.GetContextAtIndex(i, context)) {
      context.GetAddressRange(eSymbolContextEverything, 0, false, range);
      lldb::addr_t addr = range.GetBaseAddress().GetLoadAddress(&target);
      if (addr != LLDB_INVALID_ADDRESS)
        addrs.push_back(addr);
    }
  }

  if (addrs.size() > 0) {
    AddressVector::iterator start = addrs.begin();
    AddressVector::iterator end = addrs.end();

    std::sort(start, end);
    addrs.erase(std::unique(start, end), end);
    thread_plan_sp.reset(new ThreadPlanRunToAddress(thread, addrs, stop));
  }

  return thread_plan_sp;
}

void DynamicLoaderPOSIXDYLD::LoadAllCurrentModules() {
  DYLDRendezvous::iterator I;
  DYLDRendezvous::iterator E;
  ModuleList module_list;

  if (!m_rendezvous.Resolve()) {
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
    if (log)
      log->Printf("DynamicLoaderPOSIXDYLD::%s unable to resolve POSIX DYLD "
                  "rendezvous address",
                  __FUNCTION__);
    return;
  }

  // The rendezvous class doesn't enumerate the main module, so track
  // that ourselves here.
  ModuleSP executable = GetTargetExecutable();
  m_loaded_modules[executable] = m_rendezvous.GetLinkMapAddress();
  if (m_vdso_base != LLDB_INVALID_ADDRESS) {
    FileSpec file_spec("[vdso]", false);
    ModuleSP module_sp = LoadModuleAtAddress(file_spec, LLDB_INVALID_ADDRESS,
                                             m_vdso_base, false);
    if (module_sp.get()) {
      module_list.Append(module_sp);
    }
  }
  for (I = m_rendezvous.begin(), E = m_rendezvous.end(); I != E; ++I) {
    ModuleSP module_sp =
        LoadModuleAtAddress(I->file_spec, I->link_addr, I->base_addr, true);
    if (module_sp.get()) {
      module_list.Append(module_sp);
    } else {
      Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
      if (log)
        log->Printf(
            "DynamicLoaderPOSIXDYLD::%s failed loading module %s at 0x%" PRIx64,
            __FUNCTION__, I->file_spec.GetCString(), I->base_addr);
    }
  }

  m_process->GetTarget().ModulesDidLoad(module_list);
}

addr_t DynamicLoaderPOSIXDYLD::ComputeLoadOffset() {
  addr_t virt_entry;

  if (m_load_offset != LLDB_INVALID_ADDRESS)
    return m_load_offset;

  if ((virt_entry = GetEntryPoint()) == LLDB_INVALID_ADDRESS)
    return LLDB_INVALID_ADDRESS;

  ModuleSP module = m_process->GetTarget().GetExecutableModule();
  if (!module)
    return LLDB_INVALID_ADDRESS;

  ObjectFile *exe = module->GetObjectFile();
  if (!exe)
    return LLDB_INVALID_ADDRESS;

  Address file_entry = exe->GetEntryPointAddress();

  if (!file_entry.IsValid())
    return LLDB_INVALID_ADDRESS;

  m_load_offset = virt_entry - file_entry.GetFileAddress();
  return m_load_offset;
}

void DynamicLoaderPOSIXDYLD::EvalVdsoStatus() {
  AuxVector::iterator I = m_auxv->FindEntry(AuxVector::AT_SYSINFO_EHDR);

  if (I != m_auxv->end())
    m_vdso_base = I->value;
}

addr_t DynamicLoaderPOSIXDYLD::GetEntryPoint() {
  if (m_entry_point != LLDB_INVALID_ADDRESS)
    return m_entry_point;

  if (m_auxv.get() == NULL)
    return LLDB_INVALID_ADDRESS;

  AuxVector::iterator I = m_auxv->FindEntry(AuxVector::AT_ENTRY);

  if (I == m_auxv->end())
    return LLDB_INVALID_ADDRESS;

  m_entry_point = static_cast<addr_t>(I->value);

  const ArchSpec &arch = m_process->GetTarget().GetArchitecture();

  // On ppc64, the entry point is actually a descriptor.  Dereference it.
  if (arch.GetMachine() == llvm::Triple::ppc64)
    m_entry_point = ReadUnsignedIntWithSizeInBytes(m_entry_point, 8);

  return m_entry_point;
}

lldb::addr_t
DynamicLoaderPOSIXDYLD::GetThreadLocalData(const lldb::ModuleSP module_sp,
                                           const lldb::ThreadSP thread,
                                           lldb::addr_t tls_file_addr) {
  auto it = m_loaded_modules.find(module_sp);
  if (it == m_loaded_modules.end())
    return LLDB_INVALID_ADDRESS;

  addr_t link_map = it->second;
  if (link_map == LLDB_INVALID_ADDRESS)
    return LLDB_INVALID_ADDRESS;

  const DYLDRendezvous::ThreadInfo &metadata = m_rendezvous.GetThreadInfo();
  if (!metadata.valid)
    return LLDB_INVALID_ADDRESS;

  // Get the thread pointer.
  addr_t tp = thread->GetThreadPointer();
  if (tp == LLDB_INVALID_ADDRESS)
    return LLDB_INVALID_ADDRESS;

  // Find the module's modid.
  int modid_size = 4; // FIXME(spucci): This isn't right for big-endian 64-bit
  int64_t modid = ReadUnsignedIntWithSizeInBytes(
      link_map + metadata.modid_offset, modid_size);
  if (modid == -1)
    return LLDB_INVALID_ADDRESS;

  // Lookup the DTV structure for this thread.
  addr_t dtv_ptr = tp + metadata.dtv_offset;
  addr_t dtv = ReadPointer(dtv_ptr);
  if (dtv == LLDB_INVALID_ADDRESS)
    return LLDB_INVALID_ADDRESS;

  // Find the TLS block for this module.
  addr_t dtv_slot = dtv + metadata.dtv_slot_size * modid;
  addr_t tls_block = ReadPointer(dtv_slot + metadata.tls_offset);

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
  if (log)
    log->Printf("DynamicLoaderPOSIXDYLD::Performed TLS lookup: "
                "module=%s, link_map=0x%" PRIx64 ", tp=0x%" PRIx64
                ", modid=%" PRId64 ", tls_block=0x%" PRIx64 "\n",
                module_sp->GetObjectName().AsCString(""), link_map, tp,
                (int64_t)modid, tls_block);

  if (tls_block == LLDB_INVALID_ADDRESS)
    return LLDB_INVALID_ADDRESS;
  else
    return tls_block + tls_file_addr;
}

void DynamicLoaderPOSIXDYLD::ResolveExecutableModule(
    lldb::ModuleSP &module_sp) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));

  if (m_process == nullptr)
    return;

  auto &target = m_process->GetTarget();
  const auto platform_sp = target.GetPlatform();

  ProcessInstanceInfo process_info;
  if (!m_process->GetProcessInfo(process_info)) {
    if (log)
      log->Printf("DynamicLoaderPOSIXDYLD::%s - failed to get process info for "
                  "pid %" PRIu64,
                  __FUNCTION__, m_process->GetID());
    return;
  }

  if (log)
    log->Printf("DynamicLoaderPOSIXDYLD::%s - got executable by pid %" PRIu64
                ": %s",
                __FUNCTION__, m_process->GetID(),
                process_info.GetExecutableFile().GetPath().c_str());

  ModuleSpec module_spec(process_info.GetExecutableFile(),
                         process_info.GetArchitecture());
  if (module_sp && module_sp->MatchesModuleSpec(module_spec))
    return;

  const auto executable_search_paths(Target::GetDefaultExecutableSearchPaths());
  auto error = platform_sp->ResolveExecutable(
      module_spec, module_sp,
      !executable_search_paths.IsEmpty() ? &executable_search_paths : nullptr);
  if (error.Fail()) {
    StreamString stream;
    module_spec.Dump(stream);

    if (log)
      log->Printf("DynamicLoaderPOSIXDYLD::%s - failed to resolve executable "
                  "with module spec \"%s\": %s",
                  __FUNCTION__, stream.GetString().c_str(), error.AsCString());
    return;
  }

  target.SetExecutableModule(module_sp, false);
}

bool DynamicLoaderPOSIXDYLD::AlwaysRelyOnEHUnwindInfo(
    lldb_private::SymbolContext &sym_ctx) {
  ModuleSP module_sp;
  if (sym_ctx.symbol)
    module_sp = sym_ctx.symbol->GetAddressRef().GetModule();
  if (!module_sp && sym_ctx.function)
    module_sp =
        sym_ctx.function->GetAddressRange().GetBaseAddress().GetModule();
  if (!module_sp)
    return false;

  return module_sp->GetFileSpec().GetPath() == "[vdso]";
}
