//===-- DynamicLoaderPOSIX.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"

#include "AuxVector.h"
#include "DynamicLoaderPOSIXDYLD.h"

using namespace lldb;
using namespace lldb_private;

void
DynamicLoaderPOSIXDYLD::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  CreateInstance);
}

void
DynamicLoaderPOSIXDYLD::Terminate()
{
}

const char *
DynamicLoaderPOSIXDYLD::GetPluginName()
{
    return "DynamicLoaderPOSIXDYLD";
}

const char *
DynamicLoaderPOSIXDYLD::GetShortPluginName()
{
    return "linux-dyld";
}

const char *
DynamicLoaderPOSIXDYLD::GetPluginNameStatic()
{
    return "dynamic-loader.linux-dyld";
}

const char *
DynamicLoaderPOSIXDYLD::GetPluginDescriptionStatic()
{
    return "Dynamic loader plug-in that watches for shared library "
           "loads/unloads in POSIX processes.";
}

void
DynamicLoaderPOSIXDYLD::GetPluginCommandHelp(const char *command, Stream *strm)
{
}

uint32_t
DynamicLoaderPOSIXDYLD::GetPluginVersion()
{
    return 1;
}

DynamicLoader *
DynamicLoaderPOSIXDYLD::CreateInstance(Process *process, bool force)
{
    bool create = force;
    if (!create)
    {
        const llvm::Triple &triple_ref = process->GetTarget().GetArchitecture().GetTriple();
        if (triple_ref.getOS() == llvm::Triple::Linux ||
            triple_ref.getOS() == llvm::Triple::FreeBSD)
            create = true;
    }
    
    if (create)
        return new DynamicLoaderPOSIXDYLD (process);
    return NULL;
}

DynamicLoaderPOSIXDYLD::DynamicLoaderPOSIXDYLD(Process *process)
    : DynamicLoader(process),
      m_rendezvous(process),
      m_load_offset(LLDB_INVALID_ADDRESS),
      m_entry_point(LLDB_INVALID_ADDRESS),
      m_auxv(NULL)
{
}

DynamicLoaderPOSIXDYLD::~DynamicLoaderPOSIXDYLD()
{
}

void
DynamicLoaderPOSIXDYLD::DidAttach()
{
    ModuleSP executable;
    addr_t load_offset;

    m_auxv.reset(new AuxVector(m_process));

    executable = m_process->GetTarget().GetExecutableModule();
    load_offset = ComputeLoadOffset();

    if (executable.get() && load_offset != LLDB_INVALID_ADDRESS)
    {
        ModuleList module_list;
        module_list.Append(executable);
        UpdateLoadedSections(executable, load_offset);
        LoadAllCurrentModules();
        m_process->GetTarget().ModulesDidLoad(module_list);
    }
}

void
DynamicLoaderPOSIXDYLD::DidLaunch()
{
    ModuleSP executable;
    addr_t load_offset;

    m_auxv.reset(new AuxVector(m_process));

    executable = m_process->GetTarget().GetExecutableModule();
    load_offset = ComputeLoadOffset();

    if (executable.get() && load_offset != LLDB_INVALID_ADDRESS)
    {
        ModuleList module_list;
        module_list.Append(executable);
        UpdateLoadedSections(executable, load_offset);
        ProbeEntry();
        m_process->GetTarget().ModulesDidLoad(module_list);
    }
}

Error
DynamicLoaderPOSIXDYLD::ExecutePluginCommand(Args &command, Stream *strm)
{
    return Error();
}

Log *
DynamicLoaderPOSIXDYLD::EnablePluginLogging(Stream *strm, Args &command)
{
    return NULL;
}

Error
DynamicLoaderPOSIXDYLD::CanLoadImage()
{
    return Error();
}

void
DynamicLoaderPOSIXDYLD::UpdateLoadedSections(ModuleSP module, addr_t base_addr)
{
    ObjectFile *obj_file = module->GetObjectFile();
    SectionList *sections = obj_file->GetSectionList();
    SectionLoadList &load_list = m_process->GetTarget().GetSectionLoadList();
    const size_t num_sections = sections->GetSize();

    for (unsigned i = 0; i < num_sections; ++i)
    {
        SectionSP section_sp (sections->GetSectionAtIndex(i));
        lldb::addr_t new_load_addr = section_sp->GetFileAddress() + base_addr;
        lldb::addr_t old_load_addr = load_list.GetSectionLoadAddress(section_sp);

        // If the file address of the section is zero then this is not an
        // allocatable/loadable section (property of ELF sh_addr).  Skip it.
        if (new_load_addr == base_addr)
            continue;

        if (old_load_addr == LLDB_INVALID_ADDRESS ||
            old_load_addr != new_load_addr)
            load_list.SetSectionLoadAddress(section_sp, new_load_addr);
    }
}

void
DynamicLoaderPOSIXDYLD::ProbeEntry()
{
    Breakpoint *entry_break;
    addr_t entry;

    if ((entry = GetEntryPoint()) == LLDB_INVALID_ADDRESS)
        return;
    
    entry_break = m_process->GetTarget().CreateBreakpoint(entry, true).get();
    entry_break->SetCallback(EntryBreakpointHit, this, true);
}

// The runtime linker has run and initialized the rendezvous structure once the
// process has hit its entry point.  When we hit the corresponding breakpoint we
// interrogate the rendezvous structure to get the load addresses of all
// dependent modules for the process.  Similarly, we can discover the runtime
// linker function and setup a breakpoint to notify us of any dynamically loaded
// modules (via dlopen).
bool
DynamicLoaderPOSIXDYLD::EntryBreakpointHit(void *baton, 
                                           StoppointCallbackContext *context, 
                                           user_id_t break_id, 
                                           user_id_t break_loc_id)
{
    DynamicLoaderPOSIXDYLD* dyld_instance;

    dyld_instance = static_cast<DynamicLoaderPOSIXDYLD*>(baton);
    dyld_instance->LoadAllCurrentModules();
    dyld_instance->SetRendezvousBreakpoint();
    return false; // Continue running.
}

void
DynamicLoaderPOSIXDYLD::SetRendezvousBreakpoint()
{
    Breakpoint *dyld_break;
    addr_t break_addr;

    break_addr = m_rendezvous.GetBreakAddress();
    dyld_break = m_process->GetTarget().CreateBreakpoint(break_addr, true).get();
    dyld_break->SetCallback(RendezvousBreakpointHit, this, true);
}

bool
DynamicLoaderPOSIXDYLD::RendezvousBreakpointHit(void *baton, 
                                                StoppointCallbackContext *context, 
                                                user_id_t break_id, 
                                                user_id_t break_loc_id)
{
    DynamicLoaderPOSIXDYLD* dyld_instance;

    dyld_instance = static_cast<DynamicLoaderPOSIXDYLD*>(baton);
    dyld_instance->RefreshModules();

    // Return true to stop the target, false to just let the target run.
    return dyld_instance->GetStopWhenImagesChange();
}

void
DynamicLoaderPOSIXDYLD::RefreshModules()
{
    if (!m_rendezvous.Resolve())
        return;

    DYLDRendezvous::iterator I;
    DYLDRendezvous::iterator E;

    ModuleList &loaded_modules = m_process->GetTarget().GetImages();

    if (m_rendezvous.ModulesDidLoad()) 
    {
        ModuleList new_modules;

        E = m_rendezvous.loaded_end();
        for (I = m_rendezvous.loaded_begin(); I != E; ++I)
        {
            FileSpec file(I->path.c_str(), true);
            ModuleSP module_sp = LoadModuleAtAddress(file, I->base_addr);
            if (module_sp.get())
                loaded_modules.AppendIfNeeded(module_sp);
        }
    }
    
    if (m_rendezvous.ModulesDidUnload())
    {
        ModuleList old_modules;

        E = m_rendezvous.unloaded_end();
        for (I = m_rendezvous.unloaded_begin(); I != E; ++I)
        {
            FileSpec file(I->path.c_str(), true);
            ModuleSpec module_spec (file);
            ModuleSP module_sp = 
                loaded_modules.FindFirstModule (module_spec);
            if (module_sp.get())
                old_modules.Append(module_sp);
        }
        loaded_modules.Remove(old_modules);
    }
}

ThreadPlanSP
DynamicLoaderPOSIXDYLD::GetStepThroughTrampolinePlan(Thread &thread, bool stop)
{
    ThreadPlanSP thread_plan_sp;

    StackFrame *frame = thread.GetStackFrameAtIndex(0).get();
    const SymbolContext &context = frame->GetSymbolContext(eSymbolContextSymbol);
    Symbol *sym = context.symbol;

    if (sym == NULL || !sym->IsTrampoline())
        return thread_plan_sp;

    const ConstString &sym_name = sym->GetMangled().GetName(Mangled::ePreferMangled);
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
    for (size_t i = 0; i < num_targets; ++i)
    {
        SymbolContext context;
        AddressRange range;
        if (target_symbols.GetContextAtIndex(i, context))
        {
            context.GetAddressRange(eSymbolContextEverything, 0, false, range);
            lldb::addr_t addr = range.GetBaseAddress().GetLoadAddress(&target);
            if (addr != LLDB_INVALID_ADDRESS)
                addrs.push_back(addr);
        }
    }

    if (addrs.size() > 0) 
    {
        AddressVector::iterator start = addrs.begin();
        AddressVector::iterator end = addrs.end();

        std::sort(start, end);
        addrs.erase(std::unique(start, end), end);
        thread_plan_sp.reset(new ThreadPlanRunToAddress(thread, addrs, stop));
    }

    return thread_plan_sp;
}

void
DynamicLoaderPOSIXDYLD::LoadAllCurrentModules()
{
    DYLDRendezvous::iterator I;
    DYLDRendezvous::iterator E;
    ModuleList module_list;
    
    if (!m_rendezvous.Resolve())
        return;

    for (I = m_rendezvous.begin(), E = m_rendezvous.end(); I != E; ++I)
    {
        FileSpec file(I->path.c_str(), false);
        ModuleSP module_sp = LoadModuleAtAddress(file, I->base_addr);
        if (module_sp.get())
            module_list.Append(module_sp);
    }

    m_process->GetTarget().ModulesDidLoad(module_list);
}

ModuleSP
DynamicLoaderPOSIXDYLD::LoadModuleAtAddress(const FileSpec &file, addr_t base_addr)
{
    Target &target = m_process->GetTarget();
    ModuleList &modules = target.GetImages();
    ModuleSP module_sp;

    ModuleSpec module_spec (file, target.GetArchitecture());
    if ((module_sp = modules.FindFirstModule (module_spec))) 
    {
        UpdateLoadedSections(module_sp, base_addr);
    }
    else if ((module_sp = target.GetSharedModule(module_spec))) 
    {
        UpdateLoadedSections(module_sp, base_addr);
        modules.Append(module_sp);
    }

    return module_sp;
}

addr_t
DynamicLoaderPOSIXDYLD::ComputeLoadOffset()
{
    addr_t virt_entry;

    if (m_load_offset != LLDB_INVALID_ADDRESS)
        return m_load_offset;

    if ((virt_entry = GetEntryPoint()) == LLDB_INVALID_ADDRESS)
        return LLDB_INVALID_ADDRESS;

    ModuleSP module = m_process->GetTarget().GetExecutableModule();
    ObjectFile *exe = module->GetObjectFile();
    Address file_entry = exe->GetEntryPointAddress();

    if (!file_entry.IsValid())
        return LLDB_INVALID_ADDRESS;
            
    m_load_offset = virt_entry - file_entry.GetFileAddress();
    return m_load_offset;
}

addr_t
DynamicLoaderPOSIXDYLD::GetEntryPoint()
{
    if (m_entry_point != LLDB_INVALID_ADDRESS)
        return m_entry_point;

    if (m_auxv.get() == NULL)
        return LLDB_INVALID_ADDRESS;

    AuxVector::iterator I = m_auxv->FindEntry(AuxVector::AT_ENTRY);

    if (I == m_auxv->end())
        return LLDB_INVALID_ADDRESS;

    m_entry_point = static_cast<addr_t>(I->value);
    return m_entry_point;
}
