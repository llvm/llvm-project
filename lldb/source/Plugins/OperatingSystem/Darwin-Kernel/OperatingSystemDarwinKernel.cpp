//===-- OperatingSystemDarwinKernel.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OperatingSystemDarwinKernel.h"
// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/ADT/Triple.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Target/Thread.h"
#include "Plugins/Process/Utility/DynamicRegisterInfo.h"
#include "Plugins/Process/Utility/RegisterContextMemory.h"
#include "Plugins/Process/Utility/ThreadMemory.h"

using namespace lldb;
using namespace lldb_private;

static ConstString &
GetThreadGPRMemberName ()
{
    static ConstString g_gpr_member_name("gpr");
    return g_gpr_member_name;
}

void
OperatingSystemDarwinKernel::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
OperatingSystemDarwinKernel::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

OperatingSystem *
OperatingSystemDarwinKernel::CreateInstance (Process *process, bool force)
{
#if 0
    bool create = force;
    if (!create)
    {
        Module* exe_module = process->GetTarget().GetExecutableModulePointer();
        if (exe_module)
        {
            ObjectFile *object_file = exe_module->GetObjectFile();
            if (object_file)
            {
                if (object_file->GetStrata() != ObjectFile::eStrataKernel)
                {
                    return NULL;
                }
            }
        }

        // We can limit the creation of this plug-in to "*-apple-macosx" or "*-apple-ios" triples
        // if we comment out the lines below...
//        if (create)
//        {
//            const llvm::Triple &triple_ref = process->GetTarget().GetArchitecture().GetTriple();
//            create = triple_ref.getOS() == llvm::Triple::Darwin && triple_ref.getVendor() == llvm::Triple::Apple;
//        }
    }
    
    if (create)
        return new OperatingSystemDarwinKernel (process);
#endif
    return NULL;
}


const char *
OperatingSystemDarwinKernel::GetPluginNameStatic()
{
    return "macosx-kernel";
}

const char *
OperatingSystemDarwinKernel::GetPluginDescriptionStatic()
{
    return "Operating system plug-in that gathers OS information from darwin kernels.";
}


OperatingSystemDarwinKernel::OperatingSystemDarwinKernel (lldb_private::Process *process) :
    OperatingSystem (process),
    m_thread_list_valobj_sp (),
    m_register_info_ap ()
{
}

OperatingSystemDarwinKernel::~OperatingSystemDarwinKernel ()
{
}

ValueObjectSP
OperatingSystemDarwinKernel::GetThreadListValueObject ()
{
    if (m_thread_list_valobj_sp.get() == NULL)
    {
        VariableList variable_list;
        const uint32_t max_matches = 1;
        const bool append = true;
        static ConstString g_thread_list_name("g_thread_list");
        Module *exe_module = m_process->GetTarget().GetExecutableModulePointer();
        if (exe_module)
        {
            if (exe_module->FindGlobalVariables (g_thread_list_name,
                                                 NULL,
                                                 append, 
                                                 max_matches,
                                                 variable_list))
            {
                m_thread_list_valobj_sp = ValueObjectVariable::Create (m_process, variable_list.GetVariableAtIndex(0));
            }
        }
    }
    return m_thread_list_valobj_sp;
}

DynamicRegisterInfo *
OperatingSystemDarwinKernel::GetDynamicRegisterInfo ()
{
    if (m_register_info_ap.get() == NULL && m_thread_list_valobj_sp)
    {
        m_register_info_ap.reset (new DynamicRegisterInfo());
        ConstString empty_name;
        const bool can_create = true;
        AddressType addr_type;
        addr_t base_addr = LLDB_INVALID_ADDRESS;
        ValueObjectSP gpr_valobj_sp (m_thread_list_valobj_sp->GetChildMemberWithName(GetThreadGPRMemberName (), can_create));
        
        if (gpr_valobj_sp->IsPointerType ())
            base_addr = gpr_valobj_sp->GetPointerValue (&addr_type);
        else
            base_addr = gpr_valobj_sp->GetAddressOf (true, &addr_type);

        ValueObjectSP child_valobj_sp;
        if (gpr_valobj_sp)
        {
            ABI *abi = m_process->GetABI().get();
            assert (abi);
            uint32_t num_children = gpr_valobj_sp->GetNumChildren();
            
            ConstString gpr_name (gpr_valobj_sp->GetName());
            uint32_t reg_num = 0;
            for (uint32_t i=0; i<num_children; ++i)
            {
                child_valobj_sp = gpr_valobj_sp->GetChildAtIndex(i, can_create);

                ConstString reg_name(child_valobj_sp->GetName());
                if (reg_name)
                {
                    const char *reg_name_cstr = reg_name.GetCString();
                    while (reg_name_cstr[0] == '_')
                        ++reg_name_cstr;
                    if (reg_name_cstr != reg_name.GetCString())
                        reg_name.SetCString (reg_name_cstr);
                }
                
                RegisterInfo reg_info;
                if (abi->GetRegisterInfoByName(reg_name, reg_info))
                {
                    // Adjust the byte size and the offset to match the layout of registers in our struct
                    reg_info.byte_size = child_valobj_sp->GetByteSize();
                    reg_info.byte_offset = child_valobj_sp->GetAddressOf(true, &addr_type) - base_addr;
                    reg_info.kinds[eRegisterKindLLDB] = reg_num++;
                    m_register_info_ap->AddRegister (reg_info, reg_name, empty_name, gpr_name);
                }
            }
            
            m_register_info_ap->Finalize();
        }
    }
    assert (m_register_info_ap.get());
    return m_register_info_ap.get();
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
OperatingSystemDarwinKernel::GetPluginName()
{
    return "OperatingSystemDarwinKernel";
}

const char *
OperatingSystemDarwinKernel::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
OperatingSystemDarwinKernel::GetPluginVersion()
{
    return 1;
}

bool
OperatingSystemDarwinKernel::UpdateThreadList (ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    // Make any constant strings once and cache the uniqued C string values
    // so we don't have to rehash them each time through this function call
    static ConstString g_tid_member_name("tid");
    static ConstString g_next_member_name("next");

    ValueObjectSP root_valobj_sp (GetThreadListValueObject ());
    ValueObjectSP valobj_sp = root_valobj_sp;
    const bool can_create = true;
    while (valobj_sp)
    {
        if (valobj_sp->GetValueAsUnsigned(0) == 0)
            break;

        ValueObjectSP tid_valobj_sp(valobj_sp->GetChildMemberWithName(g_tid_member_name, can_create));
        if (!tid_valobj_sp)
            break;
        
        tid_t tid = tid_valobj_sp->GetValueAsUnsigned (LLDB_INVALID_THREAD_ID);
        if (tid == LLDB_INVALID_THREAD_ID)
            break;

        ThreadSP thread_sp (old_thread_list.FindThreadByID (tid, false));
        if (!thread_sp)
            thread_sp.reset (new ThreadMemory (*m_process, tid, valobj_sp));

        new_thread_list.AddThread(thread_sp);

        ValueObjectSP next_valobj_sp (valobj_sp->GetChildMemberWithName(g_next_member_name, can_create));
        
        if (next_valobj_sp)
        {
            // Watch for circular linked lists
            if (next_valobj_sp.get() == root_valobj_sp.get())
                break;
        }
        next_valobj_sp.swap(valobj_sp);
    }
    return new_thread_list.GetSize(false) > 0;
}

void
OperatingSystemDarwinKernel::ThreadWasSelected (Thread *thread)
{
}

RegisterContextSP
OperatingSystemDarwinKernel::CreateRegisterContextForThread (Thread *thread, lldb::addr_t reg_data_addr)
{
    ThreadMemory *generic_thread = (ThreadMemory *)thread;
    RegisterContextSP reg_ctx_sp;
    
    ValueObjectSP thread_valobj_sp (generic_thread->GetValueObject());
    if (thread_valobj_sp)
    {
        const bool can_create = true;
        AddressType addr_type;
        addr_t base_addr = LLDB_INVALID_ADDRESS;
        ValueObjectSP gpr_valobj_sp (thread_valobj_sp->GetChildMemberWithName(GetThreadGPRMemberName (), can_create));
        if (gpr_valobj_sp)
        {
            if (gpr_valobj_sp->IsPointerType ())
                base_addr = gpr_valobj_sp->GetPointerValue (&addr_type);
            else
                base_addr = gpr_valobj_sp->GetAddressOf (true, &addr_type);
            reg_ctx_sp.reset (new RegisterContextMemory (*thread, 0, *GetDynamicRegisterInfo (), base_addr));
        }
    }
    return reg_ctx_sp;
}

StopInfoSP
OperatingSystemDarwinKernel::CreateThreadStopReason (lldb_private::Thread *thread)
{
    StopInfoSP stop_info_sp; //(StopInfo::CreateStopReasonWithSignal (*thread, SIGSTOP));
    return stop_info_sp;
}


