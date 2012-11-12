//===-- UnwindAssembly-x86.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_UnwindAssembly_x86_h_
#define liblldb_UnwindAssembly_x86_h_

#include "llvm-c/Disassembler.h"

#include "lldb/lldb-private.h"
#include "lldb/Target/UnwindAssembly.h"

class UnwindAssembly_x86 : public lldb_private::UnwindAssembly
{
public:

    ~UnwindAssembly_x86 ();

    virtual bool
    GetNonCallSiteUnwindPlanFromAssembly (lldb_private::AddressRange& func, 
                                          lldb_private::Thread& thread, 
                                          lldb_private::UnwindPlan& unwind_plan);

    virtual bool
    GetFastUnwindPlan (lldb_private::AddressRange& func, 
                       lldb_private::Thread& thread, 
                       lldb_private::UnwindPlan &unwind_plan);

    // thread may be NULL in which case we only use the Target (e.g. if this is called pre-process-launch).
    virtual bool
    FirstNonPrologueInsn (lldb_private::AddressRange& func, 
                          const lldb_private::ExecutionContext &exe_ctx,
                          lldb_private::Address& first_non_prologue_insn);

    static lldb_private::UnwindAssembly *
    CreateInstance (const lldb_private::ArchSpec &arch);


    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static const char *
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
private:
    UnwindAssembly_x86 (const lldb_private::ArchSpec &arch, int cpu);

    int m_cpu;
    lldb_private::ArchSpec m_arch;
};


#endif // liblldb_UnwindAssembly_x86_h_
