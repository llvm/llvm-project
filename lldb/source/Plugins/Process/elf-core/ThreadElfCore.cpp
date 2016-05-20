//===-- ThreadElfCore.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Unwind.h"

#include "ThreadElfCore.h"
#include "ProcessElfCore.h"
#include "Plugins/Process/Utility/RegisterContextLinux_arm.h"
#include "Plugins/Process/Utility/RegisterContextLinux_arm64.h"
#include "Plugins/Process/Utility/RegisterContextLinux_s390x.h"
#include "Plugins/Process/Utility/RegisterContextLinux_x86_64.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_arm.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_arm64.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_i386.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_mips64.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_powerpc.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_x86_64.h"
#include "RegisterContextPOSIXCore_arm.h"
#include "RegisterContextPOSIXCore_arm64.h"
#include "RegisterContextPOSIXCore_mips64.h"
#include "RegisterContextPOSIXCore_powerpc.h"
#include "RegisterContextPOSIXCore_s390x.h"
#include "RegisterContextPOSIXCore_x86_64.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Construct a Thread object with given data
//----------------------------------------------------------------------
ThreadElfCore::ThreadElfCore (Process &process, const ThreadData &td) :
    Thread(process, td.tid),
    m_thread_name(td.name),
    m_thread_reg_ctx_sp (),
    m_signo(td.signo),
    m_gpregset_data(td.gpregset),
    m_fpregset_data(td.fpregset),
    m_vregset_data(td.vregset)
{
}

ThreadElfCore::~ThreadElfCore ()
{
    DestroyThread();
}

void
ThreadElfCore::RefreshStateAfterStop()
{
    GetRegisterContext()->InvalidateIfNeeded (false);
}

void
ThreadElfCore::ClearStackFrames ()
{
    Unwind *unwinder = GetUnwinder ();
    if (unwinder)
        unwinder->Clear();
    Thread::ClearStackFrames();
}

RegisterContextSP
ThreadElfCore::GetRegisterContext ()
{
    if (m_reg_context_sp.get() == NULL) {
        m_reg_context_sp = CreateRegisterContextForFrame (NULL);
    }
    return m_reg_context_sp;
}

RegisterContextSP
ThreadElfCore::CreateRegisterContextForFrame (StackFrame *frame)
{
    RegisterContextSP reg_ctx_sp;
    uint32_t concrete_frame_idx = 0;
    Log *log (GetLogIfAllCategoriesSet(LIBLLDB_LOG_THREAD));

    if (frame)
        concrete_frame_idx = frame->GetConcreteFrameIndex ();

    if (concrete_frame_idx == 0)
    {
        if (m_thread_reg_ctx_sp)
            return m_thread_reg_ctx_sp;

        ProcessElfCore *process = static_cast<ProcessElfCore *>(GetProcess().get());
        ArchSpec arch = process->GetArchitecture();
        RegisterInfoInterface *reg_interface = NULL;

        switch (arch.GetTriple().getOS())
        {
            case llvm::Triple::FreeBSD:
            {
                switch (arch.GetMachine())
                {
                    case llvm::Triple::aarch64:
                        reg_interface = new RegisterContextFreeBSD_arm64(arch);
                        break;
                    case llvm::Triple::arm:
                        reg_interface = new RegisterContextFreeBSD_arm(arch);
                        break;
                    case llvm::Triple::ppc:
                        reg_interface = new RegisterContextFreeBSD_powerpc32(arch);
                        break;
                    case llvm::Triple::ppc64:
                        reg_interface = new RegisterContextFreeBSD_powerpc64(arch);
                        break;
                    case llvm::Triple::mips64:
                        reg_interface = new RegisterContextFreeBSD_mips64(arch);
                        break;
                    case llvm::Triple::x86:
                        reg_interface = new RegisterContextFreeBSD_i386(arch);
                        break;
                    case llvm::Triple::x86_64:
                        reg_interface = new RegisterContextFreeBSD_x86_64(arch);
                        break;
                    default:
                        break;
                }
                break;
            }

            case llvm::Triple::Linux:
            {
                switch (arch.GetMachine())
                {
                    case llvm::Triple::arm:
                        reg_interface = new RegisterContextLinux_arm(arch);
                        break;
                    case llvm::Triple::aarch64:
                        reg_interface = new RegisterContextLinux_arm64(arch);
                        break;
                    case llvm::Triple::systemz:
                        reg_interface = new RegisterContextLinux_s390x(arch);
                        break;
                    case llvm::Triple::x86_64:
                        reg_interface = new RegisterContextLinux_x86_64(arch);
                        break;
                    default:
                        break;
                }
                break;
            }

            default:
                break;
        }

        if (!reg_interface) {
            if (log)
                log->Printf ("elf-core::%s:: Architecture(%d) or OS(%d) not supported",
                             __FUNCTION__, arch.GetMachine(), arch.GetTriple().getOS());
                assert (false && "Architecture or OS not supported");
        }

        switch (arch.GetMachine())
        {
            case llvm::Triple::aarch64:
                m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_arm64 (*this, reg_interface, m_gpregset_data, m_fpregset_data));
                break;
            case llvm::Triple::arm:
                m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_arm (*this, reg_interface, m_gpregset_data, m_fpregset_data));
                break;
            case llvm::Triple::mips64:
                m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_mips64 (*this, reg_interface, m_gpregset_data, m_fpregset_data));
                break;
            case llvm::Triple::ppc:
            case llvm::Triple::ppc64:
                m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_powerpc (*this, reg_interface, m_gpregset_data, m_fpregset_data, m_vregset_data));
                break;
            case llvm::Triple::systemz:
                m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_s390x (*this, reg_interface, m_gpregset_data, m_fpregset_data));
                break;
            case llvm::Triple::x86:
            case llvm::Triple::x86_64:
                m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_x86_64 (*this, reg_interface, m_gpregset_data, m_fpregset_data));
                break;
            default:
                break;
        }

        reg_ctx_sp = m_thread_reg_ctx_sp;
    }
    else if (m_unwinder_ap.get())
    {
        reg_ctx_sp = m_unwinder_ap->CreateRegisterContextForFrame (frame);
    }
    return reg_ctx_sp;
}

bool
ThreadElfCore::CalculateStopInfo ()
{
    ProcessSP process_sp (GetProcess());
    if (process_sp)
    {
        SetStopInfo(StopInfo::CreateStopReasonWithSignal (*this, m_signo));
        return true;
    }
    return false;
}

//----------------------------------------------------------------
// Parse PRSTATUS from NOTE entry
//----------------------------------------------------------------
ELFLinuxPrStatus::ELFLinuxPrStatus()
{
    memset(this, 0, sizeof(ELFLinuxPrStatus));
}

bool
ELFLinuxPrStatus::Parse(DataExtractor &data, ArchSpec &arch)
{
    ByteOrder byteorder = data.GetByteOrder();
    size_t len;
    switch(arch.GetCore())
    {
        case ArchSpec::eCore_s390x_generic:
        case ArchSpec::eCore_x86_64_x86_64:
            len = data.ExtractBytes(0, ELFLINUXPRSTATUS64_SIZE, byteorder, this);
            return len == ELFLINUXPRSTATUS64_SIZE;
        default:
            return false;
    }
}

//----------------------------------------------------------------
// Parse PRPSINFO from NOTE entry
//----------------------------------------------------------------
ELFLinuxPrPsInfo::ELFLinuxPrPsInfo()
{
    memset(this, 0, sizeof(ELFLinuxPrPsInfo));
}

bool
ELFLinuxPrPsInfo::Parse(DataExtractor &data, ArchSpec &arch)
{
    ByteOrder byteorder = data.GetByteOrder();
    size_t len;
    switch(arch.GetCore())
    {
        case ArchSpec::eCore_s390x_generic:
        case ArchSpec::eCore_x86_64_x86_64:
            len = data.ExtractBytes(0, ELFLINUXPRPSINFO64_SIZE, byteorder, this);
            return len == ELFLINUXPRPSINFO64_SIZE;
        default:
            return false;
    }
}

