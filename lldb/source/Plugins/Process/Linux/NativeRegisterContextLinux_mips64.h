//===-- NativeRegisterContextLinux_mips64.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined (__mips__)

#ifndef lldb_NativeRegisterContextLinux_mips64_h
#define lldb_NativeRegisterContextLinux_mips64_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Process/Utility/RegisterContext_mips64.h"
#include "Plugins/Process/Utility/lldb-mips64-register-enums.h"

#define MAX_NUM_WP 8

namespace lldb_private {
namespace process_linux {

    class NativeProcessLinux;

    class NativeRegisterContextLinux_mips64 : public NativeRegisterContextLinux
    {
    public:
        NativeRegisterContextLinux_mips64 (const ArchSpec& target_arch,
                                           NativeThreadProtocol &native_thread, 
                                           uint32_t concrete_frame_idx);

        uint32_t
        GetRegisterSetCount () const override;

        lldb::addr_t
        GetPCfromBreakpointLocation (lldb::addr_t fail_value = LLDB_INVALID_ADDRESS) override;

        lldb::addr_t
        GetWatchpointHitAddress (uint32_t wp_index) override;

        const RegisterSet *
        GetRegisterSet (uint32_t set_index) const override;

        Error
        ReadRegister (const RegisterInfo *reg_info, RegisterValue &reg_value) override;

        Error
        WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value) override;

        Error
        ReadAllRegisterValues (lldb::DataBufferSP &data_sp) override;

        Error
        WriteAllRegisterValues (const lldb::DataBufferSP &data_sp) override;

        Error
        ReadFPR() override;

        Error
        WriteFPR() override;

        Error
        IsWatchpointHit (uint32_t wp_index, bool &is_hit) override;

        Error
        GetWatchpointHitIndex(uint32_t &wp_index, lldb::addr_t trap_addr) override;

        Error
        IsWatchpointVacant (uint32_t wp_index, bool &is_vacant) override;

        bool
        ClearHardwareWatchpoint (uint32_t wp_index) override;

        Error
        ClearAllHardwareWatchpoints () override;

        Error
        SetHardwareWatchpointWithIndex (lldb::addr_t addr, size_t size,
                uint32_t watch_flags, uint32_t wp_index);

        uint32_t
        SetHardwareWatchpoint (lldb::addr_t addr, size_t size,
                uint32_t watch_flags) override;

        lldb::addr_t
        GetWatchpointAddress (uint32_t wp_index) override;

        uint32_t
        NumSupportedHardwareWatchpoints () override;

    protected:
        Error
        DoReadRegisterValue(uint32_t offset,
                            const char* reg_name,
                            uint32_t size,
                            RegisterValue &value) override;

        Error
        DoWriteRegisterValue(uint32_t offset,
                             const char* reg_name,
                             const RegisterValue &value) override;

        Error
        DoReadWatchPointRegisterValue(lldb::tid_t tid, void* watch_readback);

        Error
        DoWriteWatchPointRegisterValue(lldb::tid_t tid, void* watch_readback);

        bool
        IsFR0();

        bool
        IsFPR(uint32_t reg_index) const;

        void*
        GetGPRBuffer() override { return &m_gpr_mips64; }

        void*
        GetFPRBuffer() override { return &m_fpr; }

        size_t
        GetFPRSize() override { return sizeof(FPR_mips); }

    private:
        // Info about register ranges.
        struct RegInfo
        {
            uint32_t num_registers;
            uint32_t num_gpr_registers;
            uint32_t num_fpr_registers;

            uint32_t last_gpr;
            uint32_t first_fpr;
            uint32_t last_fpr;
        };

        RegInfo m_reg_info;

        uint64_t m_gpr_mips64[k_num_gpr_registers_mips64];

        FPR_mips m_fpr;

        lldb::addr_t hw_addr_map[MAX_NUM_WP];
    };

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_mips64_h

#endif // defined (__mips__)
