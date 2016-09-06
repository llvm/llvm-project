//===-- NativeRegisterContextLinux_x86_64.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(__x86_64__)

#ifndef lldb_NativeRegisterContextLinux_x86_64_h
#define lldb_NativeRegisterContextLinux_x86_64_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Process/Utility/RegisterContext_x86.h"
#include "Plugins/Process/Utility/lldb-x86-register-enums.h"

namespace lldb_private {
namespace process_linux {

    class NativeProcessLinux;

    class NativeRegisterContextLinux_x86_64 : public NativeRegisterContextLinux
    {
    public:
        NativeRegisterContextLinux_x86_64 (const ArchSpec& target_arch,
                                           NativeThreadProtocol &native_thread,
                                           uint32_t concrete_frame_idx);

        uint32_t
        GetRegisterSetCount () const override;

        const RegisterSet *
        GetRegisterSet (uint32_t set_index) const override;

        uint32_t
        GetUserRegisterCount() const override;

        Error
        ReadRegister (const RegisterInfo *reg_info, RegisterValue &reg_value) override;

        Error
        WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value) override;

        Error
        ReadAllRegisterValues (lldb::DataBufferSP &data_sp) override;

        Error
        WriteAllRegisterValues (const lldb::DataBufferSP &data_sp) override;

        Error
        IsWatchpointHit(uint32_t wp_index, bool &is_hit) override;

        Error
        GetWatchpointHitIndex(uint32_t &wp_index, lldb::addr_t trap_addr) override;

        Error
        IsWatchpointVacant(uint32_t wp_index, bool &is_vacant) override;

        bool
        ClearHardwareWatchpoint(uint32_t wp_index) override;

        Error
        ClearAllHardwareWatchpoints () override;

        Error
        SetHardwareWatchpointWithIndex(lldb::addr_t addr, size_t size,
                uint32_t watch_flags, uint32_t wp_index);

        uint32_t
        SetHardwareWatchpoint(lldb::addr_t addr, size_t size,
                uint32_t watch_flags) override;

        lldb::addr_t
        GetWatchpointAddress(uint32_t wp_index) override;

        uint32_t
        NumSupportedHardwareWatchpoints() override;

    protected:
        void*
        GetGPRBuffer() override { return &m_gpr_x86_64; }

        void*
        GetFPRBuffer() override;

        size_t
        GetFPRSize() override;

        Error
        ReadFPR() override;

        Error
        WriteFPR() override;

    private:

        // Private member types.
        enum FPRType
        {
            eFPRTypeNotValid = 0,
            eFPRTypeFXSAVE,
            eFPRTypeXSAVE
        };

        // Info about register ranges.
        struct RegInfo
        {
            uint32_t num_registers;
            uint32_t num_gpr_registers;
            uint32_t num_fpr_registers;
            uint32_t num_avx_registers;
            uint32_t num_mpx_registers;

            uint32_t last_gpr;
            uint32_t first_fpr;
            uint32_t last_fpr;

            uint32_t first_st;
            uint32_t last_st;
            uint32_t first_mm;
            uint32_t last_mm;
            uint32_t first_xmm;
            uint32_t last_xmm;
            uint32_t first_ymm;
            uint32_t last_ymm;
            uint32_t first_mpxr;
            uint32_t last_mpxr;
            uint32_t first_mpxc;
            uint32_t last_mpxc;

            uint32_t first_dr;
            uint32_t gpr_flags;
        };

        // Private member variables.
        mutable FPRType m_fpr_type;
        FPR m_fpr;
        IOVEC m_iovec;
        YMM m_ymm_set;
        MPX m_mpx_set;
        RegInfo m_reg_info;
        uint64_t m_gpr_x86_64[k_num_gpr_registers_x86_64];
        uint32_t m_fctrl_offset_in_userarea;

        // Private member methods.
        bool IsRegisterSetAvailable (uint32_t set_index) const;

        bool
        IsGPR(uint32_t reg_index) const;

        FPRType
        GetFPRType () const;

        bool
        IsFPR(uint32_t reg_index) const;

        bool
        IsFPR(uint32_t reg_index, FPRType fpr_type) const;

        bool
        CopyXSTATEtoYMM (uint32_t reg_index, lldb::ByteOrder byte_order);

        bool
        CopyYMMtoXSTATE(uint32_t reg, lldb::ByteOrder byte_order);

        bool
        CopyXSTATEtoMPX(uint32_t reg);

        bool
        CopyMPXtoXSTATE(uint32_t reg);

        bool
        IsAVX(uint32_t reg_index) const;

        bool
        IsMPX(uint32_t reg_index) const;
    };

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_x86_64_h

#endif // defined(__i386__) || defined(__x86_64__)
