//===-- ARM_DWARF_Registers.c -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM_DWARF_Registers.h"
#include <string.h>

#include <string.h>

using namespace lldb;
using namespace lldb_private;

const char *
GetARMDWARFRegisterName (unsigned reg_num)
{
    switch (reg_num)
    {
        case dwarf_r0:  return "r0";
        case dwarf_r1:  return "r1";
        case dwarf_r2:  return "r2";
        case dwarf_r3:  return "r3";
        case dwarf_r4:  return "r4";
        case dwarf_r5:  return "r5";
        case dwarf_r6:  return "r6";
        case dwarf_r7:  return "r7";
        case dwarf_r8:  return "r8";
        case dwarf_r9:  return "r9";        
        case dwarf_r10: return "r10";
        case dwarf_r11: return "r11";
        case dwarf_r12: return "r12";
        case dwarf_sp:  return "sp";
        case dwarf_lr:  return "lr";
        case dwarf_pc:  return "pc";
        case dwarf_cpsr:return "cpsr";
            
        case dwarf_s0:  return "s0";
        case dwarf_s1:  return "s1";
        case dwarf_s2:  return "s2";
        case dwarf_s3:  return "s3";
        case dwarf_s4:  return "s4";
        case dwarf_s5:  return "s5";
        case dwarf_s6:  return "s6";
        case dwarf_s7:  return "s7";
        case dwarf_s8:  return "s8";
        case dwarf_s9:  return "s9";
        case dwarf_s10: return "s10";
        case dwarf_s11: return "s11";
        case dwarf_s12: return "s12";
        case dwarf_s13: return "s13";
        case dwarf_s14: return "s14";
        case dwarf_s15: return "s15";
        case dwarf_s16: return "s16";
        case dwarf_s17: return "s17";
        case dwarf_s18: return "s18";
        case dwarf_s19: return "s19";
        case dwarf_s20: return "s20";
        case dwarf_s21: return "s21";
        case dwarf_s22: return "s22";
        case dwarf_s23: return "s23";
        case dwarf_s24: return "s24";
        case dwarf_s25: return "s25";
        case dwarf_s26: return "s26";
        case dwarf_s27: return "s27";
        case dwarf_s28: return "s28";
        case dwarf_s29: return "s29";
        case dwarf_s30: return "s30";
        case dwarf_s31: return "s31";
            
        // FPA Registers 0-7
        case dwarf_f0:  return "f0";
        case dwarf_f1:  return "f1";
        case dwarf_f2:  return "f2";
        case dwarf_f3:  return "f3";
        case dwarf_f4:  return "f4";
        case dwarf_f5:  return "f5";
        case dwarf_f6:  return "f6";
        case dwarf_f7:  return "f7";
            
        // Intel wireless MMX general purpose registers 0 - 7
        // XScale accumulator register 0 - 7 (they do overlap with wCGR0 - wCGR7)
        case dwarf_wCGR0: return "wCGR0/ACC0";   
        case dwarf_wCGR1: return "wCGR1/ACC1";
        case dwarf_wCGR2: return "wCGR2/ACC2";
        case dwarf_wCGR3: return "wCGR3/ACC3";
        case dwarf_wCGR4: return "wCGR4/ACC4";
        case dwarf_wCGR5: return "wCGR5/ACC5";
        case dwarf_wCGR6: return "wCGR6/ACC6";
        case dwarf_wCGR7: return "wCGR7/ACC7";
            
        // Intel wireless MMX data registers 0 - 15
        case dwarf_wR0:   return "wR0";
        case dwarf_wR1:   return "wR1";
        case dwarf_wR2:   return "wR2";
        case dwarf_wR3:   return "wR3";
        case dwarf_wR4:   return "wR4";
        case dwarf_wR5:   return "wR5";
        case dwarf_wR6:   return "wR6";
        case dwarf_wR7:   return "wR7";
        case dwarf_wR8:   return "wR8";
        case dwarf_wR9:   return "wR9";
        case dwarf_wR10:  return "wR10";
        case dwarf_wR11:  return "wR11";
        case dwarf_wR12:  return "wR12";
        case dwarf_wR13:  return "wR13";
        case dwarf_wR14:  return "wR14";
        case dwarf_wR15:  return "wR15";
            
        case dwarf_spsr:        return "spsr";
        case dwarf_spsr_fiq:    return "spsr_fiq";
        case dwarf_spsr_irq:    return "spsr_irq";
        case dwarf_spsr_abt:    return "spsr_abt";
        case dwarf_spsr_und:    return "spsr_und";
        case dwarf_spsr_svc:    return "spsr_svc";
            
        case dwarf_r8_usr:      return "r8_usr";
        case dwarf_r9_usr:      return "r9_usr";
        case dwarf_r10_usr:     return "r10_usr";
        case dwarf_r11_usr:     return "r11_usr";
        case dwarf_r12_usr:     return "r12_usr";
        case dwarf_r13_usr:     return "r13_usr";
        case dwarf_r14_usr:     return "r14_usr";
        case dwarf_r8_fiq:      return "r8_fiq";
        case dwarf_r9_fiq:      return "r9_fiq";
        case dwarf_r10_fiq:     return "r10_fiq";
        case dwarf_r11_fiq:     return "r11_fiq";
        case dwarf_r12_fiq:     return "r12_fiq";
        case dwarf_r13_fiq:     return "r13_fiq";
        case dwarf_r14_fiq:     return "r14_fiq";
        case dwarf_r13_irq:     return "r13_irq";
        case dwarf_r14_irq:     return "r14_irq";
        case dwarf_r13_abt:     return "r13_abt";
        case dwarf_r14_abt:     return "r14_abt";
        case dwarf_r13_und:     return "r13_und";
        case dwarf_r14_und:     return "r14_und";
        case dwarf_r13_svc:     return "r13_svc";
        case dwarf_r14_svc:     return "r14_svc";
            
        // Intel wireless MMX control register in co-processor 0 - 7
        case dwarf_wC0:         return "wC0";
        case dwarf_wC1:         return "wC1";
        case dwarf_wC2:         return "wC2";
        case dwarf_wC3:         return "wC3";
        case dwarf_wC4:         return "wC4";
        case dwarf_wC5:         return "wC5";
        case dwarf_wC6:         return "wC6";
        case dwarf_wC7:         return "wC7";
            
        // VFP-v3/Neon
        case dwarf_d0:          return "d0";
        case dwarf_d1:          return "d1";
        case dwarf_d2:          return "d2";
        case dwarf_d3:          return "d3";
        case dwarf_d4:          return "d4";
        case dwarf_d5:          return "d5";
        case dwarf_d6:          return "d6";
        case dwarf_d7:          return "d7";
        case dwarf_d8:          return "d8";
        case dwarf_d9:          return "d9";
        case dwarf_d10:         return "d10";
        case dwarf_d11:         return "d11";
        case dwarf_d12:         return "d12";
        case dwarf_d13:         return "d13";
        case dwarf_d14:         return "d14";
        case dwarf_d15:         return "d15";
        case dwarf_d16:         return "d16";
        case dwarf_d17:         return "d17";
        case dwarf_d18:         return "d18";
        case dwarf_d19:         return "d19";
        case dwarf_d20:         return "d20";
        case dwarf_d21:         return "d21";
        case dwarf_d22:         return "d22";
        case dwarf_d23:         return "d23";
        case dwarf_d24:         return "d24";
        case dwarf_d25:         return "d25";
        case dwarf_d26:         return "d26";
        case dwarf_d27:         return "d27";
        case dwarf_d28:         return "d28";
        case dwarf_d29:         return "d29";
        case dwarf_d30:         return "d30";
        case dwarf_d31:         return "d31";

        // NEON 128-bit vector registers (overlays the d registers)
        case dwarf_q0:          return "q0"; 
        case dwarf_q1:          return "q1"; 
        case dwarf_q2:          return "q2"; 
        case dwarf_q3:          return "q3"; 
        case dwarf_q4:          return "q4"; 
        case dwarf_q5:          return "q5"; 
        case dwarf_q6:          return "q6"; 
        case dwarf_q7:          return "q7"; 
        case dwarf_q8:          return "q8"; 
        case dwarf_q9:          return "q9"; 
        case dwarf_q10:         return "q10";
        case dwarf_q11:         return "q11";
        case dwarf_q12:         return "q12";
        case dwarf_q13:         return "q13";
        case dwarf_q14:         return "q14";
        case dwarf_q15:         return "q15";
    }
    return 0;
}

bool
GetARMDWARFRegisterInfo (unsigned reg_num, RegisterInfo &reg_info)
{
    ::memset (&reg_info, 0, sizeof(RegisterInfo));
    ::memset (reg_info.kinds, LLDB_INVALID_REGNUM, sizeof(reg_info.kinds));

    if (reg_num >= dwarf_q0 && reg_num <= dwarf_q15)
    {
        reg_info.byte_size = 16;
        reg_info.format = eFormatVectorOfUInt8;
        reg_info.encoding = eEncodingVector;
    }
    
    if (reg_num >= dwarf_d0 && reg_num <= dwarf_d31)
    {
        reg_info.byte_size = 8;
        reg_info.format = eFormatFloat;
        reg_info.encoding = eEncodingIEEE754;
    }
    else if (reg_num >= dwarf_s0 && reg_num <= dwarf_s31)
    {
        reg_info.byte_size = 4;
        reg_info.format = eFormatFloat;
        reg_info.encoding = eEncodingIEEE754;
    }
    else if (reg_num >= dwarf_f0 && reg_num <= dwarf_f7)
    {
        reg_info.byte_size = 12;
        reg_info.format = eFormatFloat;
        reg_info.encoding = eEncodingIEEE754;
    }
    else
    {
        reg_info.byte_size = 4;
        reg_info.format = eFormatHex;
        reg_info.encoding = eEncodingUint;
    }
    
    reg_info.kinds[eRegisterKindDWARF] = reg_num;

    switch (reg_num)
    {
        case dwarf_r0:  reg_info.name = "r0"; break;
        case dwarf_r1:  reg_info.name = "r1"; break;
        case dwarf_r2:  reg_info.name = "r2"; break;
        case dwarf_r3:  reg_info.name = "r3"; break;
        case dwarf_r4:  reg_info.name = "r4"; break;
        case dwarf_r5:  reg_info.name = "r5"; break;
        case dwarf_r6:  reg_info.name = "r6"; break;
        case dwarf_r7:  reg_info.name = "r7"; reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP; break;
        case dwarf_r8:  reg_info.name = "r8"; break;
        case dwarf_r9:  reg_info.name = "r9"; break;        
        case dwarf_r10: reg_info.name = "r10"; break;
        case dwarf_r11: reg_info.name = "r11"; break;
        case dwarf_r12: reg_info.name = "r12"; break;
        case dwarf_sp:  reg_info.name = "sp"; reg_info.alt_name = "r13"; reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP; break;
        case dwarf_lr:  reg_info.name = "lr"; reg_info.alt_name = "r14"; reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_RA; break;
        case dwarf_pc:  reg_info.name = "pc"; reg_info.alt_name = "r15"; reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC; break;
        case dwarf_cpsr:reg_info.name = "cpsr"; reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS; break;
            
        case dwarf_s0:  reg_info.name = "s0"; break;
        case dwarf_s1:  reg_info.name = "s1"; break;
        case dwarf_s2:  reg_info.name = "s2"; break;
        case dwarf_s3:  reg_info.name = "s3"; break;
        case dwarf_s4:  reg_info.name = "s4"; break;
        case dwarf_s5:  reg_info.name = "s5"; break;
        case dwarf_s6:  reg_info.name = "s6"; break;
        case dwarf_s7:  reg_info.name = "s7"; break;
        case dwarf_s8:  reg_info.name = "s8"; break;
        case dwarf_s9:  reg_info.name = "s9"; break;
        case dwarf_s10: reg_info.name = "s10"; break;
        case dwarf_s11: reg_info.name = "s11"; break;
        case dwarf_s12: reg_info.name = "s12"; break;
        case dwarf_s13: reg_info.name = "s13"; break;
        case dwarf_s14: reg_info.name = "s14"; break;
        case dwarf_s15: reg_info.name = "s15"; break;
        case dwarf_s16: reg_info.name = "s16"; break;
        case dwarf_s17: reg_info.name = "s17"; break;
        case dwarf_s18: reg_info.name = "s18"; break;
        case dwarf_s19: reg_info.name = "s19"; break;
        case dwarf_s20: reg_info.name = "s20"; break;
        case dwarf_s21: reg_info.name = "s21"; break;
        case dwarf_s22: reg_info.name = "s22"; break;
        case dwarf_s23: reg_info.name = "s23"; break;
        case dwarf_s24: reg_info.name = "s24"; break;
        case dwarf_s25: reg_info.name = "s25"; break;
        case dwarf_s26: reg_info.name = "s26"; break;
        case dwarf_s27: reg_info.name = "s27"; break;
        case dwarf_s28: reg_info.name = "s28"; break;
        case dwarf_s29: reg_info.name = "s29"; break;
        case dwarf_s30: reg_info.name = "s30"; break;
        case dwarf_s31: reg_info.name = "s31"; break;
            
        // FPA Registers 0-7
        case dwarf_f0:  reg_info.name = "f0"; break;
        case dwarf_f1:  reg_info.name = "f1"; break;
        case dwarf_f2:  reg_info.name = "f2"; break;
        case dwarf_f3:  reg_info.name = "f3"; break;
        case dwarf_f4:  reg_info.name = "f4"; break;
        case dwarf_f5:  reg_info.name = "f5"; break;
        case dwarf_f6:  reg_info.name = "f6"; break;
        case dwarf_f7:  reg_info.name = "f7"; break;
            
        // Intel wireless MMX general purpose registers 0 - 7
        // XScale accumulator register 0 - 7 (they do overlap with wCGR0 - wCGR7)
        case dwarf_wCGR0: reg_info.name = "wCGR0/ACC0"; break;   
        case dwarf_wCGR1: reg_info.name = "wCGR1/ACC1"; break;
        case dwarf_wCGR2: reg_info.name = "wCGR2/ACC2"; break;
        case dwarf_wCGR3: reg_info.name = "wCGR3/ACC3"; break;
        case dwarf_wCGR4: reg_info.name = "wCGR4/ACC4"; break;
        case dwarf_wCGR5: reg_info.name = "wCGR5/ACC5"; break;
        case dwarf_wCGR6: reg_info.name = "wCGR6/ACC6"; break;
        case dwarf_wCGR7: reg_info.name = "wCGR7/ACC7"; break;
            
        // Intel wireless MMX data registers 0 - 15
        case dwarf_wR0:   reg_info.name = "wR0"; break;
        case dwarf_wR1:   reg_info.name = "wR1"; break;
        case dwarf_wR2:   reg_info.name = "wR2"; break;
        case dwarf_wR3:   reg_info.name = "wR3"; break;
        case dwarf_wR4:   reg_info.name = "wR4"; break;
        case dwarf_wR5:   reg_info.name = "wR5"; break;
        case dwarf_wR6:   reg_info.name = "wR6"; break;
        case dwarf_wR7:   reg_info.name = "wR7"; break;
        case dwarf_wR8:   reg_info.name = "wR8"; break;
        case dwarf_wR9:   reg_info.name = "wR9"; break;
        case dwarf_wR10:  reg_info.name = "wR10"; break;
        case dwarf_wR11:  reg_info.name = "wR11"; break;
        case dwarf_wR12:  reg_info.name = "wR12"; break;
        case dwarf_wR13:  reg_info.name = "wR13"; break;
        case dwarf_wR14:  reg_info.name = "wR14"; break;
        case dwarf_wR15:  reg_info.name = "wR15"; break;
            
        case dwarf_spsr:        reg_info.name = "spsr"; break;
        case dwarf_spsr_fiq:    reg_info.name = "spsr_fiq"; break;
        case dwarf_spsr_irq:    reg_info.name = "spsr_irq"; break;
        case dwarf_spsr_abt:    reg_info.name = "spsr_abt"; break;
        case dwarf_spsr_und:    reg_info.name = "spsr_und"; break;
        case dwarf_spsr_svc:    reg_info.name = "spsr_svc"; break;
            
        case dwarf_r8_usr:      reg_info.name = "r8_usr"; break;
        case dwarf_r9_usr:      reg_info.name = "r9_usr"; break;
        case dwarf_r10_usr:     reg_info.name = "r10_usr"; break;
        case dwarf_r11_usr:     reg_info.name = "r11_usr"; break;
        case dwarf_r12_usr:     reg_info.name = "r12_usr"; break;
        case dwarf_r13_usr:     reg_info.name = "r13_usr"; break;
        case dwarf_r14_usr:     reg_info.name = "r14_usr"; break;
        case dwarf_r8_fiq:      reg_info.name = "r8_fiq"; break;
        case dwarf_r9_fiq:      reg_info.name = "r9_fiq"; break;
        case dwarf_r10_fiq:     reg_info.name = "r10_fiq"; break;
        case dwarf_r11_fiq:     reg_info.name = "r11_fiq"; break;
        case dwarf_r12_fiq:     reg_info.name = "r12_fiq"; break;
        case dwarf_r13_fiq:     reg_info.name = "r13_fiq"; break;
        case dwarf_r14_fiq:     reg_info.name = "r14_fiq"; break;
        case dwarf_r13_irq:     reg_info.name = "r13_irq"; break;
        case dwarf_r14_irq:     reg_info.name = "r14_irq"; break;
        case dwarf_r13_abt:     reg_info.name = "r13_abt"; break;
        case dwarf_r14_abt:     reg_info.name = "r14_abt"; break;
        case dwarf_r13_und:     reg_info.name = "r13_und"; break;
        case dwarf_r14_und:     reg_info.name = "r14_und"; break;
        case dwarf_r13_svc:     reg_info.name = "r13_svc"; break;
        case dwarf_r14_svc:     reg_info.name = "r14_svc"; break;
            
        // Intel wireless MMX control register in co-processor 0 - 7
        case dwarf_wC0:         reg_info.name = "wC0"; break;
        case dwarf_wC1:         reg_info.name = "wC1"; break;
        case dwarf_wC2:         reg_info.name = "wC2"; break;
        case dwarf_wC3:         reg_info.name = "wC3"; break;
        case dwarf_wC4:         reg_info.name = "wC4"; break;
        case dwarf_wC5:         reg_info.name = "wC5"; break;
        case dwarf_wC6:         reg_info.name = "wC6"; break;
        case dwarf_wC7:         reg_info.name = "wC7"; break;
            
        // VFP-v3/Neon
        case dwarf_d0:          reg_info.name = "d0"; break;
        case dwarf_d1:          reg_info.name = "d1"; break;
        case dwarf_d2:          reg_info.name = "d2"; break;
        case dwarf_d3:          reg_info.name = "d3"; break;
        case dwarf_d4:          reg_info.name = "d4"; break;
        case dwarf_d5:          reg_info.name = "d5"; break;
        case dwarf_d6:          reg_info.name = "d6"; break;
        case dwarf_d7:          reg_info.name = "d7"; break;
        case dwarf_d8:          reg_info.name = "d8"; break;
        case dwarf_d9:          reg_info.name = "d9"; break;
        case dwarf_d10:         reg_info.name = "d10"; break;
        case dwarf_d11:         reg_info.name = "d11"; break;
        case dwarf_d12:         reg_info.name = "d12"; break;
        case dwarf_d13:         reg_info.name = "d13"; break;
        case dwarf_d14:         reg_info.name = "d14"; break;
        case dwarf_d15:         reg_info.name = "d15"; break;
        case dwarf_d16:         reg_info.name = "d16"; break;
        case dwarf_d17:         reg_info.name = "d17"; break;
        case dwarf_d18:         reg_info.name = "d18"; break;
        case dwarf_d19:         reg_info.name = "d19"; break;
        case dwarf_d20:         reg_info.name = "d20"; break;
        case dwarf_d21:         reg_info.name = "d21"; break;
        case dwarf_d22:         reg_info.name = "d22"; break;
        case dwarf_d23:         reg_info.name = "d23"; break;
        case dwarf_d24:         reg_info.name = "d24"; break;
        case dwarf_d25:         reg_info.name = "d25"; break;
        case dwarf_d26:         reg_info.name = "d26"; break;
        case dwarf_d27:         reg_info.name = "d27"; break;
        case dwarf_d28:         reg_info.name = "d28"; break;
        case dwarf_d29:         reg_info.name = "d29"; break;
        case dwarf_d30:         reg_info.name = "d30"; break;
        case dwarf_d31:         reg_info.name = "d31"; break;

        // NEON 128-bit vector registers (overlays the d registers)
        case dwarf_q0:          reg_info.name = "q0"; break;
        case dwarf_q1:          reg_info.name = "q1"; break;
        case dwarf_q2:          reg_info.name = "q2"; break;
        case dwarf_q3:          reg_info.name = "q3"; break;
        case dwarf_q4:          reg_info.name = "q4"; break;
        case dwarf_q5:          reg_info.name = "q5"; break;
        case dwarf_q6:          reg_info.name = "q6"; break;
        case dwarf_q7:          reg_info.name = "q7"; break;
        case dwarf_q8:          reg_info.name = "q8"; break;
        case dwarf_q9:          reg_info.name = "q9"; break;
        case dwarf_q10:         reg_info.name = "q10"; break;
        case dwarf_q11:         reg_info.name = "q11"; break;
        case dwarf_q12:         reg_info.name = "q12"; break;
        case dwarf_q13:         reg_info.name = "q13"; break;
        case dwarf_q14:         reg_info.name = "q14"; break;
        case dwarf_q15:         reg_info.name = "q15"; break;

        default: return false;
    }
    return true;
}
