//===-- EmulateInstructionMIPS64.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "EmulateInstructionMIPS64.h"

#include <stdlib.h>

#include "llvm-c/Disassembler.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Opcode.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/UnwindPlan.h"

#include "llvm/ADT/STLExtras.h"

#include "Plugins/Process/Utility/InstructionUtils.h"
#include "Plugins/Process/Utility/RegisterContext_mips.h"

using namespace lldb;
using namespace lldb_private;

#define UInt(x) ((uint64_t)x)
#define integer int64_t


//----------------------------------------------------------------------
//
// EmulateInstructionMIPS64 implementation
//
//----------------------------------------------------------------------

#ifdef __mips__
extern "C" {
    void LLVMInitializeMipsTargetInfo ();
    void LLVMInitializeMipsTarget ();
    void LLVMInitializeMipsAsmPrinter ();
    void LLVMInitializeMipsTargetMC ();
    void LLVMInitializeMipsDisassembler ();
}
#endif

EmulateInstructionMIPS64::EmulateInstructionMIPS64 (const lldb_private::ArchSpec &arch) :
    EmulateInstruction (arch)
{
    /* Create instance of llvm::MCDisassembler */
    std::string Error;
    llvm::Triple triple = arch.GetTriple();
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget (triple.getTriple(), Error);

    /*
     * If we fail to get the target then we haven't registered it. The SystemInitializerCommon 
     * does not initialize targets, MCs and disassemblers. However we need the MCDisassembler 
     * to decode the instructions so that the decoding complexity stays with LLVM. 
     * Initialize the MIPS targets and disassemblers.
    */
#ifdef __mips__
    if (!target)
    {
        LLVMInitializeMipsTargetInfo ();
        LLVMInitializeMipsTarget ();
        LLVMInitializeMipsAsmPrinter ();
        LLVMInitializeMipsTargetMC ();
        LLVMInitializeMipsDisassembler ();
        target = llvm::TargetRegistry::lookupTarget (triple.getTriple(), Error);
    }
#endif

    assert (target);

    llvm::StringRef cpu;

    switch (arch.GetCore())
    {
        case ArchSpec::eCore_mips32:
        case ArchSpec::eCore_mips32el:
            cpu = "mips32"; break;
        case ArchSpec::eCore_mips32r2:
        case ArchSpec::eCore_mips32r2el:
            cpu = "mips32r2"; break;
        case ArchSpec::eCore_mips32r3:
        case ArchSpec::eCore_mips32r3el:
            cpu = "mips32r3"; break;
        case ArchSpec::eCore_mips32r5:
        case ArchSpec::eCore_mips32r5el:
            cpu = "mips32r5"; break;
        case ArchSpec::eCore_mips32r6:
        case ArchSpec::eCore_mips32r6el:
            cpu = "mips32r6"; break;
        case ArchSpec::eCore_mips64:
        case ArchSpec::eCore_mips64el:
            cpu = "mips64"; break;
        case ArchSpec::eCore_mips64r2:
        case ArchSpec::eCore_mips64r2el:
            cpu = "mips64r2"; break;
        case ArchSpec::eCore_mips64r3:
        case ArchSpec::eCore_mips64r3el:
            cpu = "mips64r3"; break;
        case ArchSpec::eCore_mips64r5:
        case ArchSpec::eCore_mips64r5el:
            cpu = "mips64r5"; break;
        case ArchSpec::eCore_mips64r6:
        case ArchSpec::eCore_mips64r6el:
            cpu = "mips64r6"; break;
        default:
            cpu = "generic"; break;
    }

    std::string features = "";
    uint32_t arch_flags = arch.GetFlags ();
    if (arch_flags & ArchSpec::eMIPSAse_msa)
        features += "+msa,";
    if (arch_flags & ArchSpec::eMIPSAse_dsp)
        features += "+dsp,";
    if (arch_flags & ArchSpec::eMIPSAse_dspr2)
        features += "+dspr2,";
    if (arch_flags & ArchSpec::eMIPSAse_mips16)
        features += "+mips16,";
    if (arch_flags & ArchSpec::eMIPSAse_micromips)
        features += "+micromips,";

    m_reg_info.reset (target->createMCRegInfo (triple.getTriple()));
    assert (m_reg_info.get());

    m_insn_info.reset (target->createMCInstrInfo());
    assert (m_insn_info.get());

    m_asm_info.reset (target->createMCAsmInfo (*m_reg_info, triple.getTriple()));
    m_subtype_info.reset (target->createMCSubtargetInfo (triple.getTriple(), cpu, features));
    assert (m_asm_info.get() && m_subtype_info.get());

    m_context.reset (new llvm::MCContext (m_asm_info.get(), m_reg_info.get(), nullptr));
    assert (m_context.get());

    m_disasm.reset (target->createMCDisassembler (*m_subtype_info, *m_context));
    assert (m_disasm.get());
}

void
EmulateInstructionMIPS64::Initialize ()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic (),
                                   GetPluginDescriptionStatic (),
                                   CreateInstance);
}

void
EmulateInstructionMIPS64::Terminate ()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

ConstString
EmulateInstructionMIPS64::GetPluginNameStatic ()
{
    ConstString g_plugin_name ("lldb.emulate-instruction.mips64");
    return g_plugin_name;
}

lldb_private::ConstString
EmulateInstructionMIPS64::GetPluginName()
{
    static ConstString g_plugin_name ("EmulateInstructionMIPS64");
    return g_plugin_name;
}

const char *
EmulateInstructionMIPS64::GetPluginDescriptionStatic ()
{
    return "Emulate instructions for the MIPS64 architecture.";
}

EmulateInstruction *
EmulateInstructionMIPS64::CreateInstance (const ArchSpec &arch, InstructionType inst_type)
{
    if (EmulateInstructionMIPS64::SupportsEmulatingInstructionsOfTypeStatic(inst_type))
    {
        if (arch.GetTriple().getArch() == llvm::Triple::mips64 
            || arch.GetTriple().getArch() == llvm::Triple::mips64el)
        {
            std::auto_ptr<EmulateInstructionMIPS64> emulate_insn_ap (new EmulateInstructionMIPS64 (arch));
            if (emulate_insn_ap.get())
                return emulate_insn_ap.release();
        }
    }
    
    return NULL;
}

bool
EmulateInstructionMIPS64::SetTargetTriple (const ArchSpec &arch)
{
    if (arch.GetTriple().getArch () == llvm::Triple::mips64
        || arch.GetTriple().getArch () == llvm::Triple::mips64el)
        return true;
    return false;
}

const char *
EmulateInstructionMIPS64::GetRegisterName (unsigned reg_num, bool alternate_name)
{
    if (alternate_name)
    {
        switch (reg_num)
        {
            case gcc_dwarf_sp_mips64:  return "r29"; 
            case gcc_dwarf_r30_mips64: return "r30"; 
            case gcc_dwarf_ra_mips64:  return "r31";
            case gcc_dwarf_f0_mips64:  return "f0";
            case gcc_dwarf_f1_mips64:  return "f1";
            case gcc_dwarf_f2_mips64:  return "f2";
            case gcc_dwarf_f3_mips64:  return "f3";
            case gcc_dwarf_f4_mips64:  return "f4";
            case gcc_dwarf_f5_mips64:  return "f5";
            case gcc_dwarf_f6_mips64:  return "f6";
            case gcc_dwarf_f7_mips64:  return "f7";
            case gcc_dwarf_f8_mips64:  return "f8";
            case gcc_dwarf_f9_mips64:  return "f9";
            case gcc_dwarf_f10_mips64: return "f10";
            case gcc_dwarf_f11_mips64: return "f11";
            case gcc_dwarf_f12_mips64: return "f12";
            case gcc_dwarf_f13_mips64: return "f13";
            case gcc_dwarf_f14_mips64: return "f14";
            case gcc_dwarf_f15_mips64: return "f15";
            case gcc_dwarf_f16_mips64: return "f16";
            case gcc_dwarf_f17_mips64: return "f17";
            case gcc_dwarf_f18_mips64: return "f18";
            case gcc_dwarf_f19_mips64: return "f19";
            case gcc_dwarf_f20_mips64: return "f20";
            case gcc_dwarf_f21_mips64: return "f21";
            case gcc_dwarf_f22_mips64: return "f22";
            case gcc_dwarf_f23_mips64: return "f23";
            case gcc_dwarf_f24_mips64: return "f24";
            case gcc_dwarf_f25_mips64: return "f25";
            case gcc_dwarf_f26_mips64: return "f26";
            case gcc_dwarf_f27_mips64: return "f27";
            case gcc_dwarf_f28_mips64: return "f28";
            case gcc_dwarf_f29_mips64: return "f29";
            case gcc_dwarf_f30_mips64: return "f30";
            case gcc_dwarf_f31_mips64: return "f31";
            default:
                break;
        }
        return nullptr;
    }

    switch (reg_num)
    {
        case gcc_dwarf_zero_mips64:     return "r0";
        case gcc_dwarf_r1_mips64:       return "r1";
        case gcc_dwarf_r2_mips64:       return "r2";
        case gcc_dwarf_r3_mips64:       return "r3";
        case gcc_dwarf_r4_mips64:       return "r4";
        case gcc_dwarf_r5_mips64:       return "r5";
        case gcc_dwarf_r6_mips64:       return "r6";
        case gcc_dwarf_r7_mips64:       return "r7";
        case gcc_dwarf_r8_mips64:       return "r8";
        case gcc_dwarf_r9_mips64:       return "r9";
        case gcc_dwarf_r10_mips64:      return "r10";
        case gcc_dwarf_r11_mips64:      return "r11";
        case gcc_dwarf_r12_mips64:      return "r12";
        case gcc_dwarf_r13_mips64:      return "r13";
        case gcc_dwarf_r14_mips64:      return "r14";
        case gcc_dwarf_r15_mips64:      return "r15";
        case gcc_dwarf_r16_mips64:      return "r16";
        case gcc_dwarf_r17_mips64:      return "r17";
        case gcc_dwarf_r18_mips64:      return "r18";
        case gcc_dwarf_r19_mips64:      return "r19";
        case gcc_dwarf_r20_mips64:      return "r20";
        case gcc_dwarf_r21_mips64:      return "r21";
        case gcc_dwarf_r22_mips64:      return "r22";
        case gcc_dwarf_r23_mips64:      return "r23";
        case gcc_dwarf_r24_mips64:      return "r24";
        case gcc_dwarf_r25_mips64:      return "r25";
        case gcc_dwarf_r26_mips64:      return "r26";
        case gcc_dwarf_r27_mips64:      return "r27";
        case gcc_dwarf_gp_mips64:       return "gp";
        case gcc_dwarf_sp_mips64:       return "sp";
        case gcc_dwarf_r30_mips64:      return "fp";
        case gcc_dwarf_ra_mips64:       return "ra";
        case gcc_dwarf_sr_mips64:       return "sr";
        case gcc_dwarf_lo_mips64:       return "lo";
        case gcc_dwarf_hi_mips64:       return "hi";
        case gcc_dwarf_bad_mips64:      return "bad";
        case gcc_dwarf_cause_mips64:    return "cause";
        case gcc_dwarf_pc_mips64:       return "pc";
        case gcc_dwarf_f0_mips64:       return "f0";
        case gcc_dwarf_f1_mips64:       return "f1";
        case gcc_dwarf_f2_mips64:       return "f2";
        case gcc_dwarf_f3_mips64:       return "f3";
        case gcc_dwarf_f4_mips64:       return "f4";
        case gcc_dwarf_f5_mips64:       return "f5";
        case gcc_dwarf_f6_mips64:       return "f6";
        case gcc_dwarf_f7_mips64:       return "f7";
        case gcc_dwarf_f8_mips64:       return "f8";
        case gcc_dwarf_f9_mips64:       return "f9";
        case gcc_dwarf_f10_mips64:      return "f10";
        case gcc_dwarf_f11_mips64:      return "f11";
        case gcc_dwarf_f12_mips64:      return "f12";
        case gcc_dwarf_f13_mips64:      return "f13";
        case gcc_dwarf_f14_mips64:      return "f14";
        case gcc_dwarf_f15_mips64:      return "f15";
        case gcc_dwarf_f16_mips64:      return "f16";
        case gcc_dwarf_f17_mips64:      return "f17";
        case gcc_dwarf_f18_mips64:      return "f18";
        case gcc_dwarf_f19_mips64:      return "f19";
        case gcc_dwarf_f20_mips64:      return "f20";
        case gcc_dwarf_f21_mips64:      return "f21";
        case gcc_dwarf_f22_mips64:      return "f22";
        case gcc_dwarf_f23_mips64:      return "f23";
        case gcc_dwarf_f24_mips64:      return "f24";
        case gcc_dwarf_f25_mips64:      return "f25";
        case gcc_dwarf_f26_mips64:      return "f26";
        case gcc_dwarf_f27_mips64:      return "f27";
        case gcc_dwarf_f28_mips64:      return "f28";
        case gcc_dwarf_f29_mips64:      return "f29";
        case gcc_dwarf_f30_mips64:      return "f30";
        case gcc_dwarf_f31_mips64:      return "f31";
        case gcc_dwarf_fcsr_mips64:     return "fcsr";
        case gcc_dwarf_fir_mips64:      return "fir";
    }
    return nullptr;
}

bool
EmulateInstructionMIPS64::GetRegisterInfo (RegisterKind reg_kind, uint32_t reg_num, RegisterInfo &reg_info)
{
    if (reg_kind == eRegisterKindGeneric)
    {
        switch (reg_num)
        {
            case LLDB_REGNUM_GENERIC_PC:    reg_kind = eRegisterKindDWARF; reg_num = gcc_dwarf_pc_mips64; break;
            case LLDB_REGNUM_GENERIC_SP:    reg_kind = eRegisterKindDWARF; reg_num = gcc_dwarf_sp_mips64; break;
            case LLDB_REGNUM_GENERIC_FP:    reg_kind = eRegisterKindDWARF; reg_num = gcc_dwarf_r30_mips64; break;
            case LLDB_REGNUM_GENERIC_RA:    reg_kind = eRegisterKindDWARF; reg_num = gcc_dwarf_ra_mips64; break;
            case LLDB_REGNUM_GENERIC_FLAGS: reg_kind = eRegisterKindDWARF; reg_num = gcc_dwarf_sr_mips64; break;
            default:
                return false;
        }
    }

    if (reg_kind == eRegisterKindDWARF)
    {
       ::memset (&reg_info, 0, sizeof(RegisterInfo));
       ::memset (reg_info.kinds, LLDB_INVALID_REGNUM, sizeof(reg_info.kinds));

       if (reg_num == gcc_dwarf_sr_mips64 || reg_num == gcc_dwarf_fcsr_mips64 || reg_num == gcc_dwarf_fir_mips64)
       {
           reg_info.byte_size = 4;
           reg_info.format = eFormatHex;
           reg_info.encoding = eEncodingUint;
       }
       else if ((int)reg_num >= gcc_dwarf_zero_mips64 && (int)reg_num <= gcc_dwarf_f31_mips64)
       {
           reg_info.byte_size = 8;
           reg_info.format = eFormatHex;
           reg_info.encoding = eEncodingUint;
       }
       else
       {
           return false;
       }

       reg_info.name = GetRegisterName (reg_num, false);
       reg_info.alt_name = GetRegisterName (reg_num, true);
       reg_info.kinds[eRegisterKindDWARF] = reg_num;

       switch (reg_num)
       {
           case gcc_dwarf_r30_mips64: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP; break;
           case gcc_dwarf_ra_mips64: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_RA; break;
           case gcc_dwarf_sp_mips64: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP; break;
           case gcc_dwarf_pc_mips64: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC; break;
           case gcc_dwarf_sr_mips64: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS; break;
           default: break;
       }
       return true;
    }
    return false;
}

EmulateInstructionMIPS64::MipsOpcode*
EmulateInstructionMIPS64::GetOpcodeForInstruction (const char *op_name)
{
    static EmulateInstructionMIPS64::MipsOpcode
    g_opcodes[] = 
    {
        //----------------------------------------------------------------------
        // Prologue/Epilogue instructions
        //----------------------------------------------------------------------
        { "DADDiu",     &EmulateInstructionMIPS64::Emulate_DADDiu,      "DADDIU rt,rs,immediate"    },
        { "SD",         &EmulateInstructionMIPS64::Emulate_SD,          "SD rt,offset(rs)"          },
        { "LD",         &EmulateInstructionMIPS64::Emulate_LD,          "LD rt,offset(base)"        },

        { "SW",         &EmulateInstructionMIPS64::Emulate_SW,          "SW rt,offset(rs)"          },
        { "LW",         &EmulateInstructionMIPS64::Emulate_LW,          "LW rt,offset(rs)"          },

        //----------------------------------------------------------------------
        // Branch instructions
        //----------------------------------------------------------------------
        { "BEQ",        &EmulateInstructionMIPS64::Emulate_BEQ,         "BEQ rs,rt,offset"          },
        { "BNE",        &EmulateInstructionMIPS64::Emulate_BNE,         "BNE rs,rt,offset"          },
        { "BEQL",       &EmulateInstructionMIPS64::Emulate_BEQL,        "BEQL rs,rt,offset"         },
        { "BNEL",       &EmulateInstructionMIPS64::Emulate_BNEL,        "BNEL rs,rt,offset"         },
        { "BGEZALL",    &EmulateInstructionMIPS64::Emulate_BGEZALL,     "BGEZALL rt,offset"         },
        { "BAL",        &EmulateInstructionMIPS64::Emulate_BAL,         "BAL offset"                },
        { "BGEZAL",     &EmulateInstructionMIPS64::Emulate_BGEZAL,      "BGEZAL rs,offset"          },
        { "BALC",       &EmulateInstructionMIPS64::Emulate_BALC,        "BALC offset"               },
        { "BC",         &EmulateInstructionMIPS64::Emulate_BC,          "BC offset"                 },
        { "BGEZ",       &EmulateInstructionMIPS64::Emulate_BGEZ,        "BGEZ rs,offset"            },
        { "BLEZALC",    &EmulateInstructionMIPS64::Emulate_BLEZALC,     "BLEZALC rs,offset"         },
        { "BGEZALC",    &EmulateInstructionMIPS64::Emulate_BGEZALC,     "BGEZALC rs,offset"         },
        { "BLTZALC",    &EmulateInstructionMIPS64::Emulate_BLTZALC,     "BLTZALC rs,offset"         },
        { "BGTZALC",    &EmulateInstructionMIPS64::Emulate_BGTZALC,     "BGTZALC rs,offset"         },
        { "BEQZALC",    &EmulateInstructionMIPS64::Emulate_BEQZALC,     "BEQZALC rs,offset"         },
        { "BNEZALC",    &EmulateInstructionMIPS64::Emulate_BNEZALC,     "BNEZALC rs,offset"         },
        { "BEQC",       &EmulateInstructionMIPS64::Emulate_BEQC,        "BEQC rs,rt,offset"         },
        { "BNEC",       &EmulateInstructionMIPS64::Emulate_BNEC,        "BNEC rs,rt,offset"         },
        { "BLTC",       &EmulateInstructionMIPS64::Emulate_BLTC,        "BLTC rs,rt,offset"         },
        { "BGEC",       &EmulateInstructionMIPS64::Emulate_BGEC,        "BGEC rs,rt,offset"         },
        { "BLTUC",      &EmulateInstructionMIPS64::Emulate_BLTUC,       "BLTUC rs,rt,offset"        },
        { "BGEUC",      &EmulateInstructionMIPS64::Emulate_BGEUC,       "BGEUC rs,rt,offset"        },
        { "BLTZC",      &EmulateInstructionMIPS64::Emulate_BLTZC,       "BLTZC rt,offset"           },
        { "BLEZC",      &EmulateInstructionMIPS64::Emulate_BLEZC,       "BLEZC rt,offset"           },
        { "BGEZC",      &EmulateInstructionMIPS64::Emulate_BGEZC,       "BGEZC rt,offset"           },
        { "BGTZC",      &EmulateInstructionMIPS64::Emulate_BGTZC,       "BGTZC rt,offset"           },
        { "BEQZC",      &EmulateInstructionMIPS64::Emulate_BEQZC,       "BEQZC rt,offset"           },
        { "BNEZC",      &EmulateInstructionMIPS64::Emulate_BNEZC,       "BNEZC rt,offset"           },
        { "BGEZL",      &EmulateInstructionMIPS64::Emulate_BGEZL,       "BGEZL rt,offset"           },
        { "BGTZ",       &EmulateInstructionMIPS64::Emulate_BGTZ,        "BGTZ rt,offset"            },
        { "BGTZL",      &EmulateInstructionMIPS64::Emulate_BGTZL,       "BGTZL rt,offset"           },
        { "BLEZ",       &EmulateInstructionMIPS64::Emulate_BLEZ,        "BLEZ rt,offset"            },
        { "BLEZL",      &EmulateInstructionMIPS64::Emulate_BLEZL,       "BLEZL rt,offset"           },
        { "BLTZ",       &EmulateInstructionMIPS64::Emulate_BLTZ,        "BLTZ rt,offset"            },
        { "BLTZAL",     &EmulateInstructionMIPS64::Emulate_BLTZAL,      "BLTZAL rt,offset"          },
        { "BLTZALL",    &EmulateInstructionMIPS64::Emulate_BLTZALL,     "BLTZALL rt,offset"         },
        { "BLTZL",      &EmulateInstructionMIPS64::Emulate_BLTZL,       "BLTZL rt,offset"           },
        { "BOVC",       &EmulateInstructionMIPS64::Emulate_BOVC,        "BOVC rs,rt,offset"         },
        { "BNVC",       &EmulateInstructionMIPS64::Emulate_BNVC,        "BNVC rs,rt,offset"         },
        { "J",          &EmulateInstructionMIPS64::Emulate_J,           "J target"                  },
        { "JAL",        &EmulateInstructionMIPS64::Emulate_JAL,         "JAL target"                },
        { "JALX",       &EmulateInstructionMIPS64::Emulate_JAL,         "JALX target"               },
        { "JALR",       &EmulateInstructionMIPS64::Emulate_JALR,        "JALR target"               },
        { "JALR_HB",    &EmulateInstructionMIPS64::Emulate_JALR,        "JALR.HB target"            },
        { "JIALC",      &EmulateInstructionMIPS64::Emulate_JIALC,       "JIALC rt,offset"           },
        { "JIC",        &EmulateInstructionMIPS64::Emulate_JIC,         "JIC rt,offset"             },
        { "JR",         &EmulateInstructionMIPS64::Emulate_JR,          "JR target"                 },
        { "JR_HB",      &EmulateInstructionMIPS64::Emulate_JR,          "JR.HB target"              },
        { "BC1F",       &EmulateInstructionMIPS64::Emulate_BC1F,        "BC1F cc, offset"           },
        { "BC1T",       &EmulateInstructionMIPS64::Emulate_BC1T,        "BC1T cc, offset"           },
        { "BC1FL",      &EmulateInstructionMIPS64::Emulate_BC1FL,       "BC1FL cc, offset"          },
        { "BC1TL",      &EmulateInstructionMIPS64::Emulate_BC1TL,       "BC1TL cc, offset"          },
        { "BC1EQZ",     &EmulateInstructionMIPS64::Emulate_BC1EQZ,      "BC1EQZ ft, offset"         },
        { "BC1NEZ",     &EmulateInstructionMIPS64::Emulate_BC1NEZ,      "BC1NEZ ft, offset"         },
        { "BC1ANY2F",   &EmulateInstructionMIPS64::Emulate_BC1ANY2F,    "BC1ANY2F cc, offset"       },
        { "BC1ANY2T",   &EmulateInstructionMIPS64::Emulate_BC1ANY2T,    "BC1ANY2T cc, offset"       },
        { "BC1ANY4F",   &EmulateInstructionMIPS64::Emulate_BC1ANY4F,    "BC1ANY4F cc, offset"       },
        { "BC1ANY4T",   &EmulateInstructionMIPS64::Emulate_BC1ANY4T,    "BC1ANY4T cc, offset"       },
    };

    static const size_t k_num_mips_opcodes = llvm::array_lengthof(g_opcodes);

    for (size_t i = 0; i < k_num_mips_opcodes; ++i)
    {
        if (! strcasecmp (g_opcodes[i].op_name, op_name))
            return &g_opcodes[i];
    }

    return NULL;
}

bool 
EmulateInstructionMIPS64::ReadInstruction ()
{
    bool success = false;
    m_addr = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_ADDRESS, &success);
    if (success)
    {
        Context read_inst_context;
        read_inst_context.type = eContextReadOpcode;
        read_inst_context.SetNoArgs ();
        m_opcode.SetOpcode32 (ReadMemoryUnsigned (read_inst_context, m_addr, 4, 0, &success), GetByteOrder());
    }
    if (!success)
        m_addr = LLDB_INVALID_ADDRESS;
    return success;
}

bool
EmulateInstructionMIPS64::EvaluateInstruction (uint32_t evaluate_options)
{
    bool success = false;
    llvm::MCInst mc_insn;
    uint64_t insn_size;
    DataExtractor data;

    /* Keep the complexity of the decode logic with the llvm::MCDisassembler class. */
    if (m_opcode.GetData (data))
    {
        llvm::MCDisassembler::DecodeStatus decode_status;
        llvm::ArrayRef<uint8_t> raw_insn (data.GetDataStart(), data.GetByteSize());
        decode_status = m_disasm->getInstruction (mc_insn, insn_size, raw_insn, m_addr, llvm::nulls(), llvm::nulls());
        if (decode_status != llvm::MCDisassembler::Success)
            return false;
    }

    /*
     * mc_insn.getOpcode() returns decoded opcode. However to make use
     * of llvm::Mips::<insn> we would need "MipsGenInstrInfo.inc".
    */
    const char *op_name = m_insn_info->getName (mc_insn.getOpcode ());

    if (op_name == NULL)
        return false;

    /*
     * Decoding has been done already. Just get the call-back function
     * and emulate the instruction.
    */
    MipsOpcode *opcode_data = GetOpcodeForInstruction (op_name);

    if (opcode_data == NULL)
        return false;

    uint64_t old_pc = 0, new_pc = 0;
    const bool auto_advance_pc = evaluate_options & eEmulateInstructionOptionAutoAdvancePC;

    if (auto_advance_pc)
    {
        old_pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
        if (!success)
            return false;
    }

    /* emulate instruction */
    success = (this->*opcode_data->callback) (mc_insn);
    if (!success)
        return false;

    if (auto_advance_pc)
    {
        new_pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
        if (!success)
            return false;

        /* If we haven't changed the PC, change it here */
        if (old_pc == new_pc)
        {
            new_pc += 4;
            Context context;
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, new_pc))
                return false;
        }
    }

    return true;
}

bool
EmulateInstructionMIPS64::CreateFunctionEntryUnwind (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    UnwindPlan::RowSP row(new UnwindPlan::Row);
    const bool can_replace = false;

    // Our previous Call Frame Address is the stack pointer
    row->GetCFAValue().SetIsRegisterPlusOffset(gcc_dwarf_sp_mips64, 0);

    // Our previous PC is in the RA
    row->SetRegisterLocationToRegister(gcc_dwarf_pc_mips64, gcc_dwarf_ra_mips64, can_replace);

    unwind_plan.AppendRow (row);

    // All other registers are the same.
    unwind_plan.SetSourceName ("EmulateInstructionMIPS64");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolYes);

    return true;
}

bool
EmulateInstructionMIPS64::nonvolatile_reg_p (uint64_t regnum)
{
    switch (regnum)
    {
        case gcc_dwarf_r16_mips64:
        case gcc_dwarf_r17_mips64:
        case gcc_dwarf_r18_mips64:
        case gcc_dwarf_r19_mips64:
        case gcc_dwarf_r20_mips64:
        case gcc_dwarf_r21_mips64:
        case gcc_dwarf_r22_mips64:
        case gcc_dwarf_r23_mips64:
        case gcc_dwarf_gp_mips64:
        case gcc_dwarf_sp_mips64:
        case gcc_dwarf_r30_mips64:
        case gcc_dwarf_ra_mips64:
            return true;
        default:
            return false;
    }
    return false;
}

bool
EmulateInstructionMIPS64::Emulate_DADDiu (llvm::MCInst& insn)
{
    bool success = false;
    const uint32_t imm16 = insn.getOperand(2).getImm();
    uint64_t imm = SignedBits(imm16, 15, 0);
    uint64_t result;
    uint32_t src, dst;

    dst = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    src = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());

    /* Check if this is daddiu sp,<src>,imm16 */
    if (dst == gcc_dwarf_sp_mips64)
    {
        /* read <src> register */
        uint64_t src_opd_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + src, 0, &success);
        if (!success)
            return false;

        result = src_opd_val + imm;

        Context context;
        RegisterInfo reg_info_sp;
        if (GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_sp_mips64, reg_info_sp))
            context.SetRegisterPlusOffset (reg_info_sp, imm);

        /* We are allocating bytes on stack */
        context.type = eContextAdjustStackPointer;

        WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_sp_mips64, result);
    }
    
    return true;
}

bool
EmulateInstructionMIPS64::Emulate_SW (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t base;
    int64_t imm, address;
    Context bad_vaddr_context;

    base = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    imm = insn.getOperand(2).getImm();

    RegisterInfo reg_info_base;
    if (!GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + base, reg_info_base))
        return false;

    /* read base register */
    address = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + base, 0, &success);
    if (!success)
        return false;

    /* destination address */
    address = address + imm;

    /* Set the bad_vaddr register with base address used in the instruction */
    bad_vaddr_context.type = eContextInvalid;
    WriteRegisterUnsigned (bad_vaddr_context, eRegisterKindDWARF, gcc_dwarf_bad_mips64, address);

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_LW (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t base;
    int64_t imm, address;
    Context bad_vaddr_context;

    base = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    imm = insn.getOperand(2).getImm();

    RegisterInfo reg_info_base;
    if (!GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + base, reg_info_base))
        return false;

    /* read base register */
    address = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + base, 0, &success);
    if (!success)
        return false;

    /* destination address */
    address = address + imm;

    /* Set the bad_vaddr register with base address used in the instruction */
    bad_vaddr_context.type = eContextInvalid;
    WriteRegisterUnsigned (bad_vaddr_context, eRegisterKindDWARF, gcc_dwarf_bad_mips64, address);

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_SD (llvm::MCInst& insn)
{
    uint64_t address;
    RegisterInfo reg_info_base;
    RegisterInfo reg_info_src;
    bool success = false;
    uint32_t imm16 = insn.getOperand(2).getImm();
    uint64_t imm = SignedBits(imm16, 15, 0);
    uint32_t src, base;
    Context bad_vaddr_context;

    src = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    base = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());

    if (!GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + base, reg_info_base)
        || !GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + src, reg_info_src))
        return false;

    /* read SP */
    address = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + base, 0, &success);
    if (!success)
        return false;

    /* destination address */
    address = address + imm;

    /* We look for sp based non-volatile register stores */
    if (base == gcc_dwarf_sp_mips64 && nonvolatile_reg_p (src))
    {
        Context context;
        RegisterValue data_src;
        context.type = eContextPushRegisterOnStack;
        context.SetRegisterToRegisterPlusOffset (reg_info_src, reg_info_base, 0);

        uint8_t buffer [RegisterValue::kMaxRegisterByteSize];
        Error error;

        if (!ReadRegister (&reg_info_base, data_src))
            return false;

        if (data_src.GetAsMemoryData (&reg_info_src, buffer, reg_info_src.byte_size, eByteOrderLittle, error) == 0)
            return false;

        if (!WriteMemory (context, address, buffer, reg_info_src.byte_size))
            return false;
    }

    /* Set the bad_vaddr register with base address used in the instruction */
    bad_vaddr_context.type = eContextInvalid;
    WriteRegisterUnsigned (bad_vaddr_context, eRegisterKindDWARF, gcc_dwarf_bad_mips64, address);

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_LD (llvm::MCInst& insn)
{
    uint32_t src, base;

    src = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    base = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());

    if (base == gcc_dwarf_sp_mips64 && nonvolatile_reg_p (src))
    {
        RegisterValue data_src;
        RegisterInfo reg_info_src;

        if (!GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + src, reg_info_src))
            return false;

        Context context;
        context.type = eContextRegisterLoad;

        if (!WriteRegister (context, &reg_info_src, data_src))
            return false;

        return true;
    }

    return false;
}

bool
EmulateInstructionMIPS64::Emulate_BEQ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target, rs_val, rt_val;

    /*
     * BEQ rs, rt, offset
     *      condition <- (GPR[rs] = GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
     *      else
     *          PC = PC + 4
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val == rt_val)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BNE (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target, rs_val, rt_val;

    /*
     * BNE rs, rt, offset
     *      condition <- (GPR[rs] != GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val != rt_val)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BEQL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target, rs_val, rt_val;

    /*
     * BEQL rs, rt, offset
     *      condition <- (GPR[rs] = GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val == rt_val)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BNEL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target, rs_val, rt_val;

    /*
     * BNEL rs, rt, offset
     *      condition <- (GPR[rs] != GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val != rt_val)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGEZL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target; 
    int64_t rs_val;

    /*
     * BGEZL rs, offset
     *      condition <- (GPR[rs] >= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val >= 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLTZL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BLTZL rs, offset
     *      condition <- (GPR[rs] < 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val < 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGTZL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BGTZL rs, offset
     *      condition <- (GPR[rs] > 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val > 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLEZL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BLEZL rs, offset
     *      condition <- (GPR[rs] <= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val <= 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGTZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BGTZ rs, offset
     *      condition <- (GPR[rs] > 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val > 0)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLEZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target; 
    int64_t rs_val;

    /*
     * BLEZ rs, offset
     *      condition <- (GPR[rs] <= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val <= 0)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLTZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BLTZ rs, offset
     *      condition <- (GPR[rs] < 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val < 0)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGEZALL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BGEZALL rt, offset
     *      condition <- (GPR[rs] >= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val >= 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BAL (llvm::MCInst& insn)
{
    bool success = false;
    int64_t offset, pc, target;

    /*
     * BAL offset
     *      offset = sign_ext (offset << 2)
     *      RA = PC + 8
     *      PC = PC + offset
    */
    offset = insn.getOperand(0).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    target = pc + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BALC (llvm::MCInst& insn)
{
    bool success = false;
    int64_t offset, pc, target;

    /* 
     * BALC offset
     *      offset = sign_ext (offset << 2)
     *      RA = PC + 4
     *      PC = PC + 4 + offset
    */
    offset = insn.getOperand(0).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    target = pc + 4 + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGEZAL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BGEZAL rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] >= 0)
     *      if condition then     
     *          RA = PC + 8
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if ((int64_t) rs_val >= 0)
        target = pc + offset;
    else
        target = pc + 8;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLTZAL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BLTZAL rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] < 0)
     *      if condition then     
     *          RA = PC + 8
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if ((int64_t) rs_val < 0)
        target = pc + offset;
    else
        target = pc + 8;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLTZALL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BLTZALL rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] < 0)
     *      if condition then     
     *          RA = PC + 8
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val < 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 8))
        return false;

    return true;
}


bool
EmulateInstructionMIPS64::Emulate_BLEZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BLEZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] <= 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val <= 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGEZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BGEZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] >= 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val >= 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLTZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BLTZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] < 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val < 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGTZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BGTZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] > 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val > 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BEQZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target, rs_val;

    /*
     * BEQZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] == 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val == 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BNEZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target, rs_val;

    /*
     * BNEZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] != 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val != 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGEZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target, rs_val;

    /*
     * BGEZ rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] >= 0)
     *      if condition then     
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val >= 0)
        target = pc + offset;
    else
        target = pc + 8;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC (llvm::MCInst& insn)
{
    bool success = false;
    int64_t offset, pc, target;

    /* 
     * BC offset
     *      offset = sign_ext (offset << 2)
     *      PC = PC + 4 + offset
    */
    offset = insn.getOperand(0).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    target = pc + 4 + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BEQC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target, rs_val, rt_val;

    /*
     * BEQC rs, rt, offset
     *      condition <- (GPR[rs] = GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val == rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BNEC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target, rs_val, rt_val;

    /*
     * BNEC rs, rt, offset
     *      condition <- (GPR[rs] != GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val != rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLTC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target;
    int64_t rs_val, rt_val;

    /*
     * BLTC rs, rt, offset
     *      condition <- (GPR[rs] < GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val < rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGEC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target;
    int64_t rs_val, rt_val;

    /*
     * BGEC rs, rt, offset
     *      condition <- (GPR[rs] > GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val > rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLTUC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target;
    uint64_t rs_val, rt_val;

    /*
     * BLTUC rs, rt, offset
     *      condition <- (GPR[rs] < GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val < rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGEUC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target;
    uint64_t rs_val, rt_val;

    /*
     * BGEUC rs, rt, offset
     *      condition <- (GPR[rs] > GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val > rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLTZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BLTZC rs, offset
     *      condition <- (GPR[rs] < 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val < 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BLEZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BLEZC rs, offset
     *      condition <- (GPR[rs] <= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val <= 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGEZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BGEZC rs, offset
     *      condition <- (GPR[rs] >= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val >= 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BGTZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    int64_t rs_val;

    /*
     * BGTZC rs, offset
     *      condition <- (GPR[rs] > 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val > 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BEQZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    uint64_t rs_val;

    /*
     * BEQZC rs, offset
     *      condition <- (GPR[rs] = 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val == 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BNEZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int64_t offset, pc, target;
    uint64_t rs_val;

    /*
     * BNEZC rs, offset
     *      condition <- (GPR[rs] != 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val != 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

static int
IsAdd64bitOverflow (int64_t a, int64_t b)
{
  int64_t r = (uint64_t) a + (uint64_t) b;
  return (a < 0 && b < 0 && r >= 0) || (a >= 0 && b >= 0 && r < 0);
}

bool
EmulateInstructionMIPS64::Emulate_BOVC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target;
    int64_t rs_val, rt_val;

    /*
     * BOVC rs, rt, offset
     *      condition <- overflow(GPR[rs] + GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (IsAdd64bitOverflow (rs_val, rt_val))
        target = pc + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BNVC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int64_t offset, pc, target;
    int64_t rs_val, rt_val;

    /*
     * BNVC rs, rt, offset
     *      condition <- overflow(GPR[rs] + GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    if (! IsAdd64bitOverflow (rs_val, rt_val))
        target = pc + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_J (llvm::MCInst& insn)
{
    bool success = false;
    uint64_t offset, pc;

    /* 
     * J offset
     *      offset = sign_ext (offset << 2)
     *      PC = PC[63-28] | offset
    */
    offset = insn.getOperand(0).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    /* This is a PC-region branch and not PC-relative */
    pc = (pc & 0xFFFFFFFFF0000000ULL) | offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, pc))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_JAL (llvm::MCInst& insn)
{
    bool success = false;
    uint64_t offset, target, pc;

    /* 
     * JAL offset
     *      offset = sign_ext (offset << 2)
     *      PC = PC[63-28] | offset
    */
    offset = insn.getOperand(0).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    /* This is a PC-region branch and not PC-relative */
    target = (pc & 0xFFFFFFFFF0000000ULL) | offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_JALR (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    uint64_t pc, rs_val;

    /* 
     * JALR rt, rs
     *      GPR[rt] = PC + 8
     *      PC = GPR[rs]
    */
    rt = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rs = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, rs_val))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_JIALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rt;
    int64_t target, offset, pc, rt_val;

    /* 
     * JIALC rt, offset
     *      offset = sign_ext (offset)
     *      PC = GPR[rt] + offset
     *      RA = PC + 4
    */
    rt = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    target = rt_val + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_JIC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rt;
    int64_t target, offset, rt_val;

    /* 
     * JIC rt, offset
     *      offset = sign_ext (offset)
     *      PC = GPR[rt] + offset
    */
    rt = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    rt_val = (int64_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rt, 0, &success);
    if (!success)
        return false;

    target = rt_val + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_JR (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    uint64_t rs_val;

    /* 
     * JR rs
     *      PC = GPR[rs]
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, rs_val))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC1F (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int64_t target, pc, offset;
    
    /*
     * BC1F cc, offset
     *  condition <- (FPConditionCode(cc) == 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    fcsr = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_fcsr_mips64, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);
    
    if ((fcsr & (1 << cc)) == 0)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC1T (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int64_t target, pc, offset;
    
    /*
     * BC1T cc, offset
     *  condition <- (FPConditionCode(cc) != 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    fcsr = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_fcsr_mips64, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);
    
    if ((fcsr & (1 << cc)) != 0)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC1FL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int64_t target, pc, offset;
    
    /*
     * BC1F cc, offset
     *  condition <- (FPConditionCode(cc) == 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    fcsr = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_fcsr_mips64, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);
    
    if ((fcsr & (1 << cc)) == 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC1TL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int64_t target, pc, offset;
    
    /*
     * BC1T cc, offset
     *  condition <- (FPConditionCode(cc) != 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    fcsr = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_fcsr_mips64, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);
    
    if ((fcsr & (1 << cc)) != 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC1EQZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t ft;
    uint64_t ft_val;
    int64_t target, pc, offset;
    
    /*
     * BC1EQZ ft, offset
     *  condition <- (FPR[ft].bit0 == 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + 4 + offset
    */
    ft = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    ft_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + ft, 0, &success);
    if (!success)
        return false;

    if ((ft_val & 1) == 0)
        target = pc + 4 + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC1NEZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t ft;
    uint64_t ft_val;
    int64_t target, pc, offset;
    
    /*
     * BC1NEZ ft, offset
     *  condition <- (FPR[ft].bit0 != 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + 4 + offset
    */
    ft = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    ft_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + ft, 0, &success);
    if (!success)
        return false;

    if ((ft_val & 1) != 0)
        target = pc + 4 + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC1ANY2F (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int64_t target, pc, offset;
    
    /*
     * BC1ANY2F cc, offset
     *  condition <- (FPConditionCode(cc) == 0 
     *                  || FPConditionCode(cc+1) == 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    fcsr = (uint32_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_fcsr_mips64, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);

    /* if any one bit is 0 */
    if (((fcsr >> cc) & 3) != 3)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC1ANY2T (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int64_t target, pc, offset;
    
    /*
     * BC1ANY2T cc, offset
     *  condition <- (FPConditionCode(cc) == 1 
     *                  || FPConditionCode(cc+1) == 1)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    fcsr = (uint32_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_fcsr_mips64, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);

    /* if any one bit is 1 */
    if (((fcsr >> cc) & 3) != 0)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC1ANY4F (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int64_t target, pc, offset;
    
    /*
     * BC1ANY4F cc, offset
     *  condition <- (FPConditionCode(cc) == 0 
     *                  || FPConditionCode(cc+1) == 0)
     *                  || FPConditionCode(cc+2) == 0)
     *                  || FPConditionCode(cc+3) == 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    fcsr = (uint32_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_fcsr_mips64, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);

    /* if any one bit is 0 */
    if (((fcsr >> cc) & 0xf) != 0xf)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BC1ANY4T (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int64_t target, pc, offset;
    
    /*
     * BC1ANY4T cc, offset
     *  condition <- (FPConditionCode(cc) == 1 
     *                  || FPConditionCode(cc+1) == 1)
     *                  || FPConditionCode(cc+2) == 1)
     *                  || FPConditionCode(cc+3) == 1)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    fcsr = (uint32_t) ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_fcsr_mips64, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);

    /* if any one bit is 1 */
    if (((fcsr >> cc) & 0xf) != 0)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}
