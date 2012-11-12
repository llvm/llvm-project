//===-- UnwindAssembly-x86.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnwindAssembly-x86.h"

#include "llvm-c/Disassembler.h"
#include "llvm/Support/TargetSelect.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnwindAssembly.h"

using namespace lldb;
using namespace lldb_private;

enum CPU {
    k_i386,
    k_x86_64
};

enum i386_register_numbers {
    k_machine_eax = 0,
    k_machine_ecx = 1,
    k_machine_edx = 2,
    k_machine_ebx = 3,
    k_machine_esp = 4,
    k_machine_ebp = 5,
    k_machine_esi = 6,
    k_machine_edi = 7,
    k_machine_eip = 8
};

enum x86_64_register_numbers {
    k_machine_rax = 0,
    k_machine_rcx = 1,
    k_machine_rdx = 2,
    k_machine_rbx = 3,
    k_machine_rsp = 4,
    k_machine_rbp = 5,
    k_machine_rsi = 6,
    k_machine_rdi = 7,
    k_machine_r8 = 8,
    k_machine_r9 = 9,
    k_machine_r10 = 10,
    k_machine_r11 = 11,
    k_machine_r12 = 12,
    k_machine_r13 = 13,
    k_machine_r14 = 14,
    k_machine_r15 = 15,
    k_machine_rip = 16
};

struct regmap_ent {
    const char *name;
    int machine_regno;
    int lldb_regno;
};

static struct regmap_ent i386_register_map[] = {
    {"eax", k_machine_eax, -1},
    {"ecx", k_machine_ecx, -1},
    {"edx", k_machine_edx, -1},
    {"ebx", k_machine_ebx, -1},
    {"esp", k_machine_esp, -1},
    {"ebp", k_machine_ebp, -1},
    {"esi", k_machine_esi, -1},
    {"edi", k_machine_edi, -1},
    {"eip", k_machine_eip, -1}
};

const int size_of_i386_register_map = sizeof (i386_register_map) / sizeof (struct regmap_ent);

static int i386_register_map_initialized = 0;

static struct regmap_ent x86_64_register_map[] = {
    {"rax", k_machine_rax, -1},
    {"rcx", k_machine_rcx, -1},
    {"rdx", k_machine_rdx, -1},
    {"rbx", k_machine_rbx, -1},
    {"rsp", k_machine_rsp, -1},
    {"rbp", k_machine_rbp, -1},
    {"rsi", k_machine_rsi, -1},
    {"rdi", k_machine_rdi, -1},
    {"r8", k_machine_r8, -1},
    {"r9", k_machine_r9, -1},
    {"r10", k_machine_r10, -1},
    {"r11", k_machine_r11, -1},
    {"r12", k_machine_r12, -1},
    {"r13", k_machine_r13, -1},
    {"r14", k_machine_r14, -1},
    {"r15", k_machine_r15, -1},
    {"rip", k_machine_rip, -1}
};

const int size_of_x86_64_register_map = sizeof (x86_64_register_map) / sizeof (struct regmap_ent);

static int x86_64_register_map_initialized = 0;

//-----------------------------------------------------------------------------------------------
//  AssemblyParse_x86 local-file class definition & implementation functions
//-----------------------------------------------------------------------------------------------

class AssemblyParse_x86 {
public:

    AssemblyParse_x86 (const ExecutionContext &exe_ctx, int cpu, ArchSpec &arch, AddressRange func);

    ~AssemblyParse_x86 ();

    bool get_non_call_site_unwind_plan (UnwindPlan &unwind_plan);

    bool get_fast_unwind_plan (AddressRange& func, UnwindPlan &unwind_plan);

    bool find_first_non_prologue_insn (Address &address);

private:
    enum { kMaxInstructionByteSize = 32 };

    bool nonvolatile_reg_p (int machine_regno);
    bool push_rbp_pattern_p ();
    bool push_0_pattern_p ();
    bool mov_rsp_rbp_pattern_p ();
    bool sub_rsp_pattern_p (int& amount);
    bool push_reg_p (int& regno);
    bool mov_reg_to_local_stack_frame_p (int& regno, int& fp_offset);
    bool ret_pattern_p ();
    uint32_t extract_4 (uint8_t *b);
    bool machine_regno_to_lldb_regno (int machine_regno, uint32_t& lldb_regno);
    bool instruction_length (Address addr, int &length);

    const ExecutionContext m_exe_ctx;

    AddressRange m_func_bounds;

    Address m_cur_insn;
    uint8_t m_cur_insn_bytes[kMaxInstructionByteSize];

    int m_machine_ip_regnum;
    int m_machine_sp_regnum;
    int m_machine_fp_regnum;

    int m_lldb_ip_regnum;
    int m_lldb_sp_regnum;
    int m_lldb_fp_regnum;

    int m_wordsize;
    int m_cpu;
    ArchSpec m_arch;
    ::LLVMDisasmContextRef m_disasm_context;

    DISALLOW_COPY_AND_ASSIGN (AssemblyParse_x86);
};

AssemblyParse_x86::AssemblyParse_x86 (const ExecutionContext &exe_ctx, int cpu, ArchSpec &arch, AddressRange func) :
    m_exe_ctx (exe_ctx), 
    m_func_bounds(func), 
    m_cur_insn (),
    m_machine_ip_regnum (LLDB_INVALID_REGNUM),
    m_machine_sp_regnum (LLDB_INVALID_REGNUM),
    m_machine_fp_regnum (LLDB_INVALID_REGNUM),
    m_lldb_ip_regnum (LLDB_INVALID_REGNUM), 
    m_lldb_sp_regnum (LLDB_INVALID_REGNUM),
    m_lldb_fp_regnum (LLDB_INVALID_REGNUM),
    m_wordsize (-1), 
    m_cpu(cpu),
    m_arch(arch)
{
    int *initialized_flag = NULL;
    if (cpu == k_i386)
    {
        m_machine_ip_regnum = k_machine_eip;
        m_machine_sp_regnum = k_machine_esp;
        m_machine_fp_regnum = k_machine_ebp;
        m_wordsize = 4;
        initialized_flag = &i386_register_map_initialized;
    }
    else
    {
        m_machine_ip_regnum = k_machine_rip;
        m_machine_sp_regnum = k_machine_rsp;
        m_machine_fp_regnum = k_machine_rbp;
        m_wordsize = 8;
        initialized_flag = &x86_64_register_map_initialized;
    }

    // we only look at prologue - it will be complete earlier than 512 bytes into func
    if (m_func_bounds.GetByteSize() == 0)
        m_func_bounds.SetByteSize(512);

    Thread *thread = m_exe_ctx.GetThreadPtr();
    if (thread && *initialized_flag == 0)
    {
        RegisterContext *reg_ctx = thread->GetRegisterContext().get();
        if (reg_ctx)
        {
            struct regmap_ent *ent;
            int count, i;
            if (cpu == k_i386)
            {
                ent = i386_register_map;
                count = size_of_i386_register_map;
            }
            else
            {
                ent = x86_64_register_map;
                count = size_of_x86_64_register_map;
            }
            for (i = 0; i < count; i++, ent++)
            {
                const RegisterInfo *ri = reg_ctx->GetRegisterInfoByName (ent->name);
                if (ri)
                    ent->lldb_regno = ri->kinds[eRegisterKindLLDB];
            }
            *initialized_flag = 1;
        }
    }

   // on initial construction we may not have a Thread so these have to remain
   // uninitialized until we can get a RegisterContext to set up the register map table
   if (*initialized_flag == 1)
   {
       uint32_t lldb_regno;
       if (machine_regno_to_lldb_regno (m_machine_sp_regnum, lldb_regno))
           m_lldb_sp_regnum = lldb_regno;
       if (machine_regno_to_lldb_regno (m_machine_fp_regnum, lldb_regno))
           m_lldb_fp_regnum = lldb_regno;
       if (machine_regno_to_lldb_regno (m_machine_ip_regnum, lldb_regno))
           m_lldb_ip_regnum = lldb_regno;
   }

   m_disasm_context = ::LLVMCreateDisasm(m_arch.GetTriple().getTriple().c_str(), 
                                          (void*)this, 
                                          /*TagType=*/1,
                                          NULL,
                                          NULL);
}

AssemblyParse_x86::~AssemblyParse_x86 ()
{
    ::LLVMDisasmDispose(m_disasm_context);
}

// This function expects an x86 native register number (i.e. the bits stripped out of the 
// actual instruction), not an lldb register number.

bool
AssemblyParse_x86::nonvolatile_reg_p (int machine_regno)
{
    if (m_cpu == k_i386)
    {
          switch (machine_regno) {
              case k_machine_ebx:
              case k_machine_ebp:  // not actually a nonvolatile but often treated as such by convention
              case k_machine_esi:
              case k_machine_edi:
              case k_machine_esp:
                  return true;
              default:
                  return false;
          }
    }
    if (m_cpu == k_x86_64)
    {
          switch (machine_regno) {
              case k_machine_rbx:
              case k_machine_rsp:
              case k_machine_rbp:  // not actually a nonvolatile but often treated as such by convention
              case k_machine_r12:
              case k_machine_r13:
              case k_machine_r14:
              case k_machine_r15:
                  return true;
              default:
                  return false;
          }
    }
    return false;
}


// Macro to detect if this is a REX mode prefix byte. 
#define REX_W_PREFIX_P(opcode) (((opcode) & (~0x5)) == 0x48)

// The high bit which should be added to the source register number (the "R" bit)
#define REX_W_SRCREG(opcode) (((opcode) & 0x4) >> 2)

// The high bit which should be added to the destination register number (the "B" bit)
#define REX_W_DSTREG(opcode) ((opcode) & 0x1)

// pushq %rbp [0x55]
bool AssemblyParse_x86::push_rbp_pattern_p () {
    uint8_t *p = m_cur_insn_bytes;
    if (*p == 0x55)
      return true;
    return false;
}

// pushq $0 ; the first instruction in start() [0x6a 0x00]
bool AssemblyParse_x86::push_0_pattern_p ()
{
    uint8_t *p = m_cur_insn_bytes;
    if (*p == 0x6a && *(p + 1) == 0x0)
        return true;
    return false;
}

// movq %rsp, %rbp [0x48 0x8b 0xec] or [0x48 0x89 0xe5]
// movl %esp, %ebp [0x8b 0xec] or [0x89 0xe5]
bool AssemblyParse_x86::mov_rsp_rbp_pattern_p () {
    uint8_t *p = m_cur_insn_bytes;
    if (m_wordsize == 8 && *p == 0x48)
      p++;
    if (*(p) == 0x8b && *(p + 1) == 0xec)
        return true;
    if (*(p) == 0x89 && *(p + 1) == 0xe5)
        return true;
    return false;
}

// subq $0x20, %rsp 
bool AssemblyParse_x86::sub_rsp_pattern_p (int& amount) {
    uint8_t *p = m_cur_insn_bytes;
    if (m_wordsize == 8 && *p == 0x48)
      p++;
    // 8-bit immediate operand
    if (*p == 0x83 && *(p + 1) == 0xec) {
        amount = (int8_t) *(p + 2);
        return true;
    }
    // 32-bit immediate operand
    if (*p == 0x81 && *(p + 1) == 0xec) {
        amount = (int32_t) extract_4 (p + 2);
        return true;
    }
    // Not handled:  [0x83 0xc4] for imm8 with neg values
    // [0x81 0xc4] for imm32 with neg values
    return false;
}

// pushq %rbx
// pushl $ebx
bool AssemblyParse_x86::push_reg_p (int& regno) {
    uint8_t *p = m_cur_insn_bytes;
    int regno_prefix_bit = 0;
    // If we have a rex prefix byte, check to see if a B bit is set
    if (m_wordsize == 8 && *p == 0x41) {
        regno_prefix_bit = 1 << 3;
        p++;
    }
    if (*p >= 0x50 && *p <= 0x57) {
        regno = (*p - 0x50) | regno_prefix_bit;
        return true;
    }
    return false;
}

// Look for an instruction sequence storing a nonvolatile register
// on to the stack frame.

//  movq %rax, -0x10(%rbp) [0x48 0x89 0x45 0xf0]
//  movl %eax, -0xc(%ebp)  [0x89 0x45 0xf4]
bool AssemblyParse_x86::mov_reg_to_local_stack_frame_p (int& regno, int& rbp_offset) {
    uint8_t *p = m_cur_insn_bytes;
    int src_reg_prefix_bit = 0;
    int target_reg_prefix_bit = 0;

    if (m_wordsize == 8 && REX_W_PREFIX_P (*p)) {
        src_reg_prefix_bit = REX_W_SRCREG (*p) << 3;
        target_reg_prefix_bit = REX_W_DSTREG (*p) << 3;
        if (target_reg_prefix_bit == 1) {
            // rbp/ebp don't need a prefix bit - we know this isn't the
            // reg we care about.
            return false;
        }
        p++;
    }

    if (*p == 0x89) {
        /* Mask off the 3-5 bits which indicate the destination register
           if this is a ModR/M byte.  */
        int opcode_destreg_masked_out = *(p + 1) & (~0x38);

        /* Is this a ModR/M byte with Mod bits 01 and R/M bits 101 
           and three bits between them, e.g. 01nnn101
           We're looking for a destination of ebp-disp8 or ebp-disp32.   */
        int immsize;
        if (opcode_destreg_masked_out == 0x45)
          immsize = 2;
        else if (opcode_destreg_masked_out == 0x85)
          immsize = 4;
        else
          return false;

        int offset = 0;
        if (immsize == 2)
          offset = (int8_t) *(p + 2);
        if (immsize == 4)
             offset = (uint32_t) extract_4 (p + 2);
        if (offset > 0)
          return false;

        regno = ((*(p + 1) >> 3) & 0x7) | src_reg_prefix_bit;
        rbp_offset = offset > 0 ? offset : -offset;
        return true;
    }
    return false;
}

// ret [0xc9] or [0xc2 imm8] or [0xca imm8]
bool 
AssemblyParse_x86::ret_pattern_p () 
{
    uint8_t *p = m_cur_insn_bytes;
    if (*p == 0xc9 || *p == 0xc2 || *p == 0xca || *p == 0xc3)
        return true;
    return false;
}

uint32_t
AssemblyParse_x86::extract_4 (uint8_t *b)
{
    uint32_t v = 0;
    for (int i = 3; i >= 0; i--)
        v = (v << 8) | b[i];
    return v;
}

bool 
AssemblyParse_x86::machine_regno_to_lldb_regno (int machine_regno, uint32_t &lldb_regno)
{
    struct regmap_ent *ent;
    int count, i;
    if (m_cpu == k_i386)
    {
        ent = i386_register_map;
        count = size_of_i386_register_map;
    }
    else
    {
        ent = x86_64_register_map;
        count = size_of_x86_64_register_map;
    }
    for (i = 0; i < count; i++, ent++)
    {
        if (ent->machine_regno == machine_regno)
            if (ent->lldb_regno != -1)
            {
                lldb_regno = ent->lldb_regno;
                return true;
            }
    }
    return false;
}

bool
AssemblyParse_x86::instruction_length (Address addr, int &length)
{
    const uint32_t max_op_byte_size = m_arch.GetMaximumOpcodeByteSize();
    llvm::SmallVector <uint8_t, 32> opcode_data;
    opcode_data.resize (max_op_byte_size);

    if (!addr.IsValid())
        return false;

    const bool prefer_file_cache = true;
    Error error;
    Target *target = m_exe_ctx.GetTargetPtr();
    if (target->ReadMemory (addr, prefer_file_cache, opcode_data.data(), max_op_byte_size, error) == -1)
    {
        return false;
    }
   
    char out_string[512];
    const addr_t pc = addr.GetFileAddress();
    const size_t inst_size = ::LLVMDisasmInstruction (m_disasm_context,
                                                      opcode_data.data(),
                                                      max_op_byte_size,
                                                      pc, // PC value
                                                      out_string,
                                                      sizeof(out_string));

    length = inst_size;
    return true;
}


bool 
AssemblyParse_x86::get_non_call_site_unwind_plan (UnwindPlan &unwind_plan)
{
    UnwindPlan::RowSP row(new UnwindPlan::Row);
    int non_prologue_insn_count = 0;
    m_cur_insn = m_func_bounds.GetBaseAddress ();
    int current_func_text_offset = 0;
    int current_sp_bytes_offset_from_cfa = 0;
    UnwindPlan::Row::RegisterLocation initial_regloc;
    Error error;

    if (!m_cur_insn.IsValid())
    {
        return false;
    }

    unwind_plan.SetPlanValidAddressRange (m_func_bounds);
    unwind_plan.SetRegisterKind (eRegisterKindLLDB);

    // At the start of the function, find the CFA by adding wordsize to the SP register
    row->SetOffset (current_func_text_offset);
    row->SetCFARegister (m_lldb_sp_regnum);
    row->SetCFAOffset (m_wordsize);

    // caller's stack pointer value before the call insn is the CFA address
    initial_regloc.SetIsCFAPlusOffset (0);
    row->SetRegisterInfo (m_lldb_sp_regnum, initial_regloc);

    // saved instruction pointer can be found at CFA - wordsize.
    current_sp_bytes_offset_from_cfa = m_wordsize;
    initial_regloc.SetAtCFAPlusOffset (-current_sp_bytes_offset_from_cfa);
    row->SetRegisterInfo (m_lldb_ip_regnum, initial_regloc);

    unwind_plan.AppendRow (row);

    // Allocate a new Row, populate it with the existing Row contents.
    UnwindPlan::Row *newrow = new UnwindPlan::Row;
    *newrow = *row.get();
    row.reset(newrow);

    const bool prefer_file_cache = true;

    Target *target = m_exe_ctx.GetTargetPtr();
    while (m_func_bounds.ContainsFileAddress (m_cur_insn) && non_prologue_insn_count < 10)
    {
        int stack_offset, insn_len;
        int machine_regno;          // register numbers masked directly out of instructions
        uint32_t lldb_regno;        // register numbers in lldb's eRegisterKindLLDB numbering scheme

        if (!instruction_length (m_cur_insn, insn_len) || insn_len == 0 || insn_len > kMaxInstructionByteSize)
        {
            // An unrecognized/junk instruction
            break;
        }
        if (target->ReadMemory (m_cur_insn, prefer_file_cache, m_cur_insn_bytes, insn_len, error) == -1)
        {
           // Error reading the instruction out of the file, stop scanning
           break;
        }

        if (push_rbp_pattern_p ())
        {
            row->SetOffset (current_func_text_offset + insn_len);
            current_sp_bytes_offset_from_cfa += m_wordsize;
            row->SetCFAOffset (current_sp_bytes_offset_from_cfa);
            UnwindPlan::Row::RegisterLocation regloc;
            regloc.SetAtCFAPlusOffset (-row->GetCFAOffset());
            row->SetRegisterInfo (m_lldb_fp_regnum, regloc);
            unwind_plan.AppendRow (row);
            // Allocate a new Row, populate it with the existing Row contents.
            newrow = new UnwindPlan::Row;
            *newrow = *row.get();
            row.reset(newrow);
            goto loopnext;
        }

        if (mov_rsp_rbp_pattern_p ())
        {
            row->SetOffset (current_func_text_offset + insn_len);
            row->SetCFARegister (m_lldb_fp_regnum);
            unwind_plan.AppendRow (row);
            // Allocate a new Row, populate it with the existing Row contents.
            newrow = new UnwindPlan::Row;
            *newrow = *row.get();
            row.reset(newrow);
            goto loopnext;
        }

        // This is the start() function (or a pthread equivalent), it starts with a pushl $0x0 which puts the
        // saved pc value of 0 on the stack.  In this case we want to pretend we didn't see a stack movement at all --
        // normally the saved pc value is already on the stack by the time the function starts executing.
        if (push_0_pattern_p ())
        {
            goto loopnext;
        }

        if (push_reg_p (machine_regno))
        {
            current_sp_bytes_offset_from_cfa += m_wordsize;
            if (nonvolatile_reg_p (machine_regno) && machine_regno_to_lldb_regno (machine_regno, lldb_regno))
            {
                row->SetOffset (current_func_text_offset + insn_len);
                if (row->GetCFARegister() == m_lldb_sp_regnum)
                {
                    row->SetCFAOffset (current_sp_bytes_offset_from_cfa);
                }
                UnwindPlan::Row::RegisterLocation regloc;
                regloc.SetAtCFAPlusOffset (-current_sp_bytes_offset_from_cfa);
                row->SetRegisterInfo (lldb_regno, regloc);
                unwind_plan.AppendRow (row);
                // Allocate a new Row, populate it with the existing Row contents.
                newrow = new UnwindPlan::Row;
                *newrow = *row.get();
                row.reset(newrow);
            }
            goto loopnext;
        }

        if (mov_reg_to_local_stack_frame_p (machine_regno, stack_offset) && nonvolatile_reg_p (machine_regno))
        {
            if (machine_regno_to_lldb_regno (machine_regno, lldb_regno))
            {
                row->SetOffset (current_func_text_offset + insn_len);
                UnwindPlan::Row::RegisterLocation regloc;
                regloc.SetAtCFAPlusOffset (-row->GetCFAOffset());
                row->SetRegisterInfo (lldb_regno, regloc);
                unwind_plan.AppendRow (row);
                // Allocate a new Row, populate it with the existing Row contents.
                newrow = new UnwindPlan::Row;
                *newrow = *row.get();
                row.reset(newrow);
                goto loopnext;
            }
        }

        if (sub_rsp_pattern_p (stack_offset))
        {
            current_sp_bytes_offset_from_cfa += stack_offset;
            if (row->GetCFARegister() == m_lldb_sp_regnum)
            {
                row->SetOffset (current_func_text_offset + insn_len);
                row->SetCFAOffset (current_sp_bytes_offset_from_cfa);
                unwind_plan.AppendRow (row);
                // Allocate a new Row, populate it with the existing Row contents.
                newrow = new UnwindPlan::Row;
                *newrow = *row.get();
                row.reset(newrow);
            }
            goto loopnext;
        }

        if (ret_pattern_p ())
        {
            // we know where the end of the function is; set the limit on the PlanValidAddressRange
            // in case our initial "high pc" value was overly large
            // int original_size = m_func_bounds.GetByteSize();
            // int calculated_size = m_cur_insn.GetOffset() - m_func_bounds.GetBaseAddress().GetOffset() + insn_len + 1;
            // m_func_bounds.SetByteSize (calculated_size);
            // unwind_plan.SetPlanValidAddressRange (m_func_bounds);
            break;
        }

        // FIXME recognize the i386 picbase setup instruction sequence,
        // 0x1f16:  call   0x1f1b                   ; main + 11 at /private/tmp/a.c:3
        // 0x1f1b:  popl   %eax
        // and record the temporary stack movements if the CFA is not expressed in terms of ebp.

        non_prologue_insn_count++;
loopnext:
        m_cur_insn.SetOffset (m_cur_insn.GetOffset() + insn_len);
        current_func_text_offset += insn_len;
    }
    
    // Now look at the byte at the end of the AddressRange for a limited attempt at describing the
    // epilogue.  We're looking for the sequence

    //  [ 0x5d ] mov %rbp, %rsp
    //  [ 0xc3 ] ret
    //  [ 0xe8 xx xx xx xx ] call __stack_chk_fail  (this is sometimes the final insn in the function)

    // We want to add a Row describing how to unwind when we're stopped on the 'ret' instruction where the
    // CFA is no longer defined in terms of rbp, but is now defined in terms of rsp like on function entry.

    uint64_t ret_insn_offset = LLDB_INVALID_ADDRESS;
    Address end_of_fun(m_func_bounds.GetBaseAddress());
    end_of_fun.SetOffset (end_of_fun.GetOffset() + m_func_bounds.GetByteSize());
    
    if (m_func_bounds.GetByteSize() > 7)
    {
        uint8_t bytebuf[7];
        Address last_seven_bytes(end_of_fun);
        last_seven_bytes.SetOffset (last_seven_bytes.GetOffset() - 7);
        if (target->ReadMemory (last_seven_bytes, prefer_file_cache, bytebuf, 7, error) != -1)
        {
            if (bytebuf[5] == 0x5d && bytebuf[6] == 0xc3)  // mov, ret
            {
                ret_insn_offset = m_func_bounds.GetByteSize() - 1;
            }
            else if (bytebuf[0] == 0x5d && bytebuf[1] == 0xc3 && bytebuf[2] == 0xe8) // mov, ret, call
            {
                ret_insn_offset = m_func_bounds.GetByteSize() - 6;
            }
        }
    } else if (m_func_bounds.GetByteSize() > 2)
    {
        uint8_t bytebuf[2];
        Address last_two_bytes(end_of_fun);
        last_two_bytes.SetOffset (last_two_bytes.GetOffset() - 2);
        if (target->ReadMemory (last_two_bytes, prefer_file_cache, bytebuf, 2, error) != -1)
        {
            if (bytebuf[0] == 0x5d && bytebuf[1] == 0xc3) // mov, ret
            {
                ret_insn_offset = m_func_bounds.GetByteSize() - 1;
            }
        }
    }

    if (ret_insn_offset != LLDB_INVALID_ADDRESS)
    {
        // Create a fresh, empty Row and RegisterLocation - don't mention any other registers
        UnwindPlan::RowSP epi_row(new UnwindPlan::Row);
        UnwindPlan::Row::RegisterLocation epi_regloc;

        // When the ret instruction is about to be executed, here's our state
        epi_row->SetOffset (ret_insn_offset);
        epi_row->SetCFARegister (m_lldb_sp_regnum);
        epi_row->SetCFAOffset (m_wordsize);
       
        // caller's stack pointer value before the call insn is the CFA address
        epi_regloc.SetIsCFAPlusOffset (0);
        epi_row->SetRegisterInfo (m_lldb_sp_regnum, epi_regloc);

        // saved instruction pointer can be found at CFA - wordsize
        epi_regloc.SetAtCFAPlusOffset (-m_wordsize);
        epi_row->SetRegisterInfo (m_lldb_ip_regnum, epi_regloc);

        unwind_plan.AppendRow (epi_row);
    }
    
    unwind_plan.SetSourceName ("assembly insn profiling");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolYes);

    return true;
}

/* The "fast unwind plan" is valid for functions that follow the usual convention of 
   using the frame pointer register (ebp, rbp), i.e. the function prologue looks like
     push   %rbp      [0x55]
     mov    %rsp,%rbp [0x48 0x89 0xe5]   (this is a 2-byte insn seq on i386)
*/

bool 
AssemblyParse_x86::get_fast_unwind_plan (AddressRange& func, UnwindPlan &unwind_plan)
{
    UnwindPlan::RowSP row(new UnwindPlan::Row);
    UnwindPlan::Row::RegisterLocation pc_reginfo;
    UnwindPlan::Row::RegisterLocation sp_reginfo;
    UnwindPlan::Row::RegisterLocation fp_reginfo;
    unwind_plan.SetRegisterKind (eRegisterKindLLDB);

    if (!func.GetBaseAddress().IsValid())
        return false;

    Target *target = m_exe_ctx.GetTargetPtr();

    uint8_t bytebuf[4];
    Error error;
    const bool prefer_file_cache = true;
    if (target->ReadMemory (func.GetBaseAddress(), prefer_file_cache, bytebuf, sizeof (bytebuf), error) == -1)
        return false;

    uint8_t i386_prologue[] = {0x55, 0x89, 0xe5};
    uint8_t x86_64_prologue[] = {0x55, 0x48, 0x89, 0xe5};
    int prologue_size;

    if (memcmp (bytebuf, i386_prologue, sizeof (i386_prologue)) == 0)
    {
        prologue_size = sizeof (i386_prologue);
    }
    else if (memcmp (bytebuf, x86_64_prologue, sizeof (x86_64_prologue)) == 0)
    {
        prologue_size = sizeof (x86_64_prologue);
    }
    else
    {
        return false;
    }

    pc_reginfo.SetAtCFAPlusOffset (-m_wordsize);
    row->SetRegisterInfo (m_lldb_ip_regnum, pc_reginfo);

    sp_reginfo.SetIsCFAPlusOffset (0);
    row->SetRegisterInfo (m_lldb_sp_regnum, sp_reginfo);

    // Zero instructions into the function
    row->SetCFARegister (m_lldb_sp_regnum);
    row->SetCFAOffset (m_wordsize);
    row->SetOffset (0);
    unwind_plan.AppendRow (row);
    UnwindPlan::Row *newrow = new UnwindPlan::Row;
    *newrow = *row.get();
    row.reset(newrow);

    // push %rbp has executed - stack moved, rbp now saved
    row->SetCFAOffset (2 * m_wordsize);
    fp_reginfo.SetAtCFAPlusOffset (2 * -m_wordsize);
    row->SetRegisterInfo (m_lldb_fp_regnum, fp_reginfo);
    row->SetOffset (1);
    unwind_plan.AppendRow (row);

    newrow = new UnwindPlan::Row;
    *newrow = *row.get();
    row.reset(newrow);
    
    // mov %rsp, %rbp has executed
    row->SetCFARegister (m_lldb_fp_regnum);
    row->SetCFAOffset (2 * m_wordsize);
    row->SetOffset (prologue_size);     /// 3 or 4 bytes depending on arch
    unwind_plan.AppendRow (row);

    newrow = new UnwindPlan::Row;
    *newrow = *row.get();
    row.reset(newrow);

    unwind_plan.SetPlanValidAddressRange (func);
    unwind_plan.SetSourceName ("fast unwind assembly profiling");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolNo);
    return true;
}

bool 
AssemblyParse_x86::find_first_non_prologue_insn (Address &address)
{
    m_cur_insn = m_func_bounds.GetBaseAddress ();
    if (!m_cur_insn.IsValid())
    {
        return false;
    }

    const bool prefer_file_cache = true;
    Target *target = m_exe_ctx.GetTargetPtr();
    while (m_func_bounds.ContainsFileAddress (m_cur_insn))
    {
        Error error;
        int insn_len, offset, regno;
        if (!instruction_length (m_cur_insn, insn_len) || insn_len > kMaxInstructionByteSize || insn_len == 0)
        {
            // An error parsing the instruction, i.e. probably data/garbage - stop scanning
            break;
        }
        if (target->ReadMemory (m_cur_insn, prefer_file_cache, m_cur_insn_bytes, insn_len, error) == -1)
        {
           // Error reading the instruction out of the file, stop scanning
           break;
        }

        if (push_rbp_pattern_p () || mov_rsp_rbp_pattern_p () || sub_rsp_pattern_p (offset)
            || push_reg_p (regno) || mov_reg_to_local_stack_frame_p (regno, offset))
        {
            m_cur_insn.SetOffset (m_cur_insn.GetOffset() + insn_len);
            continue;
        }

        // Unknown non-prologue instruction - stop scanning
        break;
    }

    address = m_cur_insn;
    return true;
}






//-----------------------------------------------------------------------------------------------
//  UnwindAssemblyParser_x86 method definitions 
//-----------------------------------------------------------------------------------------------

UnwindAssembly_x86::UnwindAssembly_x86 (const ArchSpec &arch, int cpu) : 
    lldb_private::UnwindAssembly(arch), 
    m_cpu(cpu),
    m_arch(arch)
{
}


UnwindAssembly_x86::~UnwindAssembly_x86 ()
{
}

bool
UnwindAssembly_x86::GetNonCallSiteUnwindPlanFromAssembly (AddressRange& func, Thread& thread, UnwindPlan& unwind_plan)
{
    ExecutionContext exe_ctx (thread.shared_from_this());
    AssemblyParse_x86 asm_parse(exe_ctx, m_cpu, m_arch, func);
    return asm_parse.get_non_call_site_unwind_plan (unwind_plan);
}

bool
UnwindAssembly_x86::GetFastUnwindPlan (AddressRange& func, Thread& thread, UnwindPlan &unwind_plan)
{
    ExecutionContext exe_ctx (thread.shared_from_this());
    AssemblyParse_x86 asm_parse(exe_ctx, m_cpu, m_arch, func);
    return asm_parse.get_fast_unwind_plan (func, unwind_plan);
}

bool
UnwindAssembly_x86::FirstNonPrologueInsn (AddressRange& func, const ExecutionContext &exe_ctx, Address& first_non_prologue_insn)
{
    AssemblyParse_x86 asm_parse(exe_ctx, m_cpu, m_arch, func);
    return asm_parse.find_first_non_prologue_insn (first_non_prologue_insn);
}

UnwindAssembly *
UnwindAssembly_x86::CreateInstance (const ArchSpec &arch)
{
    const llvm::Triple::ArchType cpu = arch.GetMachine ();
    if (cpu == llvm::Triple::x86)
        return new UnwindAssembly_x86 (arch, k_i386);
    else if (cpu == llvm::Triple::x86_64)
        return new UnwindAssembly_x86 (arch, k_x86_64);
    return NULL;
}


//------------------------------------------------------------------
// PluginInterface protocol in UnwindAssemblyParser_x86
//------------------------------------------------------------------

const char *
UnwindAssembly_x86::GetPluginName()
{
    return "UnwindAssembly_x86";
}

const char *
UnwindAssembly_x86::GetShortPluginName()
{
    return "unwindassembly.x86";
}


uint32_t
UnwindAssembly_x86::GetPluginVersion()
{
    return 1;
}

void
UnwindAssembly_x86::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
UnwindAssembly_x86::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
UnwindAssembly_x86::GetPluginNameStatic()
{
    return "UnwindAssembly_x86";
}

const char *
UnwindAssembly_x86::GetPluginDescriptionStatic()
{
    return "i386 and x86_64 assembly language profiler plugin.";
}
