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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/TargetSelect.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnwindAssembly.h"
#include "lldb/Utility/RegisterNumber.h"

using namespace lldb;
using namespace lldb_private;

enum CPU
{
    k_i386,
    k_x86_64
};

enum i386_register_numbers
{
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

enum x86_64_register_numbers
{
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

struct regmap_ent
{
    const char *name;
    int machine_regno;
    int lldb_regno;
};

static struct regmap_ent i386_register_map[] =
{
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

const int size_of_i386_register_map = llvm::array_lengthof (i386_register_map);

static int i386_register_map_initialized = 0;

static struct regmap_ent x86_64_register_map[] =
{
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

const int size_of_x86_64_register_map = llvm::array_lengthof (x86_64_register_map);

static int x86_64_register_map_initialized = 0;

//-----------------------------------------------------------------------------------------------
//  AssemblyParse_x86 local-file class definition & implementation functions
//-----------------------------------------------------------------------------------------------

class AssemblyParse_x86
{
public:

    AssemblyParse_x86 (const ExecutionContext &exe_ctx, int cpu, ArchSpec &arch, AddressRange func);

    ~AssemblyParse_x86 ();

    bool get_non_call_site_unwind_plan (UnwindPlan &unwind_plan);

    bool augment_unwind_plan_from_call_site (AddressRange& func, UnwindPlan &unwind_plan);

    bool get_fast_unwind_plan (AddressRange& func, UnwindPlan &unwind_plan);

    bool find_first_non_prologue_insn (Address &address);

private:
    enum { kMaxInstructionByteSize = 32 };

    bool nonvolatile_reg_p (int machine_regno);
    bool push_rbp_pattern_p ();
    bool push_0_pattern_p ();
    bool mov_rsp_rbp_pattern_p ();
    bool sub_rsp_pattern_p (int& amount);
    bool add_rsp_pattern_p (int& amount);
    bool lea_rsp_pattern_p (int& amount);
    bool push_reg_p (int& regno);
    bool pop_reg_p (int& regno);
    bool push_imm_pattern_p ();
    bool mov_reg_to_local_stack_frame_p (int& regno, int& fp_offset);
    bool ret_pattern_p ();
    bool pop_rbp_pattern_p ();
    bool leave_pattern_p ();
    bool call_next_insn_pattern_p();
    uint32_t extract_4 (uint8_t *b);
    bool machine_regno_to_lldb_regno (int machine_regno, uint32_t& lldb_regno);
    bool instruction_length (Address addr, int &length);

    const ExecutionContext m_exe_ctx;

    AddressRange m_func_bounds;

    Address m_cur_insn;
    uint8_t m_cur_insn_bytes[kMaxInstructionByteSize];

    uint32_t m_machine_ip_regnum;
    uint32_t m_machine_sp_regnum;
    uint32_t m_machine_fp_regnum;

    uint32_t m_lldb_ip_regnum;
    uint32_t m_lldb_sp_regnum;
    uint32_t m_lldb_fp_regnum;

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
          switch (machine_regno)
          {
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
          switch (machine_regno)
          {
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
bool 
AssemblyParse_x86::push_rbp_pattern_p ()
{
    uint8_t *p = m_cur_insn_bytes;
    if (*p == 0x55)
      return true;
    return false;
}

// pushq $0 ; the first instruction in start() [0x6a 0x00]
bool 
AssemblyParse_x86::push_0_pattern_p ()
{
    uint8_t *p = m_cur_insn_bytes;
    if (*p == 0x6a && *(p + 1) == 0x0)
        return true;
    return false;
}

// pushq $0
// pushl $0
bool 
AssemblyParse_x86::push_imm_pattern_p ()
{
    uint8_t *p = m_cur_insn_bytes;
    if (*p == 0x68 || *p == 0x6a)
        return true;
    return false;
}

// movq %rsp, %rbp [0x48 0x8b 0xec] or [0x48 0x89 0xe5]
// movl %esp, %ebp [0x8b 0xec] or [0x89 0xe5]
bool 
AssemblyParse_x86::mov_rsp_rbp_pattern_p ()
{
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
bool 
AssemblyParse_x86::sub_rsp_pattern_p (int& amount)
{
    uint8_t *p = m_cur_insn_bytes;
    if (m_wordsize == 8 && *p == 0x48)
      p++;
    // 8-bit immediate operand
    if (*p == 0x83 && *(p + 1) == 0xec)
    {
        amount = (int8_t) *(p + 2);
        return true;
    }
    // 32-bit immediate operand
    if (*p == 0x81 && *(p + 1) == 0xec)
    {
        amount = (int32_t) extract_4 (p + 2);
        return true;
    }
    return false;
}

// addq $0x20, %rsp
bool 
AssemblyParse_x86::add_rsp_pattern_p (int& amount)
{
    uint8_t *p = m_cur_insn_bytes;
    if (m_wordsize == 8 && *p == 0x48)
      p++;
    // 8-bit immediate operand
    if (*p == 0x83 && *(p + 1) == 0xc4)
    {
        amount = (int8_t) *(p + 2);
        return true;
    }
    // 32-bit immediate operand
    if (*p == 0x81 && *(p + 1) == 0xc4)
    {
        amount = (int32_t) extract_4 (p + 2);
        return true;
    }
    return false;
}

// lea esp, [esp - 0x28]
// lea esp, [esp + 0x28]
bool
AssemblyParse_x86::lea_rsp_pattern_p (int& amount)
{
    uint8_t *p = m_cur_insn_bytes;
    if (m_wordsize == 8 && *p == 0x48)
        p++;

    // Check opcode
    if (*p != 0x8d)
        return false;

    // 8 bit displacement
    if (*(p + 1) == 0x64 && (*(p + 2) & 0x3f) == 0x24)
    {
        amount = (int8_t) *(p + 3);
        return true;
    }

    // 32 bit displacement
    if (*(p + 1) == 0xa4 && (*(p + 2) & 0x3f) == 0x24)
    {
        amount = (int32_t) extract_4 (p + 3);
        return true;
    }

    return false;
}

// pushq %rbx
// pushl %ebx
bool 
AssemblyParse_x86::push_reg_p (int& regno)
{
    uint8_t *p = m_cur_insn_bytes;
    int regno_prefix_bit = 0;
    // If we have a rex prefix byte, check to see if a B bit is set
    if (m_wordsize == 8 && *p == 0x41)
    {
        regno_prefix_bit = 1 << 3;
        p++;
    }
    if (*p >= 0x50 && *p <= 0x57)
    {
        regno = (*p - 0x50) | regno_prefix_bit;
        return true;
    }
    return false;
}

// popq %rbx
// popl %ebx
bool 
AssemblyParse_x86::pop_reg_p (int& regno)
{
    uint8_t *p = m_cur_insn_bytes;
    int regno_prefix_bit = 0;
    // If we have a rex prefix byte, check to see if a B bit is set
    if (m_wordsize == 8 && *p == 0x41)
    {
        regno_prefix_bit = 1 << 3;
        p++;
    }
    if (*p >= 0x58 && *p <= 0x5f)
    {
        regno = (*p - 0x58) | regno_prefix_bit;
        return true;
    }
    return false;
}

// popq %rbp [0x5d]
// popl %ebp [0x5d]
bool 
AssemblyParse_x86::pop_rbp_pattern_p ()
{
    uint8_t *p = m_cur_insn_bytes;
    return (*p == 0x5d);
}

// leave [0xc9]
bool
AssemblyParse_x86::leave_pattern_p ()
{
    uint8_t *p = m_cur_insn_bytes;
    return (*p == 0xc9);
}

// call $0 [0xe8 0x0 0x0 0x0 0x0]
bool 
AssemblyParse_x86::call_next_insn_pattern_p ()
{
    uint8_t *p = m_cur_insn_bytes;
    return (*p == 0xe8) && (*(p+1) == 0x0) && (*(p+2) == 0x0)
                        && (*(p+3) == 0x0) && (*(p+4) == 0x0);
}

// Look for an instruction sequence storing a nonvolatile register
// on to the stack frame.

//  movq %rax, -0x10(%rbp) [0x48 0x89 0x45 0xf0]
//  movl %eax, -0xc(%ebp)  [0x89 0x45 0xf4]

// The offset value returned in rbp_offset will be positive --
// but it must be subtraced from the frame base register to get
// the actual location.  The positive value returned for the offset
// is a convention used elsewhere for CFA offsets et al.

bool 
AssemblyParse_x86::mov_reg_to_local_stack_frame_p (int& regno, int& rbp_offset)
{
    uint8_t *p = m_cur_insn_bytes;
    int src_reg_prefix_bit = 0;
    int target_reg_prefix_bit = 0;

    if (m_wordsize == 8 && REX_W_PREFIX_P (*p))
    {
        src_reg_prefix_bit = REX_W_SRCREG (*p) << 3;
        target_reg_prefix_bit = REX_W_DSTREG (*p) << 3;
        if (target_reg_prefix_bit == 1)
        {
            // rbp/ebp don't need a prefix bit - we know this isn't the
            // reg we care about.
            return false;
        }
        p++;
    }

    if (*p == 0x89)
    {
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
    if (target->ReadMemory (addr, prefer_file_cache, opcode_data.data(),
                            max_op_byte_size, error) == static_cast<size_t>(-1))
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
    m_cur_insn = m_func_bounds.GetBaseAddress ();
    addr_t current_func_text_offset = 0;
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
    row->GetCFAValue().SetIsRegisterPlusOffset(m_lldb_sp_regnum, m_wordsize);

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

    // Track which registers have been saved so far in the prologue.
    // If we see another push of that register, it's not part of the prologue.
    // The register numbers used here are the machine register #'s
    // (i386_register_numbers, x86_64_register_numbers).
    std::vector<bool> saved_registers(32, false);

    const bool prefer_file_cache = true;

    // Once the prologue has completed we'll save a copy of the unwind instructions
    // If there is an epilogue in the middle of the function, after that epilogue we'll reinstate
    // the unwind setup -- we assume that some code path jumps over the mid-function epilogue

    UnwindPlan::RowSP prologue_completed_row;          // copy of prologue row of CFI
    int prologue_completed_sp_bytes_offset_from_cfa;   // The sp value before the epilogue started executed
    std::vector<bool> prologue_completed_saved_registers;

    Target *target = m_exe_ctx.GetTargetPtr();
    while (m_func_bounds.ContainsFileAddress (m_cur_insn))
    {
        int stack_offset, insn_len;
        int machine_regno;          // register numbers masked directly out of instructions
        uint32_t lldb_regno;        // register numbers in lldb's eRegisterKindLLDB numbering scheme

        bool in_epilogue = false;                          // we're in the middle of an epilogue sequence
        bool row_updated = false;                          // The UnwindPlan::Row 'row' has been updated

        if (!instruction_length (m_cur_insn, insn_len) || insn_len == 0 || insn_len > kMaxInstructionByteSize)
        {
            // An unrecognized/junk instruction
            break;
        }

        if (target->ReadMemory (m_cur_insn, prefer_file_cache, m_cur_insn_bytes,
                                insn_len, error) == static_cast<size_t>(-1))
        {
           // Error reading the instruction out of the file, stop scanning
           break;
        }

        if (push_rbp_pattern_p ())
        {
            current_sp_bytes_offset_from_cfa += m_wordsize;
            row->GetCFAValue().SetOffset (current_sp_bytes_offset_from_cfa);
            UnwindPlan::Row::RegisterLocation regloc;
            regloc.SetAtCFAPlusOffset (-row->GetCFAValue().GetOffset());
            row->SetRegisterInfo (m_lldb_fp_regnum, regloc);
            saved_registers[m_machine_fp_regnum] = true;
            row_updated = true;
        }

        else if (mov_rsp_rbp_pattern_p ())
        {
            row->GetCFAValue().SetIsRegisterPlusOffset(m_lldb_fp_regnum, row->GetCFAValue().GetOffset());
            row_updated = true;
        }

        // This is the start() function (or a pthread equivalent), it starts with a pushl $0x0 which puts the
        // saved pc value of 0 on the stack.  In this case we want to pretend we didn't see a stack movement at all --
        // normally the saved pc value is already on the stack by the time the function starts executing.
        else if (push_0_pattern_p ())
        {
        }

        else if (push_reg_p (machine_regno))
        {
            current_sp_bytes_offset_from_cfa += m_wordsize;
            // the PUSH instruction has moved the stack pointer - if the CFA is set in terms of the stack pointer,
            // we need to add a new row of instructions.
            if (row->GetCFAValue().GetRegisterNumber() == m_lldb_sp_regnum)
            {
                row->GetCFAValue().SetOffset (current_sp_bytes_offset_from_cfa);
                row_updated = true;
            }
            // record where non-volatile (callee-saved, spilled) registers are saved on the stack
            if (nonvolatile_reg_p (machine_regno) 
                && machine_regno_to_lldb_regno (machine_regno, lldb_regno)
                && saved_registers[machine_regno] == false)
            {
                UnwindPlan::Row::RegisterLocation regloc;
                regloc.SetAtCFAPlusOffset (-current_sp_bytes_offset_from_cfa);
                row->SetRegisterInfo (lldb_regno, regloc);
                saved_registers[machine_regno] = true;
                row_updated = true;
            }
        }

        else if (pop_reg_p (machine_regno))
        {
            current_sp_bytes_offset_from_cfa -= m_wordsize;

            if (nonvolatile_reg_p (machine_regno) 
                && machine_regno_to_lldb_regno (machine_regno, lldb_regno)
                && saved_registers[machine_regno] == true)
            {
                saved_registers[machine_regno] = false;
                row->RemoveRegisterInfo (lldb_regno);

                if (machine_regno == (int)m_machine_fp_regnum)
                {
                    row->GetCFAValue().SetIsRegisterPlusOffset (m_lldb_sp_regnum, row->GetCFAValue().GetOffset());
                }

                in_epilogue = true;
                row_updated = true;
            }

            // the POP instruction has moved the stack pointer - if the CFA is set in terms of the stack pointer,
            // we need to add a new row of instructions.
            if (row->GetCFAValue().GetRegisterNumber() == m_lldb_sp_regnum)
            {
                row->GetCFAValue().SetIsRegisterPlusOffset (m_lldb_sp_regnum, current_sp_bytes_offset_from_cfa);
                row_updated = true;
            }
        }

        // The LEAVE instruction moves the value from rbp into rsp and pops
        // a value off the stack into rbp (restoring the caller's rbp value).  
        // It is the opposite of ENTER, or 'push rbp, mov rsp rbp'.
        else if (leave_pattern_p ())
        {
            // We're going to copy the value in rbp into rsp, so re-set the sp offset
            // based on the CFAValue.  Also, adjust it to recognize that we're popping
            // the saved rbp value off the stack.
            current_sp_bytes_offset_from_cfa = row->GetCFAValue().GetOffset();
            current_sp_bytes_offset_from_cfa -= m_wordsize;
            row->GetCFAValue().SetOffset (current_sp_bytes_offset_from_cfa);

            // rbp is restored to the caller's value
            saved_registers[m_machine_fp_regnum] = false;
            row->RemoveRegisterInfo (m_lldb_fp_regnum);

            // cfa is now in terms of rsp again.
            row->GetCFAValue().SetIsRegisterPlusOffset (m_lldb_sp_regnum, row->GetCFAValue().GetOffset());
            row->GetCFAValue().SetOffset (current_sp_bytes_offset_from_cfa);

            in_epilogue = true;
            row_updated = true;
        }

        else if (mov_reg_to_local_stack_frame_p (machine_regno, stack_offset) 
                 && nonvolatile_reg_p (machine_regno)
                 && machine_regno_to_lldb_regno (machine_regno, lldb_regno) 
                 && saved_registers[machine_regno] == false)
        {
            saved_registers[machine_regno] = true;

            UnwindPlan::Row::RegisterLocation regloc;

            // stack_offset for 'movq %r15, -80(%rbp)' will be 80.
            // In the Row, we want to express this as the offset from the CFA.  If the frame base
            // is rbp (like the above instruction), the CFA offset for rbp is probably 16.  So we
            // want to say that the value is stored at the CFA address - 96.
            regloc.SetAtCFAPlusOffset (-(stack_offset + row->GetCFAValue().GetOffset()));

            row->SetRegisterInfo (lldb_regno, regloc);

            row_updated = true;
        }

        else if (sub_rsp_pattern_p (stack_offset))
        {
            current_sp_bytes_offset_from_cfa += stack_offset;
            if (row->GetCFAValue().GetRegisterNumber() == m_lldb_sp_regnum)
            {
                row->GetCFAValue().SetOffset (current_sp_bytes_offset_from_cfa);
                row_updated = true;
            }
        }

        else if (add_rsp_pattern_p (stack_offset))
        {
            current_sp_bytes_offset_from_cfa -= stack_offset;
            if (row->GetCFAValue().GetRegisterNumber() == m_lldb_sp_regnum)
            {
                row->GetCFAValue().SetOffset (current_sp_bytes_offset_from_cfa);
                row_updated = true;
            }
            in_epilogue = true;
        }

        else if (lea_rsp_pattern_p (stack_offset))
        {
            current_sp_bytes_offset_from_cfa -= stack_offset;
            if (row->GetCFAValue().GetRegisterNumber() == m_lldb_sp_regnum)
            {
                row->GetCFAValue().SetOffset (current_sp_bytes_offset_from_cfa);
                row_updated = true;
            }
            if (stack_offset > 0)
                in_epilogue = true;
        }

        else if (ret_pattern_p () && prologue_completed_row.get())
        {
            // Reinstate the saved prologue setup for any instructions
            // that come after the ret instruction

            UnwindPlan::Row *newrow = new UnwindPlan::Row;
            *newrow = *prologue_completed_row.get();
            row.reset (newrow);
            current_sp_bytes_offset_from_cfa = prologue_completed_sp_bytes_offset_from_cfa;

            saved_registers.clear();
            saved_registers.resize(prologue_completed_saved_registers.size(), false);
            for (size_t i = 0; i < prologue_completed_saved_registers.size(); ++i)
            {
                saved_registers[i] = prologue_completed_saved_registers[i];
            }

            in_epilogue = true;
            row_updated = true;
        }

        // call next instruction
        //     call 0
        //  => pop  %ebx
        // This is used in i386 programs to get the PIC base address for finding global data
        else if (call_next_insn_pattern_p ())
        {
            current_sp_bytes_offset_from_cfa += m_wordsize;
            if (row->GetCFAValue().GetRegisterNumber() == m_lldb_sp_regnum)
            {
                row->GetCFAValue().SetOffset (current_sp_bytes_offset_from_cfa);
                row_updated = true;
            }
        }

        if (row_updated)
        {
            if (current_func_text_offset + insn_len < m_func_bounds.GetByteSize())
            {
                row->SetOffset (current_func_text_offset + insn_len);
                unwind_plan.AppendRow (row);
                // Allocate a new Row, populate it with the existing Row contents.
                newrow = new UnwindPlan::Row;
                *newrow = *row.get();
                row.reset(newrow);
            }
        }

        if (in_epilogue == false && row_updated)
        {
            // If we're not in an epilogue sequence, save the updated Row
            UnwindPlan::Row *newrow = new UnwindPlan::Row;
            *newrow = *row.get();
            prologue_completed_row.reset (newrow);

            prologue_completed_saved_registers.clear();
            prologue_completed_saved_registers.resize(saved_registers.size(), false);
            for (size_t i = 0; i < saved_registers.size(); ++i)
            {
                prologue_completed_saved_registers[i] = saved_registers[i];
            }
        }

        // We may change the sp value without adding a new Row necessarily -- keep
        // track of it either way.
        if (in_epilogue == false)
        {
            prologue_completed_sp_bytes_offset_from_cfa = current_sp_bytes_offset_from_cfa;
        }

        m_cur_insn.SetOffset (m_cur_insn.GetOffset() + insn_len);
        current_func_text_offset += insn_len;
    }

    unwind_plan.SetSourceName ("assembly insn profiling");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolYes);

    return true;
}

bool
AssemblyParse_x86::augment_unwind_plan_from_call_site (AddressRange& func, UnwindPlan &unwind_plan)
{
    // Is func address valid?
    Address addr_start = func.GetBaseAddress();
    if (!addr_start.IsValid())
        return false;

    // Is original unwind_plan valid?
    // unwind_plan should have at least one row which is ABI-default (CFA register is sp),
    // and another row in mid-function.
    if (unwind_plan.GetRowCount() < 2)
        return false;
    UnwindPlan::RowSP first_row = unwind_plan.GetRowAtIndex (0);
    if (first_row->GetOffset() != 0)
        return false;
    uint32_t cfa_reg = m_exe_ctx.GetThreadPtr()->GetRegisterContext()
                       ->ConvertRegisterKindToRegisterNumber (unwind_plan.GetRegisterKind(),
                                                              first_row->GetCFAValue().GetRegisterNumber());
    if (cfa_reg != m_lldb_sp_regnum || first_row->GetCFAValue().GetOffset() != m_wordsize)
        return false;

    UnwindPlan::RowSP original_last_row = unwind_plan.GetRowForFunctionOffset (-1);

    Target *target = m_exe_ctx.GetTargetPtr();
    m_cur_insn = func.GetBaseAddress();
    uint64_t offset = 0;
    int row_id = 1;
    bool unwind_plan_updated = false;
    UnwindPlan::RowSP row(new UnwindPlan::Row(*first_row));

    // After a mid-function epilogue we will need to re-insert the original unwind rules
    // so unwinds work for the remainder of the function.  These aren't common with clang/gcc
    // on x86 but it is possible.
    bool reinstate_unwind_state = false;

    while (func.ContainsFileAddress (m_cur_insn))
    {
        int insn_len;
        if (!instruction_length (m_cur_insn, insn_len)
            || insn_len == 0 || insn_len > kMaxInstructionByteSize)
        {
            // An unrecognized/junk instruction.
            break;
        }
        const bool prefer_file_cache = true;
        Error error;
        if (target->ReadMemory (m_cur_insn, prefer_file_cache, m_cur_insn_bytes,
                                insn_len, error) == static_cast<size_t>(-1))
        {
           // Error reading the instruction out of the file, stop scanning.
           break;
        }

        // Advance offsets.
        offset += insn_len;
        m_cur_insn.SetOffset(m_cur_insn.GetOffset() + insn_len);

        if (reinstate_unwind_state)
        {
            // that was the last instruction of this function
            if (func.ContainsFileAddress (m_cur_insn) == false)
                continue;

            UnwindPlan::RowSP new_row(new UnwindPlan::Row());
            *new_row = *original_last_row;
            new_row->SetOffset (offset);
            unwind_plan.AppendRow (new_row);
            row.reset (new UnwindPlan::Row());
            *row = *new_row;
            reinstate_unwind_state = false;
            unwind_plan_updated = true;
            continue;
        }

        // If we already have one row for this instruction, we can continue.
        while (row_id < unwind_plan.GetRowCount()
               && unwind_plan.GetRowAtIndex (row_id)->GetOffset() <= offset)
        {
            row_id++;
        }
        UnwindPlan::RowSP original_row = unwind_plan.GetRowAtIndex (row_id - 1);
        if (original_row->GetOffset() == offset)
        {
            *row = *original_row;
            continue;
        }

        if (row_id == 0)
        {
            // If we are here, compiler didn't generate CFI for prologue.
            // This won't happen to GCC or clang.
            // In this case, bail out directly.
            return false;
        }

        // Inspect the instruction to check if we need a new row for it.
        cfa_reg = m_exe_ctx.GetThreadPtr()->GetRegisterContext()
                  ->ConvertRegisterKindToRegisterNumber (unwind_plan.GetRegisterKind(),
                                                         row->GetCFAValue().GetRegisterNumber());
        if (cfa_reg == m_lldb_sp_regnum)
        {
            // CFA register is sp.

            // call next instruction
            //     call 0
            //  => pop  %ebx
            if (call_next_insn_pattern_p ())
            {
                row->SetOffset (offset);
                row->GetCFAValue().IncOffset (m_wordsize);

                UnwindPlan::RowSP new_row(new UnwindPlan::Row(*row));
                unwind_plan.InsertRow (new_row);
                unwind_plan_updated = true;
                continue;
            }

            // push/pop register
            int regno;
            if (push_reg_p (regno))
            {
                row->SetOffset (offset);
                row->GetCFAValue().IncOffset (m_wordsize);

                UnwindPlan::RowSP new_row(new UnwindPlan::Row(*row));
                unwind_plan.InsertRow (new_row);
                unwind_plan_updated = true;
                continue;
            }
            if (pop_reg_p (regno))
            {
                // Technically, this might be a nonvolatile register recover in epilogue.
                // We should reset RegisterInfo for the register.
                // But in practice, previous rule for the register is still valid...
                // So we ignore this case.

                row->SetOffset (offset);
                row->GetCFAValue().IncOffset (-m_wordsize);

                UnwindPlan::RowSP new_row(new UnwindPlan::Row(*row));
                unwind_plan.InsertRow (new_row);
                unwind_plan_updated = true;
                continue;
            }

            // push imm
            if (push_imm_pattern_p ())
            {
                row->SetOffset (offset);
                row->GetCFAValue().IncOffset (m_wordsize);
                UnwindPlan::RowSP new_row(new UnwindPlan::Row(*row));
                unwind_plan.InsertRow (new_row);
                unwind_plan_updated = true;
                continue;
            }

            // add/sub %rsp/%esp
            int amount;
            if (add_rsp_pattern_p (amount))
            {
                row->SetOffset (offset);
                row->GetCFAValue().IncOffset (-amount);

                UnwindPlan::RowSP new_row(new UnwindPlan::Row(*row));
                unwind_plan.InsertRow (new_row);
                unwind_plan_updated = true;
                continue;
            }
            if (sub_rsp_pattern_p (amount))
            {
                row->SetOffset (offset);
                row->GetCFAValue().IncOffset (amount);

                UnwindPlan::RowSP new_row(new UnwindPlan::Row(*row));
                unwind_plan.InsertRow (new_row);
                unwind_plan_updated = true;
                continue;
            }

            // lea %rsp, [%rsp + $offset]
            if (lea_rsp_pattern_p (amount))
            {
                row->SetOffset (offset);
                row->GetCFAValue().IncOffset (-amount);

                UnwindPlan::RowSP new_row(new UnwindPlan::Row(*row));
                unwind_plan.InsertRow (new_row);
                unwind_plan_updated = true;
                continue;
            }

            if (ret_pattern_p ())
            {
                reinstate_unwind_state = true;
                continue;
            }
        }
        else if (cfa_reg == m_lldb_fp_regnum)
        {
            // CFA register is fp.

            // The only case we care about is epilogue:
            //     [0x5d] pop %rbp/%ebp
            //  => [0xc3] ret
            if (pop_rbp_pattern_p () || leave_pattern_p ())
            {
                if (target->ReadMemory (m_cur_insn, prefer_file_cache, m_cur_insn_bytes,
                                        1, error) != static_cast<size_t>(-1)
                    && ret_pattern_p ())
                {
                    row->SetOffset (offset);
                    row->GetCFAValue().SetIsRegisterPlusOffset (first_row->GetCFAValue().GetRegisterNumber(), 
                                                                m_wordsize);

                    UnwindPlan::RowSP new_row(new UnwindPlan::Row(*row));
                    unwind_plan.InsertRow (new_row);
                    unwind_plan_updated = true;
                    reinstate_unwind_state = true;
                    continue;
                }
            }
        }
        else
        {
            // CFA register is not sp or fp.

            // This must be hand-written assembly.
            // Just trust eh_frame and assume we have finished.
            break;
        }
    }

    unwind_plan.SetPlanValidAddressRange (func);
    if (unwind_plan_updated)
    {
        std::string unwind_plan_source (unwind_plan.GetSourceName().AsCString());
        unwind_plan_source += " plus augmentation from assembly parsing";
        unwind_plan.SetSourceName (unwind_plan_source.c_str());
        unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
        unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolYes);
    }
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
    if (target->ReadMemory (func.GetBaseAddress(), prefer_file_cache, bytebuf,
                            sizeof (bytebuf), error) == static_cast<size_t>(-1))
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
    row->GetCFAValue().SetIsRegisterPlusOffset (m_lldb_sp_regnum, m_wordsize);
    row->SetOffset (0);
    unwind_plan.AppendRow (row);
    UnwindPlan::Row *newrow = new UnwindPlan::Row;
    *newrow = *row.get();
    row.reset(newrow);

    // push %rbp has executed - stack moved, rbp now saved
    row->GetCFAValue().IncOffset (m_wordsize);
    fp_reginfo.SetAtCFAPlusOffset (2 * -m_wordsize);
    row->SetRegisterInfo (m_lldb_fp_regnum, fp_reginfo);
    row->SetOffset (1);
    unwind_plan.AppendRow (row);

    newrow = new UnwindPlan::Row;
    *newrow = *row.get();
    row.reset(newrow);

    // mov %rsp, %rbp has executed
    row->GetCFAValue().SetIsRegisterPlusOffset (m_lldb_fp_regnum, 2 * m_wordsize);
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
        if (target->ReadMemory (m_cur_insn, prefer_file_cache, m_cur_insn_bytes,
                                insn_len, error) == static_cast<size_t>(-1))
        {
           // Error reading the instruction out of the file, stop scanning
           break;
        }

        if (push_rbp_pattern_p () || mov_rsp_rbp_pattern_p () || sub_rsp_pattern_p (offset)
            || push_reg_p (regno) || mov_reg_to_local_stack_frame_p (regno, offset)
            || (lea_rsp_pattern_p (offset) && offset < 0))
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
UnwindAssembly_x86::AugmentUnwindPlanFromCallSite (AddressRange& func, Thread& thread, UnwindPlan& unwind_plan)
{
    bool do_augment_unwindplan = true;

    UnwindPlan::RowSP first_row = unwind_plan.GetRowForFunctionOffset (0);
    UnwindPlan::RowSP last_row = unwind_plan.GetRowForFunctionOffset (-1);
    
    int wordsize = 8;
    ProcessSP process_sp (thread.GetProcess());
    if (process_sp)
    {
        wordsize = process_sp->GetTarget().GetArchitecture().GetAddressByteSize();
    }

    RegisterNumber sp_regnum (thread, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    RegisterNumber pc_regnum (thread, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);

    // Does this UnwindPlan describe the prologue?  I want to see that the CFA is set
    // in terms of the stack pointer plus an offset, and I want to see that rip is 
    // retrieved at the CFA-wordsize.
    // If there is no description of the prologue, don't try to augment this eh_frame
    // unwinder code, fall back to assembly parsing instead.

    if (first_row->GetCFAValue().GetValueType() != UnwindPlan::Row::CFAValue::isRegisterPlusOffset
        || RegisterNumber (thread, unwind_plan.GetRegisterKind(),
            first_row->GetCFAValue().GetRegisterNumber()) != sp_regnum
        || first_row->GetCFAValue().GetOffset() != wordsize)
    {
        return false;
    }
    UnwindPlan::Row::RegisterLocation first_row_pc_loc;
    if (first_row->GetRegisterInfo (pc_regnum.GetAsKind (unwind_plan.GetRegisterKind()), first_row_pc_loc) == false
        || first_row_pc_loc.IsAtCFAPlusOffset() == false
        || first_row_pc_loc.GetOffset() != -wordsize)
    {
            return false;
    }


    // It looks like the prologue is described.  
    // Is the epilogue described?  If it is, no need to do any augmentation.

    if (first_row != last_row && first_row->GetOffset() != last_row->GetOffset())
    {
        // The first & last row have the same CFA register
        // and the same CFA offset value
        // and the CFA register is esp/rsp (the stack pointer).

        // We're checking that both of them have an unwind rule like "CFA=esp+4" or CFA+rsp+8".

        if (first_row->GetCFAValue().GetValueType() == last_row->GetCFAValue().GetValueType()
            && first_row->GetCFAValue().GetRegisterNumber() == last_row->GetCFAValue().GetRegisterNumber()
            && first_row->GetCFAValue().GetOffset() == last_row->GetCFAValue().GetOffset())
        {
            // Get the register locations for eip/rip from the first & last rows.
            // Are they both CFA plus an offset?  Is it the same offset?

            UnwindPlan::Row::RegisterLocation last_row_pc_loc;
            if (last_row->GetRegisterInfo (pc_regnum.GetAsKind (unwind_plan.GetRegisterKind()), last_row_pc_loc))
            {
                if (last_row_pc_loc.IsAtCFAPlusOffset()
                    && first_row_pc_loc.GetOffset() == last_row_pc_loc.GetOffset())
                {
            
                    // One last sanity check:  Is the unwind rule for getting the caller pc value
                    // "deref the CFA-4" or "deref the CFA-8"? 

                    // If so, we have an UnwindPlan that already describes the epilogue and we don't need
                    // to modify it at all.

                    if (first_row_pc_loc.GetOffset() == -wordsize)
                    {
                        do_augment_unwindplan = false;
                    }
                }
            }
        }
    }

    if (do_augment_unwindplan)
    {
        ExecutionContext exe_ctx (thread.shared_from_this());
        AssemblyParse_x86 asm_parse(exe_ctx, m_cpu, m_arch, func);
        return asm_parse.augment_unwind_plan_from_call_site (func, unwind_plan);
    }
    
    return false;
}

bool
UnwindAssembly_x86::GetFastUnwindPlan (AddressRange& func, Thread& thread, UnwindPlan &unwind_plan)
{
    // if prologue is
    //   55     pushl %ebp
    //   89 e5  movl %esp, %ebp
    //  or
    //   55        pushq %rbp
    //   48 89 e5  movq %rsp, %rbp

    // We should pull in the ABI architecture default unwind plan and return that

    llvm::SmallVector <uint8_t, 4> opcode_data;

    ProcessSP process_sp = thread.GetProcess();
    if (process_sp)
    {
        Target &target (process_sp->GetTarget());
        const bool prefer_file_cache = true;
        Error error;
        if (target.ReadMemory (func.GetBaseAddress (), prefer_file_cache, opcode_data.data(),
                               4, error) == 4)
        {
            uint8_t i386_push_mov[] = {0x55, 0x89, 0xe5};
            uint8_t x86_64_push_mov[] = {0x55, 0x48, 0x89, 0xe5};

            if (memcmp (opcode_data.data(), i386_push_mov, sizeof (i386_push_mov)) == 0
                || memcmp (opcode_data.data(), x86_64_push_mov, sizeof (x86_64_push_mov)) == 0)
            {
                ABISP abi_sp = process_sp->GetABI();
                if (abi_sp)
                {
                    return abi_sp->CreateDefaultUnwindPlan (unwind_plan);
                }
            }
        }
    }
    return false;
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

ConstString
UnwindAssembly_x86::GetPluginName()
{
    return GetPluginNameStatic();
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


lldb_private::ConstString
UnwindAssembly_x86::GetPluginNameStatic()
{
    static ConstString g_name("x86");
    return g_name;
}

const char *
UnwindAssembly_x86::GetPluginDescriptionStatic()
{
    return "i386 and x86_64 assembly language profiler plugin.";
}
