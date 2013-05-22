//===-- DisassemblerLLVMC.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DisassemblerLLVMC.h"

#include "llvm-c/Disassembler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ADT/SmallString.h"


#include "lldb/Core/Address.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/StackFrame.h"

#include <regex.h>

using namespace lldb;
using namespace lldb_private;

class InstructionLLVMC : public lldb_private::Instruction
{
public:
    InstructionLLVMC (DisassemblerLLVMC &disasm,
                      const lldb_private::Address &address, 
                      AddressClass addr_class) :
        Instruction (address, addr_class),
        m_disasm_sp (disasm.shared_from_this()),
        m_does_branch (eLazyBoolCalculate),
        m_is_valid (false),
        m_using_file_addr (false)
    {
    }
    
    virtual
    ~InstructionLLVMC ()
    {
    }
    
    virtual bool
    DoesBranch ()
    {
        if (m_does_branch == eLazyBoolCalculate)
        {
            GetDisassemblerLLVMC().Lock(this, NULL);
            DataExtractor data;
            if (m_opcode.GetData(data))
            {
                bool is_alternate_isa;
                lldb::addr_t pc = m_address.GetFileAddress();

                DisassemblerLLVMC::LLVMCDisassembler *mc_disasm_ptr = GetDisasmToUse (is_alternate_isa);
                const uint8_t *opcode_data = data.GetDataStart();
                const size_t opcode_data_len = data.GetByteSize();
                llvm::MCInst inst;
                const size_t inst_size = mc_disasm_ptr->GetMCInst (opcode_data,
                                                                   opcode_data_len,
                                                                   pc,
                                                                   inst);
                // Be conservative, if we didn't understand the instruction, say it might branch...
                if (inst_size == 0)
                    m_does_branch = eLazyBoolYes;
                else
                {
                    const bool can_branch = mc_disasm_ptr->CanBranch(inst);
                    if (can_branch)
                        m_does_branch = eLazyBoolYes;
                    else
                        m_does_branch = eLazyBoolNo;
                }
            }
            GetDisassemblerLLVMC().Unlock();
        }
        return m_does_branch == eLazyBoolYes;
    }
    
    DisassemblerLLVMC::LLVMCDisassembler *
    GetDisasmToUse (bool &is_alternate_isa)
    {
        is_alternate_isa = false;
        DisassemblerLLVMC &llvm_disasm = GetDisassemblerLLVMC();
        if (llvm_disasm.m_alternate_disasm_ap.get() != NULL)
        {
            const AddressClass address_class = GetAddressClass ();
        
            if (address_class == eAddressClassCodeAlternateISA)
            {
                is_alternate_isa = true;
                return llvm_disasm.m_alternate_disasm_ap.get();
            }
        }
        return llvm_disasm.m_disasm_ap.get();
    }
    
    virtual size_t
    Decode (const lldb_private::Disassembler &disassembler,
            const lldb_private::DataExtractor &data,
            lldb::offset_t data_offset)
    {
        // All we have to do is read the opcode which can be easy for some
        // architetures
        bool got_op = false;
        DisassemblerLLVMC &llvm_disasm = GetDisassemblerLLVMC();
        const ArchSpec &arch = llvm_disasm.GetArchitecture();
        
        const uint32_t min_op_byte_size = arch.GetMinimumOpcodeByteSize();
        const uint32_t max_op_byte_size = arch.GetMaximumOpcodeByteSize();
        if (min_op_byte_size == max_op_byte_size)
        {
            // Fixed size instructions, just read that amount of data.
            if (!data.ValidOffsetForDataOfSize(data_offset, min_op_byte_size))
                return false;
            
            switch (min_op_byte_size)
            {
                case 1:
                    m_opcode.SetOpcode8  (data.GetU8  (&data_offset));
                    got_op = true;
                    break;

                case 2:
                    m_opcode.SetOpcode16 (data.GetU16 (&data_offset));
                    got_op = true;
                    break;

                case 4:
                    m_opcode.SetOpcode32 (data.GetU32 (&data_offset));
                    got_op = true;
                    break;

                case 8:
                    m_opcode.SetOpcode64 (data.GetU64 (&data_offset));
                    got_op = true;
                    break;

                default:
                    m_opcode.SetOpcodeBytes(data.PeekData(data_offset, min_op_byte_size), min_op_byte_size);
                    got_op = true;
                    break;
            }
        }
        if (!got_op)
        {
            bool is_alternate_isa = false;
            DisassemblerLLVMC::LLVMCDisassembler *mc_disasm_ptr = GetDisasmToUse (is_alternate_isa);
            
            const llvm::Triple::ArchType machine = arch.GetMachine();
            if (machine == llvm::Triple::arm || machine == llvm::Triple::thumb)
            {
                if (machine == llvm::Triple::thumb || is_alternate_isa)
                {
                    uint32_t thumb_opcode = data.GetU16(&data_offset);
                    if ((thumb_opcode & 0xe000) != 0xe000 || ((thumb_opcode & 0x1800u) == 0))
                    {
                        m_opcode.SetOpcode16 (thumb_opcode);
                        m_is_valid = true;
                    }
                    else
                    {
                        thumb_opcode <<= 16;
                        thumb_opcode |= data.GetU16(&data_offset);
                        m_opcode.SetOpcode16_2 (thumb_opcode);
                        m_is_valid = true;
                    }
                }
                else
                {
                    m_opcode.SetOpcode32 (data.GetU32(&data_offset));
                    m_is_valid = true;
                }
            }
            else
            {
                // The opcode isn't evenly sized, so we need to actually use the llvm
                // disassembler to parse it and get the size.
                uint8_t *opcode_data = const_cast<uint8_t *>(data.PeekData (data_offset, 1));
                const size_t opcode_data_len = data.BytesLeft(data_offset);
                const addr_t pc = m_address.GetFileAddress();
                llvm::MCInst inst;
                
                llvm_disasm.Lock(this, NULL);
                const size_t inst_size = mc_disasm_ptr->GetMCInst(opcode_data,
                                                                  opcode_data_len,
                                                                  pc,
                                                                  inst);
                llvm_disasm.Unlock();
                if (inst_size == 0)
                    m_opcode.Clear();
                else
                {
                    m_opcode.SetOpcodeBytes(opcode_data, inst_size);
                    m_is_valid = true;
                }
            }
        }
        return m_opcode.GetByteSize();
    }
    
    void
    AppendComment (std::string &description)
    {
        if (m_comment.empty())
            m_comment.swap (description);
        else
        {
            m_comment.append(", ");
            m_comment.append(description);
        }
    }
    
    virtual void
    CalculateMnemonicOperandsAndComment (const lldb_private::ExecutionContext *exe_ctx)
    {
        DataExtractor data;
        const AddressClass address_class = GetAddressClass ();

        if (m_opcode.GetData(data))
        {
            char out_string[512];
            
            DisassemblerLLVMC &llvm_disasm = GetDisassemblerLLVMC();

            DisassemblerLLVMC::LLVMCDisassembler *mc_disasm_ptr;
            
            if (address_class == eAddressClassCodeAlternateISA)
                mc_disasm_ptr = llvm_disasm.m_alternate_disasm_ap.get();
            else
                mc_disasm_ptr = llvm_disasm.m_disasm_ap.get();
            
            lldb::addr_t pc = m_address.GetFileAddress();
            m_using_file_addr = true;
            
            const bool data_from_file = GetDisassemblerLLVMC().m_data_from_file;
            if (!data_from_file)
            {
                if (exe_ctx)
                {
                    Target *target = exe_ctx->GetTargetPtr();
                    if (target)
                    {
                        const lldb::addr_t load_addr = m_address.GetLoadAddress(target);
                        if (load_addr != LLDB_INVALID_ADDRESS)
                        {
                            pc = load_addr;
                            m_using_file_addr = false;
                        }
                    }
                }
            }
            
            llvm_disasm.Lock(this, exe_ctx);
            
            const uint8_t *opcode_data = data.GetDataStart();
            const size_t opcode_data_len = data.GetByteSize();
            llvm::MCInst inst;
            size_t inst_size = mc_disasm_ptr->GetMCInst (opcode_data,
                                                         opcode_data_len,
                                                         pc,
                                                         inst);
                
            if (inst_size > 0)
                mc_disasm_ptr->PrintMCInst(inst, out_string, sizeof(out_string));
            
            llvm_disasm.Unlock();
            
            if (inst_size == 0)
            {
                m_comment.assign ("unknown opcode");
                inst_size = m_opcode.GetByteSize();
                StreamString mnemonic_strm;
                lldb::offset_t offset = 0;
                switch (inst_size)
                {
                    case 1:
                        {
                            const uint8_t uval8 = data.GetU8 (&offset);
                            m_opcode.SetOpcode8 (uval8);
                            m_opcode_name.assign (".byte");
                            mnemonic_strm.Printf("0x%2.2x", uval8);
                        }
                        break;
                    case 2:
                        {
                            const uint16_t uval16 = data.GetU16(&offset);
                            m_opcode.SetOpcode16(uval16);
                            m_opcode_name.assign (".short");
                            mnemonic_strm.Printf("0x%4.4x", uval16);
                        }
                        break;
                    case 4:
                        {
                            const uint32_t uval32 = data.GetU32(&offset);
                            m_opcode.SetOpcode32(uval32);
                            m_opcode_name.assign (".long");
                            mnemonic_strm.Printf("0x%8.8x", uval32);
                        }
                        break;
                    case 8:
                        {
                            const uint64_t uval64 = data.GetU64(&offset);
                            m_opcode.SetOpcode64(uval64);
                            m_opcode_name.assign (".quad");
                            mnemonic_strm.Printf("0x%16.16" PRIx64, uval64);
                        }
                        break;
                    default:
                        if (inst_size == 0)
                            return;
                        else
                        {
                            const uint8_t *bytes = data.PeekData(offset, inst_size);
                            if (bytes == NULL)
                                return;
                            m_opcode_name.assign (".byte");
                            m_opcode.SetOpcodeBytes(bytes, inst_size);
                            mnemonic_strm.Printf("0x%2.2x", bytes[0]);
                            for (uint32_t i=1; i<inst_size; ++i)
                                mnemonic_strm.Printf(" 0x%2.2x", bytes[i]);
                        }
                        break;
                }
                m_mnemonics.swap(mnemonic_strm.GetString());
                return;
            }
            else
            {
                if (m_does_branch == eLazyBoolCalculate)
                {
                    const bool can_branch = mc_disasm_ptr->CanBranch(inst);
                    if (can_branch)
                        m_does_branch = eLazyBoolYes;
                    else
                        m_does_branch = eLazyBoolNo;

                }
            }
            
            if (!s_regex_compiled)
            {
                ::regcomp(&s_regex, "[ \t]*([^ ^\t]+)[ \t]*([^ ^\t].*)?", REG_EXTENDED);
                s_regex_compiled = true;
            }
            
            ::regmatch_t matches[3];
            
            if (!::regexec(&s_regex, out_string, sizeof(matches) / sizeof(::regmatch_t), matches, 0))
            {
                if (matches[1].rm_so != -1)
                    m_opcode_name.assign(out_string + matches[1].rm_so, matches[1].rm_eo - matches[1].rm_so);
                if (matches[2].rm_so != -1)
                    m_mnemonics.assign(out_string + matches[2].rm_so, matches[2].rm_eo - matches[2].rm_so);
            }
        }
    }
    
    bool
    IsValid () const
    {
        return m_is_valid;
    }
    
    bool
    UsingFileAddress() const
    {
        return m_using_file_addr;
    }
    size_t
    GetByteSize () const
    {
        return m_opcode.GetByteSize();
    }
    
    DisassemblerLLVMC &
    GetDisassemblerLLVMC ()
    {
        return *(DisassemblerLLVMC *)m_disasm_sp.get();
    }
protected:
    
    DisassemblerSP          m_disasm_sp; // for ownership
    LazyBool                m_does_branch;
    bool                    m_is_valid;
    bool                    m_using_file_addr;
    
    static bool             s_regex_compiled;
    static ::regex_t        s_regex;
};

bool InstructionLLVMC::s_regex_compiled = false;
::regex_t InstructionLLVMC::s_regex;

DisassemblerLLVMC::LLVMCDisassembler::LLVMCDisassembler (const char *triple, unsigned flavor, DisassemblerLLVMC &owner):
    m_is_valid(true)
{
    std::string Error;
    const llvm::Target *curr_target = llvm::TargetRegistry::lookupTarget(triple, Error);
    if (!curr_target)
    {
        m_is_valid = false;
        return;
    }
    
    m_instr_info_ap.reset(curr_target->createMCInstrInfo());
    m_reg_info_ap.reset (curr_target->createMCRegInfo(triple));
    
    std::string features_str;

    m_subtarget_info_ap.reset(curr_target->createMCSubtargetInfo(triple, "",
                                                                features_str));
    
    m_asm_info_ap.reset(curr_target->createMCAsmInfo(triple));

    if (m_instr_info_ap.get() == NULL || m_reg_info_ap.get() == NULL || m_subtarget_info_ap.get() == NULL || m_asm_info_ap.get() == NULL)
    {
        m_is_valid = false;
        return;
    }
    
    m_context_ap.reset(new llvm::MCContext(*m_asm_info_ap.get(), *(m_reg_info_ap.get()), 0));
    
    m_disasm_ap.reset(curr_target->createMCDisassembler(*m_subtarget_info_ap.get()));
    if (m_disasm_ap.get())
    {
        m_disasm_ap->setupForSymbolicDisassembly(NULL,
                                                  DisassemblerLLVMC::SymbolLookupCallback,
                                                  (void *) &owner,
                                                  m_context_ap.get());
        
        unsigned asm_printer_variant;
        if (flavor == ~0U)
            asm_printer_variant = m_asm_info_ap->getAssemblerDialect();
        else
        {
            asm_printer_variant = flavor;
        }
        
        m_instr_printer_ap.reset(curr_target->createMCInstPrinter(asm_printer_variant,
                                                                  *m_asm_info_ap.get(),
                                                                  *m_instr_info_ap.get(),
                                                                  *m_reg_info_ap.get(),
                                                                  *m_subtarget_info_ap.get()));
        if (m_instr_printer_ap.get() == NULL)
        {
            m_disasm_ap.reset();
            m_is_valid = false;
        }
    }
    else
        m_is_valid = false;
}

DisassemblerLLVMC::LLVMCDisassembler::~LLVMCDisassembler()
{
}

namespace {
    // This is the memory object we use in GetInstruction.
    class LLDBDisasmMemoryObject : public llvm::MemoryObject {
      const uint8_t *m_bytes;
      uint64_t m_size;
      uint64_t m_base_PC;
    public:
      LLDBDisasmMemoryObject(const uint8_t *bytes, uint64_t size, uint64_t basePC) :
                         m_bytes(bytes), m_size(size), m_base_PC(basePC) {}
     
      uint64_t getBase() const { return m_base_PC; }
      uint64_t getExtent() const { return m_size; }

      int readByte(uint64_t addr, uint8_t *byte) const {
        if (addr - m_base_PC >= m_size)
          return -1;
        *byte = m_bytes[addr - m_base_PC];
        return 0;
      }
    };
} // End Anonymous Namespace

uint64_t
DisassemblerLLVMC::LLVMCDisassembler::GetMCInst (const uint8_t *opcode_data,
                                                 size_t opcode_data_len,
                                                 lldb::addr_t pc,
                                                 llvm::MCInst &mc_inst)
{
    LLDBDisasmMemoryObject memory_object (opcode_data, opcode_data_len, pc);
    llvm::MCDisassembler::DecodeStatus status;

    uint64_t new_inst_size;
    status = m_disasm_ap->getInstruction(mc_inst,
                                         new_inst_size,
                                         memory_object,
                                         pc,
                                         llvm::nulls(),
                                         llvm::nulls());
    if (status == llvm::MCDisassembler::Success)
        return new_inst_size;
    else
        return 0;
}

uint64_t
DisassemblerLLVMC::LLVMCDisassembler::PrintMCInst (llvm::MCInst &mc_inst,
                                                   char *dst,
                                                   size_t dst_len)
{
    llvm::StringRef unused_annotations;
    llvm::SmallString<64> inst_string;
    llvm::raw_svector_ostream inst_stream(inst_string);
    m_instr_printer_ap->printInst (&mc_inst, inst_stream, unused_annotations);
    inst_stream.flush();
    const size_t output_size = std::min(dst_len - 1, inst_string.size());
    std::memcpy(dst, inst_string.data(), output_size);
    dst[output_size] = '\0';
    
    return output_size;
}

bool
DisassemblerLLVMC::LLVMCDisassembler::CanBranch (llvm::MCInst &mc_inst)
{
    return m_instr_info_ap->get(mc_inst.getOpcode()).mayAffectControlFlow(mc_inst, *m_reg_info_ap.get());
}

bool
DisassemblerLLVMC::FlavorValidForArchSpec (const lldb_private::ArchSpec &arch, const char *flavor)
{
    llvm::Triple triple = arch.GetTriple();
    if (flavor == NULL || strcmp (flavor, "default") == 0)
        return true;
    
    if (triple.getArch() == llvm::Triple::x86 || triple.getArch() == llvm::Triple::x86_64)
    {
        if (strcmp (flavor, "intel") == 0 || strcmp (flavor, "att") == 0)
            return true;
        else
            return false;
    }
    else
        return false;
}
    

Disassembler *
DisassemblerLLVMC::CreateInstance (const ArchSpec &arch, const char *flavor)
{
    if (arch.GetTriple().getArch() != llvm::Triple::UnknownArch)
    {
        std::unique_ptr<DisassemblerLLVMC> disasm_ap (new DisassemblerLLVMC(arch, flavor));
    
        if (disasm_ap.get() && disasm_ap->IsValid())
            return disasm_ap.release();
    }
    return NULL;
}

DisassemblerLLVMC::DisassemblerLLVMC (const ArchSpec &arch, const char *flavor_string) :
    Disassembler(arch, flavor_string),
    m_exe_ctx (NULL),
    m_inst (NULL),
    m_data_from_file (false)
{
    if (!FlavorValidForArchSpec (arch, m_flavor.c_str()))
    {
        m_flavor.assign("default");
    }
    
    const char *triple = arch.GetTriple().getTriple().c_str();
    unsigned flavor = ~0U;
    
    // So far the only supported flavor is "intel" on x86.  The base class will set this
    // correctly coming in.
    if (arch.GetTriple().getArch() == llvm::Triple::x86
        || arch.GetTriple().getArch() == llvm::Triple::x86_64)
    {
        if (m_flavor == "intel")
        {
            flavor = 1;
        }
        else if (m_flavor == "att")
        {
            flavor = 0;
        }
    }
    
    m_disasm_ap.reset (new LLVMCDisassembler(triple, flavor, *this));
    if (!m_disasm_ap->IsValid())
    {
        // We use m_disasm_ap.get() to tell whether we are valid or not, so if this isn't good for some reason,
        // we reset it, and then we won't be valid and FindPlugin will fail and we won't get used.
        m_disasm_ap.reset();
    }
    
    if (arch.GetTriple().getArch() == llvm::Triple::arm)
    {
        ArchSpec thumb_arch(arch);
        std::string thumb_arch_name (thumb_arch.GetTriple().getArchName().str());
        // Replace "arm" with "thumb" so we get all thumb variants correct
        if (thumb_arch_name.size() > 3)
        {
            thumb_arch_name.erase(0,3);
            thumb_arch_name.insert(0, "thumb");
        }
        else
        {
            thumb_arch_name = "thumbv7";
        }
        thumb_arch.GetTriple().setArchName(llvm::StringRef(thumb_arch_name.c_str()));
        std::string thumb_triple(thumb_arch.GetTriple().getTriple());
        m_alternate_disasm_ap.reset(new LLVMCDisassembler(thumb_triple.c_str(), flavor, *this));
        if (!m_alternate_disasm_ap->IsValid())
        {
            m_disasm_ap.reset();
            m_alternate_disasm_ap.reset();
        }
    }
}

DisassemblerLLVMC::~DisassemblerLLVMC()
{
}

size_t
DisassemblerLLVMC::DecodeInstructions (const Address &base_addr,
                                       const DataExtractor& data,
                                       lldb::offset_t data_offset,
                                       size_t num_instructions,
                                       bool append,
                                       bool data_from_file)
{
    if (!append)
        m_instruction_list.Clear();
    
    if (!IsValid())
        return 0;
    
    m_data_from_file = data_from_file;
    uint32_t data_cursor = data_offset;
    const size_t data_byte_size = data.GetByteSize();
    uint32_t instructions_parsed = 0;
    Address inst_addr(base_addr);
    
    while (data_cursor < data_byte_size && instructions_parsed < num_instructions)
    {
        
        AddressClass address_class = eAddressClassCode;
        
        if (m_alternate_disasm_ap.get() != NULL)
            address_class = inst_addr.GetAddressClass ();
        
        InstructionSP inst_sp(new InstructionLLVMC(*this,
                                                   inst_addr, 
                                                   address_class));
        
        if (!inst_sp)
            break;
        
        uint32_t inst_size = inst_sp->Decode(*this, data, data_cursor);
                
        if (inst_size == 0)
            break;

        m_instruction_list.Append(inst_sp);
        data_cursor += inst_size;
        inst_addr.Slide(inst_size);
        instructions_parsed++;
    }
    
    return data_cursor - data_offset;
}

void
DisassemblerLLVMC::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "Disassembler that uses LLVM MC to disassemble i386, x86_64 and ARM.",
                                   CreateInstance);
    
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllDisassemblers();
}

void
DisassemblerLLVMC::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


ConstString
DisassemblerLLVMC::GetPluginNameStatic()
{
    static ConstString g_name("llvm-mc");
    return g_name;
}

int DisassemblerLLVMC::OpInfoCallback (void *disassembler,
                                       uint64_t pc,
                                       uint64_t offset,
                                       uint64_t size,
                                       int tag_type,
                                       void *tag_bug)
{
    return static_cast<DisassemblerLLVMC*>(disassembler)->OpInfo (pc,
                                                                  offset,
                                                                  size,
                                                                  tag_type,
                                                                  tag_bug);
}

const char *DisassemblerLLVMC::SymbolLookupCallback (void *disassembler,
                                                     uint64_t value,
                                                     uint64_t *type,
                                                     uint64_t pc,
                                                     const char **name)
{
    return static_cast<DisassemblerLLVMC*>(disassembler)->SymbolLookup(value,
                                                                       type,
                                                                       pc,
                                                                       name);
}

int DisassemblerLLVMC::OpInfo (uint64_t PC,
                               uint64_t Offset,
                               uint64_t Size,
                               int tag_type,
                               void *tag_bug)
{
    switch (tag_type)
    {
    default:
        break;
    case 1:
        bzero (tag_bug, sizeof(::LLVMOpInfo1));
        break;
    }
    return 0;
}

const char *DisassemblerLLVMC::SymbolLookup (uint64_t value,
                                             uint64_t *type_ptr,
                                             uint64_t pc,
                                             const char **name)
{
    if (*type_ptr)
    {
        if (m_exe_ctx && m_inst)
        {        
            //std::string remove_this_prior_to_checkin;
            Target *target = m_exe_ctx ? m_exe_ctx->GetTargetPtr() : NULL;
            Address value_so_addr;
            if (m_inst->UsingFileAddress())
            {
                ModuleSP module_sp(m_inst->GetAddress().GetModule());
                if (module_sp)
                    module_sp->ResolveFileAddress(value, value_so_addr);
            }
            else if (target && !target->GetSectionLoadList().IsEmpty())
            {
                target->GetSectionLoadList().ResolveLoadAddress(value, value_so_addr);
            }
            
            if (value_so_addr.IsValid() && value_so_addr.GetSection())
            {
                StreamString ss;
                
                value_so_addr.Dump (&ss,
                                    target,
                                    Address::DumpStyleResolvedDescriptionNoModule,
                                    Address::DumpStyleSectionNameOffset);
                
                if (!ss.GetString().empty())
                {
                    m_inst->AppendComment(ss.GetString());
                }
            }
        }
    }

    *type_ptr = LLVMDisassembler_ReferenceType_InOut_None;
    *name = NULL;
    return NULL;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
ConstString
DisassemblerLLVMC::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
DisassemblerLLVMC::GetPluginVersion()
{
    return 1;
}

