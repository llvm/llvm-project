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
#include "llvm/Support/TargetSelect.h"

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
        Instruction(address, addr_class),
        m_is_valid(false),
        m_disasm(disasm),
        m_disasm_sp(disasm.shared_from_this()),
        m_does_branch(eLazyBoolCalculate)
    {
    }
    
    virtual
    ~InstructionLLVMC ()
    {
    }
    
    static void
    PadToWidth (lldb_private::StreamString &ss,
                int new_width)
    {
        int old_width = ss.GetSize();
        
        if (old_width < new_width)
        {
            ss.Printf("%*s", new_width - old_width, "");
        }
    }
        
    virtual bool
    DoesBranch () const
    {
        return m_does_branch == eLazyBoolYes;
    }
    
    virtual size_t
    Decode (const lldb_private::Disassembler &disassembler,
            const lldb_private::DataExtractor &data,
            uint32_t data_offset)
    {
        // All we have to do is read the opcode which can be easy for some
        // architetures
        bool got_op = false;
        const ArchSpec &arch = m_disasm.GetArchitecture();
        
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
            ::LLVMDisasmContextRef disasm_context = m_disasm.m_disasm_context;
            
            bool is_altnernate_isa = false;
            if (m_disasm.m_alternate_disasm_context)
            {
                const AddressClass address_class = GetAddressClass ();
            
                if (address_class == eAddressClassCodeAlternateISA)
                {
                    disasm_context = m_disasm.m_alternate_disasm_context;
                    is_altnernate_isa = true;
                }
            }
            const llvm::Triple::ArchType machine = arch.GetMachine();
            if (machine == llvm::Triple::arm || machine == llvm::Triple::thumb)
            {
                if (machine == llvm::Triple::thumb || is_altnernate_isa)
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
                char out_string[512];
                m_disasm.Lock(this, NULL);
                uint8_t *opcode_data = const_cast<uint8_t *>(data.PeekData (data_offset, 1));
                const size_t opcode_data_len = data.GetByteSize() - data_offset;
                const addr_t pc = m_address.GetFileAddress();
                const size_t inst_size = ::LLVMDisasmInstruction (disasm_context,
                                                                  opcode_data,
                                                                  opcode_data_len,
                                                                  pc, // PC value
                                                                  out_string,
                                                                  sizeof(out_string));
                // The address lookup function could have caused us to fill in our comment
                m_comment.clear();
                m_disasm.Unlock();
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
            
            ::LLVMDisasmContextRef disasm_context;
            
            if (address_class == eAddressClassCodeAlternateISA)
                disasm_context = m_disasm.m_alternate_disasm_context;
            else
                disasm_context = m_disasm.m_disasm_context;
            
            lldb::addr_t pc = LLDB_INVALID_ADDRESS;
            
            if (exe_ctx)
            {
                Target *target = exe_ctx->GetTargetPtr();
                if (target)
                    pc = m_address.GetLoadAddress(target);
            }
            
            if (pc == LLDB_INVALID_ADDRESS)
                pc = m_address.GetFileAddress();
            
            m_disasm.Lock(this, exe_ctx);
            uint8_t *opcode_data = const_cast<uint8_t *>(data.PeekData (0, 1));
            const size_t opcode_data_len = data.GetByteSize();
            size_t inst_size = ::LLVMDisasmInstruction (disasm_context,
                                                        opcode_data,
                                                        opcode_data_len,
                                                        pc,
                                                        out_string,
                                                        sizeof(out_string));
            
            m_disasm.Unlock();
            
            if (inst_size == 0)
            {
                m_comment.assign ("unknown opcode");
                inst_size = m_opcode.GetByteSize();
                StreamString mnemonic_strm;
                uint32_t offset = 0;
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
                            mnemonic_strm.Printf("0x%16.16llx", uval64);
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
                m_mnemocics.swap(mnemonic_strm.GetString());
                return;
            }
            else
            {
                if (m_does_branch == eLazyBoolCalculate)
                {
                    if (StringRepresentsBranch (out_string, strlen(out_string)))
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
                    m_mnemocics.assign(out_string + matches[2].rm_so, matches[2].rm_eo - matches[2].rm_so);
            }
        }
    }
    
    bool
    IsValid ()
    {
        return m_is_valid;
    }
    
    size_t
    GetByteSize ()
    {
        return m_opcode.GetByteSize();
    }
protected:
    
    bool StringRepresentsBranch (const char *data, size_t size)
    {
        const char *cursor = data;

        bool inWhitespace = true;

        while (inWhitespace && cursor < data + size)
        {
            switch (*cursor)
            {
            default:
                inWhitespace = false;
                break;
            case ' ':
                break;
            case '\t':
                break;
            }
            
            if (inWhitespace)
                ++cursor;
        }
        
        if (cursor >= data + size)
            return false;
        
        llvm::Triple::ArchType arch = m_disasm.GetArchitecture().GetMachine();
        
        switch (arch)
        {
        default:
            return false;
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
            switch (cursor[0])
            {
            default:
                return false;
            case 'j':
                return true;
            case 'c':
                if (cursor[1] == 'a' &&
                    cursor[2] == 'l' &&
                    cursor[3] == 'l')
                    return true;
                else
                    return false;
            }
        case llvm::Triple::arm:
        case llvm::Triple::thumb:
            switch (cursor[0])
            {
            default:
                return false;
            case 'b':
                {
                    switch (cursor[1])
                    {
                    default:
                        return true;
                    case 'f':
                    case 'i':
                    case 'k':
                        return false;
                    }
                }
            case 'c':
                {
                    switch (cursor[1])
                    {
                    default:
                        return false;
                    case 'b':
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
    
    bool                    m_is_valid;
    DisassemblerLLVMC      &m_disasm;
    DisassemblerSP          m_disasm_sp; // for ownership
    LazyBool                m_does_branch;
    
    static bool             s_regex_compiled;
    static ::regex_t        s_regex;
};

bool InstructionLLVMC::s_regex_compiled = false;
::regex_t InstructionLLVMC::s_regex;

Disassembler *
DisassemblerLLVMC::CreateInstance (const ArchSpec &arch)
{
    if (arch.GetTriple().getArch() != llvm::Triple::UnknownArch)
    {
        std::auto_ptr<DisassemblerLLVMC> disasm_ap (new DisassemblerLLVMC(arch));
    
        if (disasm_ap.get() && disasm_ap->IsValid())
            return disasm_ap.release();
    }
    return NULL;
}

DisassemblerLLVMC::DisassemblerLLVMC (const ArchSpec &arch) :
    Disassembler(arch),
    m_exe_ctx (NULL),
    m_inst (NULL),
    m_disasm_context (NULL),
    m_alternate_disasm_context (NULL)
{
    m_disasm_context = ::LLVMCreateDisasm(arch.GetTriple().getTriple().c_str(), 
                                          (void*)this, 
                                          /*TagType=*/1,
                                          NULL,
                                          DisassemblerLLVMC::SymbolLookupCallback);
    
    if (arch.GetTriple().getArch() == llvm::Triple::arm)
    {
        ArchSpec thumb_arch(arch);
        thumb_arch.GetTriple().setArchName(llvm::StringRef("thumbv7"));
        std::string thumb_triple(thumb_arch.GetTriple().getTriple());

        m_alternate_disasm_context = ::LLVMCreateDisasm(thumb_triple.c_str(),
                                                        (void*)this, 
                                                        /*TagType=*/1,
                                                        NULL,
                                                        DisassemblerLLVMC::SymbolLookupCallback);
    }
}

DisassemblerLLVMC::~DisassemblerLLVMC()
{
    if (m_disasm_context)
    {
        ::LLVMDisasmDispose(m_disasm_context);
        m_disasm_context = NULL;
    }
    if (m_alternate_disasm_context)
    {
        ::LLVMDisasmDispose(m_alternate_disasm_context);
        m_alternate_disasm_context = NULL;
    }
}

size_t
DisassemblerLLVMC::DecodeInstructions (const Address &base_addr,
                                       const DataExtractor& data,
                                       uint32_t data_offset,
                                       uint32_t num_instructions,
                                       bool append)
{
    if (!append)
        m_instruction_list.Clear();
    
    if (!IsValid())
        return 0;
    
    uint32_t data_cursor = data_offset;
    const size_t data_byte_size = data.GetByteSize();
    uint32_t instructions_parsed = 0;
    Address inst_addr(base_addr);
    
    while (data_cursor < data_byte_size && instructions_parsed < num_instructions)
    {
        
        AddressClass address_class = eAddressClassCode;
        
        if (m_alternate_disasm_context)
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
                                   GetPluginDescriptionStatic(),
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


const char *
DisassemblerLLVMC::GetPluginNameStatic()
{
    return "llvm-mc";
}

const char *
DisassemblerLLVMC::GetPluginDescriptionStatic()
{
    return "Disassembler that uses LLVM MC to disassemble i386, x86_64 and ARM.";
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
            Address reference_address;
            
            Target *target = m_exe_ctx ? m_exe_ctx->GetTargetPtr() : NULL;
            
            if (target && !target->GetSectionLoadList().IsEmpty())
                target->GetSectionLoadList().ResolveLoadAddress(value, reference_address);
            else
            {
                ModuleSP module_sp(m_inst->GetAddress().GetModule());
                if (module_sp)
                    module_sp->ResolveFileAddress(value, reference_address);
            }
                
            if (reference_address.IsValid() && reference_address.GetSection())
            {
                StreamString ss;
                
                reference_address.Dump (&ss, 
                                        target, 
                                        Address::DumpStyleResolvedDescriptionNoModule, 
                                        Address::DumpStyleSectionNameOffset);
                
                if (!ss.GetString().empty())
                {
                    //remove_this_prior_to_checkin = ss.GetString();
                    //if (*type_ptr)
                    m_inst->AppendComment(ss.GetString());
                }
            }
            //printf ("DisassemblerLLVMC::SymbolLookup (value=0x%16.16llx, type=%llu, pc=0x%16.16llx, name=\"%s\") m_exe_ctx=%p, m_inst=%p\n", value, *type_ptr, pc, remove_this_prior_to_checkin.c_str(), m_exe_ctx, m_inst);
        }
    }

    *type_ptr = LLVMDisassembler_ReferenceType_InOut_None;
    *name = NULL;
    return NULL;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
DisassemblerLLVMC::GetPluginName()
{
    return "DisassemblerLLVMC";
}

const char *
DisassemblerLLVMC::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
DisassemblerLLVMC::GetPluginVersion()
{
    return 1;
}

