//===-- DisassemblerLLVMC.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DisassemblerLLVMC_h_
#define liblldb_DisassemblerLLVMC_h_


#include "llvm-c/Disassembler.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Mutex.h"

class InstructionLLVMC;

class DisassemblerLLVMC : public lldb_private::Disassembler
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();
    
    static void
    Terminate();
    
    static const char *
    GetPluginNameStatic();
    
    static const char *
    GetPluginDescriptionStatic();
    
    static lldb_private::Disassembler *
    CreateInstance(const lldb_private::ArchSpec &arch);
    
    
    DisassemblerLLVMC(const lldb_private::ArchSpec &arch);
    
    virtual
    ~DisassemblerLLVMC();
    
    size_t
    DecodeInstructions (const lldb_private::Address &base_addr,
                        const lldb_private::DataExtractor& data,
                        uint32_t data_offset,
                        uint32_t num_instructions,
                        bool append);
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
protected:
    friend class InstructionLLVMC;
    
    bool
    IsValid()
    {
        return (m_disasm_context != NULL);
    }
    
    int OpInfo(uint64_t PC,
               uint64_t Offset,
               uint64_t Size,
               int TagType,
               void *TagBug);
    
    const char *SymbolLookup (uint64_t ReferenceValue,
                              uint64_t *ReferenceType,
                              uint64_t ReferencePC,
                              const char **ReferenceName);
    
    static int OpInfoCallback (void *DisInfo,
                               uint64_t PC,
                               uint64_t Offset,
                               uint64_t Size,
                               int TagType,
                               void *TagBug);
    
    static const char *SymbolLookupCallback(void *DisInfo,
                                            uint64_t ReferenceValue,
                                            uint64_t *ReferenceType,
                                            uint64_t ReferencePC,
                                            const char **ReferenceName);
    
    void Lock(InstructionLLVMC *inst, 
              const lldb_private::ExecutionContext *exe_ctx)
    {
        m_mutex.Lock();
        m_inst = inst;
        m_exe_ctx = exe_ctx;
    }
    
    void Unlock()
    {
        m_inst = NULL;
        m_exe_ctx = NULL;
        m_mutex.Unlock();
    }
    
    const lldb_private::ExecutionContext *m_exe_ctx;
    InstructionLLVMC *m_inst;
    lldb_private::Mutex m_mutex;
    ::LLVMDisasmContextRef m_disasm_context;
    ::LLVMDisasmContextRef m_alternate_disasm_context;
};

#endif  // liblldb_DisassemblerLLVM_h_
