//===-- IRExecutionUnit.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
// Project includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Log.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"

using namespace lldb_private;

IRExecutionUnit::IRExecutionUnit (std::unique_ptr<llvm::LLVMContext> &context_ap,
                                  std::unique_ptr<llvm::Module> &module_ap,
                                  ConstString &name,
                                  const lldb::TargetSP &target_sp,
                                  std::vector<std::string> &cpu_features) :
    IRMemoryMap(target_sp),
    m_context_ap(context_ap.release()),
    m_module_ap(module_ap.release()),
    m_module(m_module_ap.get()),
    m_cpu_features(cpu_features),
    m_name(name),
    m_did_jit(false),
    m_function_load_addr(LLDB_INVALID_ADDRESS),
    m_function_end_load_addr(LLDB_INVALID_ADDRESS)
{
}

lldb::addr_t
IRExecutionUnit::WriteNow (const uint8_t *bytes,
                           size_t size,
                           Error &error)
{    
    lldb::addr_t allocation_process_addr = Malloc (size,
                                                   8,
                                                   lldb::ePermissionsWritable | lldb::ePermissionsReadable,
                                                   eAllocationPolicyMirror,
                                                   error);
    
    if (!error.Success())
        return LLDB_INVALID_ADDRESS;
    
    WriteMemory(allocation_process_addr, bytes, size, error);
    
    if (!error.Success())
    {
        Error err;
        Free (allocation_process_addr, err);
        
        return LLDB_INVALID_ADDRESS;
    }
    
    if (Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS))
    {
        DataBufferHeap my_buffer(size, 0);
        Error err;
        ReadMemory(my_buffer.GetBytes(), allocation_process_addr, size, err);
        
        if (err.Success())
        {
            DataExtractor my_extractor(my_buffer.GetBytes(), my_buffer.GetByteSize(), lldb::eByteOrderBig, 8);

            StreamString ss;
            
            my_extractor.Dump(&ss, 0, lldb::eFormatBytesWithASCII, 1, my_buffer.GetByteSize(), 32, allocation_process_addr, 0, 0);
            
            log->PutCString(ss.GetData());
        }
    }
    
    return allocation_process_addr;
}

void
IRExecutionUnit::FreeNow (lldb::addr_t allocation)
{
    if (allocation == LLDB_INVALID_ADDRESS)
        return;
    
    Error err;
    
    Free(allocation, err);
}

Error
IRExecutionUnit::DisassembleFunction (Stream &stream,
                                      lldb::ProcessSP &process_wp)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    ExecutionContext exe_ctx(process_wp);
        
    Error ret;
    
    ret.Clear();
    
    lldb::addr_t func_local_addr = LLDB_INVALID_ADDRESS;
    lldb::addr_t func_remote_addr = LLDB_INVALID_ADDRESS;
        
    for (JittedFunction &function : m_jitted_functions)
    {
        if (strstr(function.m_name.c_str(), m_name.AsCString()))
        {
            func_local_addr = function.m_local_addr;
            func_remote_addr = function.m_remote_addr;
        }
    }
    
    if (func_local_addr == LLDB_INVALID_ADDRESS)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Couldn't find function %s for disassembly", m_name.AsCString());
        return ret;
    }
    
    if (log)
        log->Printf("Found function, has local address 0x%" PRIx64 " and remote address 0x%" PRIx64, (uint64_t)func_local_addr, (uint64_t)func_remote_addr);
    
    std::pair <lldb::addr_t, lldb::addr_t> func_range;
    
    func_range = GetRemoteRangeForLocal(func_local_addr);
    
    if (func_range.first == 0 && func_range.second == 0)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Couldn't find code range for function %s", m_name.AsCString());
        return ret;
    }
    
    if (log)
        log->Printf("Function's code range is [0x%" PRIx64 "+0x%" PRIx64 "]", func_range.first, func_range.second);
    
    Target *target = exe_ctx.GetTargetPtr();
    if (!target)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorString("Couldn't find the target");
        return ret;
    }
    
    lldb::DataBufferSP buffer_sp(new DataBufferHeap(func_range.second, 0));
    
    Process *process = exe_ctx.GetProcessPtr();
    Error err;
    process->ReadMemory(func_remote_addr, buffer_sp->GetBytes(), buffer_sp->GetByteSize(), err);
    
    if (!err.Success())
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Couldn't read from process: %s", err.AsCString("unknown error"));
        return ret;
    }
    
    ArchSpec arch(target->GetArchitecture());
    
    const char *plugin_name = NULL;
    const char *flavor_string = NULL;
    lldb::DisassemblerSP disassembler = Disassembler::FindPlugin(arch, flavor_string, plugin_name);
    
    if (!disassembler)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Unable to find disassembler plug-in for %s architecture.", arch.GetArchitectureName());
        return ret;
    }
    
    if (!process)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorString("Couldn't find the process");
        return ret;
    }
    
    DataExtractor extractor(buffer_sp,
                            process->GetByteOrder(),
                            target->GetArchitecture().GetAddressByteSize());
    
    if (log)
    {
        log->Printf("Function data has contents:");
        extractor.PutToLog (log,
                            0,
                            extractor.GetByteSize(),
                            func_remote_addr,
                            16,
                            DataExtractor::TypeUInt8);
    }
    
    disassembler->DecodeInstructions (Address (func_remote_addr), extractor, 0, UINT32_MAX, false, false);
    
    InstructionList &instruction_list = disassembler->GetInstructionList();
    const uint32_t max_opcode_byte_size = instruction_list.GetMaxOpcocdeByteSize();
    
    for (size_t instruction_index = 0, num_instructions = instruction_list.GetSize();
         instruction_index < num_instructions;
         ++instruction_index)
    {
        Instruction *instruction = instruction_list.GetInstructionAtIndex(instruction_index).get();
        instruction->Dump (&stream,
                           max_opcode_byte_size,
                           true,
                           true,
                           &exe_ctx);
        stream.PutChar('\n');
    }
    
    return ret;
}

static void ReportInlineAsmError(const llvm::SMDiagnostic &diagnostic, void *Context, unsigned LocCookie)
{
    Error *err = static_cast<Error*>(Context);
    
    if (err && err->Success())
    {
        err->SetErrorToGenericError();
        err->SetErrorStringWithFormat("Inline assembly error: %s", diagnostic.getMessage().str().c_str());
    }
}

void
IRExecutionUnit::GetRunnableInfo(Error &error,
                                 lldb::addr_t &func_addr,
                                 lldb::addr_t &func_end)
{
    lldb::ProcessSP process_sp(GetProcessWP().lock());
    
    func_addr = LLDB_INVALID_ADDRESS;
    func_end = LLDB_INVALID_ADDRESS;
    
    if (!process_sp)
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Couldn't write the JIT compiled code into the process because the process is invalid");
        return;
    }
    
    if (m_did_jit)
    {
        func_addr = m_function_load_addr;
        func_end = m_function_end_load_addr;
        
        return;
    };
    
    m_did_jit = true;
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    std::string error_string;
    
    if (log)
    {
        std::string s;
        llvm::raw_string_ostream oss(s);
        
        m_module->print(oss, NULL);
        
        oss.flush();
        
        log->Printf ("Module being sent to JIT: \n%s", s.c_str());
    }
    
    llvm::Triple triple(m_module->getTargetTriple());
    llvm::Function *function = m_module->getFunction (m_name.AsCString());
    llvm::Reloc::Model relocModel;
    llvm::CodeModel::Model codeModel;
    
    if (triple.isOSBinFormatELF())
    {
        relocModel = llvm::Reloc::Static;
        // This will be small for 32-bit and large for 64-bit.
        codeModel = llvm::CodeModel::JITDefault;
    }
    else
    {
        relocModel = llvm::Reloc::PIC_;
        codeModel = llvm::CodeModel::Small;
    }
    
    m_module_ap->getContext().setInlineAsmDiagnosticHandler(ReportInlineAsmError, &error);
    
    llvm::EngineBuilder builder(m_module_ap.get());
    
    builder.setEngineKind(llvm::EngineKind::JIT)
    .setErrorStr(&error_string)
    .setRelocationModel(relocModel)
    .setJITMemoryManager(new MemoryManager(*this))
    .setOptLevel(llvm::CodeGenOpt::Less)
    .setAllocateGVsWithCode(true)
    .setCodeModel(codeModel)
    .setUseMCJIT(true);
    
    llvm::StringRef mArch;
    llvm::StringRef mCPU;
    llvm::SmallVector<std::string, 0> mAttrs;
    
    for (std::string &feature : m_cpu_features)
        mAttrs.push_back(feature);
    
    llvm::TargetMachine *target_machine = builder.selectTarget(triple,
                                                               mArch,
                                                               mCPU,
                                                               mAttrs);
    
    m_execution_engine_ap.reset(builder.create(target_machine));
    
    if (!m_execution_engine_ap.get())
    {
        error.SetErrorToGenericError();
        error.SetErrorStringWithFormat("Couldn't JIT the function: %s", error_string.c_str());
        return;
    }
    else
    {
        m_module_ap.release(); // ownership was transferred
    }
    
    m_execution_engine_ap->DisableLazyCompilation();
    
    // We don't actually need the function pointer here, this just forces it to get resolved.
    
    void *fun_ptr = m_execution_engine_ap->getPointerToFunction(function);
    
    if (!error.Success())
    {
        // We got an error through our callback!
        return;
    }
    
    if (!function)
    {
        error.SetErrorToGenericError();
        error.SetErrorStringWithFormat("Couldn't find '%s' in the JITted module", m_name.AsCString());
        return;
    }
    
    if (!fun_ptr)
    {
        error.SetErrorToGenericError();
        error.SetErrorStringWithFormat("'%s' was in the JITted module but wasn't lowered", m_name.AsCString());
        return;
    }
    
    m_jitted_functions.push_back (JittedFunction(m_name.AsCString(), (lldb::addr_t)fun_ptr));
    
    CommitAllocations(process_sp);
    ReportAllocations(*m_execution_engine_ap);
    WriteData(process_sp);
            
    for (JittedFunction &jitted_function : m_jitted_functions)
    {
        jitted_function.m_remote_addr = GetRemoteAddressForLocal (jitted_function.m_local_addr);
        
        if (!jitted_function.m_name.compare(m_name.AsCString()))
        {
            AddrRange func_range = GetRemoteRangeForLocal(jitted_function.m_local_addr);
            m_function_end_load_addr = func_range.first + func_range.second;
            m_function_load_addr = jitted_function.m_remote_addr;
        }
    }
    
    if (log)
    {
        log->Printf("Code can be run in the target.");
        
        StreamString disassembly_stream;
        
        Error err = DisassembleFunction(disassembly_stream, process_sp);
        
        if (!err.Success())
        {
            log->Printf("Couldn't disassemble function : %s", err.AsCString("unknown error"));
        }
        else
        {
            log->Printf("Function disassembly:\n%s", disassembly_stream.GetData());
        }
    }
    
    func_addr = m_function_load_addr;
    func_end = m_function_end_load_addr;
    
    return;
}

IRExecutionUnit::~IRExecutionUnit ()
{
    m_module_ap.reset();
    m_execution_engine_ap.reset();
    m_context_ap.reset();
}

IRExecutionUnit::MemoryManager::MemoryManager (IRExecutionUnit &parent) :
    m_default_mm_ap (llvm::JITMemoryManager::CreateDefaultMemManager()),
    m_parent (parent)
{
}

void
IRExecutionUnit::MemoryManager::setMemoryWritable ()
{
    m_default_mm_ap->setMemoryWritable();
}

void
IRExecutionUnit::MemoryManager::setMemoryExecutable ()
{
    m_default_mm_ap->setMemoryExecutable();
}


uint8_t *
IRExecutionUnit::MemoryManager::startFunctionBody(const llvm::Function *F,
                                                  uintptr_t &ActualSize)
{
    return m_default_mm_ap->startFunctionBody(F, ActualSize);
}

uint8_t *
IRExecutionUnit::MemoryManager::allocateStub(const llvm::GlobalValue* F,
                                             unsigned StubSize,
                                             unsigned Alignment)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    uint8_t *return_value = m_default_mm_ap->allocateStub(F, StubSize, Alignment);

    m_parent.m_records.push_back(AllocationRecord((uintptr_t)return_value,
                                                  lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                                  StubSize,
                                                  Alignment));

    if (log)
    {
        log->Printf("IRExecutionUnit::allocateStub (F=%p, StubSize=%u, Alignment=%u) = %p",
                    F, StubSize, Alignment, return_value);
    }
        
    return return_value;
}

void
IRExecutionUnit::MemoryManager::endFunctionBody(const llvm::Function *F,
                                                uint8_t *FunctionStart,
                                                uint8_t *FunctionEnd)
{
    m_default_mm_ap->endFunctionBody(F, FunctionStart, FunctionEnd);
}

uint8_t *
IRExecutionUnit::MemoryManager::allocateSpace(intptr_t Size, unsigned Alignment)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    uint8_t *return_value = m_default_mm_ap->allocateSpace(Size, Alignment);
    
    m_parent.m_records.push_back(AllocationRecord((uintptr_t)return_value,
                                                  lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                                  Size,
                                                  Alignment));
    
    if (log)
    {
        log->Printf("IRExecutionUnit::allocateSpace(Size=%" PRIu64 ", Alignment=%u) = %p",
                               (uint64_t)Size, Alignment, return_value);
    }
        
    return return_value;
}

uint8_t *
IRExecutionUnit::MemoryManager::allocateCodeSection(uintptr_t Size,
                                                    unsigned Alignment,
                                                    unsigned SectionID)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    uint8_t *return_value = m_default_mm_ap->allocateCodeSection(Size, Alignment, SectionID);
    
    m_parent.m_records.push_back(AllocationRecord((uintptr_t)return_value,
                                                  lldb::ePermissionsReadable | lldb::ePermissionsExecutable,
                                                  Size,
                                                  Alignment,
                                                  SectionID));
    
    if (log)
    {
        log->Printf("IRExecutionUnit::allocateCodeSection(Size=0x%" PRIx64 ", Alignment=%u, SectionID=%u) = %p",
                    (uint64_t)Size, Alignment, SectionID, return_value);
    }
        
    return return_value;
}

uint8_t *
IRExecutionUnit::MemoryManager::allocateDataSection(uintptr_t Size,
                                                    unsigned Alignment,
                                                    unsigned SectionID,
                                                    bool IsReadOnly)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    uint8_t *return_value = m_default_mm_ap->allocateDataSection(Size, Alignment, SectionID, IsReadOnly);
    
    m_parent.m_records.push_back(AllocationRecord((uintptr_t)return_value,
                                                  lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                                  Size,
                                                  Alignment,
                                                  SectionID));
    if (log)
    {
        log->Printf("IRExecutionUnit::allocateDataSection(Size=0x%" PRIx64 ", Alignment=%u, SectionID=%u) = %p",
                    (uint64_t)Size, Alignment, SectionID, return_value);
    }
        
    return return_value; 
}

uint8_t *
IRExecutionUnit::MemoryManager::allocateGlobal(uintptr_t Size,
                                               unsigned Alignment)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    uint8_t *return_value = m_default_mm_ap->allocateGlobal(Size, Alignment);
    
    m_parent.m_records.push_back(AllocationRecord((uintptr_t)return_value,
                                                  lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                                  Size,
                                                  Alignment));
    
    if (log)
    {
        log->Printf("IRExecutionUnit::allocateGlobal(Size=0x%" PRIx64 ", Alignment=%u) = %p",
                    (uint64_t)Size, Alignment, return_value);
    }
    
    return return_value;
}

void
IRExecutionUnit::MemoryManager::deallocateFunctionBody(void *Body)
{
    m_default_mm_ap->deallocateFunctionBody(Body);
}

uint8_t*
IRExecutionUnit::MemoryManager::startExceptionTable(const llvm::Function* F,
                                                    uintptr_t &ActualSize)
{
    return m_default_mm_ap->startExceptionTable(F, ActualSize);
}

void
IRExecutionUnit::MemoryManager::endExceptionTable(const llvm::Function *F,
                                                  uint8_t *TableStart,
                                                  uint8_t *TableEnd,
                                                  uint8_t* FrameRegister)
{
    m_default_mm_ap->endExceptionTable(F, TableStart, TableEnd, FrameRegister);
}

void
IRExecutionUnit::MemoryManager::deallocateExceptionTable(void *ET)
{
    m_default_mm_ap->deallocateExceptionTable (ET);
}

lldb::addr_t
IRExecutionUnit::GetRemoteAddressForLocal (lldb::addr_t local_address)
{
    for (AllocationRecord &record : m_records)
    {
        if (local_address >= record.m_host_address &&
            local_address < record.m_host_address + record.m_size)
        {
            if (record.m_process_address == LLDB_INVALID_ADDRESS)
                return LLDB_INVALID_ADDRESS;
        }
        
        return record.m_process_address + (local_address - record.m_host_address);
    }

    return LLDB_INVALID_ADDRESS;
}

IRExecutionUnit::AddrRange
IRExecutionUnit::GetRemoteRangeForLocal (lldb::addr_t local_address)
{
    for (AllocationRecord &record : m_records)
    {
        if (local_address >= record.m_host_address &&
            local_address < record.m_host_address + record.m_size)
        {
            if (record.m_process_address == LLDB_INVALID_ADDRESS)
                return AddrRange(0, 0);
            
            return AddrRange(record.m_process_address, record.m_size);
        }
    }
    
    return AddrRange (0, 0);
}

bool
IRExecutionUnit::CommitAllocations (lldb::ProcessSP &process_sp)
{
    bool ret = true;
    
    lldb_private::Error err;
    
    for (AllocationRecord &record : m_records)
    {
        if (record.m_process_address != LLDB_INVALID_ADDRESS)
            continue;
        
        
        record.m_process_address = Malloc(record.m_size,
                                          record.m_alignment,
                                          record.m_permissions,
                                          eAllocationPolicyProcessOnly,
                                          err);
        
        if (!err.Success())
        {
            ret = false;
            break;
        }
    }
    
    if (!ret)
    {
        for (AllocationRecord &record : m_records)
        {
            if (record.m_process_address != LLDB_INVALID_ADDRESS)
            {
                Free(record.m_process_address, err);
                record.m_process_address = LLDB_INVALID_ADDRESS;
            }
        }
    }
    
    return ret;
}

void
IRExecutionUnit::ReportAllocations (llvm::ExecutionEngine &engine)
{
    for (AllocationRecord &record : m_records)
    {
        if (record.m_process_address == LLDB_INVALID_ADDRESS)
            continue;
        
        if (record.m_section_id == eSectionIDInvalid)
            continue;
        
        engine.mapSectionAddress((void*)record.m_host_address, record.m_process_address);
    }
    
    // Trigger re-application of relocations.
    engine.finalizeObject();
}

bool
IRExecutionUnit::WriteData (lldb::ProcessSP &process_sp)
{
    for (AllocationRecord &record : m_records)
    {
        if (record.m_process_address == LLDB_INVALID_ADDRESS)
            return false;
        
        lldb_private::Error err;

        WriteMemory (record.m_process_address, (uint8_t*)record.m_host_address, record.m_size, err);
    }
    
    return true;
}

void 
IRExecutionUnit::AllocationRecord::dump (Log *log)
{
    if (!log)
        return;
    
    log->Printf("[0x%llx+0x%llx]->0x%llx (alignment %d, section ID %d)",
                (unsigned long long)m_host_address,
                (unsigned long long)m_size,
                (unsigned long long)m_process_address,
                (unsigned)m_alignment,
                (unsigned)m_section_id);
}
