//===-- AppleObjCRuntimeV1.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleObjCRuntimeV1.h"
#include "AppleObjCTrampolineHandler.h"

#include "llvm/Support/MachO.h"
#include "clang/AST/Type.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ClangUtilityFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

static const char *pluginName = "AppleObjCRuntimeV1";
static const char *pluginDesc = "Apple Objective C Language Runtime - Version 1";
static const char *pluginShort = "language.apple.objc.v1";


AppleObjCRuntimeV1::AppleObjCRuntimeV1(Process *process) :
    AppleObjCRuntime (process),
    m_hash_signature (),
    m_isa_hash_table_ptr (LLDB_INVALID_ADDRESS)
{
}

bool
AppleObjCRuntimeV1::GetDynamicTypeAndAddress (ValueObject &in_value,
                                             lldb::DynamicValueType use_dynamic, 
                                             TypeAndOrName &class_type_or_name, 
                                             Address &address)
{
    return false;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
lldb_private::LanguageRuntime *
AppleObjCRuntimeV1::CreateInstance (Process *process, lldb::LanguageType language)
{
    // FIXME: This should be a MacOS or iOS process, and we need to look for the OBJC section to make
    // sure we aren't using the V1 runtime.
    if (language == eLanguageTypeObjC)
    {
        ModuleSP objc_module_sp;
        
        if (AppleObjCRuntime::GetObjCVersion (process, objc_module_sp) == eAppleObjC_V1)
            return new AppleObjCRuntimeV1 (process);
        else
            return NULL;
    }
    else
        return NULL;
}


void
AppleObjCRuntimeV1::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);    
}

void
AppleObjCRuntimeV1::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
AppleObjCRuntimeV1::GetPluginName()
{
    return pluginName;
}

const char *
AppleObjCRuntimeV1::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
AppleObjCRuntimeV1::GetPluginVersion()
{
    return 1;
}

BreakpointResolverSP
AppleObjCRuntimeV1::CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp)
{
    BreakpointResolverSP resolver_sp;
    
    if (throw_bp)
        resolver_sp.reset (new BreakpointResolverName (bkpt,
                                                       "objc_exception_throw",
                                                       eFunctionNameTypeBase,
                                                       Breakpoint::Exact,
                                                       eLazyBoolNo));
    // FIXME: don't do catch yet.
    return resolver_sp;
}

struct BufStruct {
    char contents[2048];
};

ClangUtilityFunction *
AppleObjCRuntimeV1::CreateObjectChecker(const char *name)
{
    std::auto_ptr<BufStruct> buf(new BufStruct);
    
    assert(snprintf(&buf->contents[0], sizeof(buf->contents),
                    "struct __objc_class                                                    \n"
                    "{                                                                      \n"
                    "   struct __objc_class *isa;                                           \n"
                    "   struct __objc_class *super_class;                                   \n"
                    "   const char *name;                                                   \n"
                    "   // rest of struct elided because unused                             \n"
                    "};                                                                     \n"
                    "                                                                       \n"
                    "struct __objc_object                                                   \n"
                    "{                                                                      \n"
                    "   struct __objc_class *isa;                                           \n"
                    "};                                                                     \n"
                    "                                                                       \n"
                    "extern \"C\" void                                                      \n"
                    "%s(void *$__lldb_arg_obj, void *$__lldb_arg_selector)                  \n"
                    "{                                                                      \n"
                    "   struct __objc_object *obj = (struct __objc_object*)$__lldb_arg_obj; \n"
                    "   (int)strlen(obj->isa->name);                                        \n"
                    "}                                                                      \n",
                    name) < sizeof(buf->contents));

    return new ClangUtilityFunction(buf->contents, name);
}

// this code relies on the assumption that an Objective-C object always starts
// with an ISA at offset 0.
//ObjCLanguageRuntime::ObjCISA
//AppleObjCRuntimeV1::GetISA(ValueObject& valobj)
//{
////    if (ClangASTType::GetMinimumLanguage(valobj.GetClangAST(),valobj.GetClangType()) != eLanguageTypeObjC)
////        return 0;
//    
//    // if we get an invalid VO (which might still happen when playing around
//    // with pointers returned by the expression parser, don't consider this
//    // a valid ObjC object)
//    if (valobj.GetValue().GetContextType() == Value::eContextTypeInvalid)
//        return 0;
//    
//    addr_t isa_pointer = valobj.GetPointerValue();
//    
//    ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
//    
//    Process *process = exe_ctx.GetProcessPtr();
//    if (process)
//    {
//        uint8_t pointer_size = process->GetAddressByteSize();
//        
//        Error error;
//        return process->ReadUnsignedIntegerFromMemory (isa_pointer,
//                                                       pointer_size,
//                                                       0,
//                                                       error);
//    }
//    return 0;
//}

AppleObjCRuntimeV1::ClassDescriptorV1::ClassDescriptorV1 (ValueObject &isa_pointer)
{
    Initialize (isa_pointer.GetValueAsUnsigned(0),
                isa_pointer.GetProcessSP());
}

AppleObjCRuntimeV1::ClassDescriptorV1::ClassDescriptorV1 (ObjCISA isa, lldb::ProcessSP process_sp)
{
    Initialize (isa, process_sp);
}

void
AppleObjCRuntimeV1::ClassDescriptorV1::Initialize (ObjCISA isa, lldb::ProcessSP process_sp)
{
    if (!isa || !process_sp)
    {
        m_valid = false;
        return;
    }
    
    m_valid = true;
    
    Error error;
    
    m_isa = process_sp->ReadPointerFromMemory(isa, error);
    
    if (error.Fail())
    {
        m_valid = false;
        return;
    }
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    
    if (!IsPointerValid(m_isa,ptr_size))
    {
        m_valid = false;
        return;
    }

    m_parent_isa = process_sp->ReadPointerFromMemory(m_isa + ptr_size,error);
    
    if (error.Fail())
    {
        m_valid = false;
        return;
    }
    
    if (!IsPointerValid(m_parent_isa,ptr_size,true))
    {
        m_valid = false;
        return;
    }
    
    lldb::addr_t name_ptr = process_sp->ReadPointerFromMemory(m_isa + 2 * ptr_size,error);
    
    if (error.Fail())
    {
        m_valid = false;
        return;
    }
    
    lldb::DataBufferSP buffer_sp(new DataBufferHeap(1024, 0));
    
    size_t count = process_sp->ReadCStringFromMemory(name_ptr, (char*)buffer_sp->GetBytes(), 1024, error);
    
    if (error.Fail())
    {
        m_valid = false;
        return;
    }
    
    if (count)
        m_name = ConstString((char*)buffer_sp->GetBytes());
    else
        m_name = ConstString();
    
    m_instance_size = process_sp->ReadUnsignedIntegerFromMemory(m_isa + 5 * ptr_size, ptr_size, 0, error);
    
    if (error.Fail())
    {
        m_valid = false;
        return;
    }
    
    m_process_wp = lldb::ProcessWP(process_sp);
}

AppleObjCRuntime::ClassDescriptorSP
AppleObjCRuntimeV1::ClassDescriptorV1::GetSuperclass ()
{
    if (!m_valid)
        return AppleObjCRuntime::ClassDescriptorSP();
    ProcessSP process_sp = m_process_wp.lock();
    if (!process_sp)
        return AppleObjCRuntime::ClassDescriptorSP();
    return ObjCLanguageRuntime::ClassDescriptorSP(new AppleObjCRuntimeV1::ClassDescriptorV1(m_parent_isa,process_sp));
}

lldb::addr_t
AppleObjCRuntimeV1::GetISAHashTablePointer ()
{
    if (m_isa_hash_table_ptr == LLDB_INVALID_ADDRESS)
    {
        ModuleSP objc_module_sp(GetObjCModule());
        
        if (!objc_module_sp)
            return LLDB_INVALID_ADDRESS;
        
        static ConstString g_objc_debug_class_hash("_objc_debug_class_hash");
        
        const Symbol *symbol = objc_module_sp->FindFirstSymbolWithNameAndType(g_objc_debug_class_hash, lldb::eSymbolTypeData);
        if (symbol)
        {
            Process *process = GetProcess();
            if (process)
            {

                lldb::addr_t objc_debug_class_hash_addr = symbol->GetAddress().GetLoadAddress(&process->GetTarget());
            
                if (objc_debug_class_hash_addr != LLDB_INVALID_ADDRESS)
                {
                    Error error;
                    lldb::addr_t objc_debug_class_hash_ptr = process->ReadPointerFromMemory(objc_debug_class_hash_addr, error);
                    if (objc_debug_class_hash_ptr != 0 &&
                        objc_debug_class_hash_ptr != LLDB_INVALID_ADDRESS)
                    {
                        m_isa_hash_table_ptr = objc_debug_class_hash_ptr;
                    }
                }
            }
        }
    }
    return m_isa_hash_table_ptr;
}

void
AppleObjCRuntimeV1::UpdateISAToDescriptorMapIfNeeded()
{
    // TODO: implement HashTableSignature...
    Process *process = GetProcess();
    
    if (process)
    {
        // Update the process stop ID that indicates the last time we updated the
        // map, wether it was successful or not.
        m_isa_to_descriptor_cache_stop_id = process->GetStopID();
        

        lldb::LogSP log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
        
        ProcessSP process_sp = process->shared_from_this();
        
        ModuleSP objc_module_sp(GetObjCModule());
        
        if (!objc_module_sp)
            return;
        
        uint32_t isa_count = 0;
        
        lldb::addr_t hash_table_ptr = GetISAHashTablePointer ();
        if (hash_table_ptr != LLDB_INVALID_ADDRESS)
        {
            // Read the NXHashTable struct:
            //
            // typedef struct {
            //     const NXHashTablePrototype *prototype;
            //     unsigned   count;
            //     unsigned   nbBuckets;
            //     void       *buckets;
            //     const void *info;
            // } NXHashTable;

            Error error;
            DataBufferHeap buffer(1024, 0);
            if (process->ReadMemory(hash_table_ptr, buffer.GetBytes(), 20, error) == 20)
            {
                const uint32_t addr_size = m_process->GetAddressByteSize();
                const ByteOrder byte_order = m_process->GetByteOrder();
                DataExtractor data (buffer.GetBytes(), buffer.GetByteSize(), byte_order, addr_size);
                uint32_t offset = addr_size; // Skip prototype
                const uint32_t count = data.GetU32(&offset);
                const uint32_t num_buckets = data.GetU32(&offset);
                const addr_t buckets_ptr = data.GetPointer(&offset);
                if (m_hash_signature.NeedsUpdate (count, num_buckets, buckets_ptr))
                {
                    m_hash_signature.UpdateSignature (count, num_buckets, buckets_ptr);

                    const uint32_t data_size = num_buckets * 2 * sizeof(uint32_t);
                    buffer.SetByteSize(data_size);
                    
                    if (process->ReadMemory(buckets_ptr, buffer.GetBytes(), data_size, error) == data_size)
                    {
                        data.SetData(buffer.GetBytes(), buffer.GetByteSize(), byte_order);
                        offset = 0;
                        for (uint32_t bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx)
                        {
                            const uint32_t bucket_isa_count = data.GetU32 (&offset);
                            const lldb::addr_t bucket_data = data.GetU32 (&offset);
                            

                            if (bucket_isa_count == 0)
                                continue;
                            
                            isa_count += bucket_isa_count;

                            ObjCISA isa;
                            if (bucket_isa_count == 1)
                            {
                                // When we only have one entry in the bucket, the bucket data is the "isa"
                                isa = bucket_data;
                                if (isa)
                                {
                                    if (m_isa_to_descriptor_cache.count(isa) == 0)
                                    {
                                        ClassDescriptorSP descriptor_sp (new ClassDescriptorV1(isa, process_sp));
                                        
                                        if (log && log->GetVerbose())
                                            log->Printf("AppleObjCRuntimeV1 added (ObjCISA)0x%llx from _objc_debug_class_hash to isa->descriptor cache", isa);
                                        
                                        m_isa_to_descriptor_cache[isa] = descriptor_sp;
                                    }
                                }
                            }
                            else
                            {
                                // When we have more than one entry in the bucket, the bucket data is a pointer
                                // to an array of "isa" values
                                addr_t isa_addr = bucket_data;
                                for (uint32_t isa_idx = 0; isa_idx < bucket_isa_count; ++isa_idx, isa_addr += addr_size)
                                {
                                    isa = m_process->ReadPointerFromMemory(isa_addr, error);

                                    if (isa && isa != LLDB_INVALID_ADDRESS)
                                    {
                                        if (m_isa_to_descriptor_cache.count(isa) == 0)
                                        {
                                            ClassDescriptorSP descriptor_sp (new ClassDescriptorV1(isa, process_sp));
                                            
                                            if (log && log->GetVerbose())
                                                log->Printf("AppleObjCRuntimeV1 added (ObjCISA)0x%llx from _objc_debug_class_hash to isa->descriptor cache", isa);
                                            
                                            m_isa_to_descriptor_cache[isa] = descriptor_sp;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }        
    }
    else
    {
        m_isa_to_descriptor_cache_stop_id = UINT32_MAX;
    }
}

