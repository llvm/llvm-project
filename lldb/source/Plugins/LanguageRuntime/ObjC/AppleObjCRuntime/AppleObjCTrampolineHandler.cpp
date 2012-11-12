//===-- AppleObjCTrampolineHandler.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleObjCTrampolineHandler.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "AppleThreadPlanStepThroughObjCTrampoline.h"

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ClangUtilityFunction.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

const char *AppleObjCTrampolineHandler::g_lookup_implementation_function_name = "__lldb_objc_find_implementation_for_selector";
const char *AppleObjCTrampolineHandler::g_lookup_implementation_function_code = "                               \n\
extern \"C\"                                                                                                    \n\
{                                                                                                               \n\
    extern void *class_getMethodImplementation(void *objc_class, void *sel);                                    \n\
    extern void *class_getMethodImplementation_stret(void *objc_class, void *sel);                              \n\
    extern void * sel_getUid(char *name);                                                                       \n\
    extern int printf(const char *format, ...);                                                                 \n\
}                                                                                                               \n\
extern \"C\" void * __lldb_objc_find_implementation_for_selector (void *object,                                 \n\
                                                    void *sel,                                                  \n\
                                                    int is_stret,                                               \n\
                                                    int is_super,                                               \n\
                                                    int is_super2,                                              \n\
                                                    int is_fixup,                                               \n\
                                                    int is_fixed,                                               \n\
                                                    int debug)                                                  \n\
{                                                                                                               \n\
    struct __lldb_imp_return_struct                                                                             \n\
    {                                                                                                           \n\
        void *class_addr;                                                                                       \n\
        void *sel_addr;                                                                                         \n\
        void *impl_addr;                                                                                        \n\
    };                                                                                                          \n\
                                                                                                                \n\
    struct __lldb_objc_class {                                                                                  \n\
        void *isa;                                                                                              \n\
        void *super_ptr;                                                                                        \n\
    };                                                                                                          \n\
    struct __lldb_objc_super {                                                                                  \n\
        void *reciever;                                                                                         \n\
        struct __lldb_objc_class *class_ptr;                                                                    \n\
    };                                                                                                          \n\
    struct __lldb_msg_ref {                                                                                     \n\
        void *dont_know;                                                                                        \n\
        void *sel;                                                                                              \n\
    };                                                                                                          \n\
                                                                                                                \n\
    struct __lldb_imp_return_struct return_struct;                                                              \n\
                                                                                                                \n\
    if (debug)                                                                                                  \n\
        printf (\"\\n*** Called with obj: 0x%p sel: 0x%p is_stret: %d is_super: %d, \"                          \n\
                \"is_super2: %d, is_fixup: %d, is_fixed: %d\\n\",                                               \n\
                 object, sel, is_stret, is_super, is_super2, is_fixup, is_fixed);                               \n\
    if (is_super)                                                                                               \n\
    {                                                                                                           \n\
        if (is_super2)                                                                                          \n\
        {                                                                                                       \n\
            return_struct.class_addr = ((__lldb_objc_super *) object)->class_ptr->super_ptr;                    \n\
        }                                                                                                       \n\
        else                                                                                                    \n\
        {                                                                                                       \n\
            return_struct.class_addr = ((__lldb_objc_super *) object)->class_ptr;                               \n\
        }                                                                                                       \n\
    }                                                                                                           \n\
    else                                                                                                        \n\
    {                                                                                                           \n\
        void *class_ptr = (void *) [(id) object class];                                                         \n\
        if (class_ptr == object)                                                                                \n\
        {                                                                                                       \n\
            struct __lldb_objc_class *class_as_class_struct = (struct __lldb_objc_class *) class_ptr;           \n\
            if (debug)                                                                                          \n\
                printf (\"Found a class object, need to return the meta class 0x%p -> 0x%p\\n\",                \n\
                        class_ptr, class_as_class_struct->isa);                                                 \n\
            return_struct.class_addr = class_as_class_struct->isa;                                              \n\
        }                                                                                                       \n\
        else                                                                                                    \n\
        {                                                                                                       \n\
            if (debug)                                                                                          \n\
                printf (\"[object class] returned: 0x%p.\\n\", class_ptr);                                      \n\
            return_struct.class_addr = class_ptr;                                                               \n\
        }                                                                                                       \n\
    }                                                                                                           \n\
                                                                                                                \n\
    if (is_fixup)                                                                                               \n\
    {                                                                                                           \n\
        if (is_fixed)                                                                                           \n\
        {                                                                                                       \n\
            return_struct.sel_addr = ((__lldb_msg_ref *) sel)->sel;                                             \n\
        }                                                                                                       \n\
        else                                                                                                    \n\
        {                                                                                                       \n\
            char *sel_name = (char *) ((__lldb_msg_ref *) sel)->sel;                                            \n\
            return_struct.sel_addr = sel_getUid (sel_name);                                                     \n\
            if (debug)                                                                                          \n\
                printf (\"\\n*** Got fixed up selector: 0x%p for name %s.\\n\",                                 \n\
                        return_struct.sel_addr, sel_name);                                                      \n\
        }                                                                                                       \n\
    }                                                                                                           \n\
    else                                                                                                        \n\
    {                                                                                                           \n\
        return_struct.sel_addr = sel;                                                                           \n\
    }                                                                                                           \n\
                                                                                                                \n\
    if (is_stret)                                                                                               \n\
    {                                                                                                           \n\
        return_struct.impl_addr = class_getMethodImplementation_stret (return_struct.class_addr,                \n\
                                                                       return_struct.sel_addr);                 \n\
    }                                                                                                           \n\
    else                                                                                                        \n\
    {                                                                                                           \n\
        return_struct.impl_addr = class_getMethodImplementation (return_struct.class_addr,                      \n\
                                                                       return_struct.sel_addr);                 \n\
    }                                                                                                           \n\
    if (debug)                                                                                                  \n\
        printf (\"\\n*** Returning implementation: 0x%p.\\n\", return_struct.impl_addr);                        \n\
                                                                                                                \n\
    return return_struct.impl_addr;                                                                             \n\
}                                                                                                               \n\
";

AppleObjCTrampolineHandler::AppleObjCVTables::VTableRegion::VTableRegion(AppleObjCVTables *owner, lldb::addr_t header_addr) :
    m_valid (true),
    m_owner(owner),
    m_header_addr (header_addr),
    m_code_start_addr(0),
    m_code_end_addr (0),
    m_next_region (0)
{
    SetUpRegion ();
}

AppleObjCTrampolineHandler::~AppleObjCTrampolineHandler()
{
}

void
AppleObjCTrampolineHandler::AppleObjCVTables::VTableRegion::SetUpRegion()
{
    // The header looks like:
    //
    //   uint16_t headerSize
    //   uint16_t descSize
    //   uint32_t descCount
    //   void * next
    //
    // First read in the header:
    
    char memory_buffer[16];
    Process *process = m_owner->GetProcess();
    DataExtractor data(memory_buffer, sizeof(memory_buffer), 
                       process->GetByteOrder(), 
                       process->GetAddressByteSize());
    size_t actual_size = 8 + process->GetAddressByteSize();
    Error error;
    size_t bytes_read = process->ReadMemory (m_header_addr, memory_buffer, actual_size, error);
    if (bytes_read != actual_size)
    {
        m_valid = false;
        return;
    }
    
    uint32_t offset_ptr = 0;
    const uint16_t header_size = data.GetU16(&offset_ptr);
    const uint16_t descriptor_size = data.GetU16(&offset_ptr);
    const size_t num_descriptors = data.GetU32(&offset_ptr);
    
    m_next_region = data.GetPointer(&offset_ptr);
    
    // If the header size is 0, that means we've come in too early before this data is set up.
    // Set ourselves as not valid, and continue.
    if (header_size == 0 || num_descriptors == 0)
    {
        m_valid = false;
        return;
    }
    
    // Now read in all the descriptors:
    // The descriptor looks like:
    //
    // uint32_t offset
    // uint32_t flags
    //
    // Where offset is either 0 - in which case it is unused, or
    // it is the offset of the vtable code from the beginning of the descriptor record.
    // Below, we'll convert that into an absolute code address, since I don't want to have
    // to compute it over and over.
    
    // Ingest the whole descriptor array:
    const lldb::addr_t desc_ptr = m_header_addr + header_size;
    const size_t desc_array_size = num_descriptors * descriptor_size;
    DataBufferSP data_sp(new DataBufferHeap (desc_array_size, '\0'));
    uint8_t* dst = (uint8_t*)data_sp->GetBytes();

    DataExtractor desc_extractor (dst, desc_array_size,
                                  process->GetByteOrder(), 
                                  process->GetAddressByteSize());
    bytes_read = process->ReadMemory(desc_ptr, dst, desc_array_size, error);
    if (bytes_read != desc_array_size)
    {
        m_valid = false;
        return;
    }
    
    // The actual code for the vtables will be laid out consecutively, so I also
    // compute the start and end of the whole code block.

    offset_ptr = 0;
    m_code_start_addr = 0;
    m_code_end_addr = 0;

    for (int i = 0; i < num_descriptors; i++)
    {
        lldb::addr_t start_offset = offset_ptr;
        uint32_t offset = desc_extractor.GetU32 (&offset_ptr);
        uint32_t flags  = desc_extractor.GetU32 (&offset_ptr);
        lldb::addr_t code_addr = desc_ptr + start_offset + offset;
        m_descriptors.push_back (VTableDescriptor(flags, code_addr));
        
        if (m_code_start_addr == 0 || code_addr < m_code_start_addr)
            m_code_start_addr = code_addr;
        if (code_addr > m_code_end_addr)
            m_code_end_addr = code_addr;
            
        offset_ptr = start_offset + descriptor_size;
    }
    // Finally, a little bird told me that all the vtable code blocks are the same size.  
    // Let's compute the blocks and if they are all the same add the size to the code end address:
    lldb::addr_t code_size = 0;
    bool all_the_same = true;
    for (int i = 0; i < num_descriptors - 1; i++)
    {
        lldb::addr_t this_size = m_descriptors[i + 1].code_start - m_descriptors[i].code_start;
        if (code_size == 0)
            code_size = this_size;
        else
        {
            if (this_size != code_size)
                all_the_same = false;
            if (this_size > code_size)
                code_size = this_size;
        }
    }
    if (all_the_same)
        m_code_end_addr += code_size;
}

bool 
AppleObjCTrampolineHandler::AppleObjCVTables::VTableRegion::AddressInRegion (lldb::addr_t addr, uint32_t &flags)
{
    if (!IsValid())
        return false;
        
    if (addr < m_code_start_addr || addr > m_code_end_addr)
        return false;
        
    std::vector<VTableDescriptor>::iterator pos, end = m_descriptors.end();
    for (pos = m_descriptors.begin(); pos != end; pos++)
    {
        if (addr <= (*pos).code_start)
        {
            flags = (*pos).flags;
            return true;
        }
    }
    return false;
}

void
AppleObjCTrampolineHandler::AppleObjCVTables::VTableRegion::Dump (Stream &s)
{
    s.Printf ("Header addr: 0x%llx Code start: 0x%llx Code End: 0x%llx Next: 0x%llx\n", 
              m_header_addr, m_code_start_addr, m_code_end_addr, m_next_region);
    size_t num_elements = m_descriptors.size();
    for (size_t i = 0; i < num_elements; i++)
    {
        s.Indent();
        s.Printf ("Code start: 0x%llx Flags: %d\n", m_descriptors[i].code_start, m_descriptors[i].flags);
    }
}
        
AppleObjCTrampolineHandler::AppleObjCVTables::AppleObjCVTables (const ProcessSP &process_sp, 
                                                                const ModuleSP &objc_module_sp) :
    m_process_sp (process_sp),
    m_trampoline_header (LLDB_INVALID_ADDRESS),
    m_trampolines_changed_bp_id (LLDB_INVALID_BREAK_ID),
    m_objc_module_sp (objc_module_sp)
{
    
}

AppleObjCTrampolineHandler::AppleObjCVTables::~AppleObjCVTables()
{
    if (m_trampolines_changed_bp_id != LLDB_INVALID_BREAK_ID)
        m_process_sp->GetTarget().RemoveBreakpointByID (m_trampolines_changed_bp_id);
}
    
bool
AppleObjCTrampolineHandler::AppleObjCVTables::InitializeVTableSymbols ()
{
    if (m_trampoline_header != LLDB_INVALID_ADDRESS)
        return true;
    Target &target = m_process_sp->GetTarget();
    
    const ModuleList &target_modules = target.GetImages();
    Mutex::Locker modules_locker(target_modules.GetMutex());
    size_t num_modules = target_modules.GetSize();
    if (!m_objc_module_sp)
    {
        for (size_t i = 0; i < num_modules; i++)
        {
            if (m_process_sp->GetObjCLanguageRuntime()->IsModuleObjCLibrary (target_modules.GetModuleAtIndexUnlocked(i)))
            {
                m_objc_module_sp = target_modules.GetModuleAtIndexUnlocked(i);
                break;
            }
        }
    }
    
    if (m_objc_module_sp)
    {
        ConstString trampoline_name ("gdb_objc_trampolines");
        const Symbol *trampoline_symbol = m_objc_module_sp->FindFirstSymbolWithNameAndType (trampoline_name, 
                                                                                            eSymbolTypeData);
        if (trampoline_symbol != NULL)
        {
            if (!trampoline_symbol->GetAddress().IsValid())
                return false;
                
            m_trampoline_header = trampoline_symbol->GetAddress().GetLoadAddress(&target);
            if (m_trampoline_header == LLDB_INVALID_ADDRESS)
                return false;
            
            // Next look up the "changed" symbol and set a breakpoint on that...
            ConstString changed_name ("gdb_objc_trampolines_changed");
            const Symbol *changed_symbol = m_objc_module_sp->FindFirstSymbolWithNameAndType (changed_name, 
                                                                                             eSymbolTypeCode);
            if (changed_symbol != NULL)
            {
                if (!changed_symbol->GetAddress().IsValid())
                    return false;
                    
                lldb::addr_t changed_addr = changed_symbol->GetAddress().GetOpcodeLoadAddress (&target);
                if (changed_addr != LLDB_INVALID_ADDRESS)
                {
                    BreakpointSP trampolines_changed_bp_sp = target.CreateBreakpoint (changed_addr, true);
                    if (trampolines_changed_bp_sp)
                    {
                        m_trampolines_changed_bp_id = trampolines_changed_bp_sp->GetID();
                        trampolines_changed_bp_sp->SetCallback (RefreshTrampolines, this, true);
                        return true;
                    }
                }
            }
        }
    }
    
    return false;
}
    
bool 
AppleObjCTrampolineHandler::AppleObjCVTables::RefreshTrampolines (void *baton, 
                                                                  StoppointCallbackContext *context, 
                                                                  lldb::user_id_t break_id, 
                                                                  lldb::user_id_t break_loc_id)
{
    AppleObjCVTables *vtable_handler = (AppleObjCVTables *) baton;
    if (vtable_handler->InitializeVTableSymbols())
    {
        // The Update function is called with the address of an added region.  So we grab that address, and
        // feed it into ReadRegions.  Of course, our friend the ABI will get the values for us.
        ExecutionContext exe_ctx (context->exe_ctx_ref);
        Process *process = exe_ctx.GetProcessPtr();
        const ABI *abi = process->GetABI().get();
        
        ClangASTContext *clang_ast_context = process->GetTarget().GetScratchClangASTContext();
        ValueList argument_values;
        Value input_value;
        void *clang_void_ptr_type = clang_ast_context->GetVoidPtrType(false);
        input_value.SetValueType (Value::eValueTypeScalar);
        input_value.SetContext (Value::eContextTypeClangType, clang_void_ptr_type);
        argument_values.PushValue(input_value);
        
        bool success = abi->GetArgumentValues (exe_ctx.GetThreadRef(), argument_values);
        if (!success)
            return false;
            
        // Now get a pointer value from the zeroth argument.
        Error error;
        DataExtractor data;
        error = argument_values.GetValueAtIndex(0)->GetValueAsData (&exe_ctx, 
                                                                    clang_ast_context->getASTContext(), 
                                                                    data, 
                                                                    0,
                                                                    NULL);
        uint32_t offset_ptr = 0;
        lldb::addr_t region_addr = data.GetPointer(&offset_ptr);
        
        if (region_addr != 0)
            vtable_handler->ReadRegions(region_addr);
    }
    return false;
}

bool
AppleObjCTrampolineHandler::AppleObjCVTables::ReadRegions ()
{
    // The no argument version reads the start region from the value of the gdb_regions_header, and 
    // gets started from there.
    
    m_regions.clear();
    if (!InitializeVTableSymbols())
        return false;
    Error error;
    lldb::addr_t region_addr = m_process_sp->ReadPointerFromMemory (m_trampoline_header, error);
    if (error.Success())    
        return ReadRegions (region_addr);
    return false;
}

bool
AppleObjCTrampolineHandler::AppleObjCVTables::ReadRegions (lldb::addr_t region_addr)
{
    if (!m_process_sp)
        return false;
        
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    
    // We aren't starting at the trampoline symbol.
    InitializeVTableSymbols ();
    lldb::addr_t next_region = region_addr;
    
    // Read in the sizes of the headers.
    while (next_region != 0) 
    {
        m_regions.push_back (VTableRegion(this, next_region));
        if (!m_regions.back().IsValid())
        {
            m_regions.clear();
            return false;
        }
        if (log)
        {
            StreamString s;
            m_regions.back().Dump(s);
            log->Printf("Read vtable region: \n%s", s.GetData());
        }
        
        next_region = m_regions.back().GetNextRegionAddr();
    }
    
    return true;
}
    
bool
AppleObjCTrampolineHandler::AppleObjCVTables::IsAddressInVTables (lldb::addr_t addr, uint32_t &flags)
{
    region_collection::iterator pos, end = m_regions.end();
    for (pos = m_regions.begin(); pos != end; pos++)
    {
        if ((*pos).AddressInRegion (addr, flags))
            return true;
    }
    return false;
}

const AppleObjCTrampolineHandler::DispatchFunction
AppleObjCTrampolineHandler::g_dispatch_functions[] =
{
    // NAME                              STRET  SUPER  SUPER2  FIXUP TYPE
    {"objc_msgSend",                     false, false, false, DispatchFunction::eFixUpNone    },
    {"objc_msgSend_fixup",               false, false, false, DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_fixedup",             false, false, false, DispatchFunction::eFixUpFixed   },
    {"objc_msgSend_stret",               true,  false, false, DispatchFunction::eFixUpNone    },
    {"objc_msgSend_stret_fixup",         true,  false, false, DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_stret_fixedup",       true,  false, false, DispatchFunction::eFixUpFixed   },
    {"objc_msgSend_fpret",               false, false, false, DispatchFunction::eFixUpNone    },
    {"objc_msgSend_fpret_fixup",         false, false, false, DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_fpret_fixedup",       false, false, false, DispatchFunction::eFixUpFixed   },
    {"objc_msgSend_fp2ret",              false, false,  true, DispatchFunction::eFixUpNone    },
    {"objc_msgSend_fp2ret_fixup",        false, false,  true, DispatchFunction::eFixUpToFix   },
    {"objc_msgSend_fp2ret_fixedup",      false, false,  true, DispatchFunction::eFixUpFixed   },
    {"objc_msgSendSuper",                false, true,  false, DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper_stret",          true,  true,  false, DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper2",               false, true,   true, DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper2_fixup",         false, true,   true, DispatchFunction::eFixUpToFix   },
    {"objc_msgSendSuper2_fixedup",       false, true,   true, DispatchFunction::eFixUpFixed   },
    {"objc_msgSendSuper2_stret",         true,  true,   true, DispatchFunction::eFixUpNone    },
    {"objc_msgSendSuper2_stret_fixup",   true,  true,   true, DispatchFunction::eFixUpToFix   },
    {"objc_msgSendSuper2_stret_fixedup", true,  true,   true, DispatchFunction::eFixUpFixed   },
};

AppleObjCTrampolineHandler::AppleObjCTrampolineHandler (const ProcessSP &process_sp, 
                                                        const ModuleSP &objc_module_sp) :
    m_process_sp (process_sp),
    m_objc_module_sp (objc_module_sp),
    m_impl_fn_addr (LLDB_INVALID_ADDRESS),
    m_impl_stret_fn_addr (LLDB_INVALID_ADDRESS),
    m_msg_forward_addr (LLDB_INVALID_ADDRESS)
{
    // Look up the known resolution functions:
    
    ConstString get_impl_name("class_getMethodImplementation");
    ConstString get_impl_stret_name("class_getMethodImplementation_stret");
    ConstString msg_forward_name("_objc_msgForward");
    ConstString msg_forward_stret_name("_objc_msgForward_stret");
    
    Target *target = m_process_sp ? &m_process_sp->GetTarget() : NULL;
    const Symbol *class_getMethodImplementation = m_objc_module_sp->FindFirstSymbolWithNameAndType (get_impl_name, eSymbolTypeCode);
    const Symbol *class_getMethodImplementation_stret = m_objc_module_sp->FindFirstSymbolWithNameAndType (get_impl_stret_name, eSymbolTypeCode);
    const Symbol *msg_forward = m_objc_module_sp->FindFirstSymbolWithNameAndType (msg_forward_name, eSymbolTypeCode);
    const Symbol *msg_forward_stret = m_objc_module_sp->FindFirstSymbolWithNameAndType (msg_forward_stret_name, eSymbolTypeCode);
    
    if (class_getMethodImplementation)
        m_impl_fn_addr = class_getMethodImplementation->GetAddress().GetOpcodeLoadAddress (target);
    if  (class_getMethodImplementation_stret)
        m_impl_stret_fn_addr = class_getMethodImplementation_stret->GetAddress().GetOpcodeLoadAddress (target);
    if (msg_forward)
        m_msg_forward_addr = msg_forward->GetAddress().GetOpcodeLoadAddress(target);
    if  (msg_forward_stret)
        m_msg_forward_stret_addr = msg_forward_stret->GetAddress().GetOpcodeLoadAddress(target);
    
    // FIXME: Do some kind of logging here.
    if (m_impl_fn_addr == LLDB_INVALID_ADDRESS || m_impl_stret_fn_addr == LLDB_INVALID_ADDRESS)
        return;
        
    // Look up the addresses for the objc dispatch functions and cache them.  For now I'm inspecting the symbol
    // names dynamically to figure out how to dispatch to them.  If it becomes more complicated than this we can 
    // turn the g_dispatch_functions char * array into a template table, and populate the DispatchFunction map
    // from there.

    for (int i = 0; i != llvm::array_lengthof(g_dispatch_functions); i++)
    {
        ConstString name_const_str(g_dispatch_functions[i].name);
        const Symbol *msgSend_symbol = m_objc_module_sp->FindFirstSymbolWithNameAndType (name_const_str, eSymbolTypeCode);
        if (msgSend_symbol)
        {
            // FixMe: Make g_dispatch_functions static table of DispatchFunctions, and have the map be address->index.
            // Problem is we also need to lookup the dispatch function.  For now we could have a side table of stret & non-stret
            // dispatch functions.  If that's as complex as it gets, we're fine.
            
            lldb::addr_t sym_addr = msgSend_symbol->GetAddress().GetOpcodeLoadAddress(target);
            
            m_msgSend_map.insert(std::pair<lldb::addr_t, int>(sym_addr, i));
        }
    }
    
    // Build our vtable dispatch handler here:
    m_vtables_ap.reset(new AppleObjCVTables(process_sp, m_objc_module_sp));
    if (m_vtables_ap.get())
        m_vtables_ap->ReadRegions();        
}

lldb::addr_t
AppleObjCTrampolineHandler::SetupDispatchFunction (Thread &thread, ValueList &dispatch_values)
{
    ExecutionContext exe_ctx (thread.shared_from_this());
    Address impl_code_address;
    StreamString errors;
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    lldb::addr_t args_addr = LLDB_INVALID_ADDRESS;

    // Scope for mutex locker:
    {
        Mutex::Locker locker(m_impl_function_mutex);
        
        // First stage is to make the ClangUtility to hold our injected function:

    #define USE_BUILTIN_FUNCTION 0  // Define this to 1 and we will use the get_implementation function found in the target.
                        // This is useful for debugging additions to the get_impl function 'cause you don't have
                        // to bother with string-ifying the code into g_lookup_implementation_function_code.
        
        if (USE_BUILTIN_FUNCTION)
        {
            ConstString our_utility_function_name("__lldb_objc_find_implementation_for_selector");
            SymbolContextList sc_list;
            
            exe_ctx.GetTargetRef().GetImages().FindSymbolsWithNameAndType (our_utility_function_name, eSymbolTypeCode, sc_list);
            if (sc_list.GetSize() == 1)
            {
                SymbolContext sc;
                sc_list.GetContextAtIndex(0, sc);
                if (sc.symbol != NULL)
                    impl_code_address = sc.symbol->GetAddress();
                    
                //lldb::addr_t addr = impl_code_address.GetOpcodeLoadAddress (exe_ctx.GetTargetPtr());
                //printf ("Getting address for our_utility_function: 0x%llx.\n", addr);
            }
            else
            {
                //printf ("Could not find implementation function address.\n");
                return args_addr;
            }
        }
        else if (!m_impl_code.get())
        {
            m_impl_code.reset (new ClangUtilityFunction (g_lookup_implementation_function_code,
                                                         g_lookup_implementation_function_name));
            if (!m_impl_code->Install(errors, exe_ctx))
            {
                if (log)
                    log->Printf ("Failed to install implementation lookup: %s.", errors.GetData());
                m_impl_code.reset();
                return args_addr;
            }
            impl_code_address.Clear();
            impl_code_address.SetOffset(m_impl_code->StartAddress());
        }
        else
        {
            impl_code_address.Clear();
            impl_code_address.SetOffset(m_impl_code->StartAddress());
        }

        // Next make the runner function for our implementation utility function.
        if (!m_impl_function.get())
        {
             ClangASTContext *clang_ast_context = thread.GetProcess()->GetTarget().GetScratchClangASTContext();
            lldb::clang_type_t clang_void_ptr_type = clang_ast_context->GetVoidPtrType(false);
             m_impl_function.reset(new ClangFunction (thread,
                                                      clang_ast_context, 
                                                      clang_void_ptr_type, 
                                                      impl_code_address, 
                                                      dispatch_values));
            
            errors.Clear();        
            unsigned num_errors = m_impl_function->CompileFunction(errors);
            if (num_errors)
            {
                if (log)
                    log->Printf ("Error compiling function: \"%s\".", errors.GetData());
                return args_addr;
            }
            
            errors.Clear();
            if (!m_impl_function->WriteFunctionWrapper(exe_ctx, errors))
            {
                if (log)
                    log->Printf ("Error Inserting function: \"%s\".", errors.GetData());
                return args_addr;
            }
        }
    }
    
    errors.Clear();
    
    // Now write down the argument values for this particular call.  This looks like it might be a race condition
    // if other threads were calling into here, but actually it isn't because we allocate a new args structure for
    // this call by passing args_addr = LLDB_INVALID_ADDRESS...

    if (!m_impl_function->WriteFunctionArguments (exe_ctx, args_addr, impl_code_address, dispatch_values, errors))
    {
        if (log)
            log->Printf ("Error writing function arguments: \"%s\".", errors.GetData());
        return args_addr;
    }
        
    return args_addr;
}

ThreadPlanSP
AppleObjCTrampolineHandler::GetStepThroughDispatchPlan (Thread &thread, bool stop_others)
{
    ThreadPlanSP ret_plan_sp;
    lldb::addr_t curr_pc = thread.GetRegisterContext()->GetPC();
    
    DispatchFunction this_dispatch;
    bool found_it = false;
    
    // First step is to look and see if we are in one of the known ObjC dispatch functions.  We've already compiled
    // a table of same, so consult it.
    
    MsgsendMap::iterator pos;
    pos = m_msgSend_map.find (curr_pc);
    if (pos != m_msgSend_map.end())
    {
        this_dispatch = g_dispatch_functions[(*pos).second];
        found_it = true;
    }
    
    // Next check to see if we are in a vtable region:
    
    if (!found_it)
    {
        uint32_t flags;
        if (m_vtables_ap.get())
        {
            found_it = m_vtables_ap->IsAddressInVTables (curr_pc, flags);
            if (found_it)
            {
                this_dispatch.name = "vtable";
                this_dispatch.stret_return 
                        = (flags & AppleObjCVTables::eOBJC_TRAMPOLINE_STRET) == AppleObjCVTables::eOBJC_TRAMPOLINE_STRET;
                this_dispatch.is_super = false;
                this_dispatch.is_super2 = false;
                this_dispatch.fixedup = DispatchFunction::eFixUpFixed;
            }
        }
    }
    
    if (found_it)
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

        // We are decoding a method dispatch.  
        // First job is to pull the arguments out:
        
        lldb::StackFrameSP thread_cur_frame = thread.GetStackFrameAtIndex(0);
        
        const ABI *abi = NULL;
        ProcessSP process_sp (thread.CalculateProcess());
        if (process_sp)
            abi = process_sp->GetABI().get();
        if (abi == NULL)
            return ret_plan_sp;
            
        TargetSP target_sp (thread.CalculateTarget());
        
        ClangASTContext *clang_ast_context = target_sp->GetScratchClangASTContext();
        ValueList argument_values;
        Value void_ptr_value;
        lldb::clang_type_t clang_void_ptr_type = clang_ast_context->GetVoidPtrType(false);
        void_ptr_value.SetValueType (Value::eValueTypeScalar);
        void_ptr_value.SetContext (Value::eContextTypeClangType, clang_void_ptr_type);
        
        int obj_index;
        int sel_index;
        
        // If this is a struct return dispatch, then the first argument is the
        // return struct pointer, and the object is the second, and the selector is the third.
        // Otherwise the object is the first and the selector the second.
        if (this_dispatch.stret_return)
        {
            obj_index = 1;
            sel_index = 2;
            argument_values.PushValue(void_ptr_value);
            argument_values.PushValue(void_ptr_value);
            argument_values.PushValue(void_ptr_value);
        }
        else
        {
            obj_index = 0;
            sel_index = 1;
            argument_values.PushValue(void_ptr_value);
            argument_values.PushValue(void_ptr_value);
        }

        
        bool success = abi->GetArgumentValues (thread, argument_values);
        if (!success)
            return ret_plan_sp;
            
        ExecutionContext exe_ctx (thread.shared_from_this());
        Process *process = exe_ctx.GetProcessPtr();
        // isa_addr will store the class pointer that the method is being dispatched to - so either the class
        // directly or the super class if this is one of the objc_msgSendSuper flavors.  That's mostly used to
        // look up the class/selector pair in our cache.
        
        lldb::addr_t isa_addr = LLDB_INVALID_ADDRESS;
        lldb::addr_t sel_addr = argument_values.GetValueAtIndex(sel_index)->GetScalar().ULongLong();
         
        // Figure out the class this is being dispatched to and see if we've already cached this method call,
        // If so we can push a run-to-address plan directly.  Otherwise we have to figure out where
        // the implementation lives.

        if (this_dispatch.is_super)
        {
            if (this_dispatch.is_super2)
            {
               // In the objc_msgSendSuper2 case, we don't get the object directly, we get a structure containing
               // the object and the class to which the super message is being sent.  So we need to dig the super
               // out of the class and use that.
               
                Value super_value(*(argument_values.GetValueAtIndex(obj_index)));
                super_value.GetScalar() += process->GetAddressByteSize();
                super_value.ResolveValue (&exe_ctx, clang_ast_context->getASTContext());
                
                if (super_value.GetScalar().IsValid())
                {
                
                    // isa_value now holds the class pointer.  The second word of the class pointer is the super-class pointer:
                    super_value.GetScalar() += process->GetAddressByteSize();
                    super_value.ResolveValue (&exe_ctx, clang_ast_context->getASTContext());
                    if (super_value.GetScalar().IsValid())
                        isa_addr = super_value.GetScalar().ULongLong();
                    else
                    {
                       if (log)
                        log->Printf("Failed to extract the super class value from the class in objc_super.");
                    }
                }
                else
                {
                   if (log)
                    log->Printf("Failed to extract the class value from objc_super.");
                }
            }
            else
            {
               // In the objc_msgSendSuper case, we don't get the object directly, we get a two element structure containing
               // the object and the super class to which the super message is being sent.  So the class we want is
               // the second element of this structure.
               
                Value super_value(*(argument_values.GetValueAtIndex(obj_index)));
                super_value.GetScalar() += process->GetAddressByteSize();
                super_value.ResolveValue (&exe_ctx, clang_ast_context->getASTContext());
                
                if (super_value.GetScalar().IsValid())
                {
                    isa_addr = super_value.GetScalar().ULongLong();
                }
                else
                {
                   if (log)
                    log->Printf("Failed to extract the class value from objc_super.");
                }
            }
        }
        else
        {
            // In the direct dispatch case, the object->isa is the class pointer we want.
            
            // This is a little cheesy, but since object->isa is the first field,
            // making the object value a load address value and resolving it will get
            // the pointer sized data pointed to by that value...
            
            // Note, it isn't a fatal error not to be able to get the address from the object, since this might
            // be a "tagged pointer" which isn't a real object, but rather some word length encoded dingus.
            
            Value isa_value(*(argument_values.GetValueAtIndex(obj_index)));

            isa_value.SetValueType(Value::eValueTypeLoadAddress);
            isa_value.ResolveValue(&exe_ctx, clang_ast_context->getASTContext());
            if (isa_value.GetScalar().IsValid())
            {
                isa_addr = isa_value.GetScalar().ULongLong();
            }
            else
            {
               if (log)
                log->Printf("Failed to extract the isa value from object.");
            }

        } 
        
        // Okay, we've got the address of the class for which we're resolving this, let's see if it's in our cache:
        lldb::addr_t impl_addr = LLDB_INVALID_ADDRESS;
        
        if (isa_addr != LLDB_INVALID_ADDRESS)
        {
            if (log)
            {
                log->Printf("Resolving call for class - 0x%llx and selector - 0x%llx",
                            isa_addr, sel_addr);
            }
            ObjCLanguageRuntime *objc_runtime = m_process_sp->GetObjCLanguageRuntime ();
            assert(objc_runtime != NULL);
            
            impl_addr = objc_runtime->LookupInMethodCache (isa_addr, sel_addr);
        }                                      
                                                                                                                          
        if (impl_addr != LLDB_INVALID_ADDRESS)
        {
            // Yup, it was in the cache, so we can run to that address directly.
            
            if (log)
                log->Printf ("Found implementation address in cache: 0x%llx", impl_addr);
                 
            ret_plan_sp.reset (new ThreadPlanRunToAddress (thread, impl_addr, stop_others));
        }
        else
        {
            // We haven't seen this class/selector pair yet.  Look it up.
            StreamString errors;
            Address impl_code_address;
            
            ValueList dispatch_values;
            
            // We've will inject a little function in the target that takes the object, selector and some flags,
            // and figures out the implementation.  Looks like:
            //      void *__lldb_objc_find_implementation_for_selector (void *object, 
            //                                                          void *sel, 
            //                                                          int is_stret, 
            //                                                          int is_super, 
            //                                                          int is_super2, 
            //                                                          int is_fixup, 
            //                                                          int is_fixed,
            //                                                          int debug)
            // So set up the arguments for that call.
            
            dispatch_values.PushValue (*(argument_values.GetValueAtIndex(obj_index)));
            dispatch_values.PushValue (*(argument_values.GetValueAtIndex(sel_index)));
            
            Value flag_value;
            lldb::clang_type_t clang_int_type 
                    = clang_ast_context->GetBuiltinTypeForEncodingAndBitSize(lldb::eEncodingSint, 32);
            flag_value.SetValueType (Value::eValueTypeScalar);
            flag_value.SetContext (Value::eContextTypeClangType, clang_int_type);
            
            if (this_dispatch.stret_return)
                flag_value.GetScalar() = 1;
            else
                flag_value.GetScalar() = 0;
            dispatch_values.PushValue (flag_value);
                    
            if (this_dispatch.is_super)
                flag_value.GetScalar() = 1;
            else
                flag_value.GetScalar() = 0;
            dispatch_values.PushValue (flag_value);
                    
            if (this_dispatch.is_super2)
                flag_value.GetScalar() = 1;
            else
                flag_value.GetScalar() = 0;
            dispatch_values.PushValue (flag_value);
                    
            switch (this_dispatch.fixedup)
            {
              case DispatchFunction::eFixUpNone:
                 flag_value.GetScalar() = 0;
                 dispatch_values.PushValue (flag_value);
                 dispatch_values.PushValue (flag_value);
                 break;
              case DispatchFunction::eFixUpFixed:
                 flag_value.GetScalar() = 1;
                 dispatch_values.PushValue (flag_value);
                 flag_value.GetScalar() = 1;
                 dispatch_values.PushValue (flag_value);
                 break;
              case DispatchFunction::eFixUpToFix:
                 flag_value.GetScalar() = 1;
                 dispatch_values.PushValue (flag_value);
                 flag_value.GetScalar() = 0;
                 dispatch_values.PushValue (flag_value);
                 break;
            }
            if (log && log->GetVerbose())
                flag_value.GetScalar() = 1;
            else
                flag_value.GetScalar() = 0;  // FIXME - Set to 0 when debugging is done.
            dispatch_values.PushValue (flag_value);
            
            ret_plan_sp.reset (new AppleThreadPlanStepThroughObjCTrampoline (thread,
                                                                             this,
                                                                             dispatch_values,
                                                                             isa_addr,
                                                                             sel_addr,
                                                                             stop_others));
            if (log)
            {
                StreamString s;
                ret_plan_sp->GetDescription(&s, eDescriptionLevelFull);
                log->Printf("Using ObjC step plan: %s.\n", s.GetData());
            }
        }
    }
    
    return ret_plan_sp;
}

ClangFunction *
AppleObjCTrampolineHandler::GetLookupImplementationWrapperFunction ()
{
    return m_impl_function.get();
}
