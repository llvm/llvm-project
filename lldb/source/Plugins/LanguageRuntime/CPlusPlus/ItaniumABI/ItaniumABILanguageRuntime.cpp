//===-- ItaniumABILanguageRuntime.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ItaniumABILanguageRuntime.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

static const char *vtable_demangled_prefix = "vtable for ";

bool
ItaniumABILanguageRuntime::CouldHaveDynamicValue (ValueObject &in_value)
{
    const bool check_cxx = true;
    const bool check_objc = false;
    return in_value.GetCompilerType().IsPossibleDynamicType (NULL, check_cxx, check_objc);
}

bool
ItaniumABILanguageRuntime::GetDynamicTypeAndAddress (ValueObject &in_value, 
                                                     lldb::DynamicValueType use_dynamic, 
                                                     TypeAndOrName &class_type_or_name, 
                                                     Address &dynamic_address,
                                                     Value::ValueType &value_type)
{
    // For Itanium, if the type has a vtable pointer in the object, it will be at offset 0
    // in the object.  That will point to the "address point" within the vtable (not the beginning of the
    // vtable.)  We can then look up the symbol containing this "address point" and that symbol's name 
    // demangled will contain the full class name.
    // The second pointer above the "address point" is the "offset_to_top".  We'll use that to get the
    // start of the value object which holds the dynamic type.
    //
    
    class_type_or_name.Clear();
    value_type = Value::ValueType::eValueTypeScalar;
    
    // Only a pointer or reference type can have a different dynamic and static type:
    if (CouldHaveDynamicValue (in_value))
    {
        // First job, pull out the address at 0 offset from the object.
        AddressType address_type;
        lldb::addr_t original_ptr = in_value.GetPointerValue(&address_type);
        if (original_ptr == LLDB_INVALID_ADDRESS)
            return false;
        
        ExecutionContext exe_ctx (in_value.GetExecutionContextRef());

        Target *target = exe_ctx.GetTargetPtr();
        Process *process = exe_ctx.GetProcessPtr();

        char memory_buffer[16];
        DataExtractor data(memory_buffer, sizeof(memory_buffer), 
                           process->GetByteOrder(), 
                           process->GetAddressByteSize());
        size_t address_byte_size = process->GetAddressByteSize();
        Error error;
        size_t bytes_read = process->ReadMemory (original_ptr, 
                                                 memory_buffer, 
                                                 address_byte_size, 
                                                 error);
        if (!error.Success() || (bytes_read != address_byte_size))
        {
            return false;
        }
        
        lldb::offset_t offset = 0;
        lldb::addr_t vtable_address_point = data.GetAddress (&offset);
            
        if (offset == 0)
            return false;
        
        // Now find the symbol that contains this address:
        
        SymbolContext sc;
        Address address_point_address;
        if (target && !target->GetSectionLoadList().IsEmpty())
        {
            if (target->GetSectionLoadList().ResolveLoadAddress (vtable_address_point, address_point_address))
            {
                target->GetImages().ResolveSymbolContextForAddress (address_point_address, eSymbolContextSymbol, sc);
                Symbol *symbol = sc.symbol;
                if (symbol != NULL)
                {
                    const char *name = symbol->GetMangled().GetDemangledName(lldb::eLanguageTypeC_plus_plus).AsCString();
                    if (name && strstr(name, vtable_demangled_prefix) == name)
                    {
                        Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
                        if (log)
                            log->Printf ("0x%16.16" PRIx64 ": static-type = '%s' has vtable symbol '%s'\n",
                                         original_ptr,
                                         in_value.GetTypeName().GetCString(),
                                         name);
                        // We are a C++ class, that's good.  Get the class name and look it up:
                        const char *class_name = name + strlen(vtable_demangled_prefix);
                        class_type_or_name.SetName (class_name);
                        const bool exact_match = true;
                        TypeList class_types;
                        
                        uint32_t num_matches = 0;
                        // First look in the module that the vtable symbol came from
                        // and look for a single exact match.
                        if (sc.module_sp)
                        {
                            num_matches = sc.module_sp->FindTypes (sc,
                                                                   ConstString(class_name),
                                                                   exact_match,
                                                                   1,
                                                                   class_types);
                        }
                        
                        // If we didn't find a symbol, then move on to the entire
                        // module list in the target and get as many unique matches
                        // as possible
                        if (num_matches == 0)
                        {
                            num_matches = target->GetImages().FindTypes (sc,
                                                                         ConstString(class_name),
                                                                         exact_match,
                                                                         UINT32_MAX,
                                                                         class_types);
                        }
                        
                        lldb::TypeSP type_sp;
                        if (num_matches == 0)
                        {
                            if (log)
                                log->Printf("0x%16.16" PRIx64 ": is not dynamic\n", original_ptr);
                            return false;
                        }
                        if (num_matches == 1)
                        {
                            type_sp = class_types.GetTypeAtIndex(0);
                            if (log)
                                log->Printf ("0x%16.16" PRIx64 ": static-type = '%s' has dynamic type: uid={0x%" PRIx64 "}, type-name='%s'\n",
                                             original_ptr,
                                             in_value.GetTypeName().AsCString(),
                                             type_sp->GetID(),
                                             type_sp->GetName().GetCString());

                            class_type_or_name.SetTypeSP(class_types.GetTypeAtIndex(0));
                        }
                        else if (num_matches > 1)
                        {
                            size_t i;
                            if (log)
                            {
                                for (i = 0; i < num_matches; i++)
                                {
                                    type_sp = class_types.GetTypeAtIndex(i);
                                    if (type_sp)
                                    {
                                        if (log)
                                            log->Printf ("0x%16.16" PRIx64 ": static-type = '%s' has multiple matching dynamic types: uid={0x%" PRIx64 "}, type-name='%s'\n",
                                                         original_ptr,
                                                         in_value.GetTypeName().AsCString(),
                                                         type_sp->GetID(),
                                                         type_sp->GetName().GetCString());
                                    }
                                }
                            }

                            for (i = 0; i < num_matches; i++)
                            {
                                type_sp = class_types.GetTypeAtIndex(i);
                                if (type_sp)
                                {
                                    if (ClangASTContext::IsCXXClassType(type_sp->GetFullCompilerType ()))
                                    {
                                        if (log)
                                            log->Printf ("0x%16.16" PRIx64 ": static-type = '%s' has multiple matching dynamic types, picking this one: uid={0x%" PRIx64 "}, type-name='%s'\n",
                                                         original_ptr,
                                                         in_value.GetTypeName().AsCString(),
                                                         type_sp->GetID(),
                                                         type_sp->GetName().GetCString());
                                        class_type_or_name.SetTypeSP(type_sp);
                                        break;
                                    }
                                }
                            }
                            
                            if (i == num_matches)
                            {
                                if (log)
                                    log->Printf ("0x%16.16" PRIx64 ": static-type = '%s' has multiple matching dynamic types, didn't find a C++ match\n",
                                                 original_ptr,
                                                 in_value.GetTypeName().AsCString());
                                return false;
                            }
                        }

                        // There can only be one type with a given name,
                        // so we've just found duplicate definitions, and this
                        // one will do as well as any other.
                        // We don't consider something to have a dynamic type if
                        // it is the same as the static type.  So compare against
                        // the value we were handed.
                        if (type_sp)
                        {
                            if (ClangASTContext::AreTypesSame (in_value.GetCompilerType(),
                                                               type_sp->GetFullCompilerType ()))
                            {
                                // The dynamic type we found was the same type,
                                // so we don't have a dynamic type here...
                                return false;
                            }

                            // The offset_to_top is two pointers above the address.
                            Address offset_to_top_address = address_point_address;
                            int64_t slide = -2 * ((int64_t) target->GetArchitecture().GetAddressByteSize());
                            offset_to_top_address.Slide (slide);
                            
                            Error error;
                            lldb::addr_t offset_to_top_location = offset_to_top_address.GetLoadAddress(target);
                            
                            size_t bytes_read = process->ReadMemory (offset_to_top_location, 
                                                                     memory_buffer, 
                                                                     address_byte_size, 
                                                                     error);
                                                                     
                            if (!error.Success() || (bytes_read != address_byte_size))
                            {
                                return false;
                            }
                            
                            offset = 0;
                            int64_t offset_to_top = data.GetMaxS64(&offset, process->GetAddressByteSize());
                            
                            // So the dynamic type is a value that starts at offset_to_top
                            // above the original address.
                            lldb::addr_t dynamic_addr = original_ptr + offset_to_top;
                            if (!target->GetSectionLoadList().ResolveLoadAddress (dynamic_addr, dynamic_address))
                            {
                                dynamic_address.SetRawAddress(dynamic_addr);
                            }
                            return true;
                        }
                    }
                }
            }
        }
    }
    
    return class_type_or_name.IsEmpty() == false;
}

TypeAndOrName
ItaniumABILanguageRuntime::FixUpDynamicType(const TypeAndOrName& type_and_or_name,
                                            ValueObject& static_value)
{
    CompilerType static_type(static_value.GetCompilerType());
    Flags static_type_flags(static_type.GetTypeInfo());
    
    TypeAndOrName ret(type_and_or_name);
    if (type_and_or_name.HasType())
    {
        // The type will always be the type of the dynamic object.  If our parent's type was a pointer,
        // then our type should be a pointer to the type of the dynamic object.  If a reference, then the original type
        // should be okay...
        CompilerType orig_type = type_and_or_name.GetCompilerType();
        CompilerType corrected_type = orig_type;
        if (static_type_flags.AllSet(eTypeIsPointer))
            corrected_type = orig_type.GetPointerType ();
        else if (static_type_flags.AllSet(eTypeIsReference))
            corrected_type = orig_type.GetLValueReferenceType();
        ret.SetCompilerType(corrected_type);
    }
    else
    {
        // If we are here we need to adjust our dynamic type name to include the correct & or * symbol
        std::string corrected_name (type_and_or_name.GetName().GetCString());
        if (static_type_flags.AllSet(eTypeIsPointer))
            corrected_name.append(" *");
        else if (static_type_flags.AllSet(eTypeIsReference))
            corrected_name.append(" &");
        // the parent type should be a correctly pointer'ed or referenc'ed type
        ret.SetCompilerType(static_type);
        ret.SetName(corrected_name.c_str());
    }
    return ret;
}

bool
ItaniumABILanguageRuntime::IsVTableName (const char *name)
{
    if (name == NULL)
        return false;
        
    // Can we maybe ask Clang about this?
    if (strstr (name, "_vptr$") == name)
        return true;
    else
        return false;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
LanguageRuntime *
ItaniumABILanguageRuntime::CreateInstance (Process *process, lldb::LanguageType language)
{
    // FIXME: We have to check the process and make sure we actually know that this process supports
    // the Itanium ABI.
    if (language == eLanguageTypeC_plus_plus ||
        language == eLanguageTypeC_plus_plus_03 ||
        language == eLanguageTypeC_plus_plus_11 ||
        language == eLanguageTypeC_plus_plus_14)
        return new ItaniumABILanguageRuntime (process);
    else
        return NULL;
}

void
ItaniumABILanguageRuntime::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "Itanium ABI for the C++ language",
                                   CreateInstance);    
}

void
ItaniumABILanguageRuntime::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

lldb_private::ConstString
ItaniumABILanguageRuntime::GetPluginNameStatic()
{
    static ConstString g_name("itanium");
    return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
ItaniumABILanguageRuntime::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ItaniumABILanguageRuntime::GetPluginVersion()
{
    return 1;
}

BreakpointResolverSP
ItaniumABILanguageRuntime::CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp)
{
    return CreateExceptionResolver (bkpt, catch_bp, throw_bp, false);
}

BreakpointResolverSP
ItaniumABILanguageRuntime::CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp, bool for_expressions)
{
    // One complication here is that most users DON'T want to stop at __cxa_allocate_expression, but until we can do
    // anything better with predicting unwinding the expression parser does.  So we have two forms of the exception
    // breakpoints, one for expressions that leaves out __cxa_allocate_exception, and one that includes it.
    // The SetExceptionBreakpoints does the latter, the CreateExceptionBreakpoint in the runtime the former.
    static const char *g_catch_name = "__cxa_begin_catch";
    static const char *g_throw_name1 = "__cxa_throw";
    static const char *g_throw_name2 = "__cxa_rethrow";
    static const char *g_exception_throw_name = "__cxa_allocate_exception";
    std::vector<const char *> exception_names;
    exception_names.reserve(4);
    if (catch_bp)
        exception_names.push_back(g_catch_name);

    if (throw_bp)
    {
        exception_names.push_back(g_throw_name1);
        exception_names.push_back(g_throw_name2);
    }

    if (for_expressions)
        exception_names.push_back(g_exception_throw_name);
    
    BreakpointResolverSP resolver_sp (new BreakpointResolverName (bkpt,
                                                                  exception_names.data(),
                                                                  exception_names.size(),
                                                                  eFunctionNameTypeBase,
                                                                  eLanguageTypeUnknown,
                                                                  eLazyBoolNo));

    return resolver_sp;
}



lldb::SearchFilterSP
ItaniumABILanguageRuntime::CreateExceptionSearchFilter ()
{
    Target &target = m_process->GetTarget();

    if (target.GetArchitecture().GetTriple().getVendor() == llvm::Triple::Apple)
    {
        // Limit the number of modules that are searched for these breakpoints for
        // Apple binaries.
        FileSpecList filter_modules;
        filter_modules.Append(FileSpec("libc++abi.dylib", false));
        filter_modules.Append(FileSpec("libSystem.B.dylib", false));
        return target.GetSearchFilterForModuleList(&filter_modules);
    }
    else
    {
        return LanguageRuntime::CreateExceptionSearchFilter();
    }
}

lldb::BreakpointSP
ItaniumABILanguageRuntime::CreateExceptionBreakpoint (bool catch_bp,
                                                      bool throw_bp,
                                                      bool for_expressions,
                                                      bool is_internal)
{
    Target &target = m_process->GetTarget();
    FileSpecList filter_modules;
    BreakpointResolverSP exception_resolver_sp = CreateExceptionResolver (NULL, catch_bp, throw_bp, for_expressions);
    SearchFilterSP filter_sp (CreateExceptionSearchFilter ());
    const bool hardware = false;
    const bool resolve_indirect_functions = false;
    return target.CreateBreakpoint (filter_sp, exception_resolver_sp, is_internal, hardware, resolve_indirect_functions);
}

void
ItaniumABILanguageRuntime::SetExceptionBreakpoints ()
{
    if (!m_process)
        return;
    
    const bool catch_bp = false;
    const bool throw_bp = true;
    const bool is_internal = true;
    const bool for_expressions = true;
    
    // For the exception breakpoints set by the Expression parser, we'll be a little more aggressive and
    // stop at exception allocation as well.
    
    if (m_cxx_exception_bp_sp)
    {
        m_cxx_exception_bp_sp->SetEnabled (true);
    }
    else
    {
        m_cxx_exception_bp_sp = CreateExceptionBreakpoint (catch_bp, throw_bp, for_expressions, is_internal);
        if (m_cxx_exception_bp_sp)
            m_cxx_exception_bp_sp->SetBreakpointKind("c++ exception");
    }
    
}

void
ItaniumABILanguageRuntime::ClearExceptionBreakpoints ()
{
    if (!m_process)
        return;
    
    if (m_cxx_exception_bp_sp)
    {
        m_cxx_exception_bp_sp->SetEnabled (false);
    }    
}

bool
ItaniumABILanguageRuntime::ExceptionBreakpointsAreSet ()
{
    return m_cxx_exception_bp_sp && m_cxx_exception_bp_sp->IsEnabled();
}

bool
ItaniumABILanguageRuntime::ExceptionBreakpointsExplainStop (lldb::StopInfoSP stop_reason)
{
    if (!m_process)
        return false;
    
    if (!stop_reason || 
        stop_reason->GetStopReason() != eStopReasonBreakpoint)
        return false;
    
    uint64_t break_site_id = stop_reason->GetValue();
    return m_process->GetBreakpointSiteList().BreakpointSiteContainsBreakpoint(break_site_id,
                                                                               m_cxx_exception_bp_sp->GetID());
    
}
