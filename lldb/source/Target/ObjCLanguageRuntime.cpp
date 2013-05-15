//===-- ObjCLanguageRuntime.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/Type.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/MappedHash.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ObjCLanguageRuntime::~ObjCLanguageRuntime()
{
}

ObjCLanguageRuntime::ObjCLanguageRuntime (Process *process) :
    LanguageRuntime (process),
    m_has_new_literals_and_indexing (eLazyBoolCalculate),
    m_isa_to_descriptor(),
    m_isa_to_descriptor_stop_id (UINT32_MAX)
{

}

bool
ObjCLanguageRuntime::AddClass (ObjCISA isa, const ClassDescriptorSP &descriptor_sp, const char *class_name)
{
    if (isa != 0)
    {
        m_isa_to_descriptor[isa] = descriptor_sp;
        // class_name is assumed to be valid
        m_hash_to_isa_map.insert(std::make_pair(MappedHash::HashStringUsingDJB(class_name), isa));
        return true;
    }
    return false;
}

void
ObjCLanguageRuntime::AddToMethodCache (lldb::addr_t class_addr, lldb::addr_t selector, lldb::addr_t impl_addr)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {
        log->Printf ("Caching: class 0x%" PRIx64 " selector 0x%" PRIx64 " implementation 0x%" PRIx64 ".", class_addr, selector, impl_addr);
    }
    m_impl_cache.insert (std::pair<ClassAndSel,lldb::addr_t> (ClassAndSel(class_addr, selector), impl_addr));
}

lldb::addr_t
ObjCLanguageRuntime::LookupInMethodCache (lldb::addr_t class_addr, lldb::addr_t selector)
{
    MsgImplMap::iterator pos, end = m_impl_cache.end();
    pos = m_impl_cache.find (ClassAndSel(class_addr, selector));
    if (pos != end)
        return (*pos).second;
    return LLDB_INVALID_ADDRESS;
}


lldb::TypeSP
ObjCLanguageRuntime::LookupInCompleteClassCache (ConstString &name)
{
    CompleteClassMap::iterator complete_class_iter = m_complete_class_cache.find(name);
    
    if (complete_class_iter != m_complete_class_cache.end())
    {
        // Check the weak pointer to make sure the type hasn't been unloaded
        TypeSP complete_type_sp (complete_class_iter->second.lock());
        
        if (complete_type_sp)
            return complete_type_sp;
        else
            m_complete_class_cache.erase(name);
    }
    
    if (m_negative_complete_class_cache.count(name) > 0)
        return TypeSP();
    
    const ModuleList &modules = m_process->GetTarget().GetImages();

    SymbolContextList sc_list;
    const size_t matching_symbols = modules.FindSymbolsWithNameAndType (name,
                                                                        eSymbolTypeObjCClass,
                                                                        sc_list);
    
    if (matching_symbols)
    {
        SymbolContext sc;
        
        sc_list.GetContextAtIndex(0, sc);
        
        ModuleSP module_sp(sc.module_sp);
        
        if (!module_sp)
            return TypeSP();
        
        const SymbolContext null_sc;
        const bool exact_match = true;
        const uint32_t max_matches = UINT32_MAX;
        TypeList types;
        
        const uint32_t num_types = module_sp->FindTypes (null_sc,
                                                         name,
                                                         exact_match,
                                                         max_matches,
                                                         types);
        
        if (num_types)
        {            
            uint32_t i;
            for (i = 0; i < num_types; ++i)
            {
                TypeSP type_sp (types.GetTypeAtIndex(i));
                
                if (ClangASTContext::IsObjCClassType(type_sp->GetClangForwardType()))
                {
                    if (type_sp->IsCompleteObjCClass())
                    {
                        m_complete_class_cache[name] = type_sp;
                        return type_sp;
                    }
                }
            }
        }
    }
    m_negative_complete_class_cache.insert(name);
    return TypeSP();
}

size_t
ObjCLanguageRuntime::GetByteOffsetForIvar (ClangASTType &parent_qual_type, const char *ivar_name)
{
    return LLDB_INVALID_IVAR_OFFSET;
}

void
ObjCLanguageRuntime::MethodName::Clear()
{
    m_full.Clear();
    m_class.Clear();
    m_category.Clear();
    m_selector.Clear();
    m_type = eTypeUnspecified;
    m_category_is_valid = false;
}

//bool
//ObjCLanguageRuntime::MethodName::SetName (const char *name, bool strict)
//{
//    Clear();
//    if (name && name[0])
//    {
//        // If "strict" is true. then the method must be specified with a
//        // '+' or '-' at the beginning. If "strict" is false, then the '+'
//        // or '-' can be omitted
//        bool valid_prefix = false;
//        
//        if (name[0] == '+' || name[0] == '-')
//        {
//            valid_prefix = name[1] == '[';
//        }
//        else if (!strict)
//        {
//            // "strict" is false, the name just needs to start with '['
//            valid_prefix = name[0] == '[';
//        }
//
//        if (valid_prefix)
//        {
//            static RegularExpression g_regex("^([-+]?)\\[([A-Za-z_][A-Za-z_0-9]*)(\\([A-Za-z_][A-Za-z_0-9]*\\))? ([A-Za-z_][A-Za-z_0-9:]*)\\]$");
//            llvm::StringRef matches[4];
//            // Since we are using a global regular expression, we must use the threadsafe version of execute
//            if (g_regex.ExecuteThreadSafe(name, matches, 4))
//            {
//                m_full.SetCString(name);
//                if (matches[0].empty())
//                    m_type = eTypeUnspecified;
//                else if (matches[0][0] == '+')
//                    m_type = eTypeClassMethod;
//                else
//                    m_type = eTypeInstanceMethod;
//                m_class.SetString(matches[1]);
//                m_selector.SetString(matches[3]);
//                if (!matches[2].empty())
//                    m_category.SetString(matches[2]);
//            }
//        }
//    }
//    return IsValid(strict);
//}

bool
ObjCLanguageRuntime::MethodName::SetName (const char *name, bool strict)
{
    Clear();
    if (name && name[0])
    {
        // If "strict" is true. then the method must be specified with a
        // '+' or '-' at the beginning. If "strict" is false, then the '+'
        // or '-' can be omitted
        bool valid_prefix = false;
        
        if (name[0] == '+' || name[0] == '-')
        {
            valid_prefix = name[1] == '[';
            if (name[0] == '+')
                m_type = eTypeClassMethod;
            else
                m_type = eTypeInstanceMethod;
        }
        else if (!strict)
        {
            // "strict" is false, the name just needs to start with '['
            valid_prefix = name[0] == '[';
        }
        
        if (valid_prefix)
        {
            int name_len = strlen (name);
            // Objective C methods must have at least:
            //      "-[" or "+[" prefix
            //      One character for a class name
            //      One character for the space between the class name
            //      One character for the method name
            //      "]" suffix
            if (name_len >= (5 + (strict ? 1 : 0)) && name[name_len - 1] == ']')
            {
                m_full.SetCStringWithLength(name, name_len);
            }
        }
    }
    return IsValid(strict);
}

const ConstString &
ObjCLanguageRuntime::MethodName::GetClassName ()
{
    if (!m_class)
    {
        if (IsValid(false))
        {
            const char *full = m_full.GetCString();
            const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
            const char *paren_pos = strchr (class_start, '(');
            if (paren_pos)
            {
                m_class.SetCStringWithLength (class_start, paren_pos - class_start);
            }
            else
            {
                // No '(' was found in the full name, we can definitively say
                // that our category was valid (and empty).
                m_category_is_valid = true;
                const char *space_pos = strchr (full, ' ');
                if (space_pos)
                {
                    m_class.SetCStringWithLength (class_start, space_pos - class_start);
                    if (!m_class_category)
                    {
                        // No category in name, so we can also fill in the m_class_category
                        m_class_category = m_class;
                    }
                }
            }
        }
    }
    return m_class;
}

const ConstString &
ObjCLanguageRuntime::MethodName::GetClassNameWithCategory () 
{
    if (!m_class_category)
    {
        if (IsValid(false))
        {
            const char *full = m_full.GetCString();
            const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
            const char *space_pos = strchr (full, ' ');
            if (space_pos)
            {
                m_class_category.SetCStringWithLength (class_start, space_pos - class_start);
                // If m_class hasn't been filled in and the class with category doesn't
                // contain a '(', then we can also fill in the m_class
                if (!m_class && strchr (m_class_category.GetCString(), '(') == NULL)
                {
                    m_class = m_class_category;
                    // No '(' was found in the full name, we can definitively say
                    // that our category was valid (and empty).
                    m_category_is_valid = true;

                }
            }
        }
    }
    return m_class_category;
}

const ConstString &
ObjCLanguageRuntime::MethodName::GetSelector ()
{
    if (!m_selector)
    {
        if (IsValid(false))
        {
            const char *full = m_full.GetCString();
            const char *space_pos = strchr (full, ' ');
            if (space_pos)
            {
                ++space_pos; // skip the space
                m_selector.SetCStringWithLength (space_pos, m_full.GetLength() - (space_pos - full) - 1);
            }
        }
    }
    return m_selector;
}

const ConstString &
ObjCLanguageRuntime::MethodName::GetCategory ()
{
    if (!m_category_is_valid && !m_category)
    {
        if (IsValid(false))
        {
            m_category_is_valid = true;
            const char *full = m_full.GetCString();
            const char *class_start = (full[0] == '[' ? full + 1 : full + 2);
            const char *open_paren_pos = strchr (class_start, '(');
            if (open_paren_pos)
            {
                ++open_paren_pos; // Skip the open paren
                const char *close_paren_pos = strchr (open_paren_pos, ')');
                if (close_paren_pos)
                    m_category.SetCStringWithLength (open_paren_pos, close_paren_pos - open_paren_pos);
            }
        }
    }
    return m_category;
}

ConstString
ObjCLanguageRuntime::MethodName::GetFullNameWithoutCategory (bool empty_if_no_category)
{
    if (IsValid(false))
    {
        if (HasCategory())
        {
            StreamString strm;
            if (m_type == eTypeClassMethod)
                strm.PutChar('+');
            else if (m_type == eTypeInstanceMethod)
                strm.PutChar('-');
            strm.Printf("[%s %s]", GetClassName().GetCString(), GetSelector().GetCString());
            return ConstString(strm.GetString().c_str());
        }
        
        if (!empty_if_no_category)
        {
            // Just return the full name since it doesn't have a category
            return GetFullName();
        }
    }
    return ConstString();
}

size_t
ObjCLanguageRuntime::MethodName::GetFullNames (std::vector<ConstString> &names, bool append)
{
    if (!append)
        names.clear();
    if (IsValid(false))
    {
        StreamString strm;
        const bool is_class_method = m_type == eTypeClassMethod;
        const bool is_instance_method = m_type == eTypeInstanceMethod;
        const ConstString &category = GetCategory();
        if (is_class_method || is_instance_method)
        {
            names.push_back (m_full);
            if (category)
            {
                strm.Printf("%c[%s %s]",
                            is_class_method ? '+' : '-',
                            GetClassName().GetCString(),
                            GetSelector().GetCString());
                names.push_back(ConstString(strm.GetString().c_str()));
            }
        }
        else
        {
            const ConstString &class_name = GetClassName();
            const ConstString &selector = GetSelector();
            strm.Printf("+[%s %s]", class_name.GetCString(), selector.GetCString());
            names.push_back(ConstString(strm.GetString().c_str()));
            strm.Clear();
            strm.Printf("-[%s %s]", class_name.GetCString(), selector.GetCString());
            names.push_back(ConstString(strm.GetString().c_str()));
            strm.Clear();
            if (category)
            {
                strm.Printf("+[%s(%s) %s]", class_name.GetCString(), category.GetCString(), selector.GetCString());
                names.push_back(ConstString(strm.GetString().c_str()));
                strm.Clear();
                strm.Printf("-[%s(%s) %s]", class_name.GetCString(), category.GetCString(), selector.GetCString());
                names.push_back(ConstString(strm.GetString().c_str()));
            }
        }
    }
    return names.size();
}


bool
ObjCLanguageRuntime::ClassDescriptor::IsPointerValid (lldb::addr_t value,
                                                      uint32_t ptr_size,
                                                      bool allow_NULLs,
                                                      bool allow_tagged,
                                                      bool check_version_specific) const
{
    if (!value)
        return allow_NULLs;
    if ( (value % 2) == 1  && allow_tagged)
        return true;
    if ((value % ptr_size) == 0)
        return (check_version_specific ? CheckPointer(value,ptr_size) : true);
    else
        return false;
}

ObjCLanguageRuntime::ObjCISA
ObjCLanguageRuntime::GetISA(const ConstString &name)
{
    ISAToDescriptorIterator pos = GetDescriptorIterator (name);
    if (pos != m_isa_to_descriptor.end())
        return pos->first;
    return 0;
}

ObjCLanguageRuntime::ISAToDescriptorIterator
ObjCLanguageRuntime::GetDescriptorIterator (const ConstString &name)
{
    ISAToDescriptorIterator end = m_isa_to_descriptor.end();

    if (name)
    {
        UpdateISAToDescriptorMap();
        if (m_hash_to_isa_map.empty())
        {
            // No name hashes were provided, we need to just linearly power through the
            // names and find a match
            for (ISAToDescriptorIterator pos = m_isa_to_descriptor.begin(); pos != end; ++pos)
            {
                if (pos->second->GetClassName() == name)
                    return pos;
            }
        }
        else
        {
            // Name hashes were provided, so use them to efficiently lookup name to isa/descriptor
            const uint32_t name_hash = MappedHash::HashStringUsingDJB (name.GetCString());
            std::pair <HashToISAIterator, HashToISAIterator> range = m_hash_to_isa_map.equal_range(name_hash);
            for (HashToISAIterator range_pos = range.first; range_pos != range.second; ++range_pos)
            {
                ISAToDescriptorIterator pos = m_isa_to_descriptor.find (range_pos->second);
                if (pos != m_isa_to_descriptor.end())
                {
                    if (pos->second->GetClassName() == name)
                        return pos;
                }
            }
        }
    }
    return end;
}


ObjCLanguageRuntime::ObjCISA
ObjCLanguageRuntime::GetParentClass(ObjCLanguageRuntime::ObjCISA isa)
{
    ClassDescriptorSP objc_class_sp (GetClassDescriptorFromISA(isa));
    if (objc_class_sp)
    {
        ClassDescriptorSP objc_super_class_sp (objc_class_sp->GetSuperclass());
        if (objc_super_class_sp)
            return objc_super_class_sp->GetISA();
    }
    return 0;
}

ConstString
ObjCLanguageRuntime::GetActualTypeName(ObjCLanguageRuntime::ObjCISA isa)
{
    ClassDescriptorSP objc_class_sp (GetNonKVOClassDescriptor(isa));
    if (objc_class_sp)
        return objc_class_sp->GetClassName();
    return ConstString();
}

ObjCLanguageRuntime::ClassDescriptorSP
ObjCLanguageRuntime::GetClassDescriptorFromClassName (const ConstString &class_name)
{
    ISAToDescriptorIterator pos = GetDescriptorIterator (class_name);
    if (pos != m_isa_to_descriptor.end())
        return pos->second;
    return ClassDescriptorSP();

}

ObjCLanguageRuntime::ClassDescriptorSP
ObjCLanguageRuntime::GetClassDescriptor (ValueObject& valobj)
{
    ClassDescriptorSP objc_class_sp;
    // if we get an invalid VO (which might still happen when playing around
    // with pointers returned by the expression parser, don't consider this
    // a valid ObjC object)
    if (valobj.GetValue().GetContextType() != Value::eContextTypeInvalid)
    {
        addr_t isa_pointer = valobj.GetPointerValue();
        if (isa_pointer != LLDB_INVALID_ADDRESS)
        {
            ExecutionContext exe_ctx (valobj.GetExecutionContextRef());
            
            Process *process = exe_ctx.GetProcessPtr();
            if (process)
            {
                Error error;
                ObjCISA isa = process->ReadPointerFromMemory(isa_pointer, error);
                if (isa != LLDB_INVALID_ADDRESS)
                    objc_class_sp = GetClassDescriptorFromISA (isa);
            }
        }
    }
    return objc_class_sp;
}

ObjCLanguageRuntime::ClassDescriptorSP
ObjCLanguageRuntime::GetNonKVOClassDescriptor (ValueObject& valobj)
{
    ObjCLanguageRuntime::ClassDescriptorSP objc_class_sp (GetClassDescriptor (valobj));
    if (objc_class_sp)
    {
        if (!objc_class_sp->IsKVO())
            return objc_class_sp;
        
        ClassDescriptorSP non_kvo_objc_class_sp(objc_class_sp->GetSuperclass());
        if (non_kvo_objc_class_sp && non_kvo_objc_class_sp->IsValid())
            return non_kvo_objc_class_sp;
    }
    return ClassDescriptorSP();
}


ObjCLanguageRuntime::ClassDescriptorSP
ObjCLanguageRuntime::GetClassDescriptorFromISA (ObjCISA isa)
{
    if (isa)
    {
        UpdateISAToDescriptorMap();
        ObjCLanguageRuntime::ISAToDescriptorIterator pos = m_isa_to_descriptor.find(isa);    
        if (pos != m_isa_to_descriptor.end())
            return pos->second;
    }
    return ClassDescriptorSP();
}

ObjCLanguageRuntime::ClassDescriptorSP
ObjCLanguageRuntime::GetNonKVOClassDescriptor (ObjCISA isa)
{
    if (isa)
    {
        ClassDescriptorSP objc_class_sp = GetClassDescriptorFromISA (isa);
        if (objc_class_sp && objc_class_sp->IsValid())
        {
            if (!objc_class_sp->IsKVO())
                return objc_class_sp;

            ClassDescriptorSP non_kvo_objc_class_sp(objc_class_sp->GetSuperclass());
            if (non_kvo_objc_class_sp && non_kvo_objc_class_sp->IsValid())
                return non_kvo_objc_class_sp;
        }
    }
    return ClassDescriptorSP();
}



