//===-- CPPLanguageRuntime.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/CPPLanguageRuntime.h"

#include <string.h>

#include "lldb/Core/PluginManager.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Target/ExecutionContext.h"

using namespace lldb;
using namespace lldb_private;

class CPPRuntimeEquivalents
{
public:
    CPPRuntimeEquivalents ()
    {
        
        m_impl.Append(ConstString("std::basic_string<char, std::char_traits<char>, std::allocator<char> >").AsCString(), ConstString("basic_string<char>"));

        // these two (with a prefixed std::) occur when c++stdlib string class occurs as a template argument in some STL container
        m_impl.Append(ConstString("std::basic_string<char, std::char_traits<char>, std::allocator<char> >").AsCString(), ConstString("std::basic_string<char>"));
        
        m_impl.Sort();
    }
    
    void
    Add (ConstString& type_name,
         ConstString& type_equivalent)
    {
        m_impl.Insert(type_name.AsCString(), type_equivalent);
    }
    
    uint32_t
    FindExactMatches (ConstString& type_name,
                      std::vector<ConstString>& equivalents)
    {
        
        uint32_t count = 0;

        for (ImplData match = m_impl.FindFirstValueForName(type_name.AsCString());
             match != NULL;
             match = m_impl.FindNextValueForName(match))
        {
            equivalents.push_back(match->value);
            count++;
        }

        return count;        
    }
    
    // partial matches can occur when a name with equivalents is a template argument.
    // e.g. we may have "class Foo" be a match for "struct Bar". if we have a typename
    // such as "class Templatized<class Foo, Anything>" we want this to be replaced with
    // "class Templatized<struct Bar, Anything>". Since partial matching is time consuming
    // once we get a partial match, we add it to the exact matches list for faster retrieval
    uint32_t
    FindPartialMatches (ConstString& type_name,
                        std::vector<ConstString>& equivalents)
    {
        
        uint32_t count = 0;
        
        const char* type_name_cstr = type_name.AsCString();
        
        size_t items_count = m_impl.GetSize();
        
        for (size_t item = 0; item < items_count; item++)
        {
            const char* key_cstr = m_impl.GetCStringAtIndex(item);
            if ( strstr(type_name_cstr,key_cstr) )
            {
                count += AppendReplacements(type_name_cstr,
                                            key_cstr,
                                            equivalents);
            }
        }
        
        return count;
        
    }
    
private:
    
    std::string& replace (std::string& target,
                          std::string& pattern,
                          std::string& with)
    {
        size_t pos;
        size_t pattern_len = pattern.size();
        
        while ( (pos = target.find(pattern)) != std::string::npos )
            target.replace(pos, pattern_len, with);
        
        return target;
    }
    
    uint32_t
    AppendReplacements (const char* original,
                        const char *matching_key,
                        std::vector<ConstString>& equivalents)
    {
        
        std::string matching_key_str(matching_key);
        ConstString original_const(original);
        
        uint32_t count = 0;
        
        for (ImplData match = m_impl.FindFirstValueForName(matching_key);
             match != NULL;
             match = m_impl.FindNextValueForName(match))
        {
            std::string target(original);
            std::string equiv_class(match->value.AsCString());
            
            replace (target, matching_key_str, equiv_class);
            
            ConstString target_const(target.c_str());

// you will most probably want to leave this off since it might make this map grow indefinitely
#ifdef ENABLE_CPP_EQUIVALENTS_MAP_TO_GROW
            Add(original_const, target_const);
#endif
            equivalents.push_back(target_const);
            
            count++;
        }
        
        return count;
    }
    
    typedef UniqueCStringMap<ConstString> Impl;
    typedef const Impl::Entry* ImplData;
    Impl m_impl;
};

static CPPRuntimeEquivalents&
GetEquivalentsMap ()
{
    static CPPRuntimeEquivalents g_equivalents_map;
    return g_equivalents_map;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CPPLanguageRuntime::~CPPLanguageRuntime()
{
}

CPPLanguageRuntime::CPPLanguageRuntime (Process *process) :
    LanguageRuntime (process)
{

}

bool
CPPLanguageRuntime::GetObjectDescription (Stream &str, ValueObject &object)
{
    // C++ has no generic way to do this.
    return false;
}

bool
CPPLanguageRuntime::GetObjectDescription (Stream &str, Value &value, ExecutionContextScope *exe_scope)
{
    // C++ has no generic way to do this.
    return false;
}

bool
CPPLanguageRuntime::IsCPPMangledName (const char *name)
{
    // FIXME, we should really run through all the known C++ Language plugins and ask each one if
    // this is a C++ mangled name, but we can put that off till there is actually more than one
    // we care about.
    
    if (name && name[0] == '_' && name[1] == 'Z')
        return true;
    else
        return false;
}

bool
CPPLanguageRuntime::StripNamespacesFromVariableName (const char *name, const char *&base_name_start, const char *&base_name_end)
{
    if (base_name_end == NULL)
        base_name_end = name + strlen (name);
    
    const char *last_colon = strrchr (name, ':');
    
    if (last_colon == NULL)
    {
        base_name_start = name;
        return true;
    }
    
    // Can't have a C++ name that begins with a single ':', nor contains an internal single ':'
    if (last_colon == name)
        return false;
    else if (last_colon[-1] != ':')
        return false;
    else
    {
        // FIXME: should check if there is
        base_name_start = last_colon + 1;
        return true;
    }
}

uint32_t
CPPLanguageRuntime::FindEquivalentNames(ConstString type_name, std::vector<ConstString>& equivalents)
{
    uint32_t count = GetEquivalentsMap().FindExactMatches(type_name, equivalents);

    bool might_have_partials= 
        ( count == 0 )  // if we have a full name match just use it
        && (strchr(type_name.AsCString(), '<') != NULL  // we should only have partial matches when templates are involved, check that we have
            && strchr(type_name.AsCString(), '>') != NULL); // angle brackets in the type_name before trying to scan for partial matches
    
    if ( might_have_partials )
        count = GetEquivalentsMap().FindPartialMatches(type_name, equivalents);
    
    return count;
}

void
CPPLanguageRuntime::MethodName::Clear()
{
    m_full.Clear();
    m_basename = llvm::StringRef();
    m_context = llvm::StringRef();
    m_arguments = llvm::StringRef();
    m_qualifiers = llvm::StringRef();
    m_type = eTypeInvalid;
    m_parsed = false;
    m_parse_error = false;
}

bool
ReverseFindMatchingChars (const llvm::StringRef &s,
                          const llvm::StringRef &left_right_chars,
                          size_t &left_pos,
                          size_t &right_pos,
                          size_t pos = llvm::StringRef::npos)
{
    assert (left_right_chars.size() == 2);
    left_pos = llvm::StringRef::npos;
    const char left_char = left_right_chars[0];
    const char right_char = left_right_chars[1];
    pos = s.find_last_of(left_right_chars, pos);
    if (pos == llvm::StringRef::npos || s[pos] == left_char)
        return false;
    right_pos = pos;
    uint32_t depth = 1;
    while (pos > 0 && depth > 0)
    {
        pos = s.find_last_of(left_right_chars, pos);
        if (pos == llvm::StringRef::npos)
            return false;
        if (s[pos] == left_char)
        {
            if (--depth == 0)
            {
                left_pos = pos;
                return left_pos < right_pos;
            }            
        }
        else if (s[pos] == right_char)
        {
            ++depth;
        }
    }
    return false;
}

void
CPPLanguageRuntime::MethodName::Parse()
{
    if (!m_parsed && m_full)
    {
//        ConstString mangled;
//        m_full.GetMangledCounterpart(mangled);
//        printf ("\n   parsing = '%s'\n", m_full.GetCString());
//        if (mangled)
//            printf ("   mangled = '%s'\n", mangled.GetCString());
        m_parse_error = false;
        m_parsed = true;
        llvm::StringRef full (m_full.GetCString());
        
        size_t arg_start, arg_end;
        llvm::StringRef parens("()", 2);
        if (ReverseFindMatchingChars (full, parens, arg_start, arg_end))
        {
            m_arguments = full.substr(arg_start, arg_end - arg_start + 1);
            if (arg_end + 1 < full.size())
                m_qualifiers = full.substr(arg_end + 1);
            if (arg_start > 0)
            {
                size_t basename_end = arg_start;
                size_t context_end = llvm::StringRef::npos;
                if (basename_end > 0 && full[basename_end-1] == '>')
                {
                    // TODO: handle template junk...
                    // Templated function
                    size_t template_start, template_end;
                    llvm::StringRef lt_gt("<>", 2);
                    if (ReverseFindMatchingChars (full, lt_gt, template_start, template_end, basename_end))
                        context_end = full.rfind(':', template_start);
                }
                if (context_end == llvm::StringRef::npos)
                    context_end = full.rfind(':', basename_end);

                if (context_end == llvm::StringRef::npos)
                    m_basename = full.substr(0, basename_end);
                else
                {
                    m_context = full.substr(0, context_end - 1);
                    const size_t basename_begin = context_end + 1;
                    m_basename = full.substr(basename_begin, basename_end - basename_begin);
                }
                m_type = eTypeUnknownMethod;
            }
            else
            {
                m_parse_error = true;
                return;
            }
        
//            if (!m_context.empty())
//                printf ("   context = '%s'\n", m_context.str().c_str());
//            if (m_basename)
//                printf ("  basename = '%s'\n", m_basename.GetCString());
//            if (!m_arguments.empty())
//                printf (" arguments = '%s'\n", m_arguments.str().c_str());
//            if (!m_qualifiers.empty())
//                printf ("qualifiers = '%s'\n", m_qualifiers.str().c_str());
        }
        else
        {
            m_parse_error = true;
//            printf ("error: didn't find matching parens for arguments\n");
        }
    }
}

llvm::StringRef
CPPLanguageRuntime::MethodName::GetBasename ()
{
    if (!m_parsed)
        Parse();
    return m_basename;
}

llvm::StringRef
CPPLanguageRuntime::MethodName::GetContext ()
{
    if (!m_parsed)
        Parse();
    return m_context;
}

llvm::StringRef
CPPLanguageRuntime::MethodName::GetArguments ()
{
    if (!m_parsed)
        Parse();
    return m_arguments;
}

llvm::StringRef
CPPLanguageRuntime::MethodName::GetQualifiers ()
{
    if (!m_parsed)
        Parse();
    return m_qualifiers;
}

