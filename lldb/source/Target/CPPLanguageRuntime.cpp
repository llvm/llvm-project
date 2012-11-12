//===-- CPPLanguageRuntime.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/CPPLanguageRuntime.h"

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
    
  const char *last_colon = NULL;
  for (const char *ptr = base_name_end; ptr != name; ptr--)
    {
      if (*ptr == ':')
        {
          last_colon = ptr;
          break;
        }
    }

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
bool
CPPLanguageRuntime::IsPossibleCPPCall (const char *name, const char *&base_name_start, const char *&base_name_end)
{
    if (!name)
      return false;
    // For now, I really can't handle taking template names apart, so if you
    // have < or > I'll say "could be CPP but leave the base_name empty which
    // means I couldn't figure out what to use for that.
    // FIXME: Do I need to do more sanity checking here?

    if (strchr(name, '>') != NULL || strchr (name, '>') != NULL)
      return true;

    size_t name_len = strlen (name);

    if (name[name_len - 1] == ')')
    {
        // We've got arguments.
        base_name_end = strchr (name, '(');
        if (base_name_end == NULL)
          return false;

        // FIXME: should check that this parenthesis isn't a template specialized
        // on a function type or something gross like that...
    }
    else
        base_name_end = name + strlen (name);

    return StripNamespacesFromVariableName (name, base_name_start, base_name_end);
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
