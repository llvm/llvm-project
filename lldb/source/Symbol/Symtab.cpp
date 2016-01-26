//===-- Symtab.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <map>
#include <set>

#include "lldb/Core/Module.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Symtab.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"

using namespace lldb;
using namespace lldb_private;



Symtab::Symtab(ObjectFile *objfile) :
    m_objfile (objfile),
    m_symbols (),
    m_file_addr_to_index (),
    m_name_to_index (),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_file_addr_to_index_computed (false),
    m_name_indexes_computed (false)
{
}

Symtab::~Symtab()
{
}

void
Symtab::Reserve(size_t count)
{
    // Clients should grab the mutex from this symbol table and lock it manually
    // when calling this function to avoid performance issues.
    m_symbols.reserve (count);
}

Symbol *
Symtab::Resize(size_t count)
{
    // Clients should grab the mutex from this symbol table and lock it manually
    // when calling this function to avoid performance issues.
    m_symbols.resize (count);
    return &m_symbols[0];
}

uint32_t
Symtab::AddSymbol(const Symbol& symbol)
{
    // Clients should grab the mutex from this symbol table and lock it manually
    // when calling this function to avoid performance issues.
    uint32_t symbol_idx = m_symbols.size();
    m_name_to_index.Clear();
    m_file_addr_to_index.Clear();
    m_symbols.push_back(symbol);
    m_file_addr_to_index_computed = false;
    m_name_indexes_computed = false;
    return symbol_idx;
}

size_t
Symtab::GetNumSymbols() const
{
    Mutex::Locker locker (m_mutex);
    return m_symbols.size();
}

void
Symtab::SectionFileAddressesChanged ()
{
    m_name_to_index.Clear();
    m_file_addr_to_index_computed = false;
}

void
Symtab::Dump (Stream *s, Target *target, SortOrder sort_order)
{
    Mutex::Locker locker (m_mutex);

//    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    const FileSpec &file_spec = m_objfile->GetFileSpec();
    const char * object_name = nullptr;
    if (m_objfile->GetModule())
        object_name = m_objfile->GetModule()->GetObjectName().GetCString();

    if (file_spec)
        s->Printf("Symtab, file = %s%s%s%s, num_symbols = %" PRIu64,
        file_spec.GetPath().c_str(),
        object_name ? "(" : "",
        object_name ? object_name : "",
        object_name ? ")" : "",
        (uint64_t)m_symbols.size());
    else
        s->Printf("Symtab, num_symbols = %" PRIu64 "", (uint64_t)m_symbols.size());

    if (!m_symbols.empty())
    {
        switch (sort_order)
        {
        case eSortOrderNone:
            {
                s->PutCString (":\n");
                DumpSymbolHeader (s);
                const_iterator begin = m_symbols.begin();
                const_iterator end = m_symbols.end();
                for (const_iterator pos = m_symbols.begin(); pos != end; ++pos)
                {
                    s->Indent();
                    pos->Dump(s, target, std::distance(begin, pos));
                }
            }
            break;

        case eSortOrderByName:
            {
                // Although we maintain a lookup by exact name map, the table
                // isn't sorted by name. So we must make the ordered symbol list
                // up ourselves.
                s->PutCString (" (sorted by name):\n");
                DumpSymbolHeader (s);
                typedef std::multimap<const char*, const Symbol *, CStringCompareFunctionObject> CStringToSymbol;
                CStringToSymbol name_map;
                for (const_iterator pos = m_symbols.begin(), end = m_symbols.end(); pos != end; ++pos)
                {
                    const char *name = pos->GetName().AsCString();
                    if (name && name[0])
                        name_map.insert (std::make_pair(name, &(*pos)));
                }
                
                for (CStringToSymbol::const_iterator pos = name_map.begin(), end = name_map.end(); pos != end; ++pos)
                {
                    s->Indent();
                    pos->second->Dump (s, target, pos->second - &m_symbols[0]);
                }
            }
            break;
            
        case eSortOrderByAddress:
            s->PutCString (" (sorted by address):\n");
            DumpSymbolHeader (s);
            if (!m_file_addr_to_index_computed)
                InitAddressIndexes();
            const size_t num_entries = m_file_addr_to_index.GetSize();
            for (size_t i=0; i<num_entries; ++i)
            {
                s->Indent();
                const uint32_t symbol_idx = m_file_addr_to_index.GetEntryRef(i).data;
                m_symbols[symbol_idx].Dump(s, target, symbol_idx);
            }
            break;
        }
    }
}

void
Symtab::Dump(Stream *s, Target *target, std::vector<uint32_t>& indexes) const
{
    Mutex::Locker locker (m_mutex);

    const size_t num_symbols = GetNumSymbols();
    //s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    s->Printf("Symtab %" PRIu64 " symbol indexes (%" PRIu64 " symbols total):\n", (uint64_t)indexes.size(), (uint64_t)m_symbols.size());
    s->IndentMore();

    if (!indexes.empty())
    {
        std::vector<uint32_t>::const_iterator pos;
        std::vector<uint32_t>::const_iterator end = indexes.end();
        DumpSymbolHeader (s);
        for (pos = indexes.begin(); pos != end; ++pos)
        {
            size_t idx = *pos;
            if (idx < num_symbols)
            {
                s->Indent();
                m_symbols[idx].Dump(s, target, idx);
            }
        }
    }
    s->IndentLess ();
}

void
Symtab::DumpSymbolHeader (Stream *s)
{
    s->Indent("               Debug symbol\n");
    s->Indent("               |Synthetic symbol\n");
    s->Indent("               ||Externally Visible\n");
    s->Indent("               |||\n");
    s->Indent("Index   UserID DSX Type            File Address/Value Load Address       Size               Flags      Name\n");
    s->Indent("------- ------ --- --------------- ------------------ ------------------ ------------------ ---------- ----------------------------------\n");
}


static int
CompareSymbolID (const void *key, const void *p)
{
    const user_id_t match_uid = *(const user_id_t*) key;
    const user_id_t symbol_uid = ((const Symbol *)p)->GetID();
    if (match_uid < symbol_uid)
        return -1;
    if (match_uid > symbol_uid)
        return 1;
    return 0;
}

Symbol *
Symtab::FindSymbolByID (lldb::user_id_t symbol_uid) const
{
    Mutex::Locker locker (m_mutex);

    Symbol *symbol = (Symbol*)::bsearch (&symbol_uid, 
                                         &m_symbols[0], 
                                         m_symbols.size(), 
                                         sizeof(m_symbols[0]),
                                         CompareSymbolID);
    return symbol;
}


Symbol *
Symtab::SymbolAtIndex(size_t idx)
{
    // Clients should grab the mutex from this symbol table and lock it manually
    // when calling this function to avoid performance issues.
    if (idx < m_symbols.size())
        return &m_symbols[idx];
    return nullptr;
}


const Symbol *
Symtab::SymbolAtIndex(size_t idx) const
{
    // Clients should grab the mutex from this symbol table and lock it manually
    // when calling this function to avoid performance issues.
    if (idx < m_symbols.size())
        return &m_symbols[idx];
    return nullptr;
}

//----------------------------------------------------------------------
// InitNameIndexes
//----------------------------------------------------------------------
void
Symtab::InitNameIndexes()
{
    // Protected function, no need to lock mutex...
    if (!m_name_indexes_computed)
    {
        m_name_indexes_computed = true;
        Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
        // Create the name index vector to be able to quickly search by name
        const size_t num_symbols = m_symbols.size();
#if 1
        m_name_to_index.Reserve (num_symbols);
#else
        // TODO: benchmark this to see if we save any memory. Otherwise we
        // will always keep the memory reserved in the vector unless we pull
        // some STL swap magic and then recopy...
        uint32_t actual_count = 0;
        for (const_iterator pos = m_symbols.begin(), end = m_symbols.end();
             pos != end; 
             ++pos)
        {
            const Mangled &mangled = pos->GetMangled();
            if (mangled.GetMangledName())
                ++actual_count;
            
            if (mangled.GetDemangledName())
                ++actual_count;
        }

        m_name_to_index.Reserve (actual_count);
#endif

        NameToIndexMap::Entry entry;

        // The "const char *" in "class_contexts" must come from a ConstString::GetCString()
        std::set<const char *> class_contexts;
        UniqueCStringMap<uint32_t> mangled_name_to_index;
        std::vector<const char *> symbol_contexts(num_symbols, nullptr);

        for (entry.value = 0; entry.value<num_symbols; ++entry.value)
        {
            const Symbol *symbol = &m_symbols[entry.value];

            // Don't let trampolines get into the lookup by name map
            // If we ever need the trampoline symbols to be searchable by name
            // we can remove this and then possibly add a new bool to any of the
            // Symtab functions that lookup symbols by name to indicate if they
            // want trampolines.
            if (symbol->IsTrampoline())
                continue;

            const Mangled &mangled = symbol->GetMangled();
            entry.cstring = mangled.GetMangledName().GetCString();
            if (entry.cstring && entry.cstring[0])
            {
                m_name_to_index.Append (entry);

                if (symbol->ContainsLinkerAnnotations()) {
                    // If the symbol has linker annotations, also add the version without the
                    // annotations.
                    entry.cstring = ConstString(m_objfile->StripLinkerSymbolAnnotations(entry.cstring)).GetCString();
                    m_name_to_index.Append (entry);
                }
                
                const SymbolType symbol_type = symbol->GetType();
                if (symbol_type == eSymbolTypeCode || symbol_type == eSymbolTypeResolver)
                {
                    if (entry.cstring[0] == '_' && entry.cstring[1] == 'Z' &&
                        (entry.cstring[2] != 'T' && // avoid virtual table, VTT structure, typeinfo structure, and typeinfo name
                         entry.cstring[2] != 'G' && // avoid guard variables
                         entry.cstring[2] != 'Z'))  // named local entities (if we eventually handle eSymbolTypeData, we will want this back)
                    {
                        CPlusPlusLanguage::MethodName cxx_method (mangled.GetDemangledName(lldb::eLanguageTypeC_plus_plus));
                        entry.cstring = ConstString(cxx_method.GetBasename()).GetCString();
                        if (entry.cstring && entry.cstring[0])
                        {
                            // ConstString objects permanently store the string in the pool so calling
                            // GetCString() on the value gets us a const char * that will never go away
                            const char *const_context = ConstString(cxx_method.GetContext()).GetCString();

                            if (entry.cstring[0] == '~' || !cxx_method.GetQualifiers().empty())
                            {
                                // The first character of the demangled basename is '~' which
                                // means we have a class destructor. We can use this information
                                // to help us know what is a class and what isn't.
                                if (class_contexts.find(const_context) == class_contexts.end())
                                    class_contexts.insert(const_context);
                                m_method_to_index.Append (entry);
                            }
                            else
                            {
                                if (const_context && const_context[0])
                                {
                                    if (class_contexts.find(const_context) != class_contexts.end())
                                    {
                                        // The current decl context is in our "class_contexts" which means
                                        // this is a method on a class
                                        m_method_to_index.Append (entry);
                                    }
                                    else
                                    {
                                        // We don't know if this is a function basename or a method,
                                        // so put it into a temporary collection so once we are done
                                        // we can look in class_contexts to see if each entry is a class
                                        // or just a function and will put any remaining items into
                                        // m_method_to_index or m_basename_to_index as needed
                                        mangled_name_to_index.Append (entry);
                                        symbol_contexts[entry.value] = const_context;
                                    }
                                }
                                else
                                {
                                    // No context for this function so this has to be a basename
                                    m_basename_to_index.Append(entry);
                                }
                            }
                        }
                    }
                }
            }
            
            entry.cstring = mangled.GetDemangledName(symbol->GetLanguage()).GetCString();
            if (entry.cstring && entry.cstring[0]) {
                m_name_to_index.Append (entry);

                if (symbol->ContainsLinkerAnnotations()) {
                    // If the symbol has linker annotations, also add the version without the
                    // annotations.
                    entry.cstring = ConstString(m_objfile->StripLinkerSymbolAnnotations(entry.cstring)).GetCString();
                    m_name_to_index.Append (entry);
                }
            }
                
            // If the demangled name turns out to be an ObjC name, and
            // is a category name, add the version without categories to the index too.
            ObjCLanguage::MethodName objc_method (entry.cstring, true);
            if (objc_method.IsValid(true))
            {
                entry.cstring = objc_method.GetSelector().GetCString();
                m_selector_to_index.Append (entry);
                
                ConstString objc_method_no_category (objc_method.GetFullNameWithoutCategory(true));
                if (objc_method_no_category)
                {
                    entry.cstring = objc_method_no_category.GetCString();
                    m_name_to_index.Append (entry);
                }
            }
                                                        
        }
        
        size_t count;
        if (!mangled_name_to_index.IsEmpty())
        {
            count = mangled_name_to_index.GetSize();
            for (size_t i=0; i<count; ++i)
            {
                if (mangled_name_to_index.GetValueAtIndex(i, entry.value))
                {
                    entry.cstring = mangled_name_to_index.GetCStringAtIndex(i);
                    if (symbol_contexts[entry.value] && class_contexts.find(symbol_contexts[entry.value]) != class_contexts.end())
                    {
                        m_method_to_index.Append (entry);
                    }
                    else
                    {
                        // If we got here, we have something that had a context (was inside a namespace or class)
                        // yet we don't know if the entry
                        m_method_to_index.Append (entry);
                        m_basename_to_index.Append (entry);
                    }
                }
            }
        }
        m_name_to_index.Sort();
        m_name_to_index.SizeToFit();
        m_selector_to_index.Sort();
        m_selector_to_index.SizeToFit();
        m_basename_to_index.Sort();
        m_basename_to_index.SizeToFit();
        m_method_to_index.Sort();
        m_method_to_index.SizeToFit();
    
//        static StreamFile a ("/tmp/a.txt");
//
//        count = m_basename_to_index.GetSize();
//        if (count)
//        {
//            for (size_t i=0; i<count; ++i)
//            {
//                if (m_basename_to_index.GetValueAtIndex(i, entry.value))
//                    a.Printf ("%s BASENAME\n", m_symbols[entry.value].GetMangled().GetName().GetCString());
//            }
//        }
//        count = m_method_to_index.GetSize();
//        if (count)
//        {
//            for (size_t i=0; i<count; ++i)
//            {
//                if (m_method_to_index.GetValueAtIndex(i, entry.value))
//                    a.Printf ("%s METHOD\n", m_symbols[entry.value].GetMangled().GetName().GetCString());
//            }
//        }
    }
}

void
Symtab::AppendSymbolNamesToMap (const IndexCollection &indexes,
                                bool add_demangled,
                                bool add_mangled,
                                NameToIndexMap &name_to_index_map) const
{
    if (add_demangled || add_mangled)
    {
        Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
        Mutex::Locker locker (m_mutex);

        // Create the name index vector to be able to quickly search by name
        NameToIndexMap::Entry entry;
        const size_t num_indexes = indexes.size();
        for (size_t i=0; i<num_indexes; ++i)
        {
            entry.value = indexes[i];
            assert (i < m_symbols.size());
            const Symbol *symbol = &m_symbols[entry.value];

            const Mangled &mangled = symbol->GetMangled();
            if (add_demangled)
            {
                entry.cstring = mangled.GetDemangledName(symbol->GetLanguage()).GetCString();
                if (entry.cstring && entry.cstring[0])
                    name_to_index_map.Append (entry);
            }
                
            if (add_mangled)
            {
                entry.cstring = mangled.GetMangledName().GetCString();
                if (entry.cstring && entry.cstring[0])
                    name_to_index_map.Append (entry);
            }
        }
    }
}

uint32_t
Symtab::AppendSymbolIndexesWithType (SymbolType symbol_type, std::vector<uint32_t>& indexes, uint32_t start_idx, uint32_t end_index) const
{
    Mutex::Locker locker (m_mutex);

    uint32_t prev_size = indexes.size();

    const uint32_t count = std::min<uint32_t> (m_symbols.size(), end_index);

    for (uint32_t i = start_idx; i < count; ++i)
    {
        if (symbol_type == eSymbolTypeAny || m_symbols[i].GetType() == symbol_type)
            indexes.push_back(i);
    }

    return indexes.size() - prev_size;
}

uint32_t
Symtab::AppendSymbolIndexesWithTypeAndFlagsValue (SymbolType symbol_type, uint32_t flags_value, std::vector<uint32_t>& indexes, uint32_t start_idx, uint32_t end_index) const
{
    Mutex::Locker locker (m_mutex);

    uint32_t prev_size = indexes.size();

    const uint32_t count = std::min<uint32_t> (m_symbols.size(), end_index);

    for (uint32_t i = start_idx; i < count; ++i)
    {
        if ((symbol_type == eSymbolTypeAny || m_symbols[i].GetType() == symbol_type) && m_symbols[i].GetFlags() == flags_value)
            indexes.push_back(i);
    }

    return indexes.size() - prev_size;
}

uint32_t
Symtab::AppendSymbolIndexesWithType (SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& indexes, uint32_t start_idx, uint32_t end_index) const
{
    Mutex::Locker locker (m_mutex);

    uint32_t prev_size = indexes.size();

    const uint32_t count = std::min<uint32_t> (m_symbols.size(), end_index);

    for (uint32_t i = start_idx; i < count; ++i)
    {
        if (symbol_type == eSymbolTypeAny || m_symbols[i].GetType() == symbol_type)
        {
            if (CheckSymbolAtIndex(i, symbol_debug_type, symbol_visibility))
                indexes.push_back(i);
        }
    }

    return indexes.size() - prev_size;
}


uint32_t
Symtab::GetIndexForSymbol (const Symbol *symbol) const
{
    if (!m_symbols.empty())
    {
        const Symbol *first_symbol = &m_symbols[0];
        if (symbol >= first_symbol && symbol < first_symbol + m_symbols.size())
            return symbol - first_symbol;
    }
    return UINT32_MAX;
}

struct SymbolSortInfo
{
    const bool sort_by_load_addr;
    const Symbol *symbols;
};

namespace {
    struct SymbolIndexComparator {
        const std::vector<Symbol>& symbols;
        std::vector<lldb::addr_t>  &addr_cache;
        
        // Getting from the symbol to the Address to the File Address involves some work.
        // Since there are potentially many symbols here, and we're using this for sorting so
        // we're going to be computing the address many times, cache that in addr_cache.
        // The array passed in has to be the same size as the symbols array passed into the
        // member variable symbols, and should be initialized with LLDB_INVALID_ADDRESS.
        // NOTE: You have to make addr_cache externally and pass it in because std::stable_sort
        // makes copies of the comparator it is initially passed in, and you end up spending
        // huge amounts of time copying this array...
        
        SymbolIndexComparator(const std::vector<Symbol>& s, std::vector<lldb::addr_t> &a) : symbols(s), addr_cache(a)  {
            assert (symbols.size() == addr_cache.size());
        }
        bool operator()(uint32_t index_a, uint32_t index_b) {
            addr_t value_a = addr_cache[index_a];
            if (value_a == LLDB_INVALID_ADDRESS)
            {
                value_a = symbols[index_a].GetAddressRef().GetFileAddress();
                addr_cache[index_a] = value_a;
            }
            
            addr_t value_b = addr_cache[index_b];
            if (value_b == LLDB_INVALID_ADDRESS)
            {
                value_b = symbols[index_b].GetAddressRef().GetFileAddress();
                addr_cache[index_b] = value_b;
            }
            

            if (value_a == value_b) {
                // The if the values are equal, use the original symbol user ID
                lldb::user_id_t uid_a = symbols[index_a].GetID();
                lldb::user_id_t uid_b = symbols[index_b].GetID();
                if (uid_a < uid_b)
                    return true;
                if (uid_a > uid_b)
                    return false;
                return false;
            } else if (value_a < value_b)
                return true;
        
            return false;
        }
    };
}

void
Symtab::SortSymbolIndexesByValue (std::vector<uint32_t>& indexes, bool remove_duplicates) const
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__,__PRETTY_FUNCTION__);
    // No need to sort if we have zero or one items...
    if (indexes.size() <= 1)
        return;

    // Sort the indexes in place using std::stable_sort.
    // NOTE: The use of std::stable_sort instead of std::sort here is strictly for performance,
    // not correctness.  The indexes vector tends to be "close" to sorted, which the
    // stable sort handles better.
    
    std::vector<lldb::addr_t> addr_cache(m_symbols.size(), LLDB_INVALID_ADDRESS);
    
    SymbolIndexComparator comparator(m_symbols, addr_cache);
    std::stable_sort(indexes.begin(), indexes.end(), comparator);

    // Remove any duplicates if requested
    if (remove_duplicates)
        std::unique(indexes.begin(), indexes.end());
}

uint32_t
Symtab::AppendSymbolIndexesWithName (const ConstString& symbol_name, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
    if (symbol_name)
    {
        const char *symbol_cstr = symbol_name.GetCString();
        if (!m_name_indexes_computed)
            InitNameIndexes();

        return m_name_to_index.GetValues (symbol_cstr, indexes);
    }
    return 0;
}

uint32_t
Symtab::AppendSymbolIndexesWithName (const ConstString& symbol_name, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
    if (symbol_name)
    {
        const size_t old_size = indexes.size();
        if (!m_name_indexes_computed)
            InitNameIndexes();

        const char *symbol_cstr = symbol_name.GetCString();
        
        std::vector<uint32_t> all_name_indexes;
        const size_t name_match_count = m_name_to_index.GetValues (symbol_cstr, all_name_indexes);
        for (size_t i=0; i<name_match_count; ++i)
        {
            if (CheckSymbolAtIndex(all_name_indexes[i], symbol_debug_type, symbol_visibility))
                indexes.push_back (all_name_indexes[i]);
        }
        return indexes.size() - old_size;
    }
    return 0;
}

uint32_t
Symtab::AppendSymbolIndexesWithNameAndType (const ConstString& symbol_name, SymbolType symbol_type, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    if (AppendSymbolIndexesWithName(symbol_name, indexes) > 0)
    {
        std::vector<uint32_t>::iterator pos = indexes.begin();
        while (pos != indexes.end())
        {
            if (symbol_type == eSymbolTypeAny || m_symbols[*pos].GetType() == symbol_type)
                ++pos;
            else
                pos = indexes.erase(pos);
        }
    }
    return indexes.size();
}

uint32_t
Symtab::AppendSymbolIndexesWithNameAndType (const ConstString& symbol_name, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    if (AppendSymbolIndexesWithName(symbol_name, symbol_debug_type, symbol_visibility, indexes) > 0)
    {
        std::vector<uint32_t>::iterator pos = indexes.begin();
        while (pos != indexes.end())
        {
            if (symbol_type == eSymbolTypeAny || m_symbols[*pos].GetType() == symbol_type)
                ++pos;
            else
                pos = indexes.erase(pos);
        }
    }
    return indexes.size();
}


uint32_t
Symtab::AppendSymbolIndexesMatchingRegExAndType (const RegularExpression &regexp, SymbolType symbol_type, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    uint32_t prev_size = indexes.size();
    uint32_t sym_end = m_symbols.size();

    for (uint32_t i = 0; i < sym_end; i++)
    {
        if (symbol_type == eSymbolTypeAny || m_symbols[i].GetType() == symbol_type)
        {
            const char *name = m_symbols[i].GetName().AsCString();
            if (name)
            {
                if (regexp.Execute (name))
                    indexes.push_back(i);
            }
        }
    }
    return indexes.size() - prev_size;

}

uint32_t
Symtab::AppendSymbolIndexesMatchingRegExAndType (const RegularExpression &regexp, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    uint32_t prev_size = indexes.size();
    uint32_t sym_end = m_symbols.size();

    for (uint32_t i = 0; i < sym_end; i++)
    {
        if (symbol_type == eSymbolTypeAny || m_symbols[i].GetType() == symbol_type)
        {
            if (CheckSymbolAtIndex(i, symbol_debug_type, symbol_visibility) == false)
                continue;

            const char *name = m_symbols[i].GetName().AsCString();
            if (name)
            {
                if (regexp.Execute (name))
                    indexes.push_back(i);
            }
        }
    }
    return indexes.size() - prev_size;

}

Symbol *
Symtab::FindSymbolWithType (SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, uint32_t& start_idx)
{
    Mutex::Locker locker (m_mutex);

    const size_t count = m_symbols.size();
    for (size_t idx = start_idx; idx < count; ++idx)
    {
        if (symbol_type == eSymbolTypeAny || m_symbols[idx].GetType() == symbol_type)
        {
            if (CheckSymbolAtIndex(idx, symbol_debug_type, symbol_visibility))
            {
                start_idx = idx;
                return &m_symbols[idx];
            }
        }
    }
    return nullptr;
}

size_t
Symtab::FindAllSymbolsWithNameAndType (const ConstString &name, SymbolType symbol_type, std::vector<uint32_t>& symbol_indexes)
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
    // Initialize all of the lookup by name indexes before converting NAME
    // to a uniqued string NAME_STR below.
    if (!m_name_indexes_computed)
        InitNameIndexes();

    if (name)
    {
        // The string table did have a string that matched, but we need
        // to check the symbols and match the symbol_type if any was given.
        AppendSymbolIndexesWithNameAndType (name, symbol_type, symbol_indexes);
    }
    return symbol_indexes.size();
}

size_t
Symtab::FindAllSymbolsWithNameAndType (const ConstString &name, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& symbol_indexes)
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
    // Initialize all of the lookup by name indexes before converting NAME
    // to a uniqued string NAME_STR below.
    if (!m_name_indexes_computed)
        InitNameIndexes();

    if (name)
    {
        // The string table did have a string that matched, but we need
        // to check the symbols and match the symbol_type if any was given.
        AppendSymbolIndexesWithNameAndType (name, symbol_type, symbol_debug_type, symbol_visibility, symbol_indexes);
    }
    return symbol_indexes.size();
}

size_t
Symtab::FindAllSymbolsMatchingRexExAndType (const RegularExpression &regex, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& symbol_indexes)
{
    Mutex::Locker locker (m_mutex);

    AppendSymbolIndexesMatchingRegExAndType(regex, symbol_type, symbol_debug_type, symbol_visibility, symbol_indexes);
    return symbol_indexes.size();
}

Symbol *
Symtab::FindFirstSymbolWithNameAndType (const ConstString &name, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility)
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
    if (!m_name_indexes_computed)
        InitNameIndexes();

    if (name)
    {
        std::vector<uint32_t> matching_indexes;
        // The string table did have a string that matched, but we need
        // to check the symbols and match the symbol_type if any was given.
        if (AppendSymbolIndexesWithNameAndType (name, symbol_type, symbol_debug_type, symbol_visibility, matching_indexes))
        {
            std::vector<uint32_t>::const_iterator pos, end = matching_indexes.end();
            for (pos = matching_indexes.begin(); pos != end; ++pos)
            {
                Symbol *symbol = SymbolAtIndex(*pos);

                if (symbol->Compare(name, symbol_type))
                    return symbol;
            }
        }
    }
    return nullptr;
}

typedef struct
{
    const Symtab *symtab;
    const addr_t file_addr;
    Symbol *match_symbol;
    const uint32_t *match_index_ptr;
    addr_t match_offset;
} SymbolSearchInfo;

static int
SymbolWithClosestFileAddress (SymbolSearchInfo *info, const uint32_t *index_ptr)
{
    const Symbol *symbol = info->symtab->SymbolAtIndex (index_ptr[0]);
    if (symbol == nullptr)
        return -1;

    const addr_t info_file_addr = info->file_addr;
    if (symbol->ValueIsAddress())
    {
        const addr_t curr_file_addr = symbol->GetAddressRef().GetFileAddress();
        if (info_file_addr < curr_file_addr)
            return -1;

        // Since we are finding the closest symbol that is greater than or equal
        // to 'info->file_addr' we set the symbol here. This will get set
        // multiple times, but after the search is done it will contain the best
        // symbol match
        info->match_symbol = const_cast<Symbol *>(symbol);
        info->match_index_ptr = index_ptr;
        info->match_offset = info_file_addr - curr_file_addr;

        if (info_file_addr > curr_file_addr)
            return +1;
        return 0;
    }
    return -1;
}

void
Symtab::InitAddressIndexes()
{
    // Protected function, no need to lock mutex...
    if (!m_file_addr_to_index_computed && !m_symbols.empty())
    {
        m_file_addr_to_index_computed = true;

        FileRangeToIndexMap::Entry entry;
        const_iterator begin = m_symbols.begin();
        const_iterator end = m_symbols.end();
        for (const_iterator pos = m_symbols.begin(); pos != end; ++pos)
        {
            if (pos->ValueIsAddress())
            {
                entry.SetRangeBase(pos->GetAddressRef().GetFileAddress());
                entry.SetByteSize(pos->GetByteSize());
                entry.data = std::distance(begin, pos);
                m_file_addr_to_index.Append(entry);
            }
        }
        const size_t num_entries = m_file_addr_to_index.GetSize();
        if (num_entries > 0)
        {
            m_file_addr_to_index.Sort();
            m_file_addr_to_index.CalculateSizesOfZeroByteSizeRanges();
        
            // Now our last symbols might not have had sizes because there
            // was no subsequent symbol to calculate the size from. If this is
            // the case, then calculate the size by capping it at the end of the
            // section in which the symbol resides
            for (int i = num_entries - 1; i >= 0; --i)
            {
                const FileRangeToIndexMap::Entry &entry = m_file_addr_to_index.GetEntryRef(i);
                // As we iterate backwards, as soon as we find a symbol with a valid
                // byte size, we are done
                if (entry.GetByteSize() > 0)
                    break;

                // Cap the size to the end of the section in which the symbol resides
                SectionSP section_sp (m_objfile->GetSectionList()->FindSectionContainingFileAddress (entry.GetRangeBase()));
                if (section_sp)
                {
                    const lldb::addr_t end_section_file_addr = section_sp->GetFileAddress() + section_sp->GetByteSize();
                    const lldb::addr_t symbol_file_addr = entry.GetRangeBase();
                    if (end_section_file_addr > symbol_file_addr)
                    {
                        Symbol &symbol = m_symbols[entry.data];
                        if (!symbol.GetByteSizeIsValid())
                        {
                            symbol.SetByteSize(end_section_file_addr - symbol_file_addr);
                            symbol.SetSizeIsSynthesized(true);
                        }
                    }
                }
            }
            // Sort again in case the range size changes the ordering
            m_file_addr_to_index.Sort();
        }
    }
}

void
Symtab::CalculateSymbolSizes ()
{
    Mutex::Locker locker (m_mutex);

    if (!m_symbols.empty())
    {
        if (!m_file_addr_to_index_computed)
            InitAddressIndexes();
        
        const size_t num_entries = m_file_addr_to_index.GetSize();

        for (size_t i = 0; i < num_entries; ++i)
        {
            // The entries in the m_file_addr_to_index have calculated the sizes already
            // so we will use this size if we need to.
            const FileRangeToIndexMap::Entry &entry = m_file_addr_to_index.GetEntryRef(i);
            
            Symbol &symbol = m_symbols[entry.data];

            // If the symbol size is already valid, no need to do anything
            if (symbol.GetByteSizeIsValid())
                continue;
            
            const addr_t range_size = entry.GetByteSize();
            if (range_size > 0)
            {
                symbol.SetByteSize(range_size);
                symbol.SetSizeIsSynthesized(true);
            }
        }
    }
}

Symbol *
Symtab::FindSymbolContainingFileAddress (addr_t file_addr, const uint32_t* indexes, uint32_t num_indexes)
{
    Mutex::Locker locker (m_mutex);

    
    SymbolSearchInfo info = { this, file_addr, nullptr, nullptr, 0 };

    ::bsearch (&info, 
               indexes, 
               num_indexes, 
               sizeof(uint32_t), 
               (ComparisonFunction)SymbolWithClosestFileAddress);

    if (info.match_symbol)
    {
        if (info.match_offset == 0)
        {
            // We found an exact match!
            return info.match_symbol;
        }

        if (!info.match_symbol->GetByteSizeIsValid())
        {
            // The matched symbol dosn't have a valid byte size so lets just go with that match...
            return info.match_symbol;
        }

        // We were able to figure out a symbol size so lets make sure our 
        // offset puts "file_addr" in the symbol's address range.
        if (info.match_offset < info.match_symbol->GetByteSize())
            return info.match_symbol;
    }
    return nullptr;
}

Symbol *
Symtab::FindSymbolContainingFileAddress (addr_t file_addr)
{
    Mutex::Locker locker (m_mutex);

    if (!m_file_addr_to_index_computed)
        InitAddressIndexes();

    const FileRangeToIndexMap::Entry *entry = m_file_addr_to_index.FindEntryThatContains(file_addr);
    if (entry)
    {
        Symbol* symbol = SymbolAtIndex(entry->data);
        if (symbol->ContainsFileAddress(file_addr))
            return symbol;
    }
    return nullptr;
}

void
Symtab::ForEachSymbolContainingFileAddress(addr_t file_addr, std::function<bool(Symbol *)> const &callback)
{
    Mutex::Locker locker (m_mutex);

    if (!m_file_addr_to_index_computed)
        InitAddressIndexes();

    std::vector<uint32_t> all_addr_indexes;

    // Get all symbols with file_addr
    const size_t addr_match_count = m_file_addr_to_index.FindEntryIndexesThatContain(file_addr, all_addr_indexes);

    for (size_t i = 0; i < addr_match_count; ++i)
    {
        if (!callback(SymbolAtIndex(all_addr_indexes[i])))
        break;
    }
}

void
Symtab::SymbolIndicesToSymbolContextList (std::vector<uint32_t> &symbol_indexes, SymbolContextList &sc_list)
{
    // No need to protect this call using m_mutex all other method calls are
    // already thread safe.
    
    const bool merge_symbol_into_function = true;
    size_t num_indices = symbol_indexes.size();
    if (num_indices > 0)
    {
        SymbolContext sc;
        sc.module_sp = m_objfile->GetModule();
        for (size_t i = 0; i < num_indices; i++)
        {
            sc.symbol = SymbolAtIndex (symbol_indexes[i]);
            if (sc.symbol)
                sc_list.AppendIfUnique(sc, merge_symbol_into_function);
        }
    }
}


size_t
Symtab::FindFunctionSymbols (const ConstString &name,
                             uint32_t name_type_mask,
                             SymbolContextList& sc_list)
{
    size_t count = 0;
    std::vector<uint32_t> symbol_indexes;
    
    const char *name_cstr = name.GetCString();
    
    // eFunctionNameTypeAuto should be pre-resolved by a call to Module::PrepareForFunctionNameLookup()
    assert ((name_type_mask & eFunctionNameTypeAuto) == 0);

    if (name_type_mask & (eFunctionNameTypeBase | eFunctionNameTypeFull))
    {
        std::vector<uint32_t> temp_symbol_indexes;
        FindAllSymbolsWithNameAndType (name, eSymbolTypeAny, temp_symbol_indexes);

        unsigned temp_symbol_indexes_size = temp_symbol_indexes.size();
        if (temp_symbol_indexes_size > 0)
        {
            Mutex::Locker locker (m_mutex);
            for (unsigned i = 0; i < temp_symbol_indexes_size; i++)
            {
                SymbolContext sym_ctx;
                sym_ctx.symbol = SymbolAtIndex (temp_symbol_indexes[i]);
                if (sym_ctx.symbol)
                {
                    switch (sym_ctx.symbol->GetType())
                    {
                    case eSymbolTypeCode:
                    case eSymbolTypeResolver:
                    case eSymbolTypeReExported:
                        symbol_indexes.push_back(temp_symbol_indexes[i]);
                        break;
                    default:
                        break;
                    }
                }
            }
        }
    }
    
    if (name_type_mask & eFunctionNameTypeBase)
    {
        // From mangled names we can't tell what is a basename and what
        // is a method name, so we just treat them the same
        if (!m_name_indexes_computed)
            InitNameIndexes();

        if (!m_basename_to_index.IsEmpty())
        {
            const UniqueCStringMap<uint32_t>::Entry *match;
            for (match = m_basename_to_index.FindFirstValueForName(name_cstr);
                 match != nullptr;
                 match = m_basename_to_index.FindNextValueForName(match))
            {
                symbol_indexes.push_back(match->value);
            }
        }
    }
    
    if (name_type_mask & eFunctionNameTypeMethod)
    {
        if (!m_name_indexes_computed)
            InitNameIndexes();
        
        if (!m_method_to_index.IsEmpty())
        {
            const UniqueCStringMap<uint32_t>::Entry *match;
            for (match = m_method_to_index.FindFirstValueForName(name_cstr);
                 match != nullptr;
                 match = m_method_to_index.FindNextValueForName(match))
            {
                symbol_indexes.push_back(match->value);
            }
        }
    }

    if (name_type_mask & eFunctionNameTypeSelector)
    {
        if (!m_name_indexes_computed)
            InitNameIndexes();

        if (!m_selector_to_index.IsEmpty())
        {
            const UniqueCStringMap<uint32_t>::Entry *match;
            for (match = m_selector_to_index.FindFirstValueForName(name_cstr);
                 match != nullptr;
                 match = m_selector_to_index.FindNextValueForName(match))
            {
                symbol_indexes.push_back(match->value);
            }
        }
    }

    if (!symbol_indexes.empty())
    {
        std::sort(symbol_indexes.begin(), symbol_indexes.end());
        symbol_indexes.erase(std::unique(symbol_indexes.begin(), symbol_indexes.end()), symbol_indexes.end());
        count = symbol_indexes.size();
        SymbolIndicesToSymbolContextList (symbol_indexes, sc_list);
    }

    return count;
}


const Symbol *
Symtab::GetParent (Symbol *child_symbol) const
{
    uint32_t child_idx = GetIndexForSymbol(child_symbol);
    if (child_idx != UINT32_MAX && child_idx > 0)
    {
        for (uint32_t idx = child_idx - 1; idx != UINT32_MAX; --idx)
        {
            const Symbol *symbol = SymbolAtIndex (idx);
            const uint32_t sibling_idx = symbol->GetSiblingIndex();
            if (sibling_idx != UINT32_MAX && sibling_idx > child_idx)
                return symbol;
        }
    }
    return NULL;
}
