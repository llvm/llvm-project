//===-- NameToDIE.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_NameToDIE_h_
#define SymbolFileDWARF_NameToDIE_h_

#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/lldb-defines.h"

class SymbolFileDWARF;

typedef std::vector<uint32_t> DIEArray;

class NameToDIE
{
public:
    NameToDIE () :   
        m_map()
    {
    }
    
    ~NameToDIE ()
    {
    }
    
    void
    Dump (lldb_private::Stream *s);

    void
    Insert (const lldb_private::ConstString& name, uint32_t die_offset);

    void
    Finalize();

    size_t
    Find (const lldb_private::ConstString &name, 
          DIEArray &info_array) const;
    
    size_t
    Find (const lldb_private::RegularExpression& regex, 
          DIEArray &info_array) const;

    size_t
    FindAllEntriesForCompileUnit (uint32_t cu_offset, 
                                  uint32_t cu_end_offset, 
                                  DIEArray &info_array) const;

protected:
    lldb_private::UniqueCStringMap<uint32_t> m_map;

};

#endif  // SymbolFileDWARF_NameToDIE_h_
