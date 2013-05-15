//===-- DWARFDebugPubnames.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDebugPubnames_h_
#define SymbolFileDWARF_DWARFDebugPubnames_h_

#include "SymbolFileDWARF.h"

#include <list>

#include "DWARFDebugPubnamesSet.h"

class DWARFDebugPubnames
{
public:
            DWARFDebugPubnames();
    bool    Extract(const lldb_private::DataExtractor& data);
    bool    GeneratePubnames(SymbolFileDWARF* dwarf2Data);
    bool    GeneratePubBaseTypes(SymbolFileDWARF* dwarf2Data);

    void    Dump(lldb_private::Log *s) const;
    bool    Find(const char* name, bool ignore_case, std::vector<dw_offset_t>& die_offset_coll) const;
    bool    Find(const lldb_private::RegularExpression& regex, std::vector<dw_offset_t>& die_offsets) const;
protected:
    typedef std::list<DWARFDebugPubnamesSet>    collection;
    typedef collection::iterator                iterator;
    typedef collection::const_iterator          const_iterator;

    collection m_sets;
};

#endif  // SymbolFileDWARF_DWARFDebugPubnames_h_
