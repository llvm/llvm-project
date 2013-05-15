//===-- DWARFDebugInfo.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDebugInfo_h_
#define SymbolFileDWARF_DWARFDebugInfo_h_

#include <vector>
#include <map>

#include "lldb/lldb-private.h"
#include "lldb/lldb-private.h"
#include "SymbolFileDWARF.h"

typedef std::multimap<const char*, dw_offset_t, CStringCompareFunctionObject> CStringToDIEMap;
typedef CStringToDIEMap::iterator CStringToDIEMapIter;
typedef CStringToDIEMap::const_iterator CStringToDIEMapConstIter;

typedef std::shared_ptr<DWARFCompileUnit> DWARFCompileUnitSP;

class DWARFDebugInfo
{
public:
    typedef dw_offset_t (*Callback)(
        SymbolFileDWARF* dwarf2Data,
        DWARFCompileUnitSP& cu_shared_ptr,
        DWARFDebugInfoEntry* die,
        const dw_offset_t next_offset,
        const uint32_t depth,
        void* userData);

    DWARFDebugInfo();
    void SetDwarfData(SymbolFileDWARF* dwarf2Data);

    bool LookupAddress(
            const dw_addr_t address,
            const dw_offset_t cu_offset,    // Can be valid (find in .debug_aranges), or DW_INVALID_OFFSET if we need to search manually
            DWARFCompileUnitSP& cu_shared_ptr,
            DWARFDebugInfoEntry** function_die,
            DWARFDebugInfoEntry** block_die);

    void AddCompileUnit(DWARFCompileUnitSP& cu);
    size_t GetNumCompileUnits();
    bool ContainsCompileUnit (const DWARFCompileUnit *cu) const;
    DWARFCompileUnit* GetCompileUnitAtIndex(uint32_t idx);
    DWARFCompileUnitSP GetCompileUnit(dw_offset_t cu_offset, uint32_t* idx_ptr = NULL);
    DWARFCompileUnitSP GetCompileUnitContainingDIE(dw_offset_t die_offset);

    DWARFDebugInfoEntry* GetDIEPtr(dw_offset_t die_offset, DWARFCompileUnitSP* cu_sp_ptr);
    DWARFDebugInfoEntry* GetDIEPtrWithCompileUnitHint (dw_offset_t die_offset, DWARFCompileUnit**cu_handle);

    const DWARFDebugInfoEntry* GetDIEPtrContainingOffset(dw_offset_t die_offset, DWARFCompileUnitSP* cu_sp_ptr);

    void Dump(lldb_private::Stream *s, const uint32_t die_offset, const uint32_t recurse_depth);
    static void Parse(SymbolFileDWARF* parser, Callback callback, void* userData);
    static void Verify(lldb_private::Stream *s, SymbolFileDWARF* dwarf2Data);
    static void Dump(lldb_private::Stream *s, SymbolFileDWARF* dwarf2Data, const uint32_t die_offset, const uint32_t recurse_depth);
    bool Find(const char* name, bool ignore_case, std::vector<dw_offset_t>& die_offsets) const;
    bool Find(lldb_private::RegularExpression& re, std::vector<dw_offset_t>& die_offsets) const;

    enum
    {
        eDumpFlag_Verbose               = (1<<0),   // Verbose dumping
        eDumpFlag_ShowForm              = (1<<1),   // Show the DW_form type
        eDumpFlag_ShowAncestors         = (1<<2)    // Show all parent DIEs when dumping single DIEs
    };

    DWARFDebugAranges &
    GetCompileUnitAranges ();

protected:
    SymbolFileDWARF* m_dwarf2Data;
    typedef std::vector<DWARFCompileUnitSP>     CompileUnitColl;
    CompileUnitColl m_compile_units;
    std::unique_ptr<DWARFDebugAranges> m_cu_aranges_ap; // A quick address to compile unit table

private:
    // All parsing needs to be done partially any managed by this class as accessors are called.
    void ParseCompileUnitHeadersIfNeeded();

    DISALLOW_COPY_AND_ASSIGN (DWARFDebugInfo);
};

#endif  // SymbolFileDWARF_DWARFDebugInfo_h_
