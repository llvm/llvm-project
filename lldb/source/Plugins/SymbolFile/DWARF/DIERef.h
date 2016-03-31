//===-- DIERef.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DIERef_h_
#define SymbolFileDWARF_DIERef_h_

#include "lldb/Core/dwarf.h"
#include "lldb/lldb-defines.h"

class DWARFFormValue;
class SymbolFileDWARF;

struct DIERef
{
    DIERef();

    DIERef(dw_offset_t c, dw_offset_t d);

    //----------------------------------------------------------------------
    // In order to properly decode a lldb::user_id_t back into a DIERef we
    // need the DWARF file since it knows if DWARF in .o files is being used
    // (MacOSX) or if DWO files are being used. The encoding of the user ID
    // differs between the two types of DWARF.
    //----------------------------------------------------------------------
    explicit
    DIERef(lldb::user_id_t uid, SymbolFileDWARF *dwarf);

    explicit
    DIERef(const DWARFFormValue& form_value);

    //----------------------------------------------------------------------
    // In order to properly encode a DIERef unto a lldb::user_id_t we need
    // the DWARF file since it knows if DWARF in .o files is being used
    // (MacOSX) or if DWO files are being used. The encoding of the user ID
    // differs between the two types of DWARF.
    //----------------------------------------------------------------------
    lldb::user_id_t
    GetUID(SymbolFileDWARF *dwarf) const;

    bool
    operator< (const DIERef &ref) const
    {
        return die_offset < ref.die_offset;
    }

    bool
    operator< (const DIERef &ref)
    {
        return die_offset < ref.die_offset;
    }

    dw_offset_t cu_offset;
    dw_offset_t die_offset;
};

typedef std::vector<DIERef> DIEArray;

#endif  // SymbolFileDWARF_DIERef_h_
