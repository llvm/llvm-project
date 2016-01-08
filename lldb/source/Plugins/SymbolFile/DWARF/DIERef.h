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

struct DIERef
{
    DIERef();

    explicit
    DIERef(dw_offset_t d);

    DIERef(dw_offset_t c, dw_offset_t d);

    explicit
    DIERef(lldb::user_id_t uid);

    explicit
    DIERef(const DWARFFormValue& form_value);

    lldb::user_id_t
    GetUID() const;

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
