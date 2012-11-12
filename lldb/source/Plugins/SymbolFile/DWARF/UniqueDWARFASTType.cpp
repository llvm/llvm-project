//===-- UniqueDWARFASTType.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UniqueDWARFASTType.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Symbol/Declaration.h"

#include "DWARFDebugInfoEntry.h"

bool
UniqueDWARFASTTypeList::Find 
(
    SymbolFileDWARF *symfile,
    const DWARFCompileUnit *cu,
    const DWARFDebugInfoEntry *die, 
    const lldb_private::Declaration &decl,
    const int32_t byte_size,
    UniqueDWARFASTType &entry
) const
{
    collection::const_iterator pos, end = m_collection.end();
    for (pos = m_collection.begin(); pos != end; ++pos)
    {
        // Make sure the tags match
        if (pos->m_die->Tag() == die->Tag())
        {
            // Validate byte sizes of both types only if both are valid.
            if (pos->m_byte_size < 0 || byte_size < 0 || pos->m_byte_size == byte_size)
            {
                // Make sure the file and line match
                if (pos->m_declaration == decl)
                {
                    // The type has the same name, and was defined on the same
                    // file and line. Now verify all of the parent DIEs match.
                    const DWARFDebugInfoEntry *parent_arg_die = die->GetParent();
                    const DWARFDebugInfoEntry *parend_pos_die = pos->m_die->GetParent();
                    bool match = true;
                    bool done = false;
                    while (!done && match && parent_arg_die && parend_pos_die)
                    {
                        if (parent_arg_die->Tag() == parend_pos_die->Tag())
                        {
                            const dw_tag_t tag = parent_arg_die->Tag();
                            switch (tag)
                            {
                            case DW_TAG_class_type:
                            case DW_TAG_structure_type:
                            case DW_TAG_union_type:
                            case DW_TAG_namespace:
                                {
                                    const char *parent_arg_die_name = parent_arg_die->GetName(symfile, cu);
                                    if (parent_arg_die_name == NULL)  // Anonymous (i.e. no-name) struct
                                    {
                                        match = false;
                                    }
                                    else
                                    {
                                        const char *parent_pos_die_name = parend_pos_die->GetName(pos->m_symfile, pos->m_cu);
                                        if (parent_pos_die_name == NULL || strcmp (parent_arg_die_name, parent_pos_die_name))
                                            match = false;
                                    }
                                }
                                break;
                            
                            case DW_TAG_compile_unit:
                                done = true;
                                break;
                            }
                        }
                        parent_arg_die = parent_arg_die->GetParent();
                        parend_pos_die = parend_pos_die->GetParent();
                    }

                    if (match)
                    {
                        entry = *pos;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}
