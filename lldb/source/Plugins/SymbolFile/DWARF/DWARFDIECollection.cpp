//===-- DWARFDIECollection.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDIECollection.h"

#include <algorithm>

#include "lldb/Core/Stream.h"

#include "DWARFDebugInfoEntry.h"

using namespace lldb_private;
using namespace std;

bool
DWARFDIECollection::Insert(const DWARFDebugInfoEntry *die)
{
    iterator end_pos = m_dies.end();
    iterator insert_pos = upper_bound(m_dies.begin(), end_pos, die);
    if (insert_pos != end_pos && (*insert_pos == die))
        return false;
    m_dies.insert(insert_pos, die);
    return true;
}

void
DWARFDIECollection::Append (const DWARFDebugInfoEntry *die)
{
    m_dies.push_back (die);
}

const DWARFDebugInfoEntry *
DWARFDIECollection::GetDIEPtrAtIndex(uint32_t idx) const
{
    if (idx < m_dies.size())
        return m_dies[idx];
    return NULL;
}


size_t
DWARFDIECollection::Size() const
{
    return m_dies.size();
}

void
DWARFDIECollection::Dump(Stream *s, const char* title) const
{
    if (title && title[0] != '\0')
        s->Printf( "%s\n", title);
    const_iterator end_pos = m_dies.end();
    const_iterator pos;
    for (pos = m_dies.begin(); pos != end_pos; ++pos)
        s->Printf( "0x%8.8x\n", (*pos)->GetOffset());
}
