//===-- ClangExternalASTSourceCommon.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/ClangExternalASTSourceCommon.h"

using namespace lldb_private;

#define ClangExternalASTSourceCommon_MAGIC  (0x00112233aabbccddull)

uint64_t g_TotalSizeOfMetadata = 0;

ClangExternalASTSourceCommon::ClangExternalASTSourceCommon() : clang::ExternalASTSource()
{
    m_magic = ClangExternalASTSourceCommon_MAGIC;
    
    g_TotalSizeOfMetadata += m_metadata.size();
}

ClangExternalASTSourceCommon::~ClangExternalASTSourceCommon()
{
    g_TotalSizeOfMetadata -= m_metadata.size();
}

ClangASTMetadata *
ClangExternalASTSourceCommon::GetMetadata (uintptr_t object)
{
    assert (m_magic == ClangExternalASTSourceCommon_MAGIC);
    
    if (HasMetadata (object))
        return &m_metadata[object];
    else
        return NULL;
}

void
ClangExternalASTSourceCommon::SetMetadata (uintptr_t object, ClangASTMetadata &metadata)
{
    assert (m_magic == ClangExternalASTSourceCommon_MAGIC);
    
    uint64_t orig_size = m_metadata.size();
    m_metadata[object] = metadata;
    uint64_t new_size = m_metadata.size();
    g_TotalSizeOfMetadata += (new_size - orig_size);
}

bool
ClangExternalASTSourceCommon::HasMetadata (uintptr_t object)
{
    assert (m_magic == ClangExternalASTSourceCommon_MAGIC);

    return m_metadata.find(object) != m_metadata.end();
}
