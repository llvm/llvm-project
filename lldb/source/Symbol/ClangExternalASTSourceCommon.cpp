//===-- ClangExternalASTSourceCommon.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Core/Stream.h"

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
ClangExternalASTSourceCommon::GetMetadata (const void *object)
{
    assert (m_magic == ClangExternalASTSourceCommon_MAGIC);
    
    if (HasMetadata (object))
        return &m_metadata[object];
    else
        return NULL;
}

void
ClangExternalASTSourceCommon::SetMetadata (const void *object, ClangASTMetadata &metadata)
{
    assert (m_magic == ClangExternalASTSourceCommon_MAGIC);
    
    uint64_t orig_size = m_metadata.size();
    m_metadata[object] = metadata;
    uint64_t new_size = m_metadata.size();
    g_TotalSizeOfMetadata += (new_size - orig_size);
}

bool
ClangExternalASTSourceCommon::HasMetadata (const void *object)
{
    assert (m_magic == ClangExternalASTSourceCommon_MAGIC);

    return m_metadata.find(object) != m_metadata.end();
}

void
ClangASTMetadata::Dump (Stream *s)
{
    lldb::user_id_t uid = GetUserID ();
    
    if (uid != LLDB_INVALID_UID)
    {
        s->Printf ("uid=0x%" PRIx64, uid);
    }
    
    uint64_t isa_ptr = GetISAPtr ();
    if (isa_ptr != 0)
    {
        s->Printf ("isa_ptr=0x%" PRIx64, isa_ptr);
    }
    
    const char *obj_ptr_name = GetObjectPtrName();
    if (obj_ptr_name)
    {
        s->Printf ("obj_ptr_name=\"%s\" ", obj_ptr_name);
    }
    
    if (m_is_dynamic_cxx)
    {
        s->Printf ("is_dynamic_cxx=%i ", m_is_dynamic_cxx);
    }
    s->EOL();
}

