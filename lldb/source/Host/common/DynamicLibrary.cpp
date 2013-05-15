//===-- DynamicLibrary.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Error.h"
#include "lldb/Host/DynamicLibrary.h"

using namespace lldb_private;

DynamicLibrary::DynamicLibrary (const FileSpec& spec, uint32_t options) : m_filespec(spec)
{
    Error err;
    m_handle = Host::DynamicLibraryOpen (spec,options,err);
    if (err.Fail())
        m_handle = NULL;
}

bool
DynamicLibrary::IsValid ()
{
    return m_handle != NULL;
}

DynamicLibrary::~DynamicLibrary ()
{
    if (m_handle)
        Host::DynamicLibraryClose (m_handle);
}
