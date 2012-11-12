//===-- Symbols.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Symbols.h"

using namespace lldb;
using namespace lldb_private;

#if !defined (__APPLE__)

FileSpec
Symbols::LocateExecutableObjectFile (const ModuleSpec &module_spec)
{
    // FIXME
    return FileSpec();
}

FileSpec
Symbols::LocateExecutableSymbolFile (const ModuleSpec &module_spec)
{
    // FIXME
    return FileSpec();
}

FileSpec
Symbols::FindSymbolFileInBundle (const FileSpec& symfile_bundle,
                                 const lldb_private::UUID *uuid,
                                 const ArchSpec *arch)
{
    return FileSpec();
}

bool
Symbols::DownloadObjectAndSymbolFile (ModuleSpec &module_spec, bool force_lookup)
{
    // Fill in the module_spec.GetFileSpec() for the object file and/or the
    // module_spec.GetSymbolFileSpec() for the debug symbols file.
    return false;
}


#endif
