//===-- ProcessWindows.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProcessWindows.h"

// Other libraries and framework includes
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private
{

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessWindows::ProcessWindows(lldb::TargetSP target_sp, Listener &listener)
    : lldb_private::Process(target_sp, listener)
{
}

ProcessWindows::~ProcessWindows()
{
}

size_t
ProcessWindows::GetSTDOUT(char *buf, size_t buf_size, Error &error)
{
    error.SetErrorString("GetSTDOUT unsupported on Windows");
    return 0;
}

size_t
ProcessWindows::GetSTDERR(char *buf, size_t buf_size, Error &error)
{
    error.SetErrorString("GetSTDERR unsupported on Windows");
    return 0;
}

size_t
ProcessWindows::PutSTDIN(const char *buf, size_t buf_size, Error &error)
{
    error.SetErrorString("PutSTDIN unsupported on Windows");
    return 0;
}

//------------------------------------------------------------------------------
// ProcessInterface protocol.


lldb::addr_t
ProcessWindows::GetImageInfoAddress()
{
    Target &target = GetTarget();
    ObjectFile *obj_file = target.GetExecutableModule()->GetObjectFile();
    Address addr = obj_file->GetImageInfoAddress(&target);
    if (addr.IsValid())
        return addr.GetLoadAddress(&target);
    else
        return LLDB_INVALID_ADDRESS;
}

}
