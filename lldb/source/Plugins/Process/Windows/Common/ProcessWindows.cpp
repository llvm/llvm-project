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
#include "lldb/Host/windows/windows.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessWindows::ProcessWindows(lldb::TargetSP target_sp,
                               lldb::ListenerSP listener_sp)
    : lldb_private::Process(target_sp, listener_sp) {}

ProcessWindows::~ProcessWindows() {}

size_t ProcessWindows::GetSTDOUT(char *buf, size_t buf_size, Error &error) {
  error.SetErrorString("GetSTDOUT unsupported on Windows");
  return 0;
}

size_t ProcessWindows::GetSTDERR(char *buf, size_t buf_size, Error &error) {
  error.SetErrorString("GetSTDERR unsupported on Windows");
  return 0;
}

size_t ProcessWindows::PutSTDIN(const char *buf, size_t buf_size,
                                Error &error) {
  error.SetErrorString("PutSTDIN unsupported on Windows");
  return 0;
}

//------------------------------------------------------------------------------
// ProcessInterface protocol.

lldb::addr_t ProcessWindows::GetImageInfoAddress() {
  Target &target = GetTarget();
  ObjectFile *obj_file = target.GetExecutableModule()->GetObjectFile();
  Address addr = obj_file->GetImageInfoAddress(&target);
  if (addr.IsValid())
    return addr.GetLoadAddress(&target);
  else
    return LLDB_INVALID_ADDRESS;
}

// The Windows page protection bits are NOT independent masks that can be
// bitwise-ORed
// together.  For example, PAGE_EXECUTE_READ is not (PAGE_EXECUTE | PAGE_READ).
// To test for an access type, it's necessary to test for any of the bits that
// provide
// that access type.
bool ProcessWindows::IsPageReadable(uint32_t protect) {
  return (protect & PAGE_NOACCESS) == 0;
}

bool ProcessWindows::IsPageWritable(uint32_t protect) {
  return (protect & (PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY |
                     PAGE_READWRITE | PAGE_WRITECOPY)) != 0;
}

bool ProcessWindows::IsPageExecutable(uint32_t protect) {
  return (protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                     PAGE_EXECUTE_WRITECOPY)) != 0;
}
}
