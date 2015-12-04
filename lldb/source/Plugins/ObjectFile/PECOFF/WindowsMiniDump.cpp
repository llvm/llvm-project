//===-- WindowsMiniDump.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This function is separated out from ObjectFilePECOFF.cpp to name avoid name
// collisions with WinAPI preprocessor macros.

#include "WindowsMiniDump.h"
#include "lldb/Host/FileSpec.h"
#include "llvm/Support/ConvertUTF.h"

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#include <DbgHelp.h>  // for MiniDumpWriteDump
#endif

namespace lldb_private {

bool
SaveMiniDump(const lldb::ProcessSP &process_sp,
             const lldb_private::FileSpec &outfile,
             lldb_private::Error &error)
{
    if (!process_sp) return false;
#ifdef _WIN32
    HANDLE process_handle = ::OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, process_sp->GetID());
    const std::string file_name = outfile.GetCString();
    std::wstring wide_name;
    wide_name.resize(file_name.size() + 1);
    char * result_ptr = reinterpret_cast<char *>(&wide_name[0]);
    const UTF8 *error_ptr = nullptr;
    if (!llvm::ConvertUTF8toWide(sizeof(wchar_t), file_name, result_ptr, error_ptr)) {
        error.SetErrorString("cannot convert file name");
        return false;
    }
    HANDLE file_handle = ::CreateFileW(wide_name.c_str(), GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    const auto result = ::MiniDumpWriteDump(process_handle, process_sp->GetID(), file_handle,
                                            MiniDumpWithFullMemoryInfo, NULL, NULL, NULL);
    ::CloseHandle(file_handle);
    ::CloseHandle(process_handle);
    if (!result)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return false;
    }
    return true;
#endif
    return false;
}

}  // namesapce lldb_private
