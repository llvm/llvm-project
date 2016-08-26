//===-- FileSystem.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/windows.h"

#include <shellapi.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/windows/AutoHandle.h"

#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"

using namespace lldb_private;

const char *
FileSystem::DEV_NULL = "nul";

const char *FileSystem::PATH_CONVERSION_ERROR = "Error converting path between UTF-8 and native encoding";

FileSpec::PathSyntax
FileSystem::GetNativePathSyntax()
{
    return FileSpec::ePathSyntaxWindows;
}

Error
FileSystem::MakeDirectory(const FileSpec &file_spec, uint32_t file_permissions)
{
    // On Win32, the mode parameter is ignored, as Windows files and directories support a
    // different permission model than POSIX.
    Error error;
    const auto err_code = llvm::sys::fs::create_directories(file_spec.GetPath(), true);
    if (err_code)
    {
        error.SetErrorString(err_code.message().c_str());
    }

    return error;
}

Error
FileSystem::DeleteDirectory(const FileSpec &file_spec, bool recurse)
{
    Error error;
    std::wstring path_buffer;
    if (!llvm::ConvertUTF8toWide(file_spec.GetPath(), path_buffer))
    {
        error.SetErrorString(PATH_CONVERSION_ERROR);
        return error;
    }
    if (!recurse)
    {
        BOOL result = ::RemoveDirectoryW(path_buffer.c_str());
        if (!result)
            error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    }
    else
    {
        // SHFileOperation() accepts a list of paths, and so must be double-null-terminated to
        // indicate the end of the list. The first null terminator is there only in the backing
        // store but not the actual vector contents, and so we need to push twice.
        path_buffer.push_back(0);
        path_buffer.push_back(0);

        SHFILEOPSTRUCTW shfos = {0};
        shfos.wFunc = FO_DELETE;
        shfos.pFrom = (LPCWSTR)path_buffer.data();
        shfos.fFlags = FOF_NO_UI;

        int result = ::SHFileOperationW(&shfos);
        // TODO(zturner): Correctly handle the intricacies of SHFileOperation return values.
        if (result != 0)
            error.SetErrorStringWithFormat("SHFileOperation failed");
    }
    return error;
}

Error
FileSystem::GetFilePermissions(const FileSpec &file_spec, uint32_t &file_permissions)
{
    Error error;
    // Beware that Windows's permission model is different from Unix's, and it's
    // not clear if this API is supposed to check ACLs.  To match the caller's
    // expectations as closely as possible, we'll use Microsoft's _stat, which
    // attempts to emulate POSIX stat.  This should be good enough for basic
    // checks like FileSpec::Readable.
    struct _stat file_stats;
    if (::_stat(file_spec.GetCString(), &file_stats) == 0)
    {
        // The owner permission bits in "st_mode" currently match the definitions
        // for the owner file mode bits.
        file_permissions = file_stats.st_mode & (_S_IREAD | _S_IWRITE | _S_IEXEC);
    }
    else
    {
        error.SetErrorToErrno();
    }

    return error;
}

Error
FileSystem::SetFilePermissions(const FileSpec &file_spec, uint32_t file_permissions)
{
    Error error;
    error.SetErrorStringWithFormat("%s is not supported on this host", LLVM_PRETTY_FUNCTION);
    return error;
}

lldb::user_id_t
FileSystem::GetFileSize(const FileSpec &file_spec)
{
    return file_spec.GetByteSize();
}

bool
FileSystem::GetFileExists(const FileSpec &file_spec)
{
    return file_spec.Exists();
}

Error
FileSystem::Hardlink(const FileSpec &src, const FileSpec &dst)
{
    Error error;
    std::wstring wsrc, wdst;
    if (!llvm::ConvertUTF8toWide(src.GetCString(), wsrc) || !llvm::ConvertUTF8toWide(dst.GetCString(), wdst))
        error.SetErrorString(PATH_CONVERSION_ERROR);
    else if (!::CreateHardLinkW(wsrc.c_str(), wdst.c_str(), nullptr))
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
}

int
FileSystem::GetHardlinkCount(const FileSpec &file_spec)
{
    std::wstring path;
    if (!llvm::ConvertUTF8toWide(file_spec.GetCString(), path))
        return -1;

    HANDLE file_handle = ::CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                       FILE_ATTRIBUTE_NORMAL, nullptr);

    if (file_handle == INVALID_HANDLE_VALUE)
      return -1;

    AutoHandle auto_file_handle(file_handle);
    BY_HANDLE_FILE_INFORMATION file_info;
    if (::GetFileInformationByHandle(file_handle, &file_info))
        return file_info.nNumberOfLinks;

    return -1;
}

Error
FileSystem::Symlink(const FileSpec &src, const FileSpec &dst)
{
    Error error;
    std::wstring wsrc, wdst;
    if (!llvm::ConvertUTF8toWide(src.GetCString(), wsrc) || !llvm::ConvertUTF8toWide(dst.GetCString(), wdst))
        error.SetErrorString(PATH_CONVERSION_ERROR);
    if (error.Fail())
        return error;
    DWORD attrib = ::GetFileAttributesW(wdst.c_str());
    if (attrib == INVALID_FILE_ATTRIBUTES)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }
    bool is_directory = !!(attrib & FILE_ATTRIBUTE_DIRECTORY);
    DWORD flag = is_directory ? SYMBOLIC_LINK_FLAG_DIRECTORY : 0;
    BOOL result = ::CreateSymbolicLinkW(wsrc.c_str(), wdst.c_str(), flag);
    if (!result)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
}

Error
FileSystem::Unlink(const FileSpec &file_spec)
{
    Error error;
    std::wstring path;
    if (!llvm::ConvertUTF8toWide(file_spec.GetCString(), path))
    {
        error.SetErrorString(PATH_CONVERSION_ERROR);
        return error;
    }
    BOOL result = ::DeleteFileW(path.c_str());
    if (!result)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
}

Error
FileSystem::Readlink(const FileSpec &src, FileSpec &dst)
{
    Error error;
    std::wstring wsrc;
    if (!llvm::ConvertUTF8toWide(src.GetCString(), wsrc))
    {
        error.SetErrorString(PATH_CONVERSION_ERROR);
        return error;
    }

    HANDLE h = ::CreateFileW(wsrc.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                             FILE_FLAG_OPEN_REPARSE_POINT, NULL);
    if (h == INVALID_HANDLE_VALUE)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }

    std::vector<wchar_t> buf(PATH_MAX + 1);
    // Subtract 1 from the path length since this function does not add a null terminator.
    DWORD result = ::GetFinalPathNameByHandleW(h, buf.data(), buf.size() - 1, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    std::string path;
    if (result == 0)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    else if (!llvm::convertWideToUTF8(buf.data(), path))
        error.SetErrorString(PATH_CONVERSION_ERROR);
    else
        dst.SetFile(path, false);

    ::CloseHandle(h);
    return error;
}

Error
FileSystem::ResolveSymbolicLink(const FileSpec &src, FileSpec &dst)
{
    return Error("ResolveSymbolicLink() isn't implemented on Windows");
}

bool
FileSystem::IsLocal(const FileSpec &spec)
{
    if (spec)
    {
        // TODO: return true if the file is on a locally mounted file system
        return true;
    }

    return false;
}

FILE *
FileSystem::Fopen(const char *path, const char *mode)
{
    std::wstring wpath, wmode;
    if (!llvm::ConvertUTF8toWide(path, wpath))
        return nullptr;
    if (!llvm::ConvertUTF8toWide(mode, wmode))
        return nullptr;
    FILE *file;
    if (_wfopen_s(&file, wpath.c_str(), wmode.c_str()) != 0)
        return nullptr;
    return file;
}

int
FileSystem::Stat(const char *path, struct stat *stats)
{
    std::wstring wpath;
    if (!llvm::ConvertUTF8toWide(path, wpath))
    {
        errno = EINVAL;
        return -EINVAL;
    }
    int stat_result;
#ifdef _USE_32BIT_TIME_T
    struct _stat32 file_stats;
    stat_result = ::_wstat32(wpath.c_str(), &file_stats);
#else
    struct _stat64i32 file_stats;
    stat_result = ::_wstat64i32(wpath.c_str(), &file_stats);
#endif
    if (stat_result == 0)
    {
        static_assert(sizeof(struct stat) == sizeof(file_stats),
                      "stat and _stat32/_stat64i32 must have the same layout");
        *stats = *reinterpret_cast<struct stat *>(&file_stats);
    }
    return stat_result;
}
