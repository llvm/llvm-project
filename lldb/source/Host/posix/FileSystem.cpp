//===-- FileSystem.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSystem.h"

// C includes
#include <dirent.h>
#include <sys/mount.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef __linux__
#include <sys/statfs.h>
#include <sys/mount.h>
#include <linux/magic.h>
#endif
#if defined(__NetBSD__)
#include <sys/statvfs.h>
#endif

// lldb Includes
#include "lldb/Core/Error.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Host.h"

using namespace lldb;
using namespace lldb_private;

const char *
FileSystem::DEV_NULL = "/dev/null";

FileSpec::PathSyntax
FileSystem::GetNativePathSyntax()
{
    return FileSpec::ePathSyntaxPosix;
}

Error
FileSystem::MakeDirectory(const FileSpec &file_spec, uint32_t file_permissions)
{
    if (file_spec)
    {
        Error error;
        if (::mkdir(file_spec.GetCString(), file_permissions) == -1)
        {
            error.SetErrorToErrno();
            errno = 0;
            switch (error.GetError())
            {
                case ENOENT:
                    {
                        // Parent directory doesn't exist, so lets make it if we can
                        // Make the parent directory and try again
                        FileSpec parent_file_spec{file_spec.GetDirectory().GetCString(), false};
                        error = MakeDirectory(parent_file_spec, file_permissions);
                        if (error.Fail())
                            return error;
                        // Try and make the directory again now that the parent directory was made successfully
                        if (::mkdir(file_spec.GetCString(), file_permissions) == -1)
                        {
                            error.SetErrorToErrno();
                        }
                        return error;
                    }
                    break;
                case EEXIST:
                    {
                        if (file_spec.IsDirectory())
                            return Error(); // It is a directory and it already exists
                    }
                    break;
            }
        }
        return error;
    }
    return Error("empty path");
}

Error
FileSystem::DeleteDirectory(const FileSpec &file_spec, bool recurse)
{
    Error error;
    if (file_spec)
    {
        if (recurse)
        {
            // Save all sub directories in a list so we don't recursively call this function
            // and possibly run out of file descriptors if the directory is too deep.
            std::vector<FileSpec> sub_directories;

            FileSpec::ForEachItemInDirectory (file_spec.GetCString(), [&error, &sub_directories](FileSpec::FileType file_type, const FileSpec &spec) -> FileSpec::EnumerateDirectoryResult {
                if (file_type == FileSpec::eFileTypeDirectory)
                {
                    // Save all directorires and process them after iterating through this directory
                    sub_directories.push_back(spec);
                }
                else
                {
                    // Update sub_spec to point to the current file and delete it
                    error = FileSystem::Unlink(spec);
                }
                // If anything went wrong, stop iterating, else process the next file
                if (error.Fail())
                    return FileSpec::eEnumerateDirectoryResultQuit;
                else
                    return FileSpec::eEnumerateDirectoryResultNext;
            });

            if (error.Success())
            {
                // Now delete all sub directories with separate calls that aren't
                // recursively calling into this function _while_ this function is
                // iterating through the current directory.
                for (const auto &sub_directory : sub_directories)
                {
                    error = DeleteDirectory(sub_directory, recurse);
                    if (error.Fail())
                        break;
                }
            }
        }

        if (error.Success())
        {
            if (::rmdir(file_spec.GetCString()) != 0)
                error.SetErrorToErrno();
        }
    }
    else
    {
        error.SetErrorString("empty path");
    }
    return error;
}

Error
FileSystem::GetFilePermissions(const FileSpec &file_spec, uint32_t &file_permissions)
{
    Error error;
    struct stat file_stats;
    if (::stat(file_spec.GetCString(), &file_stats) == 0)
    {
        // The bits in "st_mode" currently match the definitions
        // for the file mode bits in unix.
        file_permissions = file_stats.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
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
    if (::chmod(file_spec.GetCString(), file_permissions) != 0)
        error.SetErrorToErrno();
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
    if (::link(dst.GetCString(), src.GetCString()) == -1)
        error.SetErrorToErrno();
    return error;
}

int
FileSystem::GetHardlinkCount(const FileSpec &file_spec)
{
    struct stat file_stat;
    if (::stat(file_spec.GetCString(), &file_stat) == 0)
        return file_stat.st_nlink;

    return -1;
}

Error
FileSystem::Symlink(const FileSpec &src, const FileSpec &dst)
{
    Error error;
    if (::symlink(dst.GetCString(), src.GetCString()) == -1)
        error.SetErrorToErrno();
    return error;
}

Error
FileSystem::Unlink(const FileSpec &file_spec)
{
    Error error;
    if (::unlink(file_spec.GetCString()) == -1)
        error.SetErrorToErrno();
    return error;
}

Error
FileSystem::Readlink(const FileSpec &src, FileSpec &dst)
{
    Error error;
    char buf[PATH_MAX];
    ssize_t count = ::readlink(src.GetCString(), buf, sizeof(buf) - 1);
    if (count < 0)
        error.SetErrorToErrno();
    else
    {
        buf[count] = '\0'; // Success
        dst.SetFile(buf, false);
    }
    return error;
}

Error
FileSystem::ResolveSymbolicLink(const FileSpec &src, FileSpec &dst)
{
    char resolved_path[PATH_MAX];
    if (!src.GetPath (resolved_path, sizeof (resolved_path)))
    {
        return Error("Couldn't get the canonical path for %s", src.GetCString());
    }
    
    char real_path[PATH_MAX + 1];
    if (realpath(resolved_path, real_path) == nullptr)
    {
        Error err;
        err.SetErrorToErrno();
        return err;
    }
    
    dst = FileSpec(real_path, false);
    
    return Error();
}

#if defined(__NetBSD__)
static bool IsLocal(const struct statvfs& info)
{
    return (info.f_flag & MNT_LOCAL) != 0;
}
#else
static bool IsLocal(const struct statfs& info)
{
#ifdef __linux__
    #define CIFS_MAGIC_NUMBER 0xFF534D42
    switch ((uint32_t)info.f_type)
    {
    case NFS_SUPER_MAGIC:
    case SMB_SUPER_MAGIC:
    case CIFS_MAGIC_NUMBER:
        return false;
    default:
        return true;
    }
#else
    return (info.f_flags & MNT_LOCAL) != 0;
#endif
}
#endif

#if defined(__NetBSD__)
bool
FileSystem::IsLocal(const FileSpec &spec)
{
    struct statvfs statfs_info;
    std::string path (spec.GetPath());
    if (statvfs(path.c_str(), &statfs_info) == 0)
        return ::IsLocal(statfs_info);
    return false;
}
#else
bool
FileSystem::IsLocal(const FileSpec &spec)
{
    struct statfs statfs_info;
    std::string path (spec.GetPath());
    if (statfs(path.c_str(), &statfs_info) == 0)
        return ::IsLocal(statfs_info);
    return false;
}
#endif

FILE *
FileSystem::Fopen(const char *path, const char *mode)
{
    return ::fopen(path, mode);
}

int
FileSystem::Stat(const char *path, struct stat *stats)
{
    return ::stat(path, stats);
}
