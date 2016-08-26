//===-- File.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/File.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#else
#include <sys/ioctl.h>
#endif

#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Process.h" // for llvm::sys::Process::FileDescriptorHasColors()

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/FileSystem.h"

using namespace lldb;
using namespace lldb_private;

static const char *
GetStreamOpenModeFromOptions (uint32_t options)
{
    if (options & File::eOpenOptionAppend)
    {
        if (options & File::eOpenOptionRead)
        {
            if (options & File::eOpenOptionCanCreateNewOnly)
                return "a+x";
            else
                return "a+";
        }
        else if (options & File::eOpenOptionWrite)
        {
            if (options & File::eOpenOptionCanCreateNewOnly)
                return "ax";
            else
                return "a";
        }
    }
    else if (options & File::eOpenOptionRead && options & File::eOpenOptionWrite)
    {
        if (options & File::eOpenOptionCanCreate)
        {
            if (options & File::eOpenOptionCanCreateNewOnly)
                return "w+x";
            else
                return "w+";
        }
        else
            return "r+";
    }
    else if (options & File::eOpenOptionRead)
    {
        return "r";
    }
    else if (options & File::eOpenOptionWrite)
    {
        return "w";
    }
    return NULL;
}

int File::kInvalidDescriptor = -1;
FILE * File::kInvalidStream = NULL;

File::File(const char *path, uint32_t options, uint32_t permissions) :
    IOObject(eFDTypeFile, false),
    m_descriptor (kInvalidDescriptor),
    m_stream (kInvalidStream),
    m_options (),
    m_own_stream (false),
    m_is_interactive (eLazyBoolCalculate),
    m_is_real_terminal (eLazyBoolCalculate),
    m_supports_colors (eLazyBoolCalculate)
{
    Open (path, options, permissions);
}

File::File (const FileSpec& filespec,
            uint32_t options,
            uint32_t permissions) :
    IOObject(eFDTypeFile, false),
    m_descriptor (kInvalidDescriptor),
    m_stream (kInvalidStream),
    m_options (0),
    m_own_stream (false),
    m_is_interactive (eLazyBoolCalculate),
    m_is_real_terminal (eLazyBoolCalculate),
    m_supports_colors (eLazyBoolCalculate)
{
    if (filespec)
    {
        Open (filespec.GetPath().c_str(), options, permissions);
    }
}

File::~File()
{
    Close ();
}


int
File::GetDescriptor() const
{
    if (DescriptorIsValid())
        return m_descriptor;

    // Don't open the file descriptor if we don't need to, just get it from the
    // stream if we have one.
    if (StreamIsValid())
    {
#if defined(LLVM_ON_WIN32)
        return _fileno(m_stream);
#else
        return fileno(m_stream);
#endif
    }

    // Invalid descriptor and invalid stream, return invalid descriptor.
    return kInvalidDescriptor;
}

IOObject::WaitableHandle
File::GetWaitableHandle()
{
    return m_descriptor;
}


void
File::SetDescriptor (int fd, bool transfer_ownership)
{
    if (IsValid())
        Close();
    m_descriptor = fd;
    m_should_close_fd = transfer_ownership;
}


FILE *
File::GetStream ()
{
    if (!StreamIsValid())
    {
        if (DescriptorIsValid())
        {
            const char *mode = GetStreamOpenModeFromOptions (m_options);
            if (mode)
            {
                if (!m_should_close_fd)
                {
                    // We must duplicate the file descriptor if we don't own it because
                    // when you call fdopen, the stream will own the fd
#ifdef _WIN32
                    m_descriptor = ::_dup(GetDescriptor());
#else
                    m_descriptor = dup(GetDescriptor());
#endif
                    m_should_close_fd = true;
                }

                do
                {
                    m_stream = ::fdopen (m_descriptor, mode);
                } while (m_stream == NULL && errno == EINTR);

                // If we got a stream, then we own the stream and should no
                // longer own the descriptor because fclose() will close it for us

                if (m_stream)
                {
                    m_own_stream = true;
                    m_should_close_fd = false;
                }
            }
        }
    }
    return m_stream;
}

void
File::SetStream (FILE *fh, bool transfer_ownership)
{
    if (IsValid())
        Close();
    m_stream = fh;
    m_own_stream = transfer_ownership;
}

Error
File::Open (const char *path, uint32_t options, uint32_t permissions)
{
    Error error;
    if (IsValid())
        Close ();

    int oflag = 0;
    const bool read = options & eOpenOptionRead;
    const bool write = options & eOpenOptionWrite;
    if (write)
    {
        if (read)
            oflag |= O_RDWR;
        else
            oflag |= O_WRONLY;
        
        if (options & eOpenOptionAppend)
            oflag |= O_APPEND;

        if (options & eOpenOptionTruncate)
            oflag |= O_TRUNC;

        if (options & eOpenOptionCanCreate)
            oflag |= O_CREAT;
        
        if (options & eOpenOptionCanCreateNewOnly)
            oflag |= O_CREAT | O_EXCL;
    }
    else if (read)
    {
        oflag |= O_RDONLY;

#ifndef _WIN32
        if (options & eOpenOptionDontFollowSymlinks)
            oflag |= O_NOFOLLOW;
#endif
    }
    
#ifndef _WIN32
    if (options & eOpenOptionNonBlocking)
        oflag |= O_NONBLOCK;
    if (options & eOpenOptionCloseOnExec)
        oflag |= O_CLOEXEC;
#else
    oflag |= O_BINARY;
#endif

    mode_t mode = 0;
    if (oflag & O_CREAT)
    {
        if (permissions & lldb::eFilePermissionsUserRead)     mode |= S_IRUSR;
        if (permissions & lldb::eFilePermissionsUserWrite)    mode |= S_IWUSR;
        if (permissions & lldb::eFilePermissionsUserExecute)  mode |= S_IXUSR;
        if (permissions & lldb::eFilePermissionsGroupRead)    mode |= S_IRGRP;
        if (permissions & lldb::eFilePermissionsGroupWrite)   mode |= S_IWGRP;
        if (permissions & lldb::eFilePermissionsGroupExecute) mode |= S_IXGRP;
        if (permissions & lldb::eFilePermissionsWorldRead)    mode |= S_IROTH;
        if (permissions & lldb::eFilePermissionsWorldWrite)   mode |= S_IWOTH;
        if (permissions & lldb::eFilePermissionsWorldExecute) mode |= S_IXOTH;
    }

    do
    {
#ifdef _WIN32
        std::wstring wpath;
        if (!llvm::ConvertUTF8toWide(path, wpath))
        {
            m_descriptor = -1;
            error.SetErrorString("Error converting path to UTF-16");
            return error;
        }
        ::_wsopen_s(&m_descriptor, wpath.c_str(), oflag, _SH_DENYNO, mode);
#else
        m_descriptor = ::open(path, oflag, mode);
#endif
    } while (m_descriptor < 0 && errno == EINTR);

    if (!DescriptorIsValid())
        error.SetErrorToErrno();
    else
    {
        m_should_close_fd = true;
        m_options = options;
    }
    
    return error;
}

uint32_t
File::GetPermissions(const FileSpec &file_spec, Error &error)
{
    if (file_spec)
    {
        struct stat file_stats;
        int stat_result = FileSystem::Stat(file_spec.GetCString(), &file_stats);
        if (stat_result == -1)
            error.SetErrorToErrno();
        else
        {
            error.Clear();
            return file_stats.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
        }
    }
    else
        error.SetErrorString ("empty file spec");
    return 0;
}

uint32_t
File::GetPermissions(Error &error) const
{
    int fd = GetDescriptor();
    if (fd != kInvalidDescriptor)
    {
        struct stat file_stats;
        if (::fstat (fd, &file_stats) == -1)
            error.SetErrorToErrno();
        else
        {
            error.Clear();
            return file_stats.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
        }
    }
    else
    {
        error.SetErrorString ("invalid file descriptor");
    }
    return 0;
}


Error
File::Close ()
{
    Error error;
    if (StreamIsValid() && m_own_stream)
    {
        if (::fclose (m_stream) == EOF)
            error.SetErrorToErrno();
    }
    
    if (DescriptorIsValid() && m_should_close_fd)
    {
        if (::close (m_descriptor) != 0)
            error.SetErrorToErrno();
    }
    m_descriptor = kInvalidDescriptor;
    m_stream = kInvalidStream;
    m_options = 0;
    m_own_stream = false;
    m_should_close_fd = false;
    m_is_interactive = eLazyBoolCalculate;
    m_is_real_terminal = eLazyBoolCalculate;
    return error;
}

void
File::Clear ()
{
    m_stream = nullptr;
    m_descriptor = -1;
    m_options = 0;
    m_own_stream = false;
    m_is_interactive = m_supports_colors = m_is_real_terminal = eLazyBoolCalculate;
}

Error
File::GetFileSpec (FileSpec &file_spec) const
{
    Error error;
#ifdef LLDB_CONFIG_FCNTL_GETPATH_SUPPORTED
    if (IsValid ())
    {
        char path[PATH_MAX];
        if (::fcntl(GetDescriptor(), F_GETPATH, path) == -1)
            error.SetErrorToErrno();
        else
            file_spec.SetFile (path, false);
    }
    else 
    {
        error.SetErrorString("invalid file handle");
    }
#elif defined(__linux__)
    char proc[64];
    char path[PATH_MAX];
    if (::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", GetDescriptor()) < 0)
        error.SetErrorString ("cannot resolve file descriptor");
    else
    {
        ssize_t len;
        if ((len = ::readlink(proc, path, sizeof(path) - 1)) == -1)
            error.SetErrorToErrno();
        else
        {
            path[len] = '\0';
            file_spec.SetFile (path, false);
        }
    }
#else
    error.SetErrorString ("File::GetFileSpec is not supported on this platform");
#endif

    if (error.Fail())
        file_spec.Clear();
    return error;
}

off_t
File::SeekFromStart (off_t offset, Error *error_ptr)
{
    off_t result = 0;
    if (DescriptorIsValid())
    {
        result = ::lseek (m_descriptor, offset, SEEK_SET);

        if (error_ptr)
        {
            if (result == -1)
                error_ptr->SetErrorToErrno();
            else
                error_ptr->Clear();
        }
    }
    else if (StreamIsValid ())
    {
        result = ::fseek(m_stream, offset, SEEK_SET);
        
        if (error_ptr)
        {
            if (result == -1)
                error_ptr->SetErrorToErrno();
            else
                error_ptr->Clear();
        }
    }
    else if (error_ptr)
    {
        error_ptr->SetErrorString("invalid file handle");
    }
    return result;
}

off_t
File::SeekFromCurrent (off_t offset,  Error *error_ptr)
{
    off_t result = -1;
    if (DescriptorIsValid())
    {
        result = ::lseek (m_descriptor, offset, SEEK_CUR);
        
        if (error_ptr)
        {
            if (result == -1)
                error_ptr->SetErrorToErrno();
            else
                error_ptr->Clear();
        }
    }
    else if (StreamIsValid ())
    {
        result = ::fseek(m_stream, offset, SEEK_CUR);
        
        if (error_ptr)
        {
            if (result == -1)
                error_ptr->SetErrorToErrno();
            else
                error_ptr->Clear();
        }
    }
    else if (error_ptr)
    {
        error_ptr->SetErrorString("invalid file handle");
    }
    return result;
}

off_t
File::SeekFromEnd (off_t offset, Error *error_ptr)
{
    off_t result = -1;
    if (DescriptorIsValid())
    {
        result = ::lseek (m_descriptor, offset, SEEK_END);
        
        if (error_ptr)
        {
            if (result == -1)
                error_ptr->SetErrorToErrno();
            else
                error_ptr->Clear();
        }
    }
    else if (StreamIsValid ())
    {
        result = ::fseek(m_stream, offset, SEEK_END);
        
        if (error_ptr)
        {
            if (result == -1)
                error_ptr->SetErrorToErrno();
            else
                error_ptr->Clear();
        }
    }
    else if (error_ptr)
    {
        error_ptr->SetErrorString("invalid file handle");
    }
    return result;
}

Error
File::Flush ()
{
    Error error;
    if (StreamIsValid())
    {
        int err = 0;
        do
        {
            err = ::fflush (m_stream);
        } while (err == EOF && errno == EINTR);
        
        if (err == EOF)
            error.SetErrorToErrno();
    }
    else if (!DescriptorIsValid())
    {
        error.SetErrorString("invalid file handle");
    }
    return error;
}


Error
File::Sync ()
{
    Error error;
    if (DescriptorIsValid())
    {
#ifdef _WIN32
        int err = FlushFileBuffers((HANDLE)_get_osfhandle(m_descriptor));
        if (err == 0)
            error.SetErrorToGenericError();
#else
        int err = 0;
        do
        {
            err = ::fsync (m_descriptor);
        } while (err == -1 && errno == EINTR);
        
        if (err == -1)
            error.SetErrorToErrno();
#endif
    }
    else 
    {
        error.SetErrorString("invalid file handle");
    }
    return error;
}

#if defined (__APPLE__)
// Darwin kernels only can read/write <= INT_MAX bytes
#define MAX_READ_SIZE INT_MAX
#define MAX_WRITE_SIZE INT_MAX
#endif

Error
File::Read (void *buf, size_t &num_bytes)
{
    Error error;

#if defined (MAX_READ_SIZE)
    if (num_bytes > MAX_READ_SIZE)
    {
        uint8_t *p = (uint8_t *)buf;
        size_t bytes_left = num_bytes;
        // Init the num_bytes read to zero
        num_bytes = 0;

        while (bytes_left > 0)
        {
            size_t curr_num_bytes;
            if (bytes_left > MAX_READ_SIZE)
                curr_num_bytes = MAX_READ_SIZE;
            else
                curr_num_bytes = bytes_left;

            error = Read (p + num_bytes, curr_num_bytes);

            // Update how many bytes were read
            num_bytes += curr_num_bytes;
            if (bytes_left < curr_num_bytes)
                bytes_left = 0;
            else
                bytes_left -= curr_num_bytes;

            if (error.Fail())
                break;
        }
        return error;
    }
#endif

    ssize_t bytes_read = -1;
    if (DescriptorIsValid())
    {
        do
        {
            bytes_read = ::read (m_descriptor, buf, num_bytes);
        } while (bytes_read < 0 && errno == EINTR);

        if (bytes_read == -1)
        {
            error.SetErrorToErrno();
            num_bytes = 0;
        }
        else
            num_bytes = bytes_read;
    }
    else if (StreamIsValid())
    {
        bytes_read = ::fread (buf, 1, num_bytes, m_stream);

        if (bytes_read == 0)
        {
            if (::feof(m_stream))
                error.SetErrorString ("feof");
            else if (::ferror (m_stream))
                error.SetErrorString ("ferror");
            num_bytes = 0;
        }
        else
            num_bytes = bytes_read;
    }
    else 
    {
        num_bytes = 0;
        error.SetErrorString("invalid file handle");
    }
    return error;
}
          
Error
File::Write (const void *buf, size_t &num_bytes)
{
    Error error;

#if defined (MAX_WRITE_SIZE)
    if (num_bytes > MAX_WRITE_SIZE)
    {
        const uint8_t *p = (const uint8_t *)buf;
        size_t bytes_left = num_bytes;
        // Init the num_bytes written to zero
        num_bytes = 0;

        while (bytes_left > 0)
        {
            size_t curr_num_bytes;
            if (bytes_left > MAX_WRITE_SIZE)
                curr_num_bytes = MAX_WRITE_SIZE;
            else
                curr_num_bytes = bytes_left;

            error = Write (p + num_bytes, curr_num_bytes);

            // Update how many bytes were read
            num_bytes += curr_num_bytes;
            if (bytes_left < curr_num_bytes)
                bytes_left = 0;
            else
                bytes_left -= curr_num_bytes;

            if (error.Fail())
                break;
        }
        return error;
    }
#endif

    ssize_t bytes_written = -1;
    if (DescriptorIsValid())
    {
        do
        {
            bytes_written = ::write (m_descriptor, buf, num_bytes);
        } while (bytes_written < 0 && errno == EINTR);

        if (bytes_written == -1)
        {
            error.SetErrorToErrno();
            num_bytes = 0;
        }
        else
            num_bytes = bytes_written;
    }
    else if (StreamIsValid())
    {
        bytes_written = ::fwrite (buf, 1, num_bytes, m_stream);

        if (bytes_written == 0)
        {
            if (::feof(m_stream))
                error.SetErrorString ("feof");
            else if (::ferror (m_stream))
                error.SetErrorString ("ferror");
            num_bytes = 0;
        }
        else
            num_bytes = bytes_written;
        
    }
    else 
    {
        num_bytes = 0;
        error.SetErrorString("invalid file handle");
    }

    return error;
}


Error
File::Read (void *buf, size_t &num_bytes, off_t &offset)
{
    Error error;

#if defined (MAX_READ_SIZE)
    if (num_bytes > MAX_READ_SIZE)
    {
        uint8_t *p = (uint8_t *)buf;
        size_t bytes_left = num_bytes;
        // Init the num_bytes read to zero
        num_bytes = 0;

        while (bytes_left > 0)
        {
            size_t curr_num_bytes;
            if (bytes_left > MAX_READ_SIZE)
                curr_num_bytes = MAX_READ_SIZE;
            else
                curr_num_bytes = bytes_left;

            error = Read (p + num_bytes, curr_num_bytes, offset);

            // Update how many bytes were read
            num_bytes += curr_num_bytes;
            if (bytes_left < curr_num_bytes)
                bytes_left = 0;
            else
                bytes_left -= curr_num_bytes;

            if (error.Fail())
                break;
        }
        return error;
    }
#endif

#ifndef _WIN32
    int fd = GetDescriptor();
    if (fd != kInvalidDescriptor)
    {
        ssize_t bytes_read = -1;
        do
        {
            bytes_read = ::pread (fd, buf, num_bytes, offset);
        } while (bytes_read < 0 && errno == EINTR);

        if (bytes_read < 0)
        {
            num_bytes = 0;
            error.SetErrorToErrno();
        }
        else
        {
            offset += bytes_read;
            num_bytes = bytes_read;
        }
    }
    else 
    {
        num_bytes = 0;
        error.SetErrorString("invalid file handle");
    }
#else
    long cur = ::lseek(m_descriptor, 0, SEEK_CUR);
    SeekFromStart(offset);
    error = Read(buf, num_bytes);
    if (!error.Fail())
        SeekFromStart(cur);
#endif
    return error;
}

Error
File::Read (size_t &num_bytes, off_t &offset, bool null_terminate, DataBufferSP &data_buffer_sp)
{
    Error error;
    
    if (num_bytes > 0)
    {
        int fd = GetDescriptor();
        if (fd != kInvalidDescriptor)
        {
            struct stat file_stats;
            if (::fstat (fd, &file_stats) == 0)
            {
                if (file_stats.st_size > offset)
                {
                    const size_t bytes_left = file_stats.st_size - offset;
                    if (num_bytes > bytes_left)
                        num_bytes = bytes_left;
                        
                    size_t num_bytes_plus_nul_char = num_bytes + (null_terminate ? 1 : 0);
                    std::unique_ptr<DataBufferHeap> data_heap_ap;
                    data_heap_ap.reset(new DataBufferHeap());
                    data_heap_ap->SetByteSize(num_bytes_plus_nul_char);
                        
                    if (data_heap_ap.get())
                    {
                        error = Read (data_heap_ap->GetBytes(), num_bytes, offset);
                        if (error.Success())
                        {
                            // Make sure we read exactly what we asked for and if we got
                            // less, adjust the array
                            if (num_bytes_plus_nul_char < data_heap_ap->GetByteSize())
                                data_heap_ap->SetByteSize(num_bytes_plus_nul_char);
                            data_buffer_sp.reset(data_heap_ap.release());
                            return error;
                        }
                    }
                }
                else 
                    error.SetErrorString("file is empty");
            }
            else
                error.SetErrorToErrno();
        }
        else 
            error.SetErrorString("invalid file handle");
    }
    else
        error.SetErrorString("invalid file handle");

    num_bytes = 0;
    data_buffer_sp.reset();
    return error;
}

Error
File::Write (const void *buf, size_t &num_bytes, off_t &offset)
{
    Error error;

#if defined (MAX_WRITE_SIZE)
    if (num_bytes > MAX_WRITE_SIZE)
    {
        const uint8_t *p = (const uint8_t *)buf;
        size_t bytes_left = num_bytes;
        // Init the num_bytes written to zero
        num_bytes = 0;

        while (bytes_left > 0)
        {
            size_t curr_num_bytes;
            if (bytes_left > MAX_WRITE_SIZE)
                curr_num_bytes = MAX_WRITE_SIZE;
            else
                curr_num_bytes = bytes_left;

            error = Write (p + num_bytes, curr_num_bytes, offset);

            // Update how many bytes were read
            num_bytes += curr_num_bytes;
            if (bytes_left < curr_num_bytes)
                bytes_left = 0;
            else
                bytes_left -= curr_num_bytes;

            if (error.Fail())
                break;
        }
        return error;
    }
#endif

    int fd = GetDescriptor();
    if (fd != kInvalidDescriptor)
    {
#ifndef _WIN32
        ssize_t bytes_written = -1;
        do
        {
            bytes_written = ::pwrite (m_descriptor, buf, num_bytes, offset);
        } while (bytes_written < 0 && errno == EINTR);

        if (bytes_written < 0)
        {
            num_bytes = 0;
            error.SetErrorToErrno();
        }
        else
        {
            offset += bytes_written;
            num_bytes = bytes_written;
        }
#else
        long cur = ::lseek(m_descriptor, 0, SEEK_CUR);
        error = Write(buf, num_bytes);
        long after = ::lseek(m_descriptor, 0, SEEK_CUR);

        if (!error.Fail())
            SeekFromStart(cur);

        offset = after;
#endif
    }
    else 
    {
        num_bytes = 0;
        error.SetErrorString("invalid file handle");
    }
    return error;
}

//------------------------------------------------------------------
// Print some formatted output to the stream.
//------------------------------------------------------------------
size_t
File::Printf (const char *format, ...)
{
    va_list args;
    va_start (args, format);
    size_t result = PrintfVarArg (format, args);
    va_end (args);
    return result;
}

//------------------------------------------------------------------
// Print some formatted output to the stream.
//------------------------------------------------------------------
size_t
File::PrintfVarArg (const char *format, va_list args)
{
    size_t result = 0;
    if (DescriptorIsValid())
    {
        char *s = NULL;
        result = vasprintf(&s, format, args);
        if (s != NULL)
        {
            if (result > 0)
            {
                size_t s_len = result;
                Write (s, s_len);
                result = s_len;
            }
            free (s);
        }
    }
    else if (StreamIsValid())
    {
        result = ::vfprintf (m_stream, format, args);
    }
    return result;
}

mode_t
File::ConvertOpenOptionsForPOSIXOpen (uint32_t open_options)
{
    mode_t mode = 0;
    if (open_options & eOpenOptionRead && open_options & eOpenOptionWrite)
        mode |= O_RDWR;
    else if (open_options & eOpenOptionWrite)
        mode |= O_WRONLY;
    
    if (open_options & eOpenOptionAppend)
        mode |= O_APPEND;

    if (open_options & eOpenOptionTruncate)
        mode |= O_TRUNC;

    if (open_options & eOpenOptionNonBlocking)
        mode |= O_NONBLOCK;

    if (open_options & eOpenOptionCanCreateNewOnly)
        mode |= O_CREAT | O_EXCL;
    else if (open_options & eOpenOptionCanCreate)
        mode |= O_CREAT;

    return mode;
}

void
File::CalculateInteractiveAndTerminal ()
{
    const int fd = GetDescriptor();
    if (fd >= 0)
    {
        m_is_interactive = eLazyBoolNo;
        m_is_real_terminal = eLazyBoolNo;
        m_supports_colors = eLazyBoolNo;
#if defined(_WIN32)
        if (_isatty(fd))
        {
            m_is_interactive = eLazyBoolYes;
            m_is_real_terminal = eLazyBoolYes;
        }
#else
        if (isatty(fd))
        {
            m_is_interactive = eLazyBoolYes;
            struct winsize window_size;
            if (::ioctl (fd, TIOCGWINSZ, &window_size) == 0)
            {
                if (window_size.ws_col > 0)
                {
                    m_is_real_terminal = eLazyBoolYes;
                    if (llvm::sys::Process::FileDescriptorHasColors(fd))
                        m_supports_colors = eLazyBoolYes;
                }
            }
        }
#endif
    }
}

bool
File::GetIsInteractive ()
{
    if (m_is_interactive == eLazyBoolCalculate)
        CalculateInteractiveAndTerminal ();
    return m_is_interactive == eLazyBoolYes;
}

bool
File::GetIsRealTerminal ()
{
    if (m_is_real_terminal == eLazyBoolCalculate)
        CalculateInteractiveAndTerminal();
    return m_is_real_terminal == eLazyBoolYes;
}

bool
File::GetIsTerminalWithColors ()
{
    if (m_supports_colors == eLazyBoolCalculate)
        CalculateInteractiveAndTerminal();
    return m_supports_colors == eLazyBoolYes;
}

