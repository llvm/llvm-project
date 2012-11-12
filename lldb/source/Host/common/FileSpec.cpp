//===-- FileSpec.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include <dirent.h>
#include <fcntl.h>
#include <libgen.h>
#include <sys/stat.h>
#include <string.h>
#include <fstream>

#include "lldb/Host/Config.h" // Have to include this before we test the define...
#ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER
#include <pwd.h>
#endif

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "lldb/Host/File.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataBufferMemoryMap.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
#include "lldb/Host/Host.h"
#include "lldb/Utility/CleanUp.h"

using namespace lldb;
using namespace lldb_private;
using namespace std;

static bool
GetFileStats (const FileSpec *file_spec, struct stat *stats_ptr)
{
    char resolved_path[PATH_MAX];
    if (file_spec->GetPath (resolved_path, sizeof(resolved_path)))
        return ::stat (resolved_path, stats_ptr) == 0;
    return false;
}

#ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER

static const char*
GetCachedGlobTildeSlash()
{
    static std::string g_tilde;
    if (g_tilde.empty())
    {
        struct passwd *user_entry;
        user_entry = getpwuid(geteuid());
        if (user_entry != NULL)
            g_tilde = user_entry->pw_dir;

        if (g_tilde.empty())
            return NULL;
    }
    return g_tilde.c_str();
}

#endif // #ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER

// Resolves the username part of a path of the form ~user/other/directories, and
// writes the result into dst_path.
// Returns 0 if there WAS a ~ in the path but the username couldn't be resolved.
// Otherwise returns the number of characters copied into dst_path.  If the return
// is >= dst_len, then the resolved path is too long...
size_t
FileSpec::ResolveUsername (const char *src_path, char *dst_path, size_t dst_len)
{
    if (src_path == NULL || src_path[0] == '\0')
        return 0;

#ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER

    char user_home[PATH_MAX];
    const char *user_name;
    
    
    // If there's no ~, then just copy src_path straight to dst_path (they may be the same string...)
    if (src_path[0] != '~')
    {
        size_t len = strlen (src_path);
        if (len >= dst_len)
        {
            ::bcopy (src_path, dst_path, dst_len - 1);
            dst_path[dst_len] = '\0';
        }
        else
            ::bcopy (src_path, dst_path, len + 1);
        
        return len;
    }
    
    const char *first_slash = ::strchr (src_path, '/');
    char remainder[PATH_MAX];
    
    if (first_slash == NULL)
    {
        // The whole name is the username (minus the ~):
        user_name = src_path + 1;
        remainder[0] = '\0';
    }
    else
    {
        int user_name_len = first_slash - src_path - 1;
        ::memcpy (user_home, src_path + 1, user_name_len);
        user_home[user_name_len] = '\0';
        user_name = user_home;
        
        ::strcpy (remainder, first_slash);
    }
    
    if (user_name == NULL)
        return 0;
    // User name of "" means the current user...
    
    struct passwd *user_entry;
    const char *home_dir = NULL;
    
    if (user_name[0] == '\0')
    {
        home_dir = GetCachedGlobTildeSlash();
    }
    else
    {
        user_entry = ::getpwnam (user_name);
        if (user_entry != NULL)
            home_dir = user_entry->pw_dir;
    }
    
    if (home_dir == NULL)
        return 0;
    else 
        return ::snprintf (dst_path, dst_len, "%s%s", home_dir, remainder);
#else
    // Resolving home directories is not supported, just copy the path...
    return ::snprintf (dst_path, dst_len, "%s", src_path);
#endif // #ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER    
}

size_t
FileSpec::ResolvePartialUsername (const char *partial_name, StringList &matches)
{
#ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER
    size_t extant_entries = matches.GetSize();
    
    setpwent();
    struct passwd *user_entry;
    const char *name_start = partial_name + 1;
    std::set<std::string> name_list;
    
    while ((user_entry = getpwent()) != NULL)
    {
        if (strstr(user_entry->pw_name, name_start) == user_entry->pw_name)
        {
            std::string tmp_buf("~");
            tmp_buf.append(user_entry->pw_name);
            tmp_buf.push_back('/');
            name_list.insert(tmp_buf);                    
        }
    }
    std::set<std::string>::iterator pos, end = name_list.end();
    for (pos = name_list.begin(); pos != end; pos++)
    {  
        matches.AppendString((*pos).c_str());
    }
    return matches.GetSize() - extant_entries;
#else
    // Resolving home directories is not supported, just copy the path...
    return 0;
#endif // #ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER    
}



size_t
FileSpec::Resolve (const char *src_path, char *dst_path, size_t dst_len)
{
    if (src_path == NULL || src_path[0] == '\0')
        return 0;

    // Glob if needed for ~/, otherwise copy in case src_path is same as dst_path...
    char unglobbed_path[PATH_MAX];
#ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER
    if (src_path[0] == '~')
    {
        size_t return_count = ResolveUsername(src_path, unglobbed_path, sizeof(unglobbed_path));
        
        // If we couldn't find the user referred to, or the resultant path was too long,
        // then just copy over the src_path.
        if (return_count == 0 || return_count >= sizeof(unglobbed_path)) 
            ::snprintf (unglobbed_path, sizeof(unglobbed_path), "%s", src_path);
    }
    else
#endif // #ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER
    {
    	::snprintf(unglobbed_path, sizeof(unglobbed_path), "%s", src_path);
    }

    // Now resolve the path if needed
    char resolved_path[PATH_MAX];
    if (::realpath (unglobbed_path, resolved_path))
    {
        // Success, copy the resolved path
        return ::snprintf(dst_path, dst_len, "%s", resolved_path);
    }
    else
    {
        // Failed, just copy the unglobbed path
        return ::snprintf(dst_path, dst_len, "%s", unglobbed_path);
    }
}

FileSpec::FileSpec() :
    m_directory(),
    m_filename()
{
}

//------------------------------------------------------------------
// Default constructor that can take an optional full path to a
// file on disk.
//------------------------------------------------------------------
FileSpec::FileSpec(const char *pathname, bool resolve_path) :
    m_directory(),
    m_filename(),
    m_is_resolved(false)
{
    if (pathname && pathname[0])
        SetFile(pathname, resolve_path);
}

//------------------------------------------------------------------
// Copy constructor
//------------------------------------------------------------------
FileSpec::FileSpec(const FileSpec& rhs) :
    m_directory (rhs.m_directory),
    m_filename (rhs.m_filename),
    m_is_resolved (rhs.m_is_resolved)
{
}

//------------------------------------------------------------------
// Copy constructor
//------------------------------------------------------------------
FileSpec::FileSpec(const FileSpec* rhs) :
    m_directory(),
    m_filename()
{
    if (rhs)
        *this = *rhs;
}

//------------------------------------------------------------------
// Virtual destrcuctor in case anyone inherits from this class.
//------------------------------------------------------------------
FileSpec::~FileSpec()
{
}

//------------------------------------------------------------------
// Assignment operator.
//------------------------------------------------------------------
const FileSpec&
FileSpec::operator= (const FileSpec& rhs)
{
    if (this != &rhs)
    {
        m_directory = rhs.m_directory;
        m_filename = rhs.m_filename;
        m_is_resolved = rhs.m_is_resolved;
    }
    return *this;
}

//------------------------------------------------------------------
// Update the contents of this object with a new path. The path will
// be split up into a directory and filename and stored as uniqued
// string values for quick comparison and efficient memory usage.
//------------------------------------------------------------------
void
FileSpec::SetFile (const char *pathname, bool resolve)
{
    m_filename.Clear();
    m_directory.Clear();
    m_is_resolved = false;
    if (pathname == NULL || pathname[0] == '\0')
        return;

    char resolved_path[PATH_MAX];
    bool path_fit = true;
    
    if (resolve)
    {
        path_fit = (FileSpec::Resolve (pathname, resolved_path, sizeof(resolved_path)) < sizeof(resolved_path) - 1);
        m_is_resolved = path_fit;
    }
    else
    {
        // Copy the path because "basename" and "dirname" want to muck with the
        // path buffer
        if (::strlen (pathname) > sizeof(resolved_path) - 1)
            path_fit = false;
        else
            ::strcpy (resolved_path, pathname);
    }

    
    if (path_fit)
    {
        char *filename = ::basename (resolved_path);
        if (filename)
        {
            m_filename.SetCString (filename);
            // Truncate the basename off the end of the resolved path

            // Only attempt to get the dirname if it looks like we have a path
            if (strchr(resolved_path, '/'))
            {
                char *directory = ::dirname (resolved_path);

                // Make sure we didn't get our directory resolved to "." without having
                // specified
                if (directory)
                    m_directory.SetCString(directory);
                else
                {
                    char *last_resolved_path_slash = strrchr(resolved_path, '/');
                    if (last_resolved_path_slash)
                    {
                        *last_resolved_path_slash = '\0';
                        m_directory.SetCString(resolved_path);
                    }
                }
            }
        }
        else
            m_directory.SetCString(resolved_path);
    }
}

//----------------------------------------------------------------------
// Convert to pointer operator. This allows code to check any FileSpec
// objects to see if they contain anything valid using code such as:
//
//  if (file_spec)
//  {}
//----------------------------------------------------------------------
FileSpec::operator bool() const
{
    return m_filename || m_directory;
}

//----------------------------------------------------------------------
// Logical NOT operator. This allows code to check any FileSpec
// objects to see if they are invalid using code such as:
//
//  if (!file_spec)
//  {}
//----------------------------------------------------------------------
bool
FileSpec::operator!() const
{
    return !m_directory && !m_filename;
}

//------------------------------------------------------------------
// Equal to operator
//------------------------------------------------------------------
bool
FileSpec::operator== (const FileSpec& rhs) const
{
    if (m_filename == rhs.m_filename)
    {
        if (m_directory == rhs.m_directory)
            return true;
        
        // TODO: determine if we want to keep this code in here.
        // The code below was added to handle a case where we were
        // trying to set a file and line breakpoint and one path
        // was resolved, and the other not and the directory was
        // in a mount point that resolved to a more complete path:
        // "/tmp/a.c" == "/private/tmp/a.c". I might end up pulling
        // this out...
        if (IsResolved() && rhs.IsResolved())
        {
            // Both paths are resolved, no need to look further...
            return false;
        }
        
        FileSpec resolved_lhs(*this);

        // If "this" isn't resolved, resolve it
        if (!IsResolved())
        {
            if (resolved_lhs.ResolvePath())
            {
                // This path wasn't resolved but now it is. Check if the resolved
                // directory is the same as our unresolved directory, and if so, 
                // we can mark this object as resolved to avoid more future resolves
                m_is_resolved = (m_directory == resolved_lhs.m_directory);
            }
            else
                return false;
        }
        
        FileSpec resolved_rhs(rhs);
        if (!rhs.IsResolved())
        {
            if (resolved_rhs.ResolvePath())
            {
                // rhs's path wasn't resolved but now it is. Check if the resolved
                // directory is the same as rhs's unresolved directory, and if so, 
                // we can mark this object as resolved to avoid more future resolves
                rhs.m_is_resolved = (rhs.m_directory == resolved_rhs.m_directory);
            }
            else
                return false;
        }

        // If we reach this point in the code we were able to resolve both paths
        // and since we only resolve the paths if the basenames are equal, then
        // we can just check if both directories are equal...
        return resolved_lhs.GetDirectory() == resolved_rhs.GetDirectory();
    }
    return false;
}

//------------------------------------------------------------------
// Not equal to operator
//------------------------------------------------------------------
bool
FileSpec::operator!= (const FileSpec& rhs) const
{
    return !(*this == rhs);
}

//------------------------------------------------------------------
// Less than operator
//------------------------------------------------------------------
bool
FileSpec::operator< (const FileSpec& rhs) const
{
    return FileSpec::Compare(*this, rhs, true) < 0;
}

//------------------------------------------------------------------
// Dump a FileSpec object to a stream
//------------------------------------------------------------------
Stream&
lldb_private::operator << (Stream &s, const FileSpec& f)
{
    f.Dump(&s);
    return s;
}

//------------------------------------------------------------------
// Clear this object by releasing both the directory and filename
// string values and making them both the empty string.
//------------------------------------------------------------------
void
FileSpec::Clear()
{
    m_directory.Clear();
    m_filename.Clear();
}

//------------------------------------------------------------------
// Compare two FileSpec objects. If "full" is true, then both
// the directory and the filename must match. If "full" is false,
// then the directory names for "a" and "b" are only compared if
// they are both non-empty. This allows a FileSpec object to only
// contain a filename and it can match FileSpec objects that have
// matching filenames with different paths.
//
// Return -1 if the "a" is less than "b", 0 if "a" is equal to "b"
// and "1" if "a" is greater than "b".
//------------------------------------------------------------------
int
FileSpec::Compare(const FileSpec& a, const FileSpec& b, bool full)
{
    int result = 0;

    // If full is true, then we must compare both the directory and filename.

    // If full is false, then if either directory is empty, then we match on
    // the basename only, and if both directories have valid values, we still
    // do a full compare. This allows for matching when we just have a filename
    // in one of the FileSpec objects.

    if (full || (a.m_directory && b.m_directory))
    {
        result = ConstString::Compare(a.m_directory, b.m_directory);
        if (result)
            return result;
    }
    return ConstString::Compare (a.m_filename, b.m_filename);
}

bool
FileSpec::Equal (const FileSpec& a, const FileSpec& b, bool full)
{
    if (!full && (a.GetDirectory().IsEmpty() || b.GetDirectory().IsEmpty()))
        return a.m_filename == b.m_filename;
    else
        return a == b;
}



//------------------------------------------------------------------
// Dump the object to the supplied stream. If the object contains
// a valid directory name, it will be displayed followed by a
// directory delimiter, and the filename.
//------------------------------------------------------------------
void
FileSpec::Dump(Stream *s) const
{
    if (s)
    {
        m_directory.Dump(s);
        if (m_directory)
            s->PutChar('/');
        m_filename.Dump(s);
    }
}

//------------------------------------------------------------------
// Returns true if the file exists.
//------------------------------------------------------------------
bool
FileSpec::Exists () const
{
    struct stat file_stats;
    return GetFileStats (this, &file_stats);
}

bool
FileSpec::ResolveExecutableLocation ()
{
    if (!m_directory)
    {
        const char *file_cstr = m_filename.GetCString();
        if (file_cstr)
        {
            const std::string file_str (file_cstr);
            llvm::sys::Path path = llvm::sys::Program::FindProgramByName (file_str);
            const std::string &path_str = path.str();
            llvm::StringRef dir_ref = llvm::sys::path::parent_path(path_str);
            //llvm::StringRef dir_ref = path.getDirname();
            if (! dir_ref.empty())
            {
                // FindProgramByName returns "." if it can't find the file.
                if (strcmp (".", dir_ref.data()) == 0)
                    return false;

                m_directory.SetCString (dir_ref.data());
                if (Exists())
                    return true;
                else
                {
                    // If FindProgramByName found the file, it returns the directory + filename in its return results.
                    // We need to separate them.
                    FileSpec tmp_file (dir_ref.data(), false);
                    if (tmp_file.Exists())
                    {
                        m_directory = tmp_file.m_directory;
                        return true;
                    }
                }
            }
        }
    }
    
    return false;
}

bool
FileSpec::ResolvePath ()
{
    if (m_is_resolved)
        return true;    // We have already resolved this path

    char path_buf[PATH_MAX];    
    if (!GetPath (path_buf, PATH_MAX))
        return false;
    // SetFile(...) will set m_is_resolved correctly if it can resolve the path
    SetFile (path_buf, true);
    return m_is_resolved; 
}

uint64_t
FileSpec::GetByteSize() const
{
    struct stat file_stats;
    if (GetFileStats (this, &file_stats))
        return file_stats.st_size;
    return 0;
}

FileSpec::FileType
FileSpec::GetFileType () const
{
    struct stat file_stats;
    if (GetFileStats (this, &file_stats))
    {
        mode_t file_type = file_stats.st_mode & S_IFMT;
        switch (file_type)
        {
        case S_IFDIR:   return eFileTypeDirectory;
        case S_IFIFO:   return eFileTypePipe;
        case S_IFREG:   return eFileTypeRegular;
        case S_IFSOCK:  return eFileTypeSocket;
        case S_IFLNK:   return eFileTypeSymbolicLink;
        default:
            break;
        }
        return eFileTypeUnknown;
    }
    return eFileTypeInvalid;
}

TimeValue
FileSpec::GetModificationTime () const
{
    TimeValue mod_time;
    struct stat file_stats;
    if (GetFileStats (this, &file_stats))
        mod_time.OffsetWithSeconds(file_stats.st_mtime);
    return mod_time;
}

//------------------------------------------------------------------
// Directory string get accessor.
//------------------------------------------------------------------
ConstString &
FileSpec::GetDirectory()
{
    return m_directory;
}

//------------------------------------------------------------------
// Directory string const get accessor.
//------------------------------------------------------------------
const ConstString &
FileSpec::GetDirectory() const
{
    return m_directory;
}

//------------------------------------------------------------------
// Filename string get accessor.
//------------------------------------------------------------------
ConstString &
FileSpec::GetFilename()
{
    return m_filename;
}

//------------------------------------------------------------------
// Filename string const get accessor.
//------------------------------------------------------------------
const ConstString &
FileSpec::GetFilename() const
{
    return m_filename;
}

//------------------------------------------------------------------
// Extract the directory and path into a fixed buffer. This is
// needed as the directory and path are stored in separate string
// values.
//------------------------------------------------------------------
size_t
FileSpec::GetPath(char *path, size_t path_max_len) const
{
    if (path_max_len)
    {
        const char *dirname = m_directory.GetCString();
        const char *filename = m_filename.GetCString();
        if (dirname)
        {
            if (filename)
                return ::snprintf (path, path_max_len, "%s/%s", dirname, filename);
            else
                return ::snprintf (path, path_max_len, "%s", dirname);
        }
        else if (filename)
        {
            return ::snprintf (path, path_max_len, "%s", filename);
        }
    }
    if (path)
        path[0] = '\0';
    return 0;
}

ConstString
FileSpec::GetFileNameExtension () const
{
    if (m_filename)
    {
        const char *filename = m_filename.GetCString();
        const char* dot_pos = strrchr(filename, '.');
        if (dot_pos && dot_pos[1] != '\0')
            return ConstString(dot_pos+1);
    }
    return ConstString();
}

ConstString
FileSpec::GetFileNameStrippingExtension () const
{
    const char *filename = m_filename.GetCString();
    if (filename == NULL)
        return ConstString();
    
    const char* dot_pos = strrchr(filename, '.');
    if (dot_pos == NULL)
        return m_filename;
    
    return ConstString(filename, dot_pos-filename);
}

//------------------------------------------------------------------
// Returns a shared pointer to a data buffer that contains all or
// part of the contents of a file. The data is memory mapped and
// will lazily page in data from the file as memory is accessed.
// The data that is mappped will start "file_offset" bytes into the
// file, and "file_size" bytes will be mapped. If "file_size" is
// greater than the number of bytes available in the file starting
// at "file_offset", the number of bytes will be appropriately
// truncated. The final number of bytes that get mapped can be
// verified using the DataBuffer::GetByteSize() function.
//------------------------------------------------------------------
DataBufferSP
FileSpec::MemoryMapFileContents(off_t file_offset, size_t file_size) const
{
    DataBufferSP data_sp;
    auto_ptr<DataBufferMemoryMap> mmap_data(new DataBufferMemoryMap());
    if (mmap_data.get())
    {
        if (mmap_data->MemoryMapFromFileSpec (this, file_offset, file_size) >= file_size)
            data_sp.reset(mmap_data.release());
    }
    return data_sp;
}


//------------------------------------------------------------------
// Return the size in bytes that this object takes in memory. This
// returns the size in bytes of this object, not any shared string
// values it may refer to.
//------------------------------------------------------------------
size_t
FileSpec::MemorySize() const
{
    return m_filename.MemorySize() + m_directory.MemorySize();
}


size_t
FileSpec::ReadFileContents (off_t file_offset, void *dst, size_t dst_len, Error *error_ptr) const
{
    Error error;
    size_t bytes_read = 0;
    char resolved_path[PATH_MAX];
    if (GetPath(resolved_path, sizeof(resolved_path)))
    {
        File file;
        error = file.Open(resolved_path, File::eOpenOptionRead);
        if (error.Success())
        {
            off_t file_offset_after_seek = file_offset;
            bytes_read = dst_len;
            error = file.Read(dst, bytes_read, file_offset_after_seek);
        }
    }
    else
    {
        error.SetErrorString("invalid file specification");
    }
    if (error_ptr)
        *error_ptr = error;
    return bytes_read;
}

//------------------------------------------------------------------
// Returns a shared pointer to a data buffer that contains all or
// part of the contents of a file. The data copies into a heap based
// buffer that lives in the DataBuffer shared pointer object returned.
// The data that is cached will start "file_offset" bytes into the
// file, and "file_size" bytes will be mapped. If "file_size" is
// greater than the number of bytes available in the file starting
// at "file_offset", the number of bytes will be appropriately
// truncated. The final number of bytes that get mapped can be
// verified using the DataBuffer::GetByteSize() function.
//------------------------------------------------------------------
DataBufferSP
FileSpec::ReadFileContents (off_t file_offset, size_t file_size, Error *error_ptr) const
{
    Error error;
    DataBufferSP data_sp;
    char resolved_path[PATH_MAX];
    if (GetPath(resolved_path, sizeof(resolved_path)))
    {
        File file;
        error = file.Open(resolved_path, File::eOpenOptionRead);
        if (error.Success())
        {
            const bool null_terminate = false;
            error = file.Read (file_size, file_offset, null_terminate, data_sp);
        }
    }
    else
    {
        error.SetErrorString("invalid file specification");
    }
    if (error_ptr)
        *error_ptr = error;
    return data_sp;
}

DataBufferSP
FileSpec::ReadFileContentsAsCString(Error *error_ptr)
{
    Error error;
    DataBufferSP data_sp;
    char resolved_path[PATH_MAX];
    if (GetPath(resolved_path, sizeof(resolved_path)))
    {
        File file;
        error = file.Open(resolved_path, File::eOpenOptionRead);
        if (error.Success())
        {
            off_t offset = 0;
            size_t length = SIZE_MAX;
            const bool null_terminate = true;
            error = file.Read (length, offset, null_terminate, data_sp);
        }
    }
    else
    {
        error.SetErrorString("invalid file specification");
    }
    if (error_ptr)
        *error_ptr = error;
    return data_sp;
}

size_t
FileSpec::ReadFileLines (STLStringArray &lines)
{
    lines.clear();
    char path[PATH_MAX];
    if (GetPath(path, sizeof(path)))
    {
        ifstream file_stream (path);

        if (file_stream)
        {
            std::string line;
            while (getline (file_stream, line))
                lines.push_back (line);
        }
    }
    return lines.size();
}

FileSpec::EnumerateDirectoryResult
FileSpec::EnumerateDirectory
(
    const char *dir_path, 
    bool find_directories,
    bool find_files,
    bool find_other,
    EnumerateDirectoryCallbackType callback,
    void *callback_baton
)
{
    if (dir_path && dir_path[0])
    {
        lldb_utility::CleanUp <DIR *, int> dir_path_dir (opendir(dir_path), NULL, closedir);
        if (dir_path_dir.is_valid())
        {
            struct dirent* dp;
            while ((dp = readdir(dir_path_dir.get())) != NULL)
            {
                // Only search directories
                if (dp->d_type == DT_DIR || dp->d_type == DT_UNKNOWN)
                {
                    size_t len = strlen(dp->d_name);

                    if (len == 1 && dp->d_name[0] == '.')
                        continue;

                    if (len == 2 && dp->d_name[0] == '.' && dp->d_name[1] == '.')
                        continue;
                }
            
                bool call_callback = false;
                FileSpec::FileType file_type = eFileTypeUnknown;

                switch (dp->d_type)
                {
                default:
                case DT_UNKNOWN:    file_type = eFileTypeUnknown;       call_callback = true;               break;
                case DT_FIFO:       file_type = eFileTypePipe;          call_callback = find_other;         break;
                case DT_CHR:        file_type = eFileTypeOther;         call_callback = find_other;         break;
                case DT_DIR:        file_type = eFileTypeDirectory;     call_callback = find_directories;   break;
                case DT_BLK:        file_type = eFileTypeOther;         call_callback = find_other;         break;
                case DT_REG:        file_type = eFileTypeRegular;       call_callback = find_files;         break;
                case DT_LNK:        file_type = eFileTypeSymbolicLink;  call_callback = find_other;         break;
                case DT_SOCK:       file_type = eFileTypeSocket;        call_callback = find_other;         break;
#if !defined(__OpenBSD__)
                case DT_WHT:        file_type = eFileTypeOther;         call_callback = find_other;         break;
#endif
                }

                if (call_callback)
                {
                    char child_path[PATH_MAX];
                    const int child_path_len = ::snprintf (child_path, sizeof(child_path), "%s/%s", dir_path, dp->d_name);
                    if (child_path_len < (int)(sizeof(child_path) - 1))
                    {
                        // Don't resolve the file type or path
                        FileSpec child_path_spec (child_path, false);

                        EnumerateDirectoryResult result = callback (callback_baton, file_type, child_path_spec);
                        
                        switch (result)
                        {
                        default:
                        case eEnumerateDirectoryResultNext:  
                            // Enumerate next entry in the current directory. We just
                            // exit this switch and will continue enumerating the
                            // current directory as we currently are...
                            break;

                        case eEnumerateDirectoryResultEnter: // Recurse into the current entry if it is a directory or symlink, or next if not
                            if (FileSpec::EnumerateDirectory (child_path, 
                                                              find_directories, 
                                                              find_files, 
                                                              find_other, 
                                                              callback, 
                                                              callback_baton) == eEnumerateDirectoryResultQuit)
                            {
                                // The subdirectory returned Quit, which means to 
                                // stop all directory enumerations at all levels.
                                return eEnumerateDirectoryResultQuit;
                            }
                            break;
                        
                        case eEnumerateDirectoryResultExit:  // Exit from the current directory at the current level.
                            // Exit from this directory level and tell parent to 
                            // keep enumerating.
                            return eEnumerateDirectoryResultNext;

                        case eEnumerateDirectoryResultQuit:  // Stop directory enumerations at any level
                            return eEnumerateDirectoryResultQuit;
                        }
                    }
                }
            }
        }
    }
    // By default when exiting a directory, we tell the parent enumeration
    // to continue enumerating.
    return eEnumerateDirectoryResultNext;    
}

//------------------------------------------------------------------
/// Returns true if the filespec represents an implementation source
/// file (files with a ".c", ".cpp", ".m", ".mm" (many more)
/// extension).
///
/// @return
///     \b true if the filespec represents an implementation source
///     file, \b false otherwise.
//------------------------------------------------------------------
bool
FileSpec::IsSourceImplementationFile () const
{
    ConstString extension (GetFileNameExtension());
    if (extension)
    {
        static RegularExpression g_source_file_regex ("^(c|m|mm|cpp|c\\+\\+|cxx|cc|cp|s|asm|f|f77|f90|f95|f03|for|ftn|fpp|ada|adb|ads)$",
                                                      REG_EXTENDED | REG_ICASE);
        return g_source_file_regex.Execute (extension.GetCString());
    }
    return false;
}

bool
FileSpec::IsRelativeToCurrentWorkingDirectory () const
{
    const char *directory = m_directory.GetCString();
    if (directory && directory[0])
    {
        // If the path doesn't start with '/' or '~', return true
        switch (directory[0])
        {
        case '/':
        case '~':
            return false;
        default:
            return true;
        }
    }
    else if (m_filename)
    {
        // No directory, just a basename, return true
        return true;
    }
    return false;
}


