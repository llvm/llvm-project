//===-- FileSpec.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _WIN32
#include <dirent.h>
#else
#include "lldb/Host/windows/windows.h"
#endif
#include <fcntl.h>
#ifndef _MSC_VER
#include <libgen.h>
#endif
#include <fstream>
#include <set>
#include <string.h>

#include "lldb/Host/Config.h" // Have to include this before we test the define...
#ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER
#include <pwd.h>
#endif

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataBufferMemoryMap.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Utility/CleanUp.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

using namespace lldb;
using namespace lldb_private;

namespace {

bool PathSyntaxIsPosix(FileSpec::PathSyntax syntax) {
  return (syntax == FileSpec::ePathSyntaxPosix ||
          (syntax == FileSpec::ePathSyntaxHostNative &&
           FileSystem::GetNativePathSyntax() == FileSpec::ePathSyntaxPosix));
}

const char *GetPathSeparators(FileSpec::PathSyntax syntax) {
  return PathSyntaxIsPosix(syntax) ? "/" : "\\/";
}

char GetPreferredPathSeparator(FileSpec::PathSyntax syntax) {
  return GetPathSeparators(syntax)[0];
}

bool IsPathSeparator(char value, FileSpec::PathSyntax syntax) {
  return value == '/' || (!PathSyntaxIsPosix(syntax) && value == '\\');
}

void Normalize(llvm::SmallVectorImpl<char> &path, FileSpec::PathSyntax syntax) {
  if (PathSyntaxIsPosix(syntax))
    return;

  std::replace(path.begin(), path.end(), '\\', '/');
  // Windows path can have \\ slashes which can be changed by replace
  // call above to //. Here we remove the duplicate.
  auto iter = std::unique(path.begin(), path.end(), [](char &c1, char &c2) {
    return (c1 == '/' && c2 == '/');
  });
  path.erase(iter, path.end());
}

void Denormalize(llvm::SmallVectorImpl<char> &path,
                 FileSpec::PathSyntax syntax) {
  if (PathSyntaxIsPosix(syntax))
    return;

  std::replace(path.begin(), path.end(), '/', '\\');
}

bool GetFileStats(const FileSpec *file_spec, struct stat *stats_ptr) {
  char resolved_path[PATH_MAX];
  if (file_spec->GetPath(resolved_path, sizeof(resolved_path)))
    return FileSystem::Stat(resolved_path, stats_ptr) == 0;
  return false;
}

size_t FilenamePos(llvm::StringRef str, FileSpec::PathSyntax syntax) {
  if (str.size() == 2 && IsPathSeparator(str[0], syntax) && str[0] == str[1])
    return 0;

  if (str.size() > 0 && IsPathSeparator(str.back(), syntax))
    return str.size() - 1;

  size_t pos = str.find_last_of(GetPathSeparators(syntax), str.size() - 1);

  if (!PathSyntaxIsPosix(syntax) && pos == llvm::StringRef::npos)
    pos = str.find_last_of(':', str.size() - 2);

  if (pos == llvm::StringRef::npos ||
      (pos == 1 && IsPathSeparator(str[0], syntax)))
    return 0;

  return pos + 1;
}

size_t RootDirStart(llvm::StringRef str, FileSpec::PathSyntax syntax) {
  // case "c:/"
  if (!PathSyntaxIsPosix(syntax) &&
      (str.size() > 2 && str[1] == ':' && IsPathSeparator(str[2], syntax)))
    return 2;

  // case "//"
  if (str.size() == 2 && IsPathSeparator(str[0], syntax) && str[0] == str[1])
    return llvm::StringRef::npos;

  // case "//net"
  if (str.size() > 3 && IsPathSeparator(str[0], syntax) && str[0] == str[1] &&
      !IsPathSeparator(str[2], syntax))
    return str.find_first_of(GetPathSeparators(syntax), 2);

  // case "/"
  if (str.size() > 0 && IsPathSeparator(str[0], syntax))
    return 0;

  return llvm::StringRef::npos;
}

size_t ParentPathEnd(llvm::StringRef path, FileSpec::PathSyntax syntax) {
  size_t end_pos = FilenamePos(path, syntax);

  bool filename_was_sep =
      path.size() > 0 && IsPathSeparator(path[end_pos], syntax);

  // Skip separators except for root dir.
  size_t root_dir_pos = RootDirStart(path.substr(0, end_pos), syntax);

  while (end_pos > 0 && (end_pos - 1) != root_dir_pos &&
         IsPathSeparator(path[end_pos - 1], syntax))
    --end_pos;

  if (end_pos == 1 && root_dir_pos == 0 && filename_was_sep)
    return llvm::StringRef::npos;

  return end_pos;
}

} // end anonymous namespace

// Resolves the username part of a path of the form ~user/other/directories, and
// writes the result into dst_path.  This will also resolve "~" to the current
// user.
// If you want to complete "~" to the list of users, pass it to
// ResolvePartialUsername.
void FileSpec::ResolveUsername(llvm::SmallVectorImpl<char> &path) {
#if LLDB_CONFIG_TILDE_RESOLVES_TO_USER
  if (path.empty() || path[0] != '~')
    return;

  llvm::StringRef path_str(path.data(), path.size());
  size_t slash_pos = path_str.find('/', 1);
  if (slash_pos == 1 || path.size() == 1) {
    // A path of ~/ resolves to the current user's home dir
    llvm::SmallString<64> home_dir;
    // llvm::sys::path::home_directory() only checks if "HOME" is set in the
    // environment and does nothing else to locate the user home directory
    if (!llvm::sys::path::home_directory(home_dir)) {
      struct passwd *pw = getpwuid(getuid());
      if (pw && pw->pw_dir && pw->pw_dir[0]) {
        // Update our environemnt so llvm::sys::path::home_directory() works
        // next time
        setenv("HOME", pw->pw_dir, 0);
        home_dir.assign(llvm::StringRef(pw->pw_dir));
      } else {
        return;
      }
    }

    // Overwrite the ~ with the first character of the homedir, and insert
    // the rest.  This way we only trigger one move, whereas an insert
    // followed by a delete (or vice versa) would trigger two.
    path[0] = home_dir[0];
    path.insert(path.begin() + 1, home_dir.begin() + 1, home_dir.end());
    return;
  }

  auto username_begin = path.begin() + 1;
  auto username_end = (slash_pos == llvm::StringRef::npos)
                          ? path.end()
                          : (path.begin() + slash_pos);
  size_t replacement_length = std::distance(path.begin(), username_end);

  llvm::SmallString<20> username(username_begin, username_end);
  struct passwd *user_entry = ::getpwnam(username.c_str());
  if (user_entry != nullptr) {
    // Copy over the first n characters of the path, where n is the smaller of
    // the length
    // of the home directory and the slash pos.
    llvm::StringRef homedir(user_entry->pw_dir);
    size_t initial_copy_length = std::min(homedir.size(), replacement_length);
    auto src_begin = homedir.begin();
    auto src_end = src_begin + initial_copy_length;
    std::copy(src_begin, src_end, path.begin());
    if (replacement_length > homedir.size()) {
      // We copied the entire home directory, but the ~username portion of the
      // path was
      // longer, so there's characters that need to be removed.
      path.erase(path.begin() + initial_copy_length, username_end);
    } else if (replacement_length < homedir.size()) {
      // We copied all the way up to the slash in the destination, but there's
      // still more
      // characters that need to be inserted.
      path.insert(username_end, src_end, homedir.end());
    }
  } else {
    // Unable to resolve username (user doesn't exist?)
    path.clear();
  }
#endif
}

size_t FileSpec::ResolvePartialUsername(llvm::StringRef partial_name,
                                        StringList &matches) {
#ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER
  size_t extant_entries = matches.GetSize();

  setpwent();
  struct passwd *user_entry;
  partial_name = partial_name.drop_front();
  std::set<std::string> name_list;

  while ((user_entry = getpwent()) != NULL) {
    if (llvm::StringRef(user_entry->pw_name).startswith(partial_name)) {
      std::string tmp_buf("~");
      tmp_buf.append(user_entry->pw_name);
      tmp_buf.push_back('/');
      name_list.insert(tmp_buf);
    }
  }

  for (auto &name : name_list) {
    matches.AppendString(name);
  }
  return matches.GetSize() - extant_entries;
#else
  // Resolving home directories is not supported, just copy the path...
  return 0;
#endif // #ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER
}

void FileSpec::Resolve(llvm::SmallVectorImpl<char> &path) {
  if (path.empty())
    return;

#ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER
  if (path[0] == '~')
    ResolveUsername(path);
#endif // #ifdef LLDB_CONFIG_TILDE_RESOLVES_TO_USER

  // Save a copy of the original path that's passed in
  llvm::SmallString<128> original_path(path.begin(), path.end());

  llvm::sys::fs::make_absolute(path);
  if (!llvm::sys::fs::exists(path)) {
    path.clear();
    path.append(original_path.begin(), original_path.end());
  }
}

FileSpec::FileSpec() : m_syntax(FileSystem::GetNativePathSyntax()) {}

//------------------------------------------------------------------
// Default constructor that can take an optional full path to a
// file on disk.
//------------------------------------------------------------------
FileSpec::FileSpec(llvm::StringRef path, bool resolve_path, PathSyntax syntax)
    : m_syntax(syntax) {
  SetFile(path, resolve_path, syntax);
}

FileSpec::FileSpec(llvm::StringRef path, bool resolve_path, ArchSpec arch)
    : FileSpec{path, resolve_path, arch.GetTriple().isOSWindows()
                                       ? ePathSyntaxWindows
                                       : ePathSyntaxPosix} {}

//------------------------------------------------------------------
// Copy constructor
//------------------------------------------------------------------
FileSpec::FileSpec(const FileSpec &rhs)
    : m_directory(rhs.m_directory), m_filename(rhs.m_filename),
      m_is_resolved(rhs.m_is_resolved), m_syntax(rhs.m_syntax) {}

//------------------------------------------------------------------
// Copy constructor
//------------------------------------------------------------------
FileSpec::FileSpec(const FileSpec *rhs) : m_directory(), m_filename() {
  if (rhs)
    *this = *rhs;
}

//------------------------------------------------------------------
// Virtual destructor in case anyone inherits from this class.
//------------------------------------------------------------------
FileSpec::~FileSpec() {}

//------------------------------------------------------------------
// Assignment operator.
//------------------------------------------------------------------
const FileSpec &FileSpec::operator=(const FileSpec &rhs) {
  if (this != &rhs) {
    m_directory = rhs.m_directory;
    m_filename = rhs.m_filename;
    m_is_resolved = rhs.m_is_resolved;
    m_syntax = rhs.m_syntax;
  }
  return *this;
}

//------------------------------------------------------------------
// Update the contents of this object with a new path. The path will
// be split up into a directory and filename and stored as uniqued
// string values for quick comparison and efficient memory usage.
//------------------------------------------------------------------
void FileSpec::SetFile(llvm::StringRef pathname, bool resolve,
                       PathSyntax syntax) {
  // CLEANUP: Use StringRef for string handling.  This function is kind of a
  // mess and the unclear semantics of RootDirStart and ParentPathEnd make
  // it very difficult to understand this function.  There's no reason this
  // function should be particularly complicated or difficult to understand.
  m_filename.Clear();
  m_directory.Clear();
  m_is_resolved = false;
  m_syntax = (syntax == ePathSyntaxHostNative)
                 ? FileSystem::GetNativePathSyntax()
                 : syntax;

  if (pathname.empty())
    return;

  llvm::SmallString<64> resolved(pathname);

  if (resolve) {
    FileSpec::Resolve(resolved);
    m_is_resolved = true;
  }

  Normalize(resolved, m_syntax);

  llvm::StringRef resolve_path_ref(resolved.c_str());
  size_t dir_end = ParentPathEnd(resolve_path_ref, m_syntax);
  if (dir_end == 0) {
    m_filename.SetString(resolve_path_ref);
    return;
  }

  m_directory.SetString(resolve_path_ref.substr(0, dir_end));

  size_t filename_begin = dir_end;
  size_t root_dir_start = RootDirStart(resolve_path_ref, m_syntax);
  while (filename_begin != llvm::StringRef::npos &&
         filename_begin < resolve_path_ref.size() &&
         filename_begin != root_dir_start &&
         IsPathSeparator(resolve_path_ref[filename_begin], m_syntax))
    ++filename_begin;
  m_filename.SetString((filename_begin == llvm::StringRef::npos ||
                        filename_begin >= resolve_path_ref.size())
                           ? "."
                           : resolve_path_ref.substr(filename_begin));
}

void FileSpec::SetFile(llvm::StringRef path, bool resolve, ArchSpec arch) {
  return SetFile(path, resolve, arch.GetTriple().isOSWindows()
                                    ? ePathSyntaxWindows
                                    : ePathSyntaxPosix);
}

//----------------------------------------------------------------------
// Convert to pointer operator. This allows code to check any FileSpec
// objects to see if they contain anything valid using code such as:
//
//  if (file_spec)
//  {}
//----------------------------------------------------------------------
FileSpec::operator bool() const { return m_filename || m_directory; }

//----------------------------------------------------------------------
// Logical NOT operator. This allows code to check any FileSpec
// objects to see if they are invalid using code such as:
//
//  if (!file_spec)
//  {}
//----------------------------------------------------------------------
bool FileSpec::operator!() const { return !m_directory && !m_filename; }

bool FileSpec::DirectoryEquals(const FileSpec &rhs) const {
  const bool case_sensitive = IsCaseSensitive() || rhs.IsCaseSensitive();
  return ConstString::Equals(m_directory, rhs.m_directory, case_sensitive);
}

bool FileSpec::FileEquals(const FileSpec &rhs) const {
  const bool case_sensitive = IsCaseSensitive() || rhs.IsCaseSensitive();
  return ConstString::Equals(m_filename, rhs.m_filename, case_sensitive);
}

//------------------------------------------------------------------
// Equal to operator
//------------------------------------------------------------------
bool FileSpec::operator==(const FileSpec &rhs) const {
  if (!FileEquals(rhs))
    return false;
  if (DirectoryEquals(rhs))
    return true;

  // TODO: determine if we want to keep this code in here.
  // The code below was added to handle a case where we were
  // trying to set a file and line breakpoint and one path
  // was resolved, and the other not and the directory was
  // in a mount point that resolved to a more complete path:
  // "/tmp/a.c" == "/private/tmp/a.c". I might end up pulling
  // this out...
  if (IsResolved() && rhs.IsResolved()) {
    // Both paths are resolved, no need to look further...
    return false;
  }

  FileSpec resolved_lhs(*this);

  // If "this" isn't resolved, resolve it
  if (!IsResolved()) {
    if (resolved_lhs.ResolvePath()) {
      // This path wasn't resolved but now it is. Check if the resolved
      // directory is the same as our unresolved directory, and if so,
      // we can mark this object as resolved to avoid more future resolves
      m_is_resolved = (m_directory == resolved_lhs.m_directory);
    } else
      return false;
  }

  FileSpec resolved_rhs(rhs);
  if (!rhs.IsResolved()) {
    if (resolved_rhs.ResolvePath()) {
      // rhs's path wasn't resolved but now it is. Check if the resolved
      // directory is the same as rhs's unresolved directory, and if so,
      // we can mark this object as resolved to avoid more future resolves
      rhs.m_is_resolved = (rhs.m_directory == resolved_rhs.m_directory);
    } else
      return false;
  }

  // If we reach this point in the code we were able to resolve both paths
  // and since we only resolve the paths if the basenames are equal, then
  // we can just check if both directories are equal...
  return DirectoryEquals(rhs);
}

//------------------------------------------------------------------
// Not equal to operator
//------------------------------------------------------------------
bool FileSpec::operator!=(const FileSpec &rhs) const { return !(*this == rhs); }

//------------------------------------------------------------------
// Less than operator
//------------------------------------------------------------------
bool FileSpec::operator<(const FileSpec &rhs) const {
  return FileSpec::Compare(*this, rhs, true) < 0;
}

//------------------------------------------------------------------
// Dump a FileSpec object to a stream
//------------------------------------------------------------------
Stream &lldb_private::operator<<(Stream &s, const FileSpec &f) {
  f.Dump(&s);
  return s;
}

//------------------------------------------------------------------
// Clear this object by releasing both the directory and filename
// string values and making them both the empty string.
//------------------------------------------------------------------
void FileSpec::Clear() {
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
int FileSpec::Compare(const FileSpec &a, const FileSpec &b, bool full) {
  int result = 0;

  // case sensitivity of compare
  const bool case_sensitive = a.IsCaseSensitive() || b.IsCaseSensitive();

  // If full is true, then we must compare both the directory and filename.

  // If full is false, then if either directory is empty, then we match on
  // the basename only, and if both directories have valid values, we still
  // do a full compare. This allows for matching when we just have a filename
  // in one of the FileSpec objects.

  if (full || (a.m_directory && b.m_directory)) {
    result = ConstString::Compare(a.m_directory, b.m_directory, case_sensitive);
    if (result)
      return result;
  }
  return ConstString::Compare(a.m_filename, b.m_filename, case_sensitive);
}

bool FileSpec::Equal(const FileSpec &a, const FileSpec &b, bool full,
                     bool remove_backups) {
  // case sensitivity of equality test
  const bool case_sensitive = a.IsCaseSensitive() || b.IsCaseSensitive();

  if (!full && (a.GetDirectory().IsEmpty() || b.GetDirectory().IsEmpty()))
    return ConstString::Equals(a.m_filename, b.m_filename, case_sensitive);

  if (remove_backups == false)
    return a == b;

  if (a == b)
    return true;

  return Equal(a.GetNormalizedPath(), b.GetNormalizedPath(), full, false);
}

FileSpec FileSpec::GetNormalizedPath() const {
  // Fast path. Do nothing if the path is not interesting.
  if (!m_directory.GetStringRef().contains(".") &&
      !m_directory.GetStringRef().contains("//") &&
      m_filename.GetStringRef() != ".." && m_filename.GetStringRef() != ".")
    return *this;

  llvm::SmallString<64> path, result;
  const bool normalize = false;
  GetPath(path, normalize);
  llvm::StringRef rest(path);

  // We will not go below root dir.
  size_t root_dir_start = RootDirStart(path, m_syntax);
  const bool absolute = root_dir_start != llvm::StringRef::npos;
  if (absolute) {
    result += rest.take_front(root_dir_start + 1);
    rest = rest.drop_front(root_dir_start + 1);
  } else {
    if (m_syntax == ePathSyntaxWindows && path.size() > 2 && path[1] == ':') {
      result += rest.take_front(2);
      rest = rest.drop_front(2);
    }
  }

  bool anything_added = false;
  llvm::SmallVector<llvm::StringRef, 0> components, processed;
  rest.split(components, '/', -1, false);
  processed.reserve(components.size());
  for (auto component : components) {
    if (component == ".")
      continue; // Skip these.
    if (component != "..") {
      processed.push_back(component);
      continue; // Regular file name.
    }
    if (!processed.empty()) {
      processed.pop_back();
      continue; // Dots. Go one level up if we can.
    }
    if (absolute)
      continue; // We're at the top level. Cannot go higher than that. Skip.

    result += component; // We're a relative path. We need to keep these.
    result += '/';
    anything_added = true;
  }
  for (auto component : processed) {
    result += component;
    result += '/';
    anything_added = true;
  }
  if (anything_added)
    result.pop_back(); // Pop last '/'.
  else if (result.empty())
    result = ".";

  return FileSpec(result, false, m_syntax);
}

//------------------------------------------------------------------
// Dump the object to the supplied stream. If the object contains
// a valid directory name, it will be displayed followed by a
// directory delimiter, and the filename.
//------------------------------------------------------------------
void FileSpec::Dump(Stream *s) const {
  if (s) {
    std::string path{GetPath(true)};
    s->PutCString(path);
    char path_separator = GetPreferredPathSeparator(m_syntax);
    if (!m_filename && !path.empty() && path.back() != path_separator)
      s->PutChar(path_separator);
  }
}

//------------------------------------------------------------------
// Returns true if the file exists.
//------------------------------------------------------------------
bool FileSpec::Exists() const {
  struct stat file_stats;
  return GetFileStats(this, &file_stats);
}

bool FileSpec::Readable() const {
  const uint32_t permissions = GetPermissions();
  if (permissions & eFilePermissionsEveryoneR)
    return true;
  return false;
}

bool FileSpec::ResolveExecutableLocation() {
  // CLEANUP: Use StringRef for string handling.
  if (!m_directory) {
    const char *file_cstr = m_filename.GetCString();
    if (file_cstr) {
      const std::string file_str(file_cstr);
      llvm::ErrorOr<std::string> error_or_path =
          llvm::sys::findProgramByName(file_str);
      if (!error_or_path)
        return false;
      std::string path = error_or_path.get();
      llvm::StringRef dir_ref = llvm::sys::path::parent_path(path);
      if (!dir_ref.empty()) {
        // FindProgramByName returns "." if it can't find the file.
        if (strcmp(".", dir_ref.data()) == 0)
          return false;

        m_directory.SetCString(dir_ref.data());
        if (Exists())
          return true;
        else {
          // If FindProgramByName found the file, it returns the directory +
          // filename in its return results.
          // We need to separate them.
          FileSpec tmp_file(dir_ref.data(), false);
          if (tmp_file.Exists()) {
            m_directory = tmp_file.m_directory;
            return true;
          }
        }
      }
    }
  }

  return false;
}

bool FileSpec::ResolvePath() {
  if (m_is_resolved)
    return true; // We have already resolved this path

  char path_buf[PATH_MAX];
  if (!GetPath(path_buf, PATH_MAX, false))
    return false;
  // SetFile(...) will set m_is_resolved correctly if it can resolve the path
  SetFile(path_buf, true);
  return m_is_resolved;
}

uint64_t FileSpec::GetByteSize() const {
  struct stat file_stats;
  if (GetFileStats(this, &file_stats))
    return file_stats.st_size;
  return 0;
}

FileSpec::PathSyntax FileSpec::GetPathSyntax() const { return m_syntax; }

FileSpec::FileType FileSpec::GetFileType() const {
  struct stat file_stats;
  if (GetFileStats(this, &file_stats)) {
    mode_t file_type = file_stats.st_mode & S_IFMT;
    switch (file_type) {
    case S_IFDIR:
      return eFileTypeDirectory;
    case S_IFREG:
      return eFileTypeRegular;
#ifndef _WIN32
    case S_IFIFO:
      return eFileTypePipe;
    case S_IFSOCK:
      return eFileTypeSocket;
    case S_IFLNK:
      return eFileTypeSymbolicLink;
#endif
    default:
      break;
    }
    return eFileTypeUnknown;
  }
  return eFileTypeInvalid;
}

bool FileSpec::IsSymbolicLink() const {
  char resolved_path[PATH_MAX];
  if (!GetPath(resolved_path, sizeof(resolved_path)))
    return false;

#ifdef _WIN32
  std::wstring wpath;
  if (!llvm::ConvertUTF8toWide(resolved_path, wpath))
    return false;
  auto attrs = ::GetFileAttributesW(wpath.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES)
    return false;

  return (attrs & FILE_ATTRIBUTE_REPARSE_POINT);
#else
  struct stat file_stats;
  if (::lstat(resolved_path, &file_stats) != 0)
    return false;

  return (file_stats.st_mode & S_IFMT) == S_IFLNK;
#endif
}

uint32_t FileSpec::GetPermissions() const {
  uint32_t file_permissions = 0;
  if (*this)
    FileSystem::GetFilePermissions(*this, file_permissions);
  return file_permissions;
}

//------------------------------------------------------------------
// Directory string get accessor.
//------------------------------------------------------------------
ConstString &FileSpec::GetDirectory() { return m_directory; }

//------------------------------------------------------------------
// Directory string const get accessor.
//------------------------------------------------------------------
const ConstString &FileSpec::GetDirectory() const { return m_directory; }

//------------------------------------------------------------------
// Filename string get accessor.
//------------------------------------------------------------------
ConstString &FileSpec::GetFilename() { return m_filename; }

//------------------------------------------------------------------
// Filename string const get accessor.
//------------------------------------------------------------------
const ConstString &FileSpec::GetFilename() const { return m_filename; }

//------------------------------------------------------------------
// Extract the directory and path into a fixed buffer. This is
// needed as the directory and path are stored in separate string
// values.
//------------------------------------------------------------------
size_t FileSpec::GetPath(char *path, size_t path_max_len,
                         bool denormalize) const {
  if (!path)
    return 0;

  std::string result = GetPath(denormalize);
  ::snprintf(path, path_max_len, "%s", result.c_str());
  return std::min(path_max_len - 1, result.length());
}

std::string FileSpec::GetPath(bool denormalize) const {
  llvm::SmallString<64> result;
  GetPath(result, denormalize);
  return std::string(result.begin(), result.end());
}

const char *FileSpec::GetCString(bool denormalize) const {
  return ConstString{GetPath(denormalize)}.AsCString(NULL);
}

void FileSpec::GetPath(llvm::SmallVectorImpl<char> &path,
                       bool denormalize) const {
  path.append(m_directory.GetStringRef().begin(),
              m_directory.GetStringRef().end());
  if (m_directory && m_filename &&
      !IsPathSeparator(m_directory.GetStringRef().back(), m_syntax))
    path.insert(path.end(), GetPreferredPathSeparator(m_syntax));
  path.append(m_filename.GetStringRef().begin(),
              m_filename.GetStringRef().end());
  Normalize(path, m_syntax);
  if (denormalize && !path.empty())
    Denormalize(path, m_syntax);
}

ConstString FileSpec::GetFileNameExtension() const {
  if (m_filename) {
    const char *filename = m_filename.GetCString();
    const char *dot_pos = strrchr(filename, '.');
    if (dot_pos && dot_pos[1] != '\0')
      return ConstString(dot_pos + 1);
  }
  return ConstString();
}

ConstString FileSpec::GetFileNameStrippingExtension() const {
  const char *filename = m_filename.GetCString();
  if (filename == NULL)
    return ConstString();

  const char *dot_pos = strrchr(filename, '.');
  if (dot_pos == NULL)
    return m_filename;

  return ConstString(filename, dot_pos - filename);
}

//------------------------------------------------------------------
// Returns a shared pointer to a data buffer that contains all or
// part of the contents of a file. The data is memory mapped and
// will lazily page in data from the file as memory is accessed.
// The data that is mapped will start "file_offset" bytes into the
// file, and "file_size" bytes will be mapped. If "file_size" is
// greater than the number of bytes available in the file starting
// at "file_offset", the number of bytes will be appropriately
// truncated. The final number of bytes that get mapped can be
// verified using the DataBuffer::GetByteSize() function.
//------------------------------------------------------------------
DataBufferSP FileSpec::MemoryMapFileContents(off_t file_offset,
                                             size_t file_size) const {
  DataBufferSP data_sp;
  std::unique_ptr<DataBufferMemoryMap> mmap_data(new DataBufferMemoryMap());
  if (mmap_data.get()) {
    const size_t mapped_length =
        mmap_data->MemoryMapFromFileSpec(this, file_offset, file_size);
    if (((file_size == SIZE_MAX) && (mapped_length > 0)) ||
        (mapped_length >= file_size))
      data_sp.reset(mmap_data.release());
  }
  return data_sp;
}

DataBufferSP FileSpec::MemoryMapFileContentsIfLocal(off_t file_offset,
                                                    size_t file_size) const {
  if (FileSystem::IsLocal(*this))
    return MemoryMapFileContents(file_offset, file_size);
  else
    return ReadFileContents(file_offset, file_size, NULL);
}

//------------------------------------------------------------------
// Return the size in bytes that this object takes in memory. This
// returns the size in bytes of this object, not any shared string
// values it may refer to.
//------------------------------------------------------------------
size_t FileSpec::MemorySize() const {
  return m_filename.MemorySize() + m_directory.MemorySize();
}

size_t FileSpec::ReadFileContents(off_t file_offset, void *dst, size_t dst_len,
                                  Error *error_ptr) const {
  Error error;
  size_t bytes_read = 0;
  char resolved_path[PATH_MAX];
  if (GetPath(resolved_path, sizeof(resolved_path))) {
    File file;
    error = file.Open(resolved_path, File::eOpenOptionRead);
    if (error.Success()) {
      off_t file_offset_after_seek = file_offset;
      bytes_read = dst_len;
      error = file.Read(dst, bytes_read, file_offset_after_seek);
    }
  } else {
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
DataBufferSP FileSpec::ReadFileContents(off_t file_offset, size_t file_size,
                                        Error *error_ptr) const {
  Error error;
  DataBufferSP data_sp;
  char resolved_path[PATH_MAX];
  if (GetPath(resolved_path, sizeof(resolved_path))) {
    File file;
    error = file.Open(resolved_path, File::eOpenOptionRead);
    if (error.Success()) {
      const bool null_terminate = false;
      error = file.Read(file_size, file_offset, null_terminate, data_sp);
    }
  } else {
    error.SetErrorString("invalid file specification");
  }
  if (error_ptr)
    *error_ptr = error;
  return data_sp;
}

DataBufferSP FileSpec::ReadFileContentsAsCString(Error *error_ptr) {
  Error error;
  DataBufferSP data_sp;
  char resolved_path[PATH_MAX];
  if (GetPath(resolved_path, sizeof(resolved_path))) {
    File file;
    error = file.Open(resolved_path, File::eOpenOptionRead);
    if (error.Success()) {
      off_t offset = 0;
      size_t length = SIZE_MAX;
      const bool null_terminate = true;
      error = file.Read(length, offset, null_terminate, data_sp);
    }
  } else {
    error.SetErrorString("invalid file specification");
  }
  if (error_ptr)
    *error_ptr = error;
  return data_sp;
}

size_t FileSpec::ReadFileLines(STLStringArray &lines) {
  lines.clear();
  char path[PATH_MAX];
  if (GetPath(path, sizeof(path))) {
    std::ifstream file_stream(path);

    if (file_stream) {
      std::string line;
      while (getline(file_stream, line))
        lines.push_back(line);
    }
  }
  return lines.size();
}

FileSpec::EnumerateDirectoryResult
FileSpec::ForEachItemInDirectory(llvm::StringRef dir_path,
                                 DirectoryCallback const &callback) {
  if (dir_path.empty())
    return eEnumerateDirectoryResultNext;

#ifdef _WIN32
    std::string szDir(dir_path);
    szDir += "\\*";

    std::wstring wszDir;
    if (!llvm::ConvertUTF8toWide(szDir, wszDir)) {
      return eEnumerateDirectoryResultNext;
    }

    WIN32_FIND_DATAW ffd;
    HANDLE hFind = FindFirstFileW(wszDir.c_str(), &ffd);

    if (hFind == INVALID_HANDLE_VALUE) {
      return eEnumerateDirectoryResultNext;
    }

    do {
      FileSpec::FileType file_type = eFileTypeUnknown;
      if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        size_t len = wcslen(ffd.cFileName);

        if (len == 1 && ffd.cFileName[0] == L'.')
          continue;

        if (len == 2 && ffd.cFileName[0] == L'.' && ffd.cFileName[1] == L'.')
          continue;

        file_type = eFileTypeDirectory;
      } else if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DEVICE) {
        file_type = eFileTypeOther;
      } else {
        file_type = eFileTypeRegular;
      }

      std::string fileName;
      if (!llvm::convertWideToUTF8(ffd.cFileName, fileName)) {
        continue;
      }

      std::string child_path = llvm::join_items("\\", dir_path, fileName);
      // Don't resolve the file type or path
      FileSpec child_path_spec(child_path.data(), false);

      EnumerateDirectoryResult result = callback(file_type, child_path_spec);

      switch (result) {
      case eEnumerateDirectoryResultNext:
        // Enumerate next entry in the current directory. We just
        // exit this switch and will continue enumerating the
        // current directory as we currently are...
        break;

      case eEnumerateDirectoryResultEnter: // Recurse into the current entry
                                           // if it is a directory or symlink,
                                           // or next if not
        if (FileSpec::ForEachItemInDirectory(child_path.data(), callback) ==
            eEnumerateDirectoryResultQuit) {
          // The subdirectory returned Quit, which means to
          // stop all directory enumerations at all levels.
          return eEnumerateDirectoryResultQuit;
        }
        break;

      case eEnumerateDirectoryResultExit: // Exit from the current directory
                                          // at the current level.
        // Exit from this directory level and tell parent to
        // keep enumerating.
        return eEnumerateDirectoryResultNext;

      case eEnumerateDirectoryResultQuit: // Stop directory enumerations at
                                          // any level
        return eEnumerateDirectoryResultQuit;
      }
    } while (FindNextFileW(hFind, &ffd) != 0);

    FindClose(hFind);
#else
  std::string dir_string(dir_path);
  lldb_utility::CleanUp<DIR *, int> dir_path_dir(opendir(dir_string.c_str()),
                                                 NULL, closedir);
  if (dir_path_dir.is_valid()) {
    char dir_path_last_char = dir_path.back();

    long path_max = fpathconf(dirfd(dir_path_dir.get()), _PC_NAME_MAX);
#if defined(__APPLE_) && defined(__DARWIN_MAXPATHLEN)
      if (path_max < __DARWIN_MAXPATHLEN)
        path_max = __DARWIN_MAXPATHLEN;
#endif
      struct dirent *buf, *dp;
      buf = (struct dirent *)malloc(offsetof(struct dirent, d_name) + path_max +
                                    1);

      while (buf && readdir_r(dir_path_dir.get(), buf, &dp) == 0 && dp) {
        // Only search directories
        if (dp->d_type == DT_DIR || dp->d_type == DT_UNKNOWN) {
          size_t len = strlen(dp->d_name);

          if (len == 1 && dp->d_name[0] == '.')
            continue;

          if (len == 2 && dp->d_name[0] == '.' && dp->d_name[1] == '.')
            continue;
        }

        FileSpec::FileType file_type = eFileTypeUnknown;

        switch (dp->d_type) {
        default:
        case DT_UNKNOWN:
          file_type = eFileTypeUnknown;
          break;
        case DT_FIFO:
          file_type = eFileTypePipe;
          break;
        case DT_CHR:
          file_type = eFileTypeOther;
          break;
        case DT_DIR:
          file_type = eFileTypeDirectory;
          break;
        case DT_BLK:
          file_type = eFileTypeOther;
          break;
        case DT_REG:
          file_type = eFileTypeRegular;
          break;
        case DT_LNK:
          file_type = eFileTypeSymbolicLink;
          break;
        case DT_SOCK:
          file_type = eFileTypeSocket;
          break;
#if !defined(__OpenBSD__)
        case DT_WHT:
          file_type = eFileTypeOther;
          break;
#endif
        }

        std::string child_path;
        // Don't make paths with "/foo//bar", that just confuses everybody.
        if (dir_path_last_char == '/')
          child_path = llvm::join_items("", dir_path, dp->d_name);
        else
          child_path = llvm::join_items('/', dir_path, dp->d_name);

          // Don't resolve the file type or path
          FileSpec child_path_spec(child_path, false);

          EnumerateDirectoryResult result =
              callback(file_type, child_path_spec);

          switch (result) {
          case eEnumerateDirectoryResultNext:
            // Enumerate next entry in the current directory. We just
            // exit this switch and will continue enumerating the
            // current directory as we currently are...
            break;

          case eEnumerateDirectoryResultEnter: // Recurse into the current entry
                                               // if it is a directory or
                                               // symlink, or next if not
            if (FileSpec::ForEachItemInDirectory(child_path, callback) ==
                eEnumerateDirectoryResultQuit) {
              // The subdirectory returned Quit, which means to
              // stop all directory enumerations at all levels.
              if (buf)
                free(buf);
              return eEnumerateDirectoryResultQuit;
            }
            break;

          case eEnumerateDirectoryResultExit: // Exit from the current directory
                                              // at the current level.
            // Exit from this directory level and tell parent to
            // keep enumerating.
            if (buf)
              free(buf);
            return eEnumerateDirectoryResultNext;

          case eEnumerateDirectoryResultQuit: // Stop directory enumerations at
                                              // any level
            if (buf)
              free(buf);
            return eEnumerateDirectoryResultQuit;
          }
      }
      if (buf) {
        free(buf);
      }
    }
#endif
  // By default when exiting a directory, we tell the parent enumeration
  // to continue enumerating.
  return eEnumerateDirectoryResultNext;
}

FileSpec::EnumerateDirectoryResult
FileSpec::EnumerateDirectory(llvm::StringRef dir_path, bool find_directories,
                             bool find_files, bool find_other,
                             EnumerateDirectoryCallbackType callback,
                             void *callback_baton) {
  return ForEachItemInDirectory(
      dir_path,
      [&find_directories, &find_files, &find_other, &callback,
       &callback_baton](FileType file_type, const FileSpec &file_spec) {
        switch (file_type) {
        case FileType::eFileTypeDirectory:
          if (find_directories)
            return callback(callback_baton, file_type, file_spec);
          break;
        case FileType::eFileTypeRegular:
          if (find_files)
            return callback(callback_baton, file_type, file_spec);
          break;
        default:
          if (find_other)
            return callback(callback_baton, file_type, file_spec);
          break;
        }
        return eEnumerateDirectoryResultNext;
      });
}

FileSpec
FileSpec::CopyByAppendingPathComponent(llvm::StringRef component) const {
  FileSpec ret = *this;
  ret.AppendPathComponent(component);
  return ret;
}

FileSpec FileSpec::CopyByRemovingLastPathComponent() const {
  // CLEANUP: Use StringRef for string handling.
  const bool resolve = false;
  if (m_filename.IsEmpty() && m_directory.IsEmpty())
    return FileSpec("", resolve);
  if (m_directory.IsEmpty())
    return FileSpec("", resolve);
  if (m_filename.IsEmpty()) {
    const char *dir_cstr = m_directory.GetCString();
    const char *last_slash_ptr = ::strrchr(dir_cstr, '/');

    // check for obvious cases before doing the full thing
    if (!last_slash_ptr)
      return FileSpec("", resolve);
    if (last_slash_ptr == dir_cstr)
      return FileSpec("/", resolve);

    size_t last_slash_pos = last_slash_ptr - dir_cstr + 1;
    ConstString new_path(dir_cstr, last_slash_pos);
    return FileSpec(new_path.GetCString(), resolve);
  } else
    return FileSpec(m_directory.GetCString(), resolve);
}

ConstString FileSpec::GetLastPathComponent() const {
  // CLEANUP: Use StringRef for string handling.
  if (m_filename)
    return m_filename;
  if (m_directory) {
    const char *dir_cstr = m_directory.GetCString();
    const char *last_slash_ptr = ::strrchr(dir_cstr, '/');
    if (last_slash_ptr == NULL)
      return m_directory;
    if (last_slash_ptr == dir_cstr) {
      if (last_slash_ptr[1] == 0)
        return ConstString(last_slash_ptr);
      else
        return ConstString(last_slash_ptr + 1);
    }
    if (last_slash_ptr[1] != 0)
      return ConstString(last_slash_ptr + 1);
    const char *penultimate_slash_ptr = last_slash_ptr;
    while (*penultimate_slash_ptr) {
      --penultimate_slash_ptr;
      if (penultimate_slash_ptr == dir_cstr)
        break;
      if (*penultimate_slash_ptr == '/')
        break;
    }
    ConstString result(penultimate_slash_ptr + 1,
                       last_slash_ptr - penultimate_slash_ptr);
    return result;
  }
  return ConstString();
}

static std::string
join_path_components(FileSpec::PathSyntax syntax,
                     const std::vector<llvm::StringRef> components) {
  std::string result;
  for (size_t i = 0; i < components.size(); ++i) {
    if (components[i].empty())
      continue;
    result += components[i];
    if (i != components.size() - 1 &&
        !IsPathSeparator(components[i].back(), syntax))
      result += GetPreferredPathSeparator(syntax);
  }

  return result;
}

void FileSpec::PrependPathComponent(llvm::StringRef component) {
  if (component.empty())
    return;

  const bool resolve = false;
  if (m_filename.IsEmpty() && m_directory.IsEmpty()) {
    SetFile(component, resolve);
    return;
  }

  std::string result =
      join_path_components(m_syntax, {component, m_directory.GetStringRef(),
                                      m_filename.GetStringRef()});
  SetFile(result, resolve, m_syntax);
}

void FileSpec::PrependPathComponent(const FileSpec &new_path) {
  return PrependPathComponent(new_path.GetPath(false));
}

void FileSpec::AppendPathComponent(llvm::StringRef component) {
  if (component.empty())
    return;

  component = component.drop_while(
      [this](char c) { return IsPathSeparator(c, m_syntax); });

  std::string result =
      join_path_components(m_syntax, {m_directory.GetStringRef(),
                                      m_filename.GetStringRef(), component});

  SetFile(result, false, m_syntax);
}

void FileSpec::AppendPathComponent(const FileSpec &new_path) {
  return AppendPathComponent(new_path.GetPath(false));
}

void FileSpec::RemoveLastPathComponent() {
  // CLEANUP: Use StringRef for string handling.

  const bool resolve = false;
  if (m_filename.IsEmpty() && m_directory.IsEmpty()) {
    SetFile("", resolve);
    return;
  }
  if (m_directory.IsEmpty()) {
    SetFile("", resolve);
    return;
  }
  if (m_filename.IsEmpty()) {
    const char *dir_cstr = m_directory.GetCString();
    const char *last_slash_ptr = ::strrchr(dir_cstr, '/');

    // check for obvious cases before doing the full thing
    if (!last_slash_ptr) {
      SetFile("", resolve);
      return;
    }
    if (last_slash_ptr == dir_cstr) {
      SetFile("/", resolve);
      return;
    }
    size_t last_slash_pos = last_slash_ptr - dir_cstr + 1;
    ConstString new_path(dir_cstr, last_slash_pos);
    SetFile(new_path.GetCString(), resolve);
  } else
    SetFile(m_directory.GetCString(), resolve);
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
bool FileSpec::IsSourceImplementationFile() const {
  ConstString extension(GetFileNameExtension());
  if (!extension)
    return false;

  static RegularExpression g_source_file_regex(llvm::StringRef(
      "^([cC]|[mM]|[mM][mM]|[cC][pP][pP]|[cC]\\+\\+|[cC][xX][xX]|[cC][cC]|["
      "cC][pP]|[sS]|[aA][sS][mM]|[fF]|[fF]77|[fF]90|[fF]95|[fF]03|[fF][oO]["
      "rR]|[fF][tT][nN]|[fF][pP][pP]|[aA][dD][aA]|[aA][dD][bB]|[aA][dD][sS])"
      "$"));
  return g_source_file_regex.Execute(extension.GetStringRef());
}

bool FileSpec::IsRelative() const {
  const char *dir = m_directory.GetCString();
  llvm::StringRef directory(dir ? dir : "");

  if (directory.size() > 0) {
    if (PathSyntaxIsPosix(m_syntax)) {
      // If the path doesn't start with '/' or '~', return true
      switch (directory[0]) {
      case '/':
      case '~':
        return false;
      default:
        return true;
      }
    } else {
      if (directory.size() >= 2 && directory[1] == ':')
        return false;
      if (directory[0] == '/')
        return false;
      return true;
    }
  } else if (m_filename) {
    // No directory, just a basename, return true
    return true;
  }
  return false;
}

bool FileSpec::IsAbsolute() const { return !FileSpec::IsRelative(); }

void llvm::format_provider<FileSpec>::format(const FileSpec &F,
                                             raw_ostream &Stream,
                                             StringRef Style) {
  assert(
      (Style.empty() || Style.equals_lower("F") || Style.equals_lower("D")) &&
      "Invalid FileSpec style!");

  StringRef dir = F.GetDirectory().GetStringRef();
  StringRef file = F.GetFilename().GetStringRef();

  if (dir.empty() && file.empty()) {
    Stream << "(empty)";
    return;
  }

  if (Style.equals_lower("F")) {
    Stream << (file.empty() ? "(empty)" : file);
    return;
  }

  // Style is either D or empty, either way we need to print the directory.
  if (!dir.empty()) {
    // Directory is stored in normalized form, which might be different
    // than preferred form.  In order to handle this, we need to cut off
    // the filename, then denormalize, then write the entire denorm'ed
    // directory.
    llvm::SmallString<64> denormalized_dir = dir;
    Denormalize(denormalized_dir, F.GetPathSyntax());
    Stream << denormalized_dir;
    Stream << GetPreferredPathSeparator(F.GetPathSyntax());
  }

  if (Style.equals_lower("D")) {
    // We only want to print the directory, so now just exit.
    if (dir.empty())
      Stream << "(empty)";
    return;
  }

  if (!file.empty())
    Stream << file;
}
