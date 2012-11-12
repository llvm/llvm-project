//===-- PathV2.cpp - Implement OS Path Concept ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the operating system PathV2 API.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PathV2.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
#include <cstdio>
#include <cstring>

namespace {
  using llvm::StringRef;
  using llvm::sys::path::is_separator;

#ifdef LLVM_ON_WIN32
  const char *separators = "\\/";
  const char  prefered_separator = '\\';
#else
  const char  separators = '/';
  const char  prefered_separator = '/';
#endif

  StringRef find_first_component(StringRef path) {
    // Look for this first component in the following order.
    // * empty (in this case we return an empty string)
    // * either C: or {//,\\}net.
    // * {/,\}
    // * {.,..}
    // * {file,directory}name

    if (path.empty())
      return path;

#ifdef LLVM_ON_WIN32
    // C:
    if (path.size() >= 2 && std::isalpha(path[0]) && path[1] == ':')
      return path.substr(0, 2);
#endif

    // //net
    if ((path.size() > 2) &&
        is_separator(path[0]) &&
        path[0] == path[1] &&
        !is_separator(path[2])) {
      // Find the next directory separator.
      size_t end = path.find_first_of(separators, 2);
      return path.substr(0, end);
    }

    // {/,\}
    if (is_separator(path[0]))
      return path.substr(0, 1);

    if (path.startswith(".."))
      return path.substr(0, 2);

    if (path[0] == '.')
      return path.substr(0, 1);

    // * {file,directory}name
    size_t end = path.find_first_of(separators, 2);
    return path.substr(0, end);
  }

  size_t filename_pos(StringRef str) {
    if (str.size() == 2 &&
        is_separator(str[0]) &&
        str[0] == str[1])
      return 0;

    if (str.size() > 0 && is_separator(str[str.size() - 1]))
      return str.size() - 1;

    size_t pos = str.find_last_of(separators, str.size() - 1);

#ifdef LLVM_ON_WIN32
    if (pos == StringRef::npos)
      pos = str.find_last_of(':', str.size() - 2);
#endif

    if (pos == StringRef::npos ||
        (pos == 1 && is_separator(str[0])))
      return 0;

    return pos + 1;
  }

  size_t root_dir_start(StringRef str) {
    // case "c:/"
#ifdef LLVM_ON_WIN32
    if (str.size() > 2 &&
        str[1] == ':' &&
        is_separator(str[2]))
      return 2;
#endif

    // case "//"
    if (str.size() == 2 &&
        is_separator(str[0]) &&
        str[0] == str[1])
      return StringRef::npos;

    // case "//net"
    if (str.size() > 3 &&
        is_separator(str[0]) &&
        str[0] == str[1] &&
        !is_separator(str[2])) {
      return str.find_first_of(separators, 2);
    }

    // case "/"
    if (str.size() > 0 && is_separator(str[0]))
      return 0;

    return StringRef::npos;
  }

  size_t parent_path_end(StringRef path) {
    size_t end_pos = filename_pos(path);

    bool filename_was_sep = path.size() > 0 && is_separator(path[end_pos]);

    // Skip separators except for root dir.
    size_t root_dir_pos = root_dir_start(path.substr(0, end_pos));

    while(end_pos > 0 &&
          (end_pos - 1) != root_dir_pos &&
          is_separator(path[end_pos - 1]))
      --end_pos;

    if (end_pos == 1 && root_dir_pos == 0 && filename_was_sep)
      return StringRef::npos;

    return end_pos;
  }
} // end unnamed namespace

namespace llvm {
namespace sys  {
namespace path {

const_iterator begin(StringRef path) {
  const_iterator i;
  i.Path      = path;
  i.Component = find_first_component(path);
  i.Position  = 0;
  return i;
}

const_iterator end(StringRef path) {
  const_iterator i;
  i.Path      = path;
  i.Position  = path.size();
  return i;
}

const_iterator &const_iterator::operator++() {
  assert(Position < Path.size() && "Tried to increment past end!");

  // Increment Position to past the current component
  Position += Component.size();

  // Check for end.
  if (Position == Path.size()) {
    Component = StringRef();
    return *this;
  }

  // Both POSIX and Windows treat paths that begin with exactly two separators
  // specially.
  bool was_net = Component.size() > 2 &&
    is_separator(Component[0]) &&
    Component[1] == Component[0] &&
    !is_separator(Component[2]);

  // Handle separators.
  if (is_separator(Path[Position])) {
    // Root dir.
    if (was_net
#ifdef LLVM_ON_WIN32
        // c:/
        || Component.endswith(":")
#endif
        ) {
      Component = Path.substr(Position, 1);
      return *this;
    }

    // Skip extra separators.
    while (Position != Path.size() &&
           is_separator(Path[Position])) {
      ++Position;
    }

    // Treat trailing '/' as a '.'.
    if (Position == Path.size()) {
      --Position;
      Component = ".";
      return *this;
    }
  }

  // Find next component.
  size_t end_pos = Path.find_first_of(separators, Position);
  Component = Path.slice(Position, end_pos);

  return *this;
}

const_iterator &const_iterator::operator--() {
  // If we're at the end and the previous char was a '/', return '.'.
  if (Position == Path.size() &&
      Path.size() > 1 &&
      is_separator(Path[Position - 1])
#ifdef LLVM_ON_WIN32
      && Path[Position - 2] != ':'
#endif
      ) {
    --Position;
    Component = ".";
    return *this;
  }

  // Skip separators unless it's the root directory.
  size_t root_dir_pos = root_dir_start(Path);
  size_t end_pos = Position;

  while(end_pos > 0 &&
        (end_pos - 1) != root_dir_pos &&
        is_separator(Path[end_pos - 1]))
    --end_pos;

  // Find next separator.
  size_t start_pos = filename_pos(Path.substr(0, end_pos));
  Component = Path.slice(start_pos, end_pos);
  Position = start_pos;
  return *this;
}

bool const_iterator::operator==(const const_iterator &RHS) const {
  return Path.begin() == RHS.Path.begin() &&
         Position == RHS.Position;
}

bool const_iterator::operator!=(const const_iterator &RHS) const {
  return !(*this == RHS);
}

ptrdiff_t const_iterator::operator-(const const_iterator &RHS) const {
  return Position - RHS.Position;
}

const StringRef root_path(StringRef path) {
  const_iterator b = begin(path),
                 pos = b,
                 e = end(path);
  if (b != e) {
    bool has_net = b->size() > 2 && is_separator((*b)[0]) && (*b)[1] == (*b)[0];
    bool has_drive =
#ifdef LLVM_ON_WIN32
      b->endswith(":");
#else
      false;
#endif

    if (has_net || has_drive) {
      if ((++pos != e) && is_separator((*pos)[0])) {
        // {C:/,//net/}, so get the first two components.
        return path.substr(0, b->size() + pos->size());
      } else {
        // just {C:,//net}, return the first component.
        return *b;
      }
    }

    // POSIX style root directory.
    if (is_separator((*b)[0])) {
      return *b;
    }
  }

  return StringRef();
}

const StringRef root_name(StringRef path) {
  const_iterator b = begin(path),
                 e = end(path);
  if (b != e) {
    bool has_net = b->size() > 2 && is_separator((*b)[0]) && (*b)[1] == (*b)[0];
    bool has_drive =
#ifdef LLVM_ON_WIN32
      b->endswith(":");
#else
      false;
#endif

    if (has_net || has_drive) {
      // just {C:,//net}, return the first component.
      return *b;
    }
  }

  // No path or no name.
  return StringRef();
}

const StringRef root_directory(StringRef path) {
  const_iterator b = begin(path),
                 pos = b,
                 e = end(path);
  if (b != e) {
    bool has_net = b->size() > 2 && is_separator((*b)[0]) && (*b)[1] == (*b)[0];
    bool has_drive =
#ifdef LLVM_ON_WIN32
      b->endswith(":");
#else
      false;
#endif

    if ((has_net || has_drive) &&
        // {C:,//net}, skip to the next component.
        (++pos != e) && is_separator((*pos)[0])) {
      return *pos;
    }

    // POSIX style root directory.
    if (!has_net && is_separator((*b)[0])) {
      return *b;
    }
  }

  // No path or no root.
  return StringRef();
}

const StringRef relative_path(StringRef path) {
  StringRef root = root_path(path);
  return path.substr(root.size());
}

void append(SmallVectorImpl<char> &path, const Twine &a,
                                         const Twine &b,
                                         const Twine &c,
                                         const Twine &d) {
  SmallString<32> a_storage;
  SmallString<32> b_storage;
  SmallString<32> c_storage;
  SmallString<32> d_storage;

  SmallVector<StringRef, 4> components;
  if (!a.isTriviallyEmpty()) components.push_back(a.toStringRef(a_storage));
  if (!b.isTriviallyEmpty()) components.push_back(b.toStringRef(b_storage));
  if (!c.isTriviallyEmpty()) components.push_back(c.toStringRef(c_storage));
  if (!d.isTriviallyEmpty()) components.push_back(d.toStringRef(d_storage));

  for (SmallVectorImpl<StringRef>::const_iterator i = components.begin(),
                                                  e = components.end();
                                                  i != e; ++i) {
    bool path_has_sep = !path.empty() && is_separator(path[path.size() - 1]);
    bool component_has_sep = !i->empty() && is_separator((*i)[0]);
    bool is_root_name = has_root_name(*i);

    if (path_has_sep) {
      // Strip separators from beginning of component.
      size_t loc = i->find_first_not_of(separators);
      StringRef c = i->substr(loc);

      // Append it.
      path.append(c.begin(), c.end());
      continue;
    }

    if (!component_has_sep && !(path.empty() || is_root_name)) {
      // Add a separator.
      path.push_back(prefered_separator);
    }

    path.append(i->begin(), i->end());
  }
}

void append(SmallVectorImpl<char> &path,
            const_iterator begin, const_iterator end) {
  for (; begin != end; ++begin)
    path::append(path, *begin);
}

const StringRef parent_path(StringRef path) {
  size_t end_pos = parent_path_end(path);
  if (end_pos == StringRef::npos)
    return StringRef();
  else
    return path.substr(0, end_pos);
}

void remove_filename(SmallVectorImpl<char> &path) {
  size_t end_pos = parent_path_end(StringRef(path.begin(), path.size()));
  if (end_pos != StringRef::npos)
    path.set_size(end_pos);
}

void replace_extension(SmallVectorImpl<char> &path, const Twine &extension) {
  StringRef p(path.begin(), path.size());
  SmallString<32> ext_storage;
  StringRef ext = extension.toStringRef(ext_storage);

  // Erase existing extension.
  size_t pos = p.find_last_of('.');
  if (pos != StringRef::npos && pos >= filename_pos(p))
    path.set_size(pos);

  // Append '.' if needed.
  if (ext.size() > 0 && ext[0] != '.')
    path.push_back('.');

  // Append extension.
  path.append(ext.begin(), ext.end());
}

void native(const Twine &path, SmallVectorImpl<char> &result) {
  // Clear result.
  result.clear();
#ifdef LLVM_ON_WIN32
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);
  result.reserve(p.size());
  for (StringRef::const_iterator i = p.begin(),
                                 e = p.end();
                                 i != e;
                                 ++i) {
    if (*i == '/')
      result.push_back('\\');
    else
      result.push_back(*i);
  }
#else
  path.toVector(result);
#endif
}

const StringRef filename(StringRef path) {
  return *(--end(path));
}

const StringRef stem(StringRef path) {
  StringRef fname = filename(path);
  size_t pos = fname.find_last_of('.');
  if (pos == StringRef::npos)
    return fname;
  else
    if ((fname.size() == 1 && fname == ".") ||
        (fname.size() == 2 && fname == ".."))
      return fname;
    else
      return fname.substr(0, pos);
}

const StringRef extension(StringRef path) {
  StringRef fname = filename(path);
  size_t pos = fname.find_last_of('.');
  if (pos == StringRef::npos)
    return StringRef();
  else
    if ((fname.size() == 1 && fname == ".") ||
        (fname.size() == 2 && fname == ".."))
      return StringRef();
    else
      return fname.substr(pos);
}

bool is_separator(char value) {
  switch(value) {
#ifdef LLVM_ON_WIN32
    case '\\': // fall through
#endif
    case '/': return true;
    default: return false;
  }
}

void system_temp_directory(bool erasedOnReboot, SmallVectorImpl<char> &result) {
  result.clear();

  // Check whether the temporary directory is specified by an environment
  // variable.
  const char *EnvironmentVariable;
#ifdef LLVM_ON_WIN32
  EnvironmentVariable = "TEMP";
#else
  EnvironmentVariable = "TMPDIR";
#endif
  if (char *RequestedDir = getenv(EnvironmentVariable)) {
    result.append(RequestedDir, RequestedDir + strlen(RequestedDir));
    return;
  }

  // Fall back to a system default.
  const char *DefaultResult;
#ifdef LLVM_ON_WIN32
  (void)erasedOnReboot;
  DefaultResult = "C:\\TEMP";
#else
  if (erasedOnReboot)
    DefaultResult = "/tmp";
  else
    DefaultResult = "/var/tmp";
#endif
  result.append(DefaultResult, DefaultResult + strlen(DefaultResult));
}

bool has_root_name(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !root_name(p).empty();
}

bool has_root_directory(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !root_directory(p).empty();
}

bool has_root_path(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !root_path(p).empty();
}

bool has_relative_path(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !relative_path(p).empty();
}

bool has_filename(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !filename(p).empty();
}

bool has_parent_path(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !parent_path(p).empty();
}

bool has_stem(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !stem(p).empty();
}

bool has_extension(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  return !extension(p).empty();
}

bool is_absolute(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  bool rootDir = has_root_directory(p),
#ifdef LLVM_ON_WIN32
       rootName = has_root_name(p);
#else
       rootName = true;
#endif

  return rootDir && rootName;
}

bool is_relative(const Twine &path) {
  return !is_absolute(path);
}

} // end namespace path

namespace fs {

error_code make_absolute(SmallVectorImpl<char> &path) {
  StringRef p(path.data(), path.size());

  bool rootDirectory = path::has_root_directory(p),
#ifdef LLVM_ON_WIN32
       rootName = path::has_root_name(p);
#else
       rootName = true;
#endif

  // Already absolute.
  if (rootName && rootDirectory)
    return error_code::success();

  // All of the following conditions will need the current directory.
  SmallString<128> current_dir;
  if (error_code ec = current_path(current_dir)) return ec;

  // Relative path. Prepend the current directory.
  if (!rootName && !rootDirectory) {
    // Append path to the current directory.
    path::append(current_dir, p);
    // Set path to the result.
    path.swap(current_dir);
    return error_code::success();
  }

  if (!rootName && rootDirectory) {
    StringRef cdrn = path::root_name(current_dir);
    SmallString<128> curDirRootName(cdrn.begin(), cdrn.end());
    path::append(curDirRootName, p);
    // Set path to the result.
    path.swap(curDirRootName);
    return error_code::success();
  }

  if (rootName && !rootDirectory) {
    StringRef pRootName      = path::root_name(p);
    StringRef bRootDirectory = path::root_directory(current_dir);
    StringRef bRelativePath  = path::relative_path(current_dir);
    StringRef pRelativePath  = path::relative_path(p);

    SmallString<128> res;
    path::append(res, pRootName, bRootDirectory, bRelativePath, pRelativePath);
    path.swap(res);
    return error_code::success();
  }

  llvm_unreachable("All rootName and rootDirectory combinations should have "
                   "occurred above!");
}

error_code create_directories(const Twine &path, bool &existed) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  StringRef parent = path::parent_path(p);
  if (!parent.empty()) {
    bool parent_exists;
    if (error_code ec = fs::exists(parent, parent_exists)) return ec;

    if (!parent_exists)
      if (error_code ec = create_directories(parent, existed)) return ec;
  }

  return create_directory(p, existed);
}

bool exists(file_status status) {
  return status_known(status) && status.type() != file_type::file_not_found;
}

bool status_known(file_status s) {
  return s.type() != file_type::status_error;
}

bool is_directory(file_status status) {
  return status.type() == file_type::directory_file;
}

error_code is_directory(const Twine &path, bool &result) {
  file_status st;
  if (error_code ec = status(path, st))
    return ec;
  result = is_directory(st);
  return error_code::success();
}

bool is_regular_file(file_status status) {
  return status.type() == file_type::regular_file;
}

error_code is_regular_file(const Twine &path, bool &result) {
  file_status st;
  if (error_code ec = status(path, st))
    return ec;
  result = is_regular_file(st);
  return error_code::success();
}

bool is_symlink(file_status status) {
  return status.type() == file_type::symlink_file;
}

error_code is_symlink(const Twine &path, bool &result) {
  file_status st;
  if (error_code ec = status(path, st))
    return ec;
  result = is_symlink(st);
  return error_code::success();
}

bool is_other(file_status status) {
  return exists(status) &&
         !is_regular_file(status) &&
         !is_directory(status) &&
         !is_symlink(status);
}

void directory_entry::replace_filename(const Twine &filename, file_status st) {
  SmallString<128> path(Path.begin(), Path.end());
  path::remove_filename(path);
  path::append(path, filename);
  Path = path.str();
  Status = st;
}

error_code has_magic(const Twine &path, const Twine &magic, bool &result) {
  SmallString<32>  MagicStorage;
  StringRef Magic = magic.toStringRef(MagicStorage);
  SmallString<32> Buffer;

  if (error_code ec = get_magic(path, Magic.size(), Buffer)) {
    if (ec == errc::value_too_large) {
      // Magic.size() > file_size(Path).
      result = false;
      return error_code::success();
    }
    return ec;
  }

  result = Magic == Buffer;
  return error_code::success();
}

/// @brief Identify the magic in magic.
file_magic identify_magic(StringRef magic) {
  if (magic.size() < 4)
    return file_magic::unknown;
  switch ((unsigned char)magic[0]) {
    case 0xDE:  // 0x0B17C0DE = BC wraper
      if (magic[1] == (char)0xC0 && magic[2] == (char)0x17 &&
          magic[3] == (char)0x0B)
        return file_magic::bitcode;
      break;
    case 'B':
      if (magic[1] == 'C' && magic[2] == (char)0xC0 && magic[3] == (char)0xDE)
        return file_magic::bitcode;
      break;
    case '!':
      if (magic.size() >= 8)
        if (memcmp(magic.data(),"!<arch>\n",8) == 0)
          return file_magic::archive;
      break;

    case '\177':
      if (magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F') {
        if (magic.size() >= 18 && magic[17] == 0)
          switch (magic[16]) {
            default: break;
            case 1: return file_magic::elf_relocatable;
            case 2: return file_magic::elf_executable;
            case 3: return file_magic::elf_shared_object;
            case 4: return file_magic::elf_core;
          }
      }
      break;

    case 0xCA:
      if (magic[1] == char(0xFE) && magic[2] == char(0xBA) &&
          magic[3] == char(0xBE)) {
        // This is complicated by an overlap with Java class files.
        // See the Mach-O section in /usr/share/file/magic for details.
        if (magic.size() >= 8 && magic[7] < 43)
          // FIXME: Universal Binary of any type.
          return file_magic::macho_dynamically_linked_shared_lib;
      }
      break;

      // The two magic numbers for mach-o are:
      // 0xfeedface - 32-bit mach-o
      // 0xfeedfacf - 64-bit mach-o
    case 0xFE:
    case 0xCE:
    case 0xCF: {
      uint16_t type = 0;
      if (magic[0] == char(0xFE) && magic[1] == char(0xED) &&
          magic[2] == char(0xFA) &&
          (magic[3] == char(0xCE) || magic[3] == char(0xCF))) {
        /* Native endian */
        if (magic.size() >= 16) type = magic[14] << 8 | magic[15];
      } else if ((magic[0] == char(0xCE) || magic[0] == char(0xCF)) &&
                 magic[1] == char(0xFA) && magic[2] == char(0xED) &&
                 magic[3] == char(0xFE)) {
        /* Reverse endian */
        if (magic.size() >= 14) type = magic[13] << 8 | magic[12];
      }
      switch (type) {
        default: break;
        case 1: return file_magic::macho_object;
        case 2: return file_magic::macho_executable;
        case 3: return file_magic::macho_fixed_virtual_memory_shared_lib;
        case 4: return file_magic::macho_core;
        case 5: return file_magic::macho_preload_executabl;
        case 6: return file_magic::macho_dynamically_linked_shared_lib;
        case 7: return file_magic::macho_dynamic_linker;
        case 8: return file_magic::macho_bundle;
        case 9: return file_magic::macho_dynamic_linker;
        case 10: return file_magic::macho_dsym_companion;
      }
      break;
    }
    case 0xF0: // PowerPC Windows
    case 0x83: // Alpha 32-bit
    case 0x84: // Alpha 64-bit
    case 0x66: // MPS R4000 Windows
    case 0x50: // mc68K
    case 0x4c: // 80386 Windows
      if (magic[1] == 0x01)
        return file_magic::coff_object;

    case 0x90: // PA-RISC Windows
    case 0x68: // mc68K Windows
      if (magic[1] == 0x02)
        return file_magic::coff_object;
      break;

    case 0x4d: // Possible MS-DOS stub on Windows PE file
      if (magic[1] == 0x5a) {
        uint32_t off =
          *reinterpret_cast<const support::ulittle32_t*>(magic.data() + 0x3c);
        // PE/COFF file, either EXE or DLL.
        if (off < magic.size() && memcmp(magic.data() + off, "PE\0\0",4) == 0)
          return file_magic::pecoff_executable;
      }
      break;

    case 0x64: // x86-64 Windows.
      if (magic[1] == char(0x86))
        return file_magic::coff_object;
      break;

    default:
      break;
  }
  return file_magic::unknown;
}

error_code identify_magic(const Twine &path, file_magic &result) {
  SmallString<32> Magic;
  error_code ec = get_magic(path, Magic.capacity(), Magic);
  if (ec && ec != errc::value_too_large)
    return ec;

  result = identify_magic(Magic);
  return error_code::success();
}

namespace {
error_code remove_all_r(StringRef path, file_type ft, uint32_t &count) {
  if (ft == file_type::directory_file) {
    // This code would be a lot better with exceptions ;/.
    error_code ec;
    directory_iterator i(path, ec);
    if (ec) return ec;
    for (directory_iterator e; i != e; i.increment(ec)) {
      if (ec) return ec;
      file_status st;
      if (error_code ec = i->status(st)) return ec;
      if (error_code ec = remove_all_r(i->path(), st.type(), count)) return ec;
    }
    bool obviously_this_exists;
    if (error_code ec = remove(path, obviously_this_exists)) return ec;
    assert(obviously_this_exists);
    ++count; // Include the directory itself in the items removed.
  } else {
    bool obviously_this_exists;
    if (error_code ec = remove(path, obviously_this_exists)) return ec;
    assert(obviously_this_exists);
    ++count;
  }

  return error_code::success();
}
} // end unnamed namespace

error_code remove_all(const Twine &path, uint32_t &num_removed) {
  SmallString<128> path_storage;
  StringRef p = path.toStringRef(path_storage);

  file_status fs;
  if (error_code ec = status(path, fs))
    return ec;
  num_removed = 0;
  return remove_all_r(p, fs.type(), num_removed);
}

error_code directory_entry::status(file_status &result) const {
  return fs::status(Path, result);
}

} // end namespace fs
} // end namespace sys
} // end namespace llvm

// Include the truly platform-specific parts.
#if defined(LLVM_ON_UNIX)
#include "Unix/PathV2.inc"
#endif
#if defined(LLVM_ON_WIN32)
#include "Windows/PathV2.inc"
#endif
