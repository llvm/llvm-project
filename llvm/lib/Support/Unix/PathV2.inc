//===- llvm/Support/Unix/PathV2.cpp - Unix Path Implementation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific implementation of the PathV2 API.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/Support/Process.h"
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif
#if HAVE_DIRENT_H
# include <dirent.h>
# define NAMLEN(dirent) strlen((dirent)->d_name)
#else
# define dirent direct
# define NAMLEN(dirent) (dirent)->d_namlen
# if HAVE_SYS_NDIR_H
#  include <sys/ndir.h>
# endif
# if HAVE_SYS_DIR_H
#  include <sys/dir.h>
# endif
# if HAVE_NDIR_H
#  include <ndir.h>
# endif
#endif
#if HAVE_STDIO_H
#include <stdio.h>
#endif
#if HAVE_LIMITS_H
#include <limits.h>
#endif

// Both stdio.h and cstdio are included via different pathes and
// stdcxx's cstdio doesn't include stdio.h, so it doesn't #undef the macros
// either.
#undef ferror
#undef feof

// For GNU Hurd
#if defined(__GNU__) && !defined(PATH_MAX)
# define PATH_MAX 4096
#endif

using namespace llvm;

namespace {
  /// This class automatically closes the given file descriptor when it goes out
  /// of scope. You can take back explicit ownership of the file descriptor by
  /// calling take(). The destructor does not verify that close was successful.
  /// Therefore, never allow this class to call close on a file descriptor that
  /// has been read from or written to.
  struct AutoFD {
    int FileDescriptor;

    AutoFD(int fd) : FileDescriptor(fd) {}
    ~AutoFD() {
      if (FileDescriptor >= 0)
        ::close(FileDescriptor);
    }

    int take() {
      int ret = FileDescriptor;
      FileDescriptor = -1;
      return ret;
    }

    operator int() const {return FileDescriptor;}
  };

  error_code TempDir(SmallVectorImpl<char> &result) {
    // FIXME: Don't use TMPDIR if program is SUID or SGID enabled.
    const char *dir = 0;
    (dir = std::getenv("TMPDIR" )) ||
    (dir = std::getenv("TMP"    )) ||
    (dir = std::getenv("TEMP"   )) ||
    (dir = std::getenv("TEMPDIR")) ||
#ifdef P_tmpdir
    (dir = P_tmpdir) ||
#endif
    (dir = "/tmp");

    result.clear();
    StringRef d(dir);
    result.append(d.begin(), d.end());
    return error_code::success();
  }
}

namespace llvm {
namespace sys  {
namespace fs {

error_code current_path(SmallVectorImpl<char> &result) {
#ifdef MAXPATHLEN
  result.reserve(MAXPATHLEN);
#else
// For GNU Hurd
  result.reserve(1024);
#endif

  while (true) {
    if (::getcwd(result.data(), result.capacity()) == 0) {
      // See if there was a real error.
      if (errno != errc::not_enough_memory)
        return error_code(errno, system_category());
      // Otherwise there just wasn't enough space.
      result.reserve(result.capacity() * 2);
    } else
      break;
  }

  result.set_size(strlen(result.data()));
  return error_code::success();
}

error_code copy_file(const Twine &from, const Twine &to, copy_option copt) {
 // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  const size_t buf_sz = 32768;
  char buffer[buf_sz];
  int from_file = -1, to_file = -1;

  // Open from.
  if ((from_file = ::open(f.begin(), O_RDONLY)) < 0)
    return error_code(errno, system_category());
  AutoFD from_fd(from_file);

  // Stat from.
  struct stat from_stat;
  if (::stat(f.begin(), &from_stat) != 0)
    return error_code(errno, system_category());

  // Setup to flags.
  int to_flags = O_CREAT | O_WRONLY;
  if (copt == copy_option::fail_if_exists)
    to_flags |= O_EXCL;

  // Open to.
  if ((to_file = ::open(t.begin(), to_flags, from_stat.st_mode)) < 0)
    return error_code(errno, system_category());
  AutoFD to_fd(to_file);

  // Copy!
  ssize_t sz, sz_read = 1, sz_write;
  while (sz_read > 0 &&
         (sz_read = ::read(from_fd, buffer, buf_sz)) > 0) {
    // Allow for partial writes - see Advanced Unix Programming (2nd Ed.),
    // Marc Rochkind, Addison-Wesley, 2004, page 94
    sz_write = 0;
    do {
      if ((sz = ::write(to_fd, buffer + sz_write, sz_read - sz_write)) < 0) {
        sz_read = sz;  // cause read loop termination.
        break;         // error.
      }
      sz_write += sz;
    } while (sz_write < sz_read);
  }

  // After all the file operations above the return value of close actually
  // matters.
  if (::close(from_fd.take()) < 0) sz_read = -1;
  if (::close(to_fd.take()) < 0) sz_read = -1;

  // Check for errors.
  if (sz_read < 0)
    return error_code(errno, system_category());

  return error_code::success();
}

error_code create_directory(const Twine &path, bool &existed) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::mkdir(p.begin(), S_IRWXU | S_IRWXG) == -1) {
    if (errno != errc::file_exists)
      return error_code(errno, system_category());
    existed = true;
  } else
    existed = false;

  return error_code::success();
}

error_code create_hard_link(const Twine &to, const Twine &from) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::link(t.begin(), f.begin()) == -1)
    return error_code(errno, system_category());

  return error_code::success();
}

error_code create_symlink(const Twine &to, const Twine &from) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::symlink(t.begin(), f.begin()) == -1)
    return error_code(errno, system_category());

  return error_code::success();
}

error_code remove(const Twine &path, bool &existed) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::remove(p.begin()) == -1) {
    if (errno != errc::no_such_file_or_directory)
      return error_code(errno, system_category());
    existed = false;
  } else
    existed = true;

  return error_code::success();
}

error_code rename(const Twine &from, const Twine &to) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::rename(f.begin(), t.begin()) == -1) {
    // If it's a cross device link, copy then delete, otherwise return the error
    if (errno == EXDEV) {
      if (error_code ec = copy_file(from, to, copy_option::overwrite_if_exists))
        return ec;
      bool Existed;
      if (error_code ec = remove(from, Existed))
        return ec;
    } else
      return error_code(errno, system_category());
  }

  return error_code::success();
}

error_code resize_file(const Twine &path, uint64_t size) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::truncate(p.begin(), size) == -1)
    return error_code(errno, system_category());

  return error_code::success();
}

error_code exists(const Twine &path, bool &result) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::access(p.begin(), F_OK) == -1) {
    if (errno != errc::no_such_file_or_directory)
      return error_code(errno, system_category());
    result = false;
  } else
    result = true;

  return error_code::success();
}

bool equivalent(file_status A, file_status B) {
  assert(status_known(A) && status_known(B));
  return A.fs_st_dev == B.fs_st_dev &&
         A.fs_st_ino == B.fs_st_ino;
}

error_code equivalent(const Twine &A, const Twine &B, bool &result) {
  file_status fsA, fsB;
  if (error_code ec = status(A, fsA)) return ec;
  if (error_code ec = status(B, fsB)) return ec;
  result = equivalent(fsA, fsB);
  return error_code::success();
}

error_code file_size(const Twine &path, uint64_t &result) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  struct stat status;
  if (::stat(p.begin(), &status) == -1)
    return error_code(errno, system_category());
  if (!S_ISREG(status.st_mode))
    return make_error_code(errc::operation_not_permitted);

  result = status.st_size;
  return error_code::success();
}

error_code status(const Twine &path, file_status &result) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  struct stat status;
  if (::stat(p.begin(), &status) != 0) {
    error_code ec(errno, system_category());
    if (ec == errc::no_such_file_or_directory)
      result = file_status(file_type::file_not_found);
    else
      result = file_status(file_type::status_error);
    return ec;
  }

  perms prms = static_cast<perms>(status.st_mode & perms_mask);
  
  if (S_ISDIR(status.st_mode))
    result = file_status(file_type::directory_file, prms);
  else if (S_ISREG(status.st_mode))
    result = file_status(file_type::regular_file, prms);
  else if (S_ISBLK(status.st_mode))
    result = file_status(file_type::block_file, prms);
  else if (S_ISCHR(status.st_mode))
    result = file_status(file_type::character_file, prms);
  else if (S_ISFIFO(status.st_mode))
    result = file_status(file_type::fifo_file, prms);
  else if (S_ISSOCK(status.st_mode))
    result = file_status(file_type::socket_file, prms);
  else
    result = file_status(file_type::type_unknown, prms);

  result.fs_st_dev = status.st_dev;
  result.fs_st_ino = status.st_ino;

  return error_code::success();
}

// Modifies permissions on a file.
error_code permissions(const Twine &path, perms prms) {
  if ((prms & add_perms) && (prms & remove_perms))
    llvm_unreachable("add_perms and remove_perms are mutually exclusive");

  // Get current permissions
  file_status info;
  if (error_code ec = status(path, info)) {
    return ec;
  }
  
  // Set updated permissions.
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);
  perms permsToSet;
  if (prms & add_perms) {
    permsToSet = (info.permissions() | prms) & perms_mask;
  } else if (prms & remove_perms) {
    permsToSet = (info.permissions() & ~prms) & perms_mask;
  } else {
    permsToSet = prms & perms_mask;
  }
  if (::chmod(p.begin(), static_cast<mode_t>(permsToSet))) {
    return error_code(errno, system_category()); 
  }

  return error_code::success();
}

// Since this is most often used for temporary files, mode defaults to 0600.
error_code unique_file(const Twine &model, int &result_fd,
                       SmallVectorImpl<char> &result_path,
                       bool makeAbsolute, unsigned mode) {
  SmallString<128> Model;
  model.toVector(Model);
  // Null terminate.
  Model.c_str();

  if (makeAbsolute) {
    // Make model absolute by prepending a temp directory if it's not already.
    bool absolute = path::is_absolute(Twine(Model));
    if (!absolute) {
      SmallString<128> TDir;
      if (error_code ec = TempDir(TDir)) return ec;
      path::append(TDir, Twine(Model));
      Model.swap(TDir);
    }
  }

  // From here on, DO NOT modify model. It may be needed if the randomly chosen
  // path already exists.
  SmallString<128> RandomPath = Model;

retry_random_path:
  // Replace '%' with random chars.
  for (unsigned i = 0, e = Model.size(); i != e; ++i) {
    if (Model[i] == '%')
      RandomPath[i] = "0123456789abcdef"[sys::Process::GetRandomNumber() & 15];
  }

  // Try to open + create the file.
rety_open_create:
  int RandomFD = ::open(RandomPath.c_str(), O_RDWR | O_CREAT | O_EXCL, mode);
  if (RandomFD == -1) {
    // If the file existed, try again, otherwise, error.
    if (errno == errc::file_exists)
      goto retry_random_path;
    // The path prefix doesn't exist.
    if (errno == errc::no_such_file_or_directory) {
      StringRef p(RandomPath.begin(), RandomPath.size());
      SmallString<64> dir_to_create;
      for (path::const_iterator i = path::begin(p),
                                e = --path::end(p); i != e; ++i) {
        path::append(dir_to_create, *i);
        bool Exists;
        if (error_code ec = exists(Twine(dir_to_create), Exists)) return ec;
        if (!Exists) {
          // Don't try to create network paths.
          if (i->size() > 2 && (*i)[0] == '/' &&
                               (*i)[1] == '/' &&
                               (*i)[2] != '/')
            return make_error_code(errc::no_such_file_or_directory);
          if (::mkdir(dir_to_create.c_str(), 0700) == -1)
            return error_code(errno, system_category());
        }
      }
      goto rety_open_create;
    }
    return error_code(errno, system_category());
  }

   // Make the path absolute.
  char real_path_buff[PATH_MAX + 1];
  if (realpath(RandomPath.c_str(), real_path_buff) == NULL) {
    int error = errno;
    ::close(RandomFD);
    ::unlink(RandomPath.c_str());
    return error_code(error, system_category());
  }

  result_path.clear();
  StringRef d(real_path_buff);
  result_path.append(d.begin(), d.end());

  result_fd = RandomFD;
  return error_code::success();
}

error_code mapped_file_region::init(int fd, uint64_t offset) {
  AutoFD FD(fd);

  // Figure out how large the file is.
  struct stat FileInfo;
  if (fstat(fd, &FileInfo) == -1)
    return error_code(errno, system_category());
  uint64_t FileSize = FileInfo.st_size;

  if (Size == 0)
    Size = FileSize;
  else if (FileSize < Size) {
    // We need to grow the file.
    if (ftruncate(fd, Size) == -1)
      return error_code(errno, system_category());
  }

  int flags = (Mode == readwrite) ? MAP_SHARED : MAP_PRIVATE;
  int prot = (Mode == readonly) ? PROT_READ : (PROT_READ | PROT_WRITE);
#ifdef MAP_FILE
  flags |= MAP_FILE;
#endif
  Mapping = ::mmap(0, Size, prot, flags, fd, offset);
  if (Mapping == MAP_FAILED)
    return error_code(errno, system_category());
  return error_code::success();
}

mapped_file_region::mapped_file_region(const Twine &path,
                                       mapmode mode,
                                       uint64_t length,
                                       uint64_t offset,
                                       error_code &ec)
  : Mode(mode)
  , Size(length)
  , Mapping() {
  // Make sure that the requested size fits within SIZE_T.
  if (length > std::numeric_limits<size_t>::max()) {
    ec = make_error_code(errc::invalid_argument);
    return;
  }

  SmallString<128> path_storage;
  StringRef name = path.toNullTerminatedStringRef(path_storage);
  int oflags = (mode == readonly) ? O_RDONLY : O_RDWR;
  int ofd = ::open(name.begin(), oflags);
  if (ofd == -1) {
    ec = error_code(errno, system_category());
    return;
  }

  ec = init(ofd, offset);
  if (ec)
    Mapping = 0;
}

mapped_file_region::mapped_file_region(int fd,
                                       mapmode mode,
                                       uint64_t length,
                                       uint64_t offset,
                                       error_code &ec)
  : Mode(mode)
  , Size(length)
  , Mapping() {
  // Make sure that the requested size fits within SIZE_T.
  if (length > std::numeric_limits<size_t>::max()) {
    ec = make_error_code(errc::invalid_argument);
    return;
  }

  ec = init(fd, offset);
  if (ec)
    Mapping = 0;
}

mapped_file_region::~mapped_file_region() {
  if (Mapping)
    ::munmap(Mapping, Size);
}

#if LLVM_USE_RVALUE_REFERENCES
mapped_file_region::mapped_file_region(mapped_file_region &&other)
  : Mode(other.Mode), Size(other.Size), Mapping(other.Mapping) {
  other.Mapping = 0;
}
#endif

mapped_file_region::mapmode mapped_file_region::flags() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return Mode;
}

uint64_t mapped_file_region::size() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return Size;
}

char *mapped_file_region::data() const {
  assert(Mapping && "Mapping failed but used anyway!");
  assert(Mode != readonly && "Cannot get non const data for readonly mapping!");
  return reinterpret_cast<char*>(Mapping);
}

const char *mapped_file_region::const_data() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return reinterpret_cast<const char*>(Mapping);
}

int mapped_file_region::alignment() {
  return Process::GetPageSize();
}

error_code detail::directory_iterator_construct(detail::DirIterState &it,
                                                StringRef path){
  SmallString<128> path_null(path);
  DIR *directory = ::opendir(path_null.c_str());
  if (directory == 0)
    return error_code(errno, system_category());

  it.IterationHandle = reinterpret_cast<intptr_t>(directory);
  // Add something for replace_filename to replace.
  path::append(path_null, ".");
  it.CurrentEntry = directory_entry(path_null.str());
  return directory_iterator_increment(it);
}

error_code detail::directory_iterator_destruct(detail::DirIterState &it) {
  if (it.IterationHandle)
    ::closedir(reinterpret_cast<DIR *>(it.IterationHandle));
  it.IterationHandle = 0;
  it.CurrentEntry = directory_entry();
  return error_code::success();
}

error_code detail::directory_iterator_increment(detail::DirIterState &it) {
  errno = 0;
  dirent *cur_dir = ::readdir(reinterpret_cast<DIR *>(it.IterationHandle));
  if (cur_dir == 0 && errno != 0) {
    return error_code(errno, system_category());
  } else if (cur_dir != 0) {
    StringRef name(cur_dir->d_name, NAMLEN(cur_dir));
    if ((name.size() == 1 && name[0] == '.') ||
        (name.size() == 2 && name[0] == '.' && name[1] == '.'))
      return directory_iterator_increment(it);
    it.CurrentEntry.replace_filename(name);
  } else
    return directory_iterator_destruct(it);

  return error_code::success();
}

error_code get_magic(const Twine &path, uint32_t len,
                     SmallVectorImpl<char> &result) {
  SmallString<128> PathStorage;
  StringRef Path = path.toNullTerminatedStringRef(PathStorage);
  result.set_size(0);

  // Open path.
  std::FILE *file = std::fopen(Path.data(), "rb");
  if (file == 0)
    return error_code(errno, system_category());

  // Reserve storage.
  result.reserve(len);

  // Read magic!
  size_t size = std::fread(result.data(), 1, len, file);
  if (std::ferror(file) != 0) {
    std::fclose(file);
    return error_code(errno, system_category());
  } else if (size != result.size()) {
    if (std::feof(file) != 0) {
      std::fclose(file);
      result.set_size(size);
      return make_error_code(errc::value_too_large);
    }
  }
  std::fclose(file);
  result.set_size(len);
  return error_code::success();
}

error_code map_file_pages(const Twine &path, off_t file_offset, size_t size,  
                                            bool map_writable, void *&result) {
  SmallString<128> path_storage;
  StringRef name = path.toNullTerminatedStringRef(path_storage);
  int oflags = map_writable ? O_RDWR : O_RDONLY;
  int ofd = ::open(name.begin(), oflags);
  if ( ofd == -1 )
    return error_code(errno, system_category());
  AutoFD fd(ofd);
  int flags = map_writable ? MAP_SHARED : MAP_PRIVATE;
  int prot = map_writable ? (PROT_READ|PROT_WRITE) : PROT_READ;
#ifdef MAP_FILE
  flags |= MAP_FILE;
#endif
  result = ::mmap(0, size, prot, flags, fd, file_offset);
  if (result == MAP_FAILED) {
    return error_code(errno, system_category());
  }
  
  return error_code::success();
}

error_code unmap_file_pages(void *base, size_t size) {
  if ( ::munmap(base, size) == -1 )
    return error_code(errno, system_category());
   
  return error_code::success();
}


} // end namespace fs
} // end namespace sys
} // end namespace llvm
