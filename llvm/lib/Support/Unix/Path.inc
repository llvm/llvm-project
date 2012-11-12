//===- llvm/Support/Unix/Path.cpp - Unix Path Implementation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific portion of the Path class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_UTIME_H
#include <utime.h>
#endif
#if HAVE_TIME_H
#include <time.h>
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

#if HAVE_DLFCN_H
#include <dlfcn.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

// For GNU Hurd
#if defined(__GNU__) && !defined(MAXPATHLEN)
# define MAXPATHLEN 4096
#endif

// Put in a hack for Cygwin which falsely reports that the mkdtemp function
// is available when it is not.
#ifdef __CYGWIN__
# undef HAVE_MKDTEMP
#endif

namespace {
inline bool lastIsSlash(const std::string& path) {
  return !path.empty() && path[path.length() - 1] == '/';
}

}

namespace llvm {
using namespace sys;

const char sys::PathSeparator = ':';

StringRef Path::GetEXESuffix() {
  return StringRef();
}

Path::Path(StringRef p)
  : path(p) {}

Path::Path(const char *StrStart, unsigned StrLen)
  : path(StrStart, StrLen) {}

Path&
Path::operator=(StringRef that) {
  path.assign(that.data(), that.size());
  return *this;
}

bool
Path::isValid() const {
  // Empty paths are considered invalid here.
  // This code doesn't check MAXPATHLEN because there's no need. Nothing in
  // LLVM manipulates Paths with fixed-sizes arrays, and if the OS can't
  // handle names longer than some limit, it'll report this on demand using
  // ENAMETOLONG.
  return !path.empty();
}

bool
Path::isAbsolute(const char *NameStart, unsigned NameLen) {
  assert(NameStart);
  if (NameLen == 0)
    return false;
  return NameStart[0] == '/';
}

bool
Path::isAbsolute() const {
  if (path.empty())
    return false;
  return path[0] == '/';
}

Path
Path::GetRootDirectory() {
  Path result;
  result.set("/");
  return result;
}

Path
Path::GetTemporaryDirectory(std::string *ErrMsg) {
#if defined(HAVE_MKDTEMP)
  // The best way is with mkdtemp but that's not available on many systems,
  // Linux and FreeBSD have it. Others probably won't.
  char pathname[] = "/tmp/llvm_XXXXXX";
  if (0 == mkdtemp(pathname)) {
    MakeErrMsg(ErrMsg,
               std::string(pathname) + ": can't create temporary directory");
    return Path();
  }
  return Path(pathname);
#elif defined(HAVE_MKSTEMP)
  // If no mkdtemp is available, mkstemp can be used to create a temporary file
  // which is then removed and created as a directory. We prefer this over
  // mktemp because of mktemp's inherent security and threading risks. We still
  // have a slight race condition from the time the temporary file is created to
  // the time it is re-created as a directoy.
  char pathname[] = "/tmp/llvm_XXXXXX";
  int fd = 0;
  if (-1 == (fd = mkstemp(pathname))) {
    MakeErrMsg(ErrMsg,
      std::string(pathname) + ": can't create temporary directory");
    return Path();
  }
  ::close(fd);
  ::unlink(pathname); // start race condition, ignore errors
  if (-1 == ::mkdir(pathname, S_IRWXU)) { // end race condition
    MakeErrMsg(ErrMsg,
      std::string(pathname) + ": can't create temporary directory");
    return Path();
  }
  return Path(pathname);
#elif defined(HAVE_MKTEMP)
  // If a system doesn't have mkdtemp(3) or mkstemp(3) but it does have
  // mktemp(3) then we'll assume that system (e.g. AIX) has a reasonable
  // implementation of mktemp(3) and doesn't follow BSD 4.3's lead of replacing
  // the XXXXXX with the pid of the process and a letter. That leads to only
  // twenty six temporary files that can be generated.
  char pathname[] = "/tmp/llvm_XXXXXX";
  char *TmpName = ::mktemp(pathname);
  if (TmpName == 0) {
    MakeErrMsg(ErrMsg,
      std::string(TmpName) + ": can't create unique directory name");
    return Path();
  }
  if (-1 == ::mkdir(TmpName, S_IRWXU)) {
    MakeErrMsg(ErrMsg,
        std::string(TmpName) + ": can't create temporary directory");
    return Path();
  }
  return Path(TmpName);
#else
  // This is the worst case implementation. tempnam(3) leaks memory unless its
  // on an SVID2 (or later) system. On BSD 4.3 it leaks. tmpnam(3) has thread
  // issues. The mktemp(3) function doesn't have enough variability in the
  // temporary name generated. So, we provide our own implementation that
  // increments an integer from a random number seeded by the current time. This
  // should be sufficiently unique that we don't have many collisions between
  // processes. Generally LLVM processes don't run very long and don't use very
  // many temporary files so this shouldn't be a big issue for LLVM.
  static time_t num = ::time(0);
  char pathname[MAXPATHLEN];
  do {
    num++;
    sprintf(pathname, "/tmp/llvm_%010u", unsigned(num));
  } while ( 0 == access(pathname, F_OK ) );
  if (-1 == ::mkdir(pathname, S_IRWXU)) {
    MakeErrMsg(ErrMsg,
      std::string(pathname) + ": can't create temporary directory");
    return Path();
  }
  return Path(pathname);
#endif
}

void
Path::GetSystemLibraryPaths(std::vector<sys::Path>& Paths) {
#ifdef LTDL_SHLIBPATH_VAR
  char* env_var = getenv(LTDL_SHLIBPATH_VAR);
  if (env_var != 0) {
    getPathList(env_var,Paths);
  }
#endif
  // FIXME: Should this look at LD_LIBRARY_PATH too?
  Paths.push_back(sys::Path("/usr/local/lib/"));
  Paths.push_back(sys::Path("/usr/X11R6/lib/"));
  Paths.push_back(sys::Path("/usr/lib/"));
  Paths.push_back(sys::Path("/lib/"));
}

void
Path::GetBitcodeLibraryPaths(std::vector<sys::Path>& Paths) {
  char * env_var = getenv("LLVM_LIB_SEARCH_PATH");
  if (env_var != 0) {
    getPathList(env_var,Paths);
  }
#ifdef LLVM_LIBDIR
  {
    Path tmpPath;
    if (tmpPath.set(LLVM_LIBDIR))
      if (tmpPath.canRead())
        Paths.push_back(tmpPath);
  }
#endif
  GetSystemLibraryPaths(Paths);
}

Path
Path::GetUserHomeDirectory() {
  const char* home = getenv("HOME");
  Path result;
  if (home && result.set(home))
    return result;
  result.set("/");
  return result;
}

Path
Path::GetCurrentDirectory() {
  char pathname[MAXPATHLEN];
  if (!getcwd(pathname, MAXPATHLEN)) {
    assert(false && "Could not query current working directory.");
    return Path();
  }

  return Path(pathname);
}

#if defined(__FreeBSD__) || defined (__NetBSD__) || defined(__Bitrig__) || \
    defined(__OpenBSD__) || defined(__minix) || defined(__FreeBSD_kernel__) || \
    defined(__linux__) || defined(__CYGWIN__)
static int
test_dir(char buf[PATH_MAX], char ret[PATH_MAX],
    const char *dir, const char *bin)
{
  struct stat sb;

  snprintf(buf, PATH_MAX, "%s/%s", dir, bin);
  if (realpath(buf, ret) == NULL)
    return (1);
  if (stat(buf, &sb) != 0)
    return (1);

  return (0);
}

static char *
getprogpath(char ret[PATH_MAX], const char *bin)
{
  char *pv, *s, *t, buf[PATH_MAX];

  /* First approach: absolute path. */
  if (bin[0] == '/') {
    if (test_dir(buf, ret, "/", bin) == 0)
      return (ret);
    return (NULL);
  }

  /* Second approach: relative path. */
  if (strchr(bin, '/') != NULL) {
    if (getcwd(buf, PATH_MAX) == NULL)
      return (NULL);
    if (test_dir(buf, ret, buf, bin) == 0)
      return (ret);
    return (NULL);
  }

  /* Third approach: $PATH */
  if ((pv = getenv("PATH")) == NULL)
    return (NULL);
  s = pv = strdup(pv);
  if (pv == NULL)
    return (NULL);
  while ((t = strsep(&s, ":")) != NULL) {
    if (test_dir(buf, ret, t, bin) == 0) {
      free(pv);
      return (ret);
    }
  }
  free(pv);
  return (NULL);
}
#endif // __FreeBSD__ || __NetBSD__ || __FreeBSD_kernel__

/// GetMainExecutable - Return the path to the main executable, given the
/// value of argv[0] from program startup.
Path Path::GetMainExecutable(const char *argv0, void *MainAddr) {
#if defined(__APPLE__)
  // On OS X the executable path is saved to the stack by dyld. Reading it
  // from there is much faster than calling dladdr, especially for large
  // binaries with symbols.
  char exe_path[MAXPATHLEN];
  uint32_t size = sizeof(exe_path);
  if (_NSGetExecutablePath(exe_path, &size) == 0) {
    char link_path[MAXPATHLEN];
    if (realpath(exe_path, link_path))
      return Path(link_path);
  }
#elif defined(__FreeBSD__) || defined (__NetBSD__) || defined(__Bitrig__) || \
      defined(__OpenBSD__) || defined(__minix) || defined(__FreeBSD_kernel__)
  char exe_path[PATH_MAX];

  if (getprogpath(exe_path, argv0) != NULL)
    return Path(exe_path);
#elif defined(__linux__) || defined(__CYGWIN__)
  char exe_path[MAXPATHLEN];
  StringRef aPath("/proc/self/exe");
  if (sys::fs::exists(aPath)) {
      // /proc is not always mounted under Linux (chroot for example).
      ssize_t len = readlink(aPath.str().c_str(), exe_path, sizeof(exe_path));
      if (len >= 0)
          return Path(StringRef(exe_path, len));
  } else {
      // Fall back to the classical detection.
      if (getprogpath(exe_path, argv0) != NULL)
          return Path(exe_path);
  }
#elif defined(HAVE_DLFCN_H)
  // Use dladdr to get executable path if available.
  Dl_info DLInfo;
  int err = dladdr(MainAddr, &DLInfo);
  if (err == 0)
    return Path();

  // If the filename is a symlink, we need to resolve and return the location of
  // the actual executable.
  char link_path[MAXPATHLEN];
  if (realpath(DLInfo.dli_fname, link_path))
    return Path(link_path);
#else
#error GetMainExecutable is not implemented on this host yet.
#endif
  return Path();
}


StringRef Path::getDirname() const {
  return getDirnameCharSep(path, "/");
}

StringRef
Path::getBasename() const {
  // Find the last slash
  std::string::size_type slash = path.rfind('/');
  if (slash == std::string::npos)
    slash = 0;
  else
    slash++;

  std::string::size_type dot = path.rfind('.');
  if (dot == std::string::npos || dot < slash)
    return StringRef(path).substr(slash);
  else
    return StringRef(path).substr(slash, dot - slash);
}

StringRef
Path::getSuffix() const {
  // Find the last slash
  std::string::size_type slash = path.rfind('/');
  if (slash == std::string::npos)
    slash = 0;
  else
    slash++;

  std::string::size_type dot = path.rfind('.');
  if (dot == std::string::npos || dot < slash)
    return StringRef();
  else
    return StringRef(path).substr(dot + 1);
}

bool Path::getMagicNumber(std::string &Magic, unsigned len) const {
  assert(len < 1024 && "Request for magic string too long");
  char Buf[1025];
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0)
    return false;
  ssize_t bytes_read = ::read(fd, Buf, len);
  ::close(fd);
  if (ssize_t(len) != bytes_read)
    return false;
  Magic.assign(Buf, len);
  return true;
}

bool
Path::exists() const {
  return 0 == access(path.c_str(), F_OK );
}

bool
Path::isDirectory() const {
  struct stat buf;
  if (0 != stat(path.c_str(), &buf))
    return false;
  return ((buf.st_mode & S_IFMT) == S_IFDIR) ? true : false;
}

bool
Path::isSymLink() const {
  struct stat buf;
  if (0 != lstat(path.c_str(), &buf))
    return false;
  return S_ISLNK(buf.st_mode);
}


bool
Path::canRead() const {
  return 0 == access(path.c_str(), R_OK);
}

bool
Path::canWrite() const {
  return 0 == access(path.c_str(), W_OK);
}

bool
Path::isRegularFile() const {
  // Get the status so we can determine if it's a file or directory
  struct stat buf;

  if (0 != stat(path.c_str(), &buf))
    return false;

  if (S_ISREG(buf.st_mode))
    return true;

  return false;
}

bool
Path::canExecute() const {
  if (0 != access(path.c_str(), R_OK | X_OK ))
    return false;
  struct stat buf;
  if (0 != stat(path.c_str(), &buf))
    return false;
  if (!S_ISREG(buf.st_mode))
    return false;
  return true;
}

StringRef
Path::getLast() const {
  // Find the last slash
  size_t pos = path.rfind('/');

  // Handle the corner cases
  if (pos == std::string::npos)
    return path;

  // If the last character is a slash
  if (pos == path.length()-1) {
    // Find the second to last slash
    size_t pos2 = path.rfind('/', pos-1);
    if (pos2 == std::string::npos)
      return StringRef(path).substr(0,pos);
    else
      return StringRef(path).substr(pos2+1,pos-pos2-1);
  }
  // Return everything after the last slash
  return StringRef(path).substr(pos+1);
}

const FileStatus *
PathWithStatus::getFileStatus(bool update, std::string *ErrStr) const {
  if (!fsIsValid || update) {
    struct stat buf;
    if (0 != stat(path.c_str(), &buf)) {
      MakeErrMsg(ErrStr, path + ": can't get status of file");
      return 0;
    }
    status.fileSize = buf.st_size;
    status.modTime.fromEpochTime(buf.st_mtime);
    status.mode = buf.st_mode;
    status.user = buf.st_uid;
    status.group = buf.st_gid;
    status.uniqueID = uint64_t(buf.st_ino);
    status.isDir  = S_ISDIR(buf.st_mode);
    status.isFile = S_ISREG(buf.st_mode);
    fsIsValid = true;
  }
  return &status;
}

static bool AddPermissionBits(const Path &File, int bits) {
  // Get the umask value from the operating system.  We want to use it
  // when changing the file's permissions. Since calling umask() sets
  // the umask and returns its old value, we must call it a second
  // time to reset it to the user's preference.
  int mask = umask(0777); // The arg. to umask is arbitrary.
  umask(mask);            // Restore the umask.

  // Get the file's current mode.
  struct stat buf;
  if (0 != stat(File.c_str(), &buf))
    return false;
  // Change the file to have whichever permissions bits from 'bits'
  // that the umask would not disable.
  if ((chmod(File.c_str(), (buf.st_mode | (bits & ~mask)))) == -1)
      return false;
  return true;
}

bool Path::makeReadableOnDisk(std::string* ErrMsg) {
  if (!AddPermissionBits(*this, 0444))
    return MakeErrMsg(ErrMsg, path + ": can't make file readable");
  return false;
}

bool Path::makeWriteableOnDisk(std::string* ErrMsg) {
  if (!AddPermissionBits(*this, 0222))
    return MakeErrMsg(ErrMsg, path + ": can't make file writable");
  return false;
}

bool Path::makeExecutableOnDisk(std::string* ErrMsg) {
  if (!AddPermissionBits(*this, 0111))
    return MakeErrMsg(ErrMsg, path + ": can't make file executable");
  return false;
}

bool
Path::getDirectoryContents(std::set<Path>& result, std::string* ErrMsg) const {
  DIR* direntries = ::opendir(path.c_str());
  if (direntries == 0)
    return MakeErrMsg(ErrMsg, path + ": can't open directory");

  std::string dirPath = path;
  if (!lastIsSlash(dirPath))
    dirPath += '/';

  result.clear();
  struct dirent* de = ::readdir(direntries);
  for ( ; de != 0; de = ::readdir(direntries)) {
    if (de->d_name[0] != '.') {
      Path aPath(dirPath + (const char*)de->d_name);
      struct stat st;
      if (0 != lstat(aPath.path.c_str(), &st)) {
        if (S_ISLNK(st.st_mode))
          continue; // dangling symlink -- ignore
        return MakeErrMsg(ErrMsg,
                          aPath.path +  ": can't determine file object type");
      }
      result.insert(aPath);
    }
  }

  closedir(direntries);
  return false;
}

bool
Path::set(StringRef a_path) {
  if (a_path.empty())
    return false;
  path = a_path;
  return true;
}

bool
Path::appendComponent(StringRef name) {
  if (name.empty())
    return false;
  if (!lastIsSlash(path))
    path += '/';
  path += name;
  return true;
}

bool
Path::eraseComponent() {
  size_t slashpos = path.rfind('/',path.size());
  if (slashpos == 0 || slashpos == std::string::npos) {
    path.erase();
    return true;
  }
  if (slashpos == path.size() - 1)
    slashpos = path.rfind('/',slashpos-1);
  if (slashpos == std::string::npos) {
    path.erase();
    return true;
  }
  path.erase(slashpos);
  return true;
}

bool
Path::eraseSuffix() {
  size_t dotpos = path.rfind('.',path.size());
  size_t slashpos = path.rfind('/',path.size());
  if (dotpos != std::string::npos) {
    if (slashpos == std::string::npos || dotpos > slashpos+1) {
      path.erase(dotpos, path.size()-dotpos);
      return true;
    }
  }
  return false;
}

static bool createDirectoryHelper(char* beg, char* end, bool create_parents) {

  if (access(beg, R_OK | W_OK) == 0)
    return false;

  if (create_parents) {

    char* c = end;

    for (; c != beg; --c)
      if (*c == '/') {

        // Recurse to handling the parent directory.
        *c = '\0';
        bool x = createDirectoryHelper(beg, c, create_parents);
        *c = '/';

        // Return if we encountered an error.
        if (x)
          return true;

        break;
      }
  }

  return mkdir(beg, S_IRWXU | S_IRWXG) != 0;
}

bool
Path::createDirectoryOnDisk( bool create_parents, std::string* ErrMsg ) {
  // Get a writeable copy of the path name
  std::string pathname(path);

  // Null-terminate the last component
  size_t lastchar = path.length() - 1 ;

  if (pathname[lastchar] != '/')
    ++lastchar;

  pathname[lastchar] = '\0';

  if (createDirectoryHelper(&pathname[0], &pathname[lastchar], create_parents))
    return MakeErrMsg(ErrMsg, pathname + ": can't create directory");

  return false;
}

bool
Path::createFileOnDisk(std::string* ErrMsg) {
  // Create the file
  int fd = ::creat(path.c_str(), S_IRUSR | S_IWUSR);
  if (fd < 0)
    return MakeErrMsg(ErrMsg, path + ": can't create file");
  ::close(fd);
  return false;
}

bool
Path::createTemporaryFileOnDisk(bool reuse_current, std::string* ErrMsg) {
  // Make this into a unique file name
  if (makeUnique( reuse_current, ErrMsg ))
    return true;

  // create the file
  int fd = ::open(path.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0666);
  if (fd < 0)
    return MakeErrMsg(ErrMsg, path + ": can't create temporary file");
  ::close(fd);
  return false;
}

bool
Path::eraseFromDisk(bool remove_contents, std::string *ErrStr) const {
  // Get the status so we can determine if it's a file or directory.
  struct stat buf;
  if (0 != stat(path.c_str(), &buf)) {
    MakeErrMsg(ErrStr, path + ": can't get status of file");
    return true;
  }

  // Note: this check catches strange situations. In all cases, LLVM should
  // only be involved in the creation and deletion of regular files.  This
  // check ensures that what we're trying to erase is a regular file. It
  // effectively prevents LLVM from erasing things like /dev/null, any block
  // special file, or other things that aren't "regular" files.
  if (S_ISREG(buf.st_mode)) {
    if (unlink(path.c_str()) != 0)
      return MakeErrMsg(ErrStr, path + ": can't destroy file");
    return false;
  }

  if (!S_ISDIR(buf.st_mode)) {
    if (ErrStr) *ErrStr = "not a file or directory";
    return true;
  }

  if (remove_contents) {
    // Recursively descend the directory to remove its contents.
    std::string cmd = "/bin/rm -rf " + path;
    if (system(cmd.c_str()) != 0) {
      MakeErrMsg(ErrStr, path + ": failed to recursively remove directory.");
      return true;
    }
    return false;
  }

  // Otherwise, try to just remove the one directory.
  std::string pathname(path);
  size_t lastchar = path.length() - 1;
  if (pathname[lastchar] == '/')
    pathname[lastchar] = '\0';
  else
    pathname[lastchar+1] = '\0';

  if (rmdir(pathname.c_str()) != 0)
    return MakeErrMsg(ErrStr, pathname + ": can't erase directory");
  return false;
}

bool
Path::renamePathOnDisk(const Path& newName, std::string* ErrMsg) {
  if (0 != ::rename(path.c_str(), newName.c_str()))
    return MakeErrMsg(ErrMsg, std::string("can't rename '") + path + "' as '" +
               newName.str() + "'");
  return false;
}

bool
Path::setStatusInfoOnDisk(const FileStatus &si, std::string *ErrStr) const {
  struct utimbuf utb;
  utb.actime = si.modTime.toPosixTime();
  utb.modtime = utb.actime;
  if (0 != ::utime(path.c_str(),&utb))
    return MakeErrMsg(ErrStr, path + ": can't set file modification time");
  if (0 != ::chmod(path.c_str(),si.mode))
    return MakeErrMsg(ErrStr, path + ": can't set mode");
  return false;
}

bool
sys::CopyFile(const sys::Path &Dest, const sys::Path &Src, std::string* ErrMsg){
  int inFile = -1;
  int outFile = -1;
  inFile = ::open(Src.c_str(), O_RDONLY);
  if (inFile == -1)
    return MakeErrMsg(ErrMsg, Src.str() +
      ": can't open source file to copy");

  outFile = ::open(Dest.c_str(), O_WRONLY|O_CREAT, 0666);
  if (outFile == -1) {
    ::close(inFile);
    return MakeErrMsg(ErrMsg, Dest.str() +
      ": can't create destination file for copy");
  }

  char Buffer[16*1024];
  while (ssize_t Amt = ::read(inFile, Buffer, 16*1024)) {
    if (Amt == -1) {
      if (errno != EINTR && errno != EAGAIN) {
        ::close(inFile);
        ::close(outFile);
        return MakeErrMsg(ErrMsg, Src.str()+": can't read source file");
      }
    } else {
      char *BufPtr = Buffer;
      while (Amt) {
        ssize_t AmtWritten = ::write(outFile, BufPtr, Amt);
        if (AmtWritten == -1) {
          if (errno != EINTR && errno != EAGAIN) {
            ::close(inFile);
            ::close(outFile);
            return MakeErrMsg(ErrMsg, Dest.str() +
              ": can't write destination file");
          }
        } else {
          Amt -= AmtWritten;
          BufPtr += AmtWritten;
        }
      }
    }
  }
  ::close(inFile);
  ::close(outFile);
  return false;
}

bool
Path::makeUnique(bool reuse_current, std::string* ErrMsg) {
  bool Exists;
  if (reuse_current && (fs::exists(path, Exists) || !Exists))
    return false; // File doesn't exist already, just use it!

  // Append an XXXXXX pattern to the end of the file for use with mkstemp,
  // mktemp or our own implementation.
  // This uses std::vector instead of SmallVector to avoid a dependence on
  // libSupport. And performance isn't critical here.
  std::vector<char> Buf;
  Buf.resize(path.size()+8);
  char *FNBuffer = &Buf[0];
    path.copy(FNBuffer,path.size());
  bool isdir;
  if (!fs::is_directory(path, isdir) && isdir)
    strcpy(FNBuffer+path.size(), "/XXXXXX");
  else
    strcpy(FNBuffer+path.size(), "-XXXXXX");

#if defined(HAVE_MKSTEMP)
  int TempFD;
  if ((TempFD = mkstemp(FNBuffer)) == -1)
    return MakeErrMsg(ErrMsg, path + ": can't make unique filename");

  // We don't need to hold the temp file descriptor... we will trust that no one
  // will overwrite/delete the file before we can open it again.
  close(TempFD);

  // Save the name
  path = FNBuffer;

  // By default mkstemp sets the mode to 0600, so update mode bits now.
  AddPermissionBits (*this, 0666);
#elif defined(HAVE_MKTEMP)
  // If we don't have mkstemp, use the old and obsolete mktemp function.
  if (mktemp(FNBuffer) == 0)
    return MakeErrMsg(ErrMsg, path + ": can't make unique filename");

  // Save the name
  path = FNBuffer;
#else
  // Okay, looks like we have to do it all by our lonesome.
  static unsigned FCounter = 0;
  // Try to initialize with unique value.
  if (FCounter == 0) FCounter = ((unsigned)getpid() & 0xFFFF) << 8;
  char* pos = strstr(FNBuffer, "XXXXXX");
  do {
    if (++FCounter > 0xFFFFFF) {
      return MakeErrMsg(ErrMsg,
        path + ": can't make unique filename: too many files");
    }
    sprintf(pos, "%06X", FCounter);
    path = FNBuffer;
  } while (exists());
  // POSSIBLE SECURITY BUG: An attacker can easily guess the name and exploit
  // LLVM.
#endif
  return false;
}

const char *Path::MapInFilePages(int FD, size_t FileSize, off_t Offset) {
  int Flags = MAP_PRIVATE;
#ifdef MAP_FILE
  Flags |= MAP_FILE;
#endif
  void *BasePtr = ::mmap(0, FileSize, PROT_READ, Flags, FD, Offset);
  if (BasePtr == MAP_FAILED)
    return 0;
  return (const char*)BasePtr;
}

void Path::UnMapFilePages(const char *BasePtr, size_t FileSize) {
  const void *Addr = static_cast<const void *>(BasePtr);
  ::munmap(const_cast<void *>(Addr), FileSize);
}

} // end llvm namespace
