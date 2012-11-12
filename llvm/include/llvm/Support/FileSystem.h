//===- llvm/Support/FileSystem.h - File System OS Concept -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::fs namespace. It is designed after
// TR2/boost filesystem (v3), but modified to remove exception handling and the
// path class.
//
// All functions return an error_code and their actual work via the last out
// argument. The out argument is defined if and only if errc::success is
// returned. A function may return any error code in the generic or system
// category. However, they shall be equivalent to any error conditions listed
// in each functions respective documentation if the condition applies. [ note:
// this does not guarantee that error_code will be in the set of explicitly
// listed codes, but it does guarantee that if any of the explicitly listed
// errors occur, the correct error_code will be used ]. All functions may
// return errc::not_enough_memory if there is not enough memory to complete the
// operation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILE_SYSTEM_H
#define LLVM_SUPPORT_FILE_SYSTEM_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/system_error.h"
#include <ctime>
#include <iterator>
#include <stack>
#include <string>
#include <vector>

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

namespace llvm {
namespace sys {
namespace fs {

/// file_type - An "enum class" enumeration for the file system's view of the
///             type.
struct file_type {
  enum _ {
    status_error,
    file_not_found,
    regular_file,
    directory_file,
    symlink_file,
    block_file,
    character_file,
    fifo_file,
    socket_file,
    type_unknown
  };

  file_type(_ v) : v_(v) {}
  explicit file_type(int v) : v_(_(v)) {}
  operator int() const {return v_;}

private:
  int v_;
};

/// copy_option - An "enum class" enumeration of copy semantics for copy
///               operations.
struct copy_option {
  enum _ {
    fail_if_exists,
    overwrite_if_exists
  };

  copy_option(_ v) : v_(v) {}
  explicit copy_option(int v) : v_(_(v)) {}
  operator int() const {return v_;}

private:
  int v_;
};

/// space_info - Self explanatory.
struct space_info {
  uint64_t capacity;
  uint64_t free;
  uint64_t available;
};


enum perms {
  no_perms     = 0,
  owner_read   = 0400, 
  owner_write  = 0200, 
  owner_exe    = 0100, 
  owner_all    = owner_read | owner_write | owner_exe,
  group_read   =  040, 
  group_write  =  020, 
  group_exe    =  010, 
  group_all    = group_read | group_write | group_exe,
  others_read  =   04, 
  others_write =   02, 
  others_exe   =   01, 
  others_all   = others_read | others_write | others_exe, 
  all_all      = owner_all | group_all | others_all,
  set_uid_on_exe  = 04000, 
  set_gid_on_exe  = 02000, 
  sticky_bit      = 01000,
  perms_mask      = all_all | set_uid_on_exe | set_gid_on_exe | sticky_bit, 
  perms_not_known = 0xFFFF,
  add_perms       = 0x1000,
  remove_perms    = 0x2000, 
  symlink_perms   = 0x4000
};

// Helper functions so that you can use & and | to manipulate perms bits:
inline perms operator|(perms l , perms r) {
  return static_cast<perms>(
             static_cast<unsigned short>(l) | static_cast<unsigned short>(r)); 
}
inline perms operator&(perms l , perms r) {
  return static_cast<perms>(
             static_cast<unsigned short>(l) & static_cast<unsigned short>(r)); 
}
inline perms &operator|=(perms &l, perms r) {
  l = l | r; 
  return l; 
}
inline perms &operator&=(perms &l, perms r) {
  l = l & r; 
  return l; 
}
inline perms operator~(perms x) {
  return static_cast<perms>(~static_cast<unsigned short>(x));
}


 
/// file_status - Represents the result of a call to stat and friends. It has
///               a platform specific member to store the result.
class file_status
{
  #if defined(LLVM_ON_UNIX)
  dev_t fs_st_dev;
  ino_t fs_st_ino;
  #elif defined (LLVM_ON_WIN32)
  uint32_t LastWriteTimeHigh;
  uint32_t LastWriteTimeLow;
  uint32_t VolumeSerialNumber;
  uint32_t FileSizeHigh;
  uint32_t FileSizeLow;
  uint32_t FileIndexHigh;
  uint32_t FileIndexLow;
  #endif
  friend bool equivalent(file_status A, file_status B);
  friend error_code status(const Twine &path, file_status &result);
  file_type Type;
  perms Perms;
public:
  explicit file_status(file_type v=file_type::status_error, 
                      perms prms=perms_not_known)
    : Type(v), Perms(prms) {}

  // getters
  file_type type() const { return Type; }
  perms permissions() const { return Perms; }
  
  // setters
  void type(file_type v) { Type = v; }
  void permissions(perms p) { Perms = p; }
};

/// file_magic - An "enum class" enumeration of file types based on magic (the first
///         N bytes of the file).
struct file_magic {
  enum _ {
    unknown = 0,              ///< Unrecognized file
    bitcode,                  ///< Bitcode file
    archive,                  ///< ar style archive file
    elf_relocatable,          ///< ELF Relocatable object file
    elf_executable,           ///< ELF Executable image
    elf_shared_object,        ///< ELF dynamically linked shared lib
    elf_core,                 ///< ELF core image
    macho_object,             ///< Mach-O Object file
    macho_executable,         ///< Mach-O Executable
    macho_fixed_virtual_memory_shared_lib, ///< Mach-O Shared Lib, FVM
    macho_core,               ///< Mach-O Core File
    macho_preload_executabl,  ///< Mach-O Preloaded Executable
    macho_dynamically_linked_shared_lib, ///< Mach-O dynlinked shared lib
    macho_dynamic_linker,     ///< The Mach-O dynamic linker
    macho_bundle,             ///< Mach-O Bundle file
    macho_dynamically_linked_shared_lib_stub, ///< Mach-O Shared lib stub
    macho_dsym_companion,     ///< Mach-O dSYM companion file
    coff_object,              ///< COFF object file
    pecoff_executable         ///< PECOFF executable file
  };

  bool is_object() const {
    return v_ == unknown ? false : true;
  }

  file_magic() : v_(unknown) {}
  file_magic(_ v) : v_(v) {}
  explicit file_magic(int v) : v_(_(v)) {}
  operator int() const {return v_;}

private:
  int v_;
};

/// @}
/// @name Physical Operators
/// @{

/// @brief Make \a path an absolute path.
///
/// Makes \a path absolute using the current directory if it is not already. An
/// empty \a path will result in the current directory.
///
/// /absolute/path   => /absolute/path
/// relative/../path => <current-directory>/relative/../path
///
/// @param path A path that is modified to be an absolute path.
/// @returns errc::success if \a path has been made absolute, otherwise a
///          platform specific error_code.
error_code make_absolute(SmallVectorImpl<char> &path);

/// @brief Copy the file at \a from to the path \a to.
///
/// @param from The path to copy the file from.
/// @param to The path to copy the file to.
/// @param copt Behavior if \a to already exists.
/// @returns errc::success if the file has been successfully copied.
///          errc::file_exists if \a to already exists and \a copt ==
///          copy_option::fail_if_exists. Otherwise a platform specific
///          error_code.
error_code copy_file(const Twine &from, const Twine &to,
                     copy_option copt = copy_option::fail_if_exists);

/// @brief Create all the non-existent directories in path.
///
/// @param path Directories to create.
/// @param existed Set to true if \a path already existed, false otherwise.
/// @returns errc::success if is_directory(path) and existed have been set,
///          otherwise a platform specific error_code.
error_code create_directories(const Twine &path, bool &existed);

/// @brief Create the directory in path.
///
/// @param path Directory to create.
/// @param existed Set to true if \a path already existed, false otherwise.
/// @returns errc::success if is_directory(path) and existed have been set,
///          otherwise a platform specific error_code.
error_code create_directory(const Twine &path, bool &existed);

/// @brief Create a hard link from \a from to \a to.
///
/// @param to The path to hard link to.
/// @param from The path to hard link from. This is created.
/// @returns errc::success if exists(to) && exists(from) && equivalent(to, from)
///          , otherwise a platform specific error_code.
error_code create_hard_link(const Twine &to, const Twine &from);

/// @brief Create a symbolic link from \a from to \a to.
///
/// @param to The path to symbolically link to.
/// @param from The path to symbolically link from. This is created.
/// @returns errc::success if exists(to) && exists(from) && is_symlink(from),
///          otherwise a platform specific error_code.
error_code create_symlink(const Twine &to, const Twine &from);

/// @brief Get the current path.
///
/// @param result Holds the current path on return.
/// @returns errc::success if the current path has been stored in result,
///          otherwise a platform specific error_code.
error_code current_path(SmallVectorImpl<char> &result);

/// @brief Remove path. Equivalent to POSIX remove().
///
/// @param path Input path.
/// @param existed Set to true if \a path existed, false if it did not.
///                undefined otherwise.
/// @returns errc::success if path has been removed and existed has been
///          successfully set, otherwise a platform specific error_code.
error_code remove(const Twine &path, bool &existed);

/// @brief Recursively remove all files below \a path, then \a path. Files are
///        removed as if by POSIX remove().
///
/// @param path Input path.
/// @param num_removed Number of files removed.
/// @returns errc::success if path has been removed and num_removed has been
///          successfully set, otherwise a platform specific error_code.
error_code remove_all(const Twine &path, uint32_t &num_removed);

/// @brief Rename \a from to \a to. Files are renamed as if by POSIX rename().
///
/// @param from The path to rename from.
/// @param to The path to rename to. This is created.
error_code rename(const Twine &from, const Twine &to);

/// @brief Resize path to size. File is resized as if by POSIX truncate().
///
/// @param path Input path.
/// @param size Size to resize to.
/// @returns errc::success if \a path has been resized to \a size, otherwise a
///          platform specific error_code.
error_code resize_file(const Twine &path, uint64_t size);

/// @}
/// @name Physical Observers
/// @{

/// @brief Does file exist?
///
/// @param status A file_status previously returned from stat.
/// @returns True if the file represented by status exists, false if it does
///          not.
bool exists(file_status status);

/// @brief Does file exist?
///
/// @param path Input path.
/// @param result Set to true if the file represented by status exists, false if
///               it does not. Undefined otherwise.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code exists(const Twine &path, bool &result);

/// @brief Simpler version of exists for clients that don't need to
///        differentiate between an error and false.
inline bool exists(const Twine &path) {
  bool result;
  return !exists(path, result) && result;
}

/// @brief Do file_status's represent the same thing?
///
/// @param A Input file_status.
/// @param B Input file_status.
///
/// assert(status_known(A) || status_known(B));
///
/// @returns True if A and B both represent the same file system entity, false
///          otherwise.
bool equivalent(file_status A, file_status B);

/// @brief Do paths represent the same thing?
///
/// assert(status_known(A) || status_known(B));
///
/// @param A Input path A.
/// @param B Input path B.
/// @param result Set to true if stat(A) and stat(B) have the same device and
///               inode (or equivalent).
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code equivalent(const Twine &A, const Twine &B, bool &result);

/// @brief Simpler version of equivalent for clients that don't need to
///        differentiate between an error and false.
inline bool equivalent(const Twine &A, const Twine &B) {
  bool result;
  return !equivalent(A, B, result) && result;
}

/// @brief Get file size.
///
/// @param path Input path.
/// @param result Set to the size of the file in \a path.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code file_size(const Twine &path, uint64_t &result);

/// @brief Does status represent a directory?
///
/// @param status A file_status previously returned from status.
/// @returns status.type() == file_type::directory_file.
bool is_directory(file_status status);

/// @brief Is path a directory?
///
/// @param path Input path.
/// @param result Set to true if \a path is a directory, false if it is not.
///               Undefined otherwise.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code is_directory(const Twine &path, bool &result);

/// @brief Does status represent a regular file?
///
/// @param status A file_status previously returned from status.
/// @returns status_known(status) && status.type() == file_type::regular_file.
bool is_regular_file(file_status status);

/// @brief Is path a regular file?
///
/// @param path Input path.
/// @param result Set to true if \a path is a regular file, false if it is not.
///               Undefined otherwise.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code is_regular_file(const Twine &path, bool &result);

/// @brief Does this status represent something that exists but is not a
///        directory, regular file, or symlink?
///
/// @param status A file_status previously returned from status.
/// @returns exists(s) && !is_regular_file(s) && !is_directory(s) &&
///          !is_symlink(s)
bool is_other(file_status status);

/// @brief Is path something that exists but is not a directory,
///        regular file, or symlink?
///
/// @param path Input path.
/// @param result Set to true if \a path exists, but is not a directory, regular
///               file, or a symlink, false if it does not. Undefined otherwise.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code is_other(const Twine &path, bool &result);

/// @brief Does status represent a symlink?
///
/// @param status A file_status previously returned from stat.
/// @returns status.type() == symlink_file.
bool is_symlink(file_status status);

/// @brief Is path a symlink?
///
/// @param path Input path.
/// @param result Set to true if \a path is a symlink, false if it is not.
///               Undefined otherwise.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code is_symlink(const Twine &path, bool &result);

/// @brief Get file status as if by POSIX stat().
///
/// @param path Input path.
/// @param result Set to the file status.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code status(const Twine &path, file_status &result);

/// @brief Modifies permission bits on a file
///
/// @param path Input path.
/// @returns errc::success if permissions have been changed, otherwise a
///          platform specific error_code.
error_code permissions(const Twine &path, perms prms);

/// @brief Is status available?
///
/// @param s Input file status.
/// @returns True if status() != status_error.
bool status_known(file_status s);

/// @brief Is status available?
///
/// @param path Input path.
/// @param result Set to true if status() != status_error.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code status_known(const Twine &path, bool &result);

/// @brief Generate a unique path and open it as a file.
///
/// Generates a unique path suitable for a temporary file and then opens it as a
/// file. The name is based on \a model with '%' replaced by a random char in
/// [0-9a-f]. If \a model is not an absolute path, a suitable temporary
/// directory will be prepended.
///
/// This is an atomic operation. Either the file is created and opened, or the
/// file system is left untouched.
///
/// clang-%%-%%-%%-%%-%%.s => /tmp/clang-a0-b1-c2-d3-e4.s
///
/// @param model Name to base unique path off of.
/// @param result_fd Set to the opened file's file descriptor.
/// @param result_path Set to the opened file's absolute path.
/// @param makeAbsolute If true and \a model is not an absolute path, a temp
///        directory will be prepended.
/// @returns errc::success if result_{fd,path} have been successfully set,
///          otherwise a platform specific error_code.
error_code unique_file(const Twine &model, int &result_fd,
                       SmallVectorImpl<char> &result_path,
                       bool makeAbsolute = true, unsigned mode = 0600);

/// @brief Canonicalize path.
///
/// Sets result to the file system's idea of what path is. The result is always
/// absolute and has the same capitalization as the file system.
///
/// @param path Input path.
/// @param result Set to the canonicalized version of \a path.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code canonicalize(const Twine &path, SmallVectorImpl<char> &result);

/// @brief Are \a path's first bytes \a magic?
///
/// @param path Input path.
/// @param magic Byte sequence to compare \a path's first len(magic) bytes to.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code has_magic(const Twine &path, const Twine &magic, bool &result);

/// @brief Get \a path's first \a len bytes.
///
/// @param path Input path.
/// @param len Number of magic bytes to get.
/// @param result Set to the first \a len bytes in the file pointed to by
///               \a path. Or the entire file if file_size(path) < len, in which
///               case result.size() returns the size of the file.
/// @returns errc::success if result has been successfully set,
///          errc::value_too_large if len is larger then the file pointed to by
///          \a path, otherwise a platform specific error_code.
error_code get_magic(const Twine &path, uint32_t len,
                     SmallVectorImpl<char> &result);

/// @brief Identify the type of a binary file based on how magical it is.
file_magic identify_magic(StringRef magic);

/// @brief Get and identify \a path's type based on its content.
///
/// @param path Input path.
/// @param result Set to the type of file, or LLVMFileType::Unknown_FileType.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code identify_magic(const Twine &path, file_magic &result);

/// @brief Get library paths the system linker uses.
///
/// @param result Set to the list of system library paths.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code GetSystemLibraryPaths(SmallVectorImpl<std::string> &result);

/// @brief Get bitcode library paths the system linker uses
///        + LLVM_LIB_SEARCH_PATH + LLVM_LIBDIR.
///
/// @param result Set to the list of bitcode library paths.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code GetBitcodeLibraryPaths(SmallVectorImpl<std::string> &result);

/// @brief Find a library.
///
/// Find the path to a library using its short name. Use the system
/// dependent library paths to locate the library.
///
/// c => /usr/lib/libc.so
///
/// @param short_name Library name one would give to the system linker.
/// @param result Set to the absolute path \a short_name represents.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code FindLibrary(const Twine &short_name, SmallVectorImpl<char> &result);

/// @brief Get absolute path of main executable.
///
/// @param argv0 The program name as it was spelled on the command line.
/// @param MainAddr Address of some symbol in the executable (not in a library).
/// @param result Set to the absolute path of the current executable.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code GetMainExecutable(const char *argv0, void *MainAddr,
                             SmallVectorImpl<char> &result);

/// This class represents a memory mapped file. It is based on
/// boost::iostreams::mapped_file.
class mapped_file_region {
  mapped_file_region() LLVM_DELETED_FUNCTION;
  mapped_file_region(mapped_file_region&) LLVM_DELETED_FUNCTION;
  mapped_file_region &operator =(mapped_file_region&) LLVM_DELETED_FUNCTION;

public:
  enum mapmode {
    readonly, ///< May only access map via const_data as read only.
    readwrite, ///< May access map via data and modify it. Written to path.
    priv ///< May modify via data, but changes are lost on destruction.
  };

private:
  /// Platform specific mapping state.
  mapmode Mode;
  uint64_t Size;
  void *Mapping;
#ifdef LLVM_ON_WIN32
  int FileDescriptor;
  void *FileHandle;
  void *FileMappingHandle;
#endif

  error_code init(int FD, uint64_t Offset);

public:
  typedef char char_type;

#if LLVM_USE_RVALUE_REFERENCES
  mapped_file_region(mapped_file_region&&);
  mapped_file_region &operator =(mapped_file_region&&);
#endif

  /// Construct a mapped_file_region at \a path starting at \a offset of length
  /// \a length and with access \a mode.
  ///
  /// \param path Path to the file to map. If it does not exist it will be
  ///             created.
  /// \param mode How to map the memory.
  /// \param length Number of bytes to map in starting at \a offset. If the file
  ///               is shorter than this, it will be extended. If \a length is
  ///               0, the entire file will be mapped.
  /// \param offset Byte offset from the beginning of the file where the map
  ///               should begin. Must be a multiple of
  ///               mapped_file_region::alignment().
  /// \param ec This is set to errc::success if the map was constructed
  ///           sucessfully. Otherwise it is set to a platform dependent error.
  mapped_file_region(const Twine &path,
                     mapmode mode,
                     uint64_t length,
                     uint64_t offset,
                     error_code &ec);

  /// \param fd An open file descriptor to map. mapped_file_region takes
  ///           ownership. It must have been opended in the correct mode.
  mapped_file_region(int fd,
                     mapmode mode,
                     uint64_t length,
                     uint64_t offset,
                     error_code &ec);

  ~mapped_file_region();

  mapmode flags() const;
  uint64_t size() const;
  char *data() const;

  /// Get a const view of the data. Modifying this memory has undefined
  /// behaivor.
  const char *const_data() const;

  /// \returns The minimum alignment offset must be.
  static int alignment();
};

/// @brief Memory maps the contents of a file
///
/// @param path Path to file to map.
/// @param file_offset Byte offset in file where mapping should begin.
/// @param size Byte length of range of the file to map.
/// @param map_writable If true, the file will be mapped in r/w such
///        that changes to the mapped buffer will be flushed back
///        to the file.  If false, the file will be mapped read-only
///        and the buffer will be read-only.
/// @param result Set to the start address of the mapped buffer.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code map_file_pages(const Twine &path, off_t file_offset, size_t size,  
                          bool map_writable, void *&result);


/// @brief Memory unmaps the contents of a file
///
/// @param base Pointer to the start of the buffer.
/// @param size Byte length of the range to unmmap.
/// @returns errc::success if result has been successfully set, otherwise a
///          platform specific error_code.
error_code unmap_file_pages(void *base, size_t size);



/// @}
/// @name Iterators
/// @{

/// directory_entry - A single entry in a directory. Caches the status either
/// from the result of the iteration syscall, or the first time status is
/// called.
class directory_entry {
  std::string Path;
  mutable file_status Status;

public:
  explicit directory_entry(const Twine &path, file_status st = file_status())
    : Path(path.str())
    , Status(st) {}

  directory_entry() {}

  void assign(const Twine &path, file_status st = file_status()) {
    Path = path.str();
    Status = st;
  }

  void replace_filename(const Twine &filename, file_status st = file_status());

  const std::string &path() const { return Path; }
  error_code status(file_status &result) const;

  bool operator==(const directory_entry& rhs) const { return Path == rhs.Path; }
  bool operator!=(const directory_entry& rhs) const { return !(*this == rhs); }
  bool operator< (const directory_entry& rhs) const;
  bool operator<=(const directory_entry& rhs) const;
  bool operator> (const directory_entry& rhs) const;
  bool operator>=(const directory_entry& rhs) const;
};

namespace detail {
  struct DirIterState;

  error_code directory_iterator_construct(DirIterState&, StringRef);
  error_code directory_iterator_increment(DirIterState&);
  error_code directory_iterator_destruct(DirIterState&);

  /// DirIterState - Keeps state for the directory_iterator. It is reference
  /// counted in order to preserve InputIterator semantics on copy.
  struct DirIterState : public RefCountedBase<DirIterState> {
    DirIterState()
      : IterationHandle(0) {}

    ~DirIterState() {
      directory_iterator_destruct(*this);
    }

    intptr_t IterationHandle;
    directory_entry CurrentEntry;
  };
}

/// directory_iterator - Iterates through the entries in path. There is no
/// operator++ because we need an error_code. If it's really needed we can make
/// it call report_fatal_error on error.
class directory_iterator {
  IntrusiveRefCntPtr<detail::DirIterState> State;

public:
  explicit directory_iterator(const Twine &path, error_code &ec) {
    State = new detail::DirIterState;
    SmallString<128> path_storage;
    ec = detail::directory_iterator_construct(*State,
            path.toStringRef(path_storage));
  }

  explicit directory_iterator(const directory_entry &de, error_code &ec) {
    State = new detail::DirIterState;
    ec = detail::directory_iterator_construct(*State, de.path());
  }

  /// Construct end iterator.
  directory_iterator() : State(new detail::DirIterState) {}

  // No operator++ because we need error_code.
  directory_iterator &increment(error_code &ec) {
    ec = directory_iterator_increment(*State);
    return *this;
  }

  const directory_entry &operator*() const { return State->CurrentEntry; }
  const directory_entry *operator->() const { return &State->CurrentEntry; }

  bool operator==(const directory_iterator &RHS) const {
    return State->CurrentEntry == RHS.State->CurrentEntry;
  }

  bool operator!=(const directory_iterator &RHS) const {
    return !(*this == RHS);
  }
  // Other members as required by
  // C++ Std, 24.1.1 Input iterators [input.iterators]
};

namespace detail {
  /// RecDirIterState - Keeps state for the recursive_directory_iterator. It is
  /// reference counted in order to preserve InputIterator semantics on copy.
  struct RecDirIterState : public RefCountedBase<RecDirIterState> {
    RecDirIterState()
      : Level(0)
      , HasNoPushRequest(false) {}

    std::stack<directory_iterator, std::vector<directory_iterator> > Stack;
    uint16_t Level;
    bool HasNoPushRequest;
  };
}

/// recursive_directory_iterator - Same as directory_iterator except for it
/// recurses down into child directories.
class recursive_directory_iterator {
  IntrusiveRefCntPtr<detail::RecDirIterState> State;

public:
  recursive_directory_iterator() {}
  explicit recursive_directory_iterator(const Twine &path, error_code &ec)
    : State(new detail::RecDirIterState) {
    State->Stack.push(directory_iterator(path, ec));
    if (State->Stack.top() == directory_iterator())
      State.reset();
  }
  // No operator++ because we need error_code.
  recursive_directory_iterator &increment(error_code &ec) {
    static const directory_iterator end_itr;

    if (State->HasNoPushRequest)
      State->HasNoPushRequest = false;
    else {
      file_status st;
      if ((ec = State->Stack.top()->status(st))) return *this;
      if (is_directory(st)) {
        State->Stack.push(directory_iterator(*State->Stack.top(), ec));
        if (ec) return *this;
        if (State->Stack.top() != end_itr) {
          ++State->Level;
          return *this;
        }
        State->Stack.pop();
      }
    }

    while (!State->Stack.empty()
           && State->Stack.top().increment(ec) == end_itr) {
      State->Stack.pop();
      --State->Level;
    }

    // Check if we are done. If so, create an end iterator.
    if (State->Stack.empty())
      State.reset();

    return *this;
  }

  const directory_entry &operator*() const { return *State->Stack.top(); }
  const directory_entry *operator->() const { return &*State->Stack.top(); }

  // observers
  /// Gets the current level. Starting path is at level 0.
  int level() const { return State->Level; }

  /// Returns true if no_push has been called for this directory_entry.
  bool no_push_request() const { return State->HasNoPushRequest; }

  // modifiers
  /// Goes up one level if Level > 0.
  void pop() {
    assert(State && "Cannot pop and end itertor!");
    assert(State->Level > 0 && "Cannot pop an iterator with level < 1");

    static const directory_iterator end_itr;
    error_code ec;
    do {
      if (ec)
        report_fatal_error("Error incrementing directory iterator.");
      State->Stack.pop();
      --State->Level;
    } while (!State->Stack.empty()
             && State->Stack.top().increment(ec) == end_itr);

    // Check if we are done. If so, create an end iterator.
    if (State->Stack.empty())
      State.reset();
  }

  /// Does not go down into the current directory_entry.
  void no_push() { State->HasNoPushRequest = true; }

  bool operator==(const recursive_directory_iterator &RHS) const {
    return State == RHS.State;
  }

  bool operator!=(const recursive_directory_iterator &RHS) const {
    return !(*this == RHS);
  }
  // Other members as required by
  // C++ Std, 24.1.1 Input iterators [input.iterators]
};

/// @}

} // end namespace fs
} // end namespace sys
} // end namespace llvm

#endif
