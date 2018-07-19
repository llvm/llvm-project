//===-- BinaryHolder.h - Utility class for accessing binaries -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that aims to be a dropin replacement for
// Darwin's dsymutil.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_DSYMUTIL_BINARYHOLDER_H
#define LLVM_TOOLS_DSYMUTIL_BINARYHOLDER_H

#include "llvm/ADT/Triple.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"

namespace llvm {
namespace dsymutil {

/// \brief The BinaryHolder class is responsible for creating and
/// owning ObjectFile objects and their underlying MemoryBuffer. This
/// is different from a simple OwningBinary in that it handles
/// accessing to archive members.
///
/// As an optimization, this class will reuse an already mapped and
/// parsed Archive object if 2 successive requests target the same
/// archive file (Which is always the case in debug maps).
/// Currently it only owns one memory buffer at any given time,
/// meaning that a mapping request will invalidate the previous memory
/// mapping.
class BinaryHolder {
  std::vector<std::unique_ptr<object::Archive>> CurrentArchives;
  std::unique_ptr<MemoryBuffer> CurrentMemoryBuffer;
  std::vector<std::unique_ptr<object::ObjectFile>> CurrentObjectFiles;
  std::unique_ptr<object::MachOUniversalBinary> CurrentFatBinary;
  std::string CurrentFatBinaryName;
  bool Verbose;

  /// Get the MemoryBufferRefs for the file specification in \p
  /// Filename from the current archive. Multiple buffers are returned
  /// when there are multiple architectures available for the
  /// requested file.
  ///
  /// This function performs no system calls, it just looks up a
  /// potential match for the given \p Filename in the currently
  /// mapped archive if there is one.
  ErrorOr<std::vector<MemoryBufferRef>>
  GetArchiveMemberBuffers(StringRef Filename,
                          sys::TimePoint<std::chrono::seconds> Timestamp);

  /// Interpret Filename as an archive member specification map the
  /// corresponding archive to memory and return the MemoryBufferRefs
  /// corresponding to the described member. Multiple buffers are
  /// returned when there are multiple architectures available for the
  /// requested file.
  ErrorOr<std::vector<MemoryBufferRef>>
  MapArchiveAndGetMemberBuffers(StringRef Filename,
                                sys::TimePoint<std::chrono::seconds> Timestamp);

  /// Return the MemoryBufferRef that holds the memory mapping for the
  /// given \p Filename. This function will try to parse archive
  /// member specifications of the form /path/to/archive.a(member.o).
  ///
  /// The returned MemoryBufferRefs points to a buffer owned by this
  /// object. The buffer is valid until the next call to
  /// GetMemoryBufferForFile() on this object.
  /// Multiple buffers are returned when there are multiple
  /// architectures available for the requested file.
  ErrorOr<std::vector<MemoryBufferRef>>
  GetMemoryBuffersForFile(StringRef Filename,
                          sys::TimePoint<std::chrono::seconds> Timestamp);

  void changeBackingMemoryBuffer(std::unique_ptr<MemoryBuffer> &&MemBuf);
  ErrorOr<const object::ObjectFile &> getObjfileForArch(const Triple &T);

public:
  BinaryHolder(bool Verbose) : Verbose(Verbose) {}

  /// Get the ObjectFiles designated by the \p Filename. This
  /// might be an archive member specification of the form
  /// /path/to/archive.a(member.o).
  ///
  /// Calling this function invalidates the previous mapping owned by
  /// the BinaryHolder. Multiple buffers are returned when there are
  /// multiple architectures available for the requested file.
  ErrorOr<std::vector<const object::ObjectFile *>>
  GetObjectFiles(StringRef Filename,
                 sys::TimePoint<std::chrono::seconds> Timestamp =
                     sys::TimePoint<std::chrono::seconds>());

  /// Wraps GetObjectFiles() to return a derived ObjectFile type.
  template <typename ObjectFileType>
  ErrorOr<std::vector<const ObjectFileType *>>
  GetFilesAs(StringRef Filename,
             sys::TimePoint<std::chrono::seconds> Timestamp =
                 sys::TimePoint<std::chrono::seconds>()) {
    auto ErrOrObjFile = GetObjectFiles(Filename, Timestamp);
    if (auto Err = ErrOrObjFile.getError())
      return Err;

    std::vector<const ObjectFileType *> Objects;
    Objects.reserve((*ErrOrObjFile).size());
    for (const auto &Obj : *ErrOrObjFile) {
      const auto *Derived = dyn_cast<ObjectFileType>(Obj);
      if (!Derived)
        return make_error_code(object::object_error::invalid_file_type);
      Objects.push_back(Derived);
    }
    return std::move(Objects);
  }

  /// Access the currently owned ObjectFile with architecture \p T. As
  /// successfull call to GetObjectFiles() or GetFilesAs() must have
  /// been performed before calling this.
  ErrorOr<const object::ObjectFile &> Get(const Triple &T) {
    return getObjfileForArch(T);
  }

  /// Access to a derived version of the currently owned
  /// ObjectFile. The conversion must be known to be valid.
  template <typename ObjectFileType>
  ErrorOr<const ObjectFileType &> GetAs(const Triple &T) {
    auto ErrOrObj = Get(T);
    if (auto Err = ErrOrObj.getError())
      return Err;
    return cast<ObjectFileType>(*ErrOrObj);
  }
};
}
}
#endif
