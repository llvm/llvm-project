//===-- FileCollector.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileCollector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

using namespace llvm;

static bool isCaseSensitivePath(StringRef Path) {
  SmallString<256> TmpDest = Path, UpperDest, RealDest;

  // Remove component traversals, links, etc.
  if (!sys::fs::real_path(Path, TmpDest))
    return true; // Current default value in vfs.yaml
  Path = TmpDest;

  // Change path to all upper case and ask for its real path, if the latter
  // exists and is equal to path, it's not case sensitive. Default to case
  // sensitive in the absence of real_path, since this is the YAMLVFSWriter
  // default.
  UpperDest = Path.upper();
  if (sys::fs::real_path(UpperDest, RealDest) && Path.equals(RealDest))
    return false;
  return true;
}

FileCollector::FileCollector(std::string Root, std::string OverlayRoot)
    : Root(std::move(Root)), OverlayRoot(std::move(OverlayRoot)) {
  sys::fs::create_directories(this->Root, true);
}

bool FileCollector::getRealPath(StringRef SrcPath,
                                SmallVectorImpl<char> &Result) {
  SmallString<256> RealPath;
  StringRef FileName = sys::path::filename(SrcPath);
  std::string Directory = sys::path::parent_path(SrcPath).str();
  auto DirWithSymlink = SymlinkMap.find(Directory);

  // Use real_path to fix any symbolic link component present in a path.
  // Computing the real path is expensive, cache the search through the parent
  // path Directory.
  if (DirWithSymlink == SymlinkMap.end()) {
    auto EC = sys::fs::real_path(Directory, RealPath);
    if (EC)
      return false;
    SymlinkMap[Directory] = RealPath.str();
  } else {
    RealPath = DirWithSymlink->second;
  }

  sys::path::append(RealPath, FileName);
  Result.swap(RealPath);
  return true;
}

void FileCollector::addFile(const Twine &file) {
  std::lock_guard<std::mutex> lock(Mutex);
  std::string FileStr = file.str();
  if (markAsSeen(FileStr))
    addFileImpl(FileStr);
}

void FileCollector::addFileImpl(StringRef SrcPath) {
  // We need an absolute src path to append to the root.
  SmallString<256> AbsoluteSrc = SrcPath;
  sys::fs::make_absolute(AbsoluteSrc);

  // Canonicalize src to a native path to avoid mixed separator styles.
  sys::path::native(AbsoluteSrc);

  // Remove redundant leading "./" pieces and consecutive separators.
  AbsoluteSrc = sys::path::remove_leading_dotslash(AbsoluteSrc);

  // Canonicalize the source path by removing "..", "." components.
  SmallString<256> VirtualPath = AbsoluteSrc;
  sys::path::remove_dots(VirtualPath, /*remove_dot_dot=*/true);

  // If a ".." component is present after a symlink component, remove_dots may
  // lead to the wrong real destination path. Let the source be canonicalized
  // like that but make sure we always use the real path for the destination.
  SmallString<256> CopyFrom;
  if (!getRealPath(AbsoluteSrc, CopyFrom))
    CopyFrom = VirtualPath;

  SmallString<256> DstPath = StringRef(Root);
  sys::path::append(DstPath, sys::path::relative_path(CopyFrom));

  // Always map a canonical src path to its real path into the YAML, by doing
  // this we map different virtual src paths to the same entry in the VFS
  // overlay, which is a way to emulate symlink inside the VFS; this is also
  // needed for correctness, not doing that can lead to module redefinition
  // errors.
  addFileToMapping(VirtualPath, DstPath);
}

/// Set the access and modification time for the given file from the given
/// status object.
static std::error_code
copyAccessAndModificationTime(StringRef Filename,
                              const sys::fs::file_status &Stat) {
  int FD;

  if (auto EC =
          sys::fs::openFileForWrite(Filename, FD, sys::fs::CD_OpenExisting))
    return EC;

  if (auto EC = sys::fs::setLastAccessAndModificationTime(
          FD, Stat.getLastAccessedTime(), Stat.getLastModificationTime()))
    return EC;

  if (auto EC = sys::Process::SafelyCloseFileDescriptor(FD))
    return EC;

  return {};
}

std::error_code FileCollector::copyFiles(bool StopOnError) {
  for (auto &entry : VFSWriter.getMappings()) {
    // Create directory tree.
    if (std::error_code EC =
            sys::fs::create_directories(sys::path::parent_path(entry.RPath),
                                        /*IgnoreExisting=*/true)) {
      if (StopOnError)
        return EC;
    }

    // Get the status of the original file/directory.
    sys::fs::file_status Stat;
    if (std::error_code EC = sys::fs::status(entry.VPath, Stat)) {
      if (StopOnError)
        return EC;
      continue;
    }

    if (Stat.type() == sys::fs::file_type::directory_file) {
      // Construct a directory when it's just a directory entry.
      if (std::error_code EC =
              sys::fs::create_directories(entry.RPath,
                                          /*IgnoreExisting=*/true)) {
        if (StopOnError)
          return EC;
      }
      continue;
    }

    // Copy file over.
    if (std::error_code EC = sys::fs::copy_file(entry.VPath, entry.RPath)) {
      if (StopOnError)
        return EC;
    }

    // Copy over permissions.
    if (auto perms = sys::fs::getPermissions(entry.VPath)) {
      if (std::error_code EC = sys::fs::setPermissions(entry.RPath, *perms)) {
        if (StopOnError)
          return EC;
      }
    }

    // Copy over modification time.
    copyAccessAndModificationTime(entry.RPath, Stat);
  }
  return {};
}

std::error_code FileCollector::writeMapping(StringRef mapping_file) {
  std::lock_guard<std::mutex> lock(Mutex);

  VFSWriter.setOverlayDir(OverlayRoot);
  VFSWriter.setCaseSensitivity(isCaseSensitivePath(OverlayRoot));
  VFSWriter.setUseExternalNames(false);

  std::error_code EC;
  raw_fd_ostream os(mapping_file, EC, sys::fs::OF_Text);
  if (EC)
    return EC;

  VFSWriter.write(os);

  return {};
}

namespace {

class FileCollectorFileSystem : public vfs::FileSystem {
public:
  explicit FileCollectorFileSystem(IntrusiveRefCntPtr<vfs::FileSystem> FS,
                                   std::shared_ptr<FileCollector> Collector)
      : FS(std::move(FS)), Collector(std::move(Collector)) {}

  llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override {
    auto Result = FS->status(Path);
    if (Result && Result->exists())
      Collector->addFile(Path);
    return Result;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const Twine &Path) override {
    auto Result = FS->openFileForRead(Path);
    if (Result && *Result)
      Collector->addFile(Path);
    return Result;
  }

  llvm::vfs::directory_iterator dir_begin(const llvm::Twine &Dir,
                                          std::error_code &EC) override {
    auto It = FS->dir_begin(Dir, EC);
    if (EC)
      return It;
    // Collect everything that's listed in case the user needs it.
    Collector->addFile(Dir);
    for (; !EC && It != llvm::vfs::directory_iterator(); It.increment(EC)) {
      if (It->type() == sys::fs::file_type::regular_file ||
          It->type() == sys::fs::file_type::directory_file ||
          It->type() == sys::fs::file_type::symlink_file) {
        Collector->addFile(It->path());
      }
    }
    if (EC)
      return It;
    // Return a new iterator.
    return FS->dir_begin(Dir, EC);
  }

  std::error_code getRealPath(const Twine &Path,
                              SmallVectorImpl<char> &Output) const override {
    auto EC = FS->getRealPath(Path, Output);
    if (!EC) {
      Collector->addFile(Path);
      if (Output.size() > 0)
        Collector->addFile(Output);
    }
    return EC;
  }

  std::error_code isLocal(const Twine &Path, bool &Result) override {
    return FS->isLocal(Path, Result);
  }

  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    return FS->getCurrentWorkingDirectory();
  }

  std::error_code setCurrentWorkingDirectory(const llvm::Twine &Path) override {
    return FS->setCurrentWorkingDirectory(Path);
  }

private:
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
  std::shared_ptr<FileCollector> Collector;
};

} // end anonymous namespace

IntrusiveRefCntPtr<vfs::FileSystem>
FileCollector::createCollectorVFS(IntrusiveRefCntPtr<vfs::FileSystem> BaseFS,
                                  std::shared_ptr<FileCollector> Collector) {
  return new FileCollectorFileSystem(std::move(BaseFS), std::move(Collector));
}
