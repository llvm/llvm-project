//===-- CMakeFileAPI.cpp - CMake File API ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CMakeFileAPI.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;

/// \returns true if \p Path is a nested directory in \p ParentPath.
/// \p Path should be absolute. Returns false if \p ParentPath is relative.
static bool isPathNestedIn(StringRef Path, StringRef ParentPath) {
  assert(sys::path::is_absolute(Path));
  if (Path.size() <= ParentPath.size())
    return false;
  if (!Path.startswith(ParentPath))
    return false;
  return sys::path::is_separator(Path.drop_front(ParentPath.size()).front());
}

/// Performs key lookups with \p KeyPath in the form of "key/subkey/anotherkey"
/// and returns the final nested value.
static Expected<const json::Value &> getValueFromPath(const json::Object &Root,
                                                      StringRef KeyPath,
                                                      StringRef ErrPrefix) {
  assert(!KeyPath.empty());
  StringRef Remaining = KeyPath;
  size_t KeyPathPos = 0;

  auto error = [&]() -> Error {
    return createStringError(llvm::inconvertibleErrorCode(),
                             ErrPrefix + ": missing '" +
                                 KeyPath.take_front(KeyPathPos - 1) +
                                 "' entry");
  };

  const json::Object *Obj = &Root;
  while (true) {
    StringRef Key;
    std::tie(Key, Remaining) = Remaining.split('/');
    KeyPathPos += Key.size() + 1;
    if (Remaining.empty()) {
      const json::Value *Val = Obj->get(Key);
      if (!Val)
        return error();
      return *Val;
    }
    Obj = Obj->getObject(Key);
    if (!Obj)
      return error();
  }
}

/// Performs key lookups with \p KeyPath in the form of "key/subkey/anotherkey"
/// and returns the final nested string.
static Expected<StringRef> getStringFromPath(const json::Object &Root,
                                             StringRef KeyPath,
                                             StringRef ErrPrefix) {
  auto Val = getValueFromPath(Root, KeyPath, ErrPrefix);
  if (!Val)
    return Val.takeError();
  Optional<StringRef> Str = Val->getAsString();
  if (!Str)
    return createStringError(llvm::inconvertibleErrorCode(),
                             ErrPrefix + ": expected string '" + KeyPath +
                                 "' entry");

  return *Str;
}

Expected<cmake_file_api::Index>
cmake_file_api::Index::fromPath(StringRef CMakeBuildPath) {
  // Detect presence of index file like
  // '.cmake/api/v1/reply/index-2019-05-25T23-04-47-0494.json'
  SmallString<128> ReplyPath(CMakeBuildPath);
  sys::path::append(ReplyPath, ".cmake", "api", "v1", "reply");

  std::error_code EC;
  Optional<std::string> RecentIndexPath;
  for (sys::fs::directory_iterator DirIt(ReplyPath, EC), DirE;
       !EC && DirIt != DirE; DirIt.increment(EC)) {
    StringRef Filename = sys::path::filename(DirIt->path());
    if (Filename.startswith("index-") && Filename.endswith(".json")) {
      if (!RecentIndexPath || sys::path::filename(*RecentIndexPath) < Filename)
        RecentIndexPath = DirIt->path();
    }
  }
  if (EC)
    return createStringError(EC, "failed reading contents of '" + ReplyPath +
                                     "': " + EC.message());

  if (!RecentIndexPath)
    return createStringError(llvm::inconvertibleErrorCode(),
                             "missing 'index' file");
  StringRef IndexPath = *RecentIndexPath;

  ErrorOr<std::unique_ptr<MemoryBuffer>> File =
      MemoryBuffer::getFile(IndexPath);
  if (!File)
    return createStringError(File.getError(),
                             "failed opening '" + IndexPath +
                                 "': " + File.getError().message());

  Expected<json::Value> Val = json::parse((*File)->getBuffer());
  if (!Val)
    return createStringError(llvm::inconvertibleErrorCode(),
                             "json error parsing '" + IndexPath +
                                 "': " + toString(Val.takeError()));
  json::Object *Obj = Val->getAsObject();
  if (!Obj)
    return createStringError(llvm::inconvertibleErrorCode(),
                             "expected json dictionary for 'index' file");
  return cmake_file_api::Index(std::move(*Obj), ReplyPath.str().str());
}

Expected<cmake_file_api::CodeModel>
cmake_file_api::Index::getCodeModel() const {
  Expected<StringRef> CodeModelPath =
      getStringFromPath(Obj, "reply/codemodel-v2/jsonFile", "index");
  if (!CodeModelPath)
    return CodeModelPath.takeError();

  SmallString<128> FullCodeModelPath(CMakeFileAPIPath);
  sys::path::append(FullCodeModelPath, *CodeModelPath);

  ErrorOr<std::unique_ptr<MemoryBuffer>> File =
      MemoryBuffer::getFile(FullCodeModelPath);
  if (!File)
    return createStringError(File.getError(),
                             "failed opening '" + FullCodeModelPath +
                                 "': " + File.getError().message());

  Expected<json::Value> Val = json::parse((*File)->getBuffer());
  if (!Val)
    return createStringError(llvm::inconvertibleErrorCode(),
                             "json error parsing '" + FullCodeModelPath +
                                 "': " + toString(Val.takeError()));
  json::Object *Obj = Val->getAsObject();
  if (!Obj)
    return createStringError(llvm::inconvertibleErrorCode(),
                             "expected json dictionary for 'codemodel' file");
  return cmake_file_api::CodeModel(std::move(*Obj));
}

Expected<StringRef> cmake_file_api::CodeModel::getBuildPath() const {
  Expected<StringRef> BuildPath =
      getStringFromPath(Obj, "paths/build", "codemodel");
  if (!BuildPath)
    return BuildPath.takeError();
  return *BuildPath;
}

Expected<StringRef> cmake_file_api::CodeModel::getSourcePath() const {
  Expected<StringRef> SourcePath =
      getStringFromPath(Obj, "paths/source", "codemodel");
  if (!SourcePath)
    return SourcePath.takeError();
  return *SourcePath;
}

Error cmake_file_api::CodeModel::getExtraTopLevelSourcePaths(
    SmallVectorImpl<StringRef> &TopLevelPaths) const {
  const json::Array *Configs = Obj.getArray("configurations");
  if (!Configs || Configs->empty())
    return createStringError(llvm::inconvertibleErrorCode(),
                             "codemodel: missing 'configurations' entry");
  const json::Object *Config = Configs->front().getAsObject();
  if (!Config)
    return createStringError(
        llvm::inconvertibleErrorCode(),
        "codemodel: expected 'configurations' as dictionary entries");
  const json::Array *Dirs = Config->getArray("directories");
  if (!Dirs)
    return createStringError(
        llvm::inconvertibleErrorCode(),
        "codemodel: missing 'configurations[0]/directories' entry");

  struct DirInfoTy {
    StringRef SourcePath;
    Optional<int64_t> ParentIndex;
  };

  auto getDirInfo = [Dirs](unsigned Index) -> Expected<DirInfoTy> {
    const json::Object *DirObj = (*Dirs)[Index].getAsObject();
    if (!DirObj)
      return createStringError(
          llvm::inconvertibleErrorCode(),
          "codemodel: expected 'directories' as dictionary entries");
    Optional<StringRef> SourcePath = DirObj->getString("source");
    if (!SourcePath)
      return createStringError(
          llvm::inconvertibleErrorCode(),
          "codemodel: missing 'configurations[0]/directories/source' entry");
    return DirInfoTy{*SourcePath, DirObj->getInteger("parentIndex")};
  };

  auto isTopLevelDir =
      [&getDirInfo](const DirInfoTy &DirInfo) -> Expected<bool> {
    if (sys::path::is_relative(DirInfo.SourcePath))
      return false;
    if (!DirInfo.ParentIndex)
      return true;
    auto Parent = getDirInfo(*DirInfo.ParentIndex);
    if (!Parent)
      return Parent.takeError();
    return !isPathNestedIn(DirInfo.SourcePath, Parent->SourcePath);
  };

  for (unsigned I = 0, E = Dirs->size(); I != E; ++I) {
    auto DirInfo = getDirInfo(I);
    if (!DirInfo)
      return DirInfo.takeError();
    Expected<bool> TopLevelCheck = isTopLevelDir(*DirInfo);
    if (!TopLevelCheck)
      return TopLevelCheck.takeError();
    if (*TopLevelCheck)
      TopLevelPaths.push_back(DirInfo->SourcePath);
  }

  return Error::success();
}
