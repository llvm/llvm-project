//===-- cas-fuzzer.cpp - Fuzzer for CAS ObjectStore::validate() -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fuzzer for on-disk CAS validation. Creates a valid CAS database, stores
// objects, corrupts the on-disk files using fuzzer-provided bytes, then calls
// validate(). The invariant: validate() must either succeed or return an error,
// never crash.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ScopeExit.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <cstring>

using namespace llvm;
using namespace llvm::cas;

namespace {

/// Read a little-endian uint32 from Data, or 0 if not enough bytes.
static uint32_t readU32(ArrayRef<uint8_t> Data, size_t Offset) {
  if (Offset + sizeof(uint32_t) > Data.size())
    return 0;
  return support::endian::read32le(Data.data() + Offset);
}

/// Read a little-endian uint16 from Data, or 0 if not enough bytes.
static uint16_t readU16(ArrayRef<uint8_t> Data, size_t Offset) {
  if (Offset + sizeof(uint16_t) > Data.size())
    return 0;
  return support::endian::read16le(Data.data() + Offset);
}

/// Find the versioned subdirectory (v1.N) inside the CAS root.
static std::string findVersionedSubdir(StringRef CASDir) {
  std::error_code EC;
  std::string Best;
  uint64_t BestOrder = 0;
  for (sys::fs::directory_iterator DirI(CASDir, EC), DirE; !EC && DirI != DirE;
       DirI.increment(EC)) {
    if (DirI->type() != sys::fs::file_type::directory_file)
      continue;
    StringRef Name = sys::path::filename(DirI->path());
    if (!Name.starts_with("v1."))
      continue;
    uint64_t Order;
    if (Name.substr(3).getAsInteger(10, Order))
      continue;
    if (Best.empty() || Order > BestOrder) {
      Best = DirI->path();
      BestOrder = Order;
    }
  }
  return Best;
}

/// Collect paths of files matching a prefix in a directory.
static void collectFilesWithPrefix(StringRef Dir, StringRef Prefix,
                                   SmallVectorImpl<std::string> &Results) {
  std::error_code EC;
  for (sys::fs::directory_iterator DirI(Dir, EC), DirE; !EC && DirI != DirE;
       DirI.increment(EC)) {
    StringRef Name = sys::path::filename(DirI->path());
    if (Name.starts_with(Prefix))
      Results.push_back(DirI->path());
  }
}

/// Read an entire file into a buffer.
static bool readFileBytes(StringRef Path, SmallVectorImpl<char> &Buf) {
  auto MBOrErr = MemoryBuffer::getFile(Path, /*IsText=*/false,
                                       /*RequiresNullTerminator=*/false);
  if (!MBOrErr)
    return false;
  Buf.assign((*MBOrErr)->getBufferStart(), (*MBOrErr)->getBufferEnd());
  return true;
}

/// Write buffer contents to a file, replacing it entirely.
static bool writeFileBytes(StringRef Path, ArrayRef<char> Buf) {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::OF_None);
  if (EC)
    return false;
  OS.write(Buf.data(), Buf.size());
  return !OS.has_error();
}

/// Create a CAS database and store some baseline objects.
/// Returns true on success, populating CAS and AC via output parameters.
static bool createAndPopulateCAS(StringRef TmpDir,
                                 std::unique_ptr<ObjectStore> &CAS,
                                 std::unique_ptr<ActionCache> &AC) {
  auto Result = createOnDiskUnifiedCASDatabases(TmpDir);
  if (!Result) {
    consumeError(Result.takeError());
    return false;
  }
  CAS = std::move(Result->first);
  AC = std::move(Result->second);

  // Store a leaf node (no refs, small data).
  const char LeafData[] = "hello-cas-fuzzer-leaf-data";
  auto Leaf = CAS->store({}, arrayRefFromStringRef<char>(LeafData));
  if (!Leaf) {
    consumeError(Leaf.takeError());
    return false;
  }

  // Store a node with a ref to the leaf.
  const char NodeData[] = "node-with-one-ref";
  auto Node1 = CAS->store({*Leaf}, arrayRefFromStringRef<char>(NodeData));
  if (!Node1) {
    consumeError(Node1.takeError());
    return false;
  }

  // Store a node referencing both previous nodes.
  const char Node2Data[] = "node-with-two-refs";
  auto Node2 =
      CAS->store({*Leaf, *Node1}, arrayRefFromStringRef<char>(Node2Data));
  if (!Node2) {
    consumeError(Node2.takeError());
    return false;
  }

  // Store a larger data node to potentially exercise different size encodings.
  std::string LargeData(4096, 'X');
  auto LargeNode =
      CAS->store({}, arrayRefFromStringRef<char>(StringRef(LargeData)));
  if (!LargeNode) {
    consumeError(LargeNode.takeError());
    return false;
  }

  return true;
}

/// Apply byte-level mutations to a file.
static void applyByteMutations(StringRef Path, ArrayRef<uint8_t> Data) {
  SmallVector<char> Buf;
  if (!readFileBytes(Path, Buf) || Buf.empty())
    return;

  // Parse as 7-byte chunks: [offset(4)][op(1)][value(1)][unused(1)]
  for (size_t I = 0; I + 6 <= Data.size(); I += 7) {
    uint32_t Offset = readU32(Data, I) % Buf.size();
    uint8_t Op = Data[I + 4] % 3;
    uint8_t Value = Data[I + 5];
    switch (Op) {
    case 0: // XOR
      Buf[Offset] ^= Value;
      break;
    case 1: // SET
      Buf[Offset] = Value;
      break;
    case 2: // Zero
      Buf[Offset] = 0;
      break;
    }
  }
  writeFileBytes(Path, Buf);
}

/// Truncate a file to a given fraction of its size.
static void truncateFile(StringRef Path, uint8_t Fraction) {
  SmallVector<char> Buf;
  if (!readFileBytes(Path, Buf) || Buf.empty())
    return;
  // Fraction is 0-255, map to 0-100% of file size.
  size_t NewSize =
      static_cast<size_t>(static_cast<uint64_t>(Buf.size()) * Fraction / 255);
  // Don't zero out the size.
  if (NewSize == 0)
    NewSize = 1;
  Buf.resize(NewSize);
  writeFileBytes(Path, Buf);
}

/// Append garbage bytes to a file.
static void appendGarbage(StringRef Path, ArrayRef<uint8_t> Data) {
  SmallVector<char> Buf;
  if (!readFileBytes(Path, Buf))
    return;
  Buf.append(Data.begin(), Data.end());
  writeFileBytes(Path, Buf);
}

/// Zero out a range in a file.
static void zeroRange(StringRef Path, uint32_t Offset, uint16_t Length) {
  SmallVector<char> Buf;
  if (!readFileBytes(Path, Buf) || Buf.empty())
    return;
  size_t Start = Offset % Buf.size();
  size_t End = std::min(Start + static_cast<size_t>(Length), Buf.size());
  std::memset(Buf.data() + Start, 0, End - Start);
  writeFileBytes(Path, Buf);
}

/// Corrupt standalone files (obj.*, leaf.*, leaf+0.*).
static void corruptStandaloneFiles(StringRef SubDir, ArrayRef<uint8_t> Data) {
  SmallVector<std::string> StandaloneFiles;
  collectFilesWithPrefix(SubDir, "obj.", StandaloneFiles);
  collectFilesWithPrefix(SubDir, "leaf.", StandaloneFiles);
  collectFilesWithPrefix(SubDir, "leaf+0.", StandaloneFiles);

  if (StandaloneFiles.empty())
    return;

  for (size_t I = 0; I < Data.size() && !StandaloneFiles.empty(); I += 3) {
    size_t FileIdx = Data[I] % StandaloneFiles.size();
    uint8_t Action = (I + 1 < Data.size()) ? Data[I + 1] % 4 : 0;
    uint8_t Param = (I + 2 < Data.size()) ? Data[I + 2] : 128;

    StringRef FilePath = StandaloneFiles[FileIdx];
    switch (Action) {
    case 0: // Delete the file
      sys::fs::remove(FilePath);
      break;
    case 1: // Truncate
      truncateFile(FilePath, Param);
      break;
    case 2: // Corrupt bytes
      if (I + 3 < Data.size())
        applyByteMutations(
            FilePath, Data.slice(I + 3, std::min(Data.size() - I - 3,
                                                 static_cast<size_t>(21))));
      break;
    case 3: // Zero out beginning
      zeroRange(FilePath, 0, Param);
      break;
    }
  }
}

/// Select which data file to target (index.v1 or data.v1).
static std::string selectTargetFile(StringRef SubDir, uint8_t Selector) {
  SmallString<256> Path(SubDir);
  if (Selector % 2 == 0)
    sys::path::append(Path, "index.v1");
  else
    sys::path::append(Path, "data.v1");
  return std::string(Path);
}

/// Try to exercise the CAS after corruption: store and load.
static void exerciseCAS(ObjectStore &CAS) {
  // Try storing a new object.
  const char NewData[] = "post-corruption-data";
  auto NewObj = CAS.store({}, arrayRefFromStringRef<char>(NewData));
  if (!NewObj)
    consumeError(NewObj.takeError());

  // Try validate again with CheckHash=false.
  if (auto E = CAS.validate(/*CheckHash=*/false))
    consumeError(std::move(E));
}

} // end anonymous namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size == 0)
    return 0;

  // Create a unique temp directory for this fuzzer run.
  SmallString<256> TmpDir;
  if (sys::fs::createUniqueDirectory("cas-fuzzer", TmpDir))
    return 0;

  // Ensure cleanup on exit.
  auto Cleanup = scope_exit([&]() { sys::fs::remove_directories(TmpDir); });

  // Step 1: Create and populate a valid CAS.
  std::unique_ptr<ObjectStore> CAS;
  std::unique_ptr<ActionCache> AC;
  if (!createAndPopulateCAS(TmpDir, CAS, AC))
    return 0;

  // Step 2: Validate baseline - should succeed.
  if (auto E = CAS->validate(/*CheckHash=*/true)) {
    // If baseline validation fails, something is wrong with the setup.
    consumeError(std::move(E));
    return 0;
  }

  // Step 3: Close the CAS so files are unmapped.
  CAS.reset();
  AC.reset();

  // Step 4: Find the versioned subdirectory.
  std::string SubDir = findVersionedSubdir(TmpDir);
  if (SubDir.empty())
    return 0;

  // Step 5: Apply corruption based on mode selector (first byte).
  ArrayRef<uint8_t> Input(Data, Size);
  uint8_t Mode = Input[0] % 6;
  ArrayRef<uint8_t> Rest = Input.drop_front(1);

  switch (Mode) {
  case 0: { // Byte-level mutations
    if (Rest.empty())
      break;
    std::string Target = selectTargetFile(SubDir, Rest[0]);
    if (Rest.size() > 1)
      applyByteMutations(Target, Rest.drop_front(1));
    break;
  }
  case 1: { // File truncation
    if (Rest.size() < 2)
      break;
    std::string Target = selectTargetFile(SubDir, Rest[0]);
    truncateFile(Target, Rest[1]);
    break;
  }
  case 2: { // Append garbage
    if (Rest.empty())
      break;
    std::string Target = selectTargetFile(SubDir, Rest[0]);
    if (Rest.size() > 1)
      appendGarbage(Target, Rest.drop_front(1));
    break;
  }
  case 3: { // Zero out a range
    if (Rest.size() < 7)
      break;
    std::string Target = selectTargetFile(SubDir, Rest[0]);
    uint32_t Offset = readU32(Rest, 1);
    uint16_t Length = readU16(Rest, 5);
    zeroRange(Target, Offset, Length);
    break;
  }
  case 4: { // Standalone file corruption
    corruptStandaloneFiles(SubDir, Rest);
    break;
  }
  case 5: { // Combined: byte mutations + exercise CAS
    if (Rest.empty())
      break;
    std::string Target = selectTargetFile(SubDir, Rest[0]);
    if (Rest.size() > 1)
      applyByteMutations(Target, Rest.drop_front(1));
    break;
  }
  }

  // Step 6: Reopen the CAS after corruption.
  auto Reopened = createOnDiskUnifiedCASDatabases(TmpDir);
  if (!Reopened) {
    // Reopen failing is acceptable — corruption may have broken the format.
    consumeError(Reopened.takeError());
    return 0;
  }
  CAS = std::move(Reopened->first);
  AC = std::move(Reopened->second);

  // Step 7: Validate — must not crash.
  bool ValidationFailed = false;
  if (auto E = CAS->validate(/*CheckHash=*/true)) {
    consumeError(std::move(E));
    ValidationFailed = true;
  }
  if (auto E = CAS->validate(/*CheckHash=*/false)) {
    consumeError(std::move(E));
    ValidationFailed = true;
  }

  // Step 8: For mode 5, exercise the CAS only if validation passed.
  if (Mode == 5 && !ValidationFailed)
    exerciseCAS(*CAS);

  return 0;
}
