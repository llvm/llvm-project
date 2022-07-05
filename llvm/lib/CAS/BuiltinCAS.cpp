//===- BuiltinCAS.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BuiltinCAS.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CAS/BuiltinObjectHasher.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::builtin;

static StringRef getCASIDPrefix() { return "llvmcas://"; }

Expected<CASID> BuiltinCAS::parseID(StringRef Reference) {
  if (!Reference.consume_front(getCASIDPrefix()))
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "invalid cas-id '" + Reference + "'");

  // FIXME: Allow shortened references?
  if (Reference.size() != 2 * sizeof(HashType))
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "wrong size for cas-id hash '" + Reference + "'");

  std::string Binary;
  if (!tryGetFromHex(Reference, Binary))
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "invalid hash in cas-id '" + Reference + "'");

  return parseIDImpl(arrayRefFromStringRef(Binary));
}

void BuiltinCAS::printIDImpl(raw_ostream &OS, const CASID &ID) const {
  assert(&ID.getContext() == this);
  assert(ID.getHash().size() == sizeof(HashType));

  SmallString<64> Hash;
  toHex(ID.getHash(), /*LowerCase=*/true, Hash);
  OS << getCASIDPrefix() << Hash;
}

static size_t getPageSize() {
  static int PageSize = sys::Process::getPageSizeEstimate();
  return PageSize;
}

Expected<ObjectHandle>
BuiltinCAS::storeFromOpenFileImpl(sys::fs::file_t FD,
                                  Optional<sys::fs::file_status> Status) {
  int PageSize = getPageSize();

  if (!Status) {
    Status.emplace();
    if (std::error_code EC = sys::fs::status(FD, *Status))
      return errorCodeToError(EC);
  }

  constexpr size_t MinMappedSize = 4 * 4096;
  auto readWithStream = [&]() -> Expected<ObjectHandle> {
    // FIXME: MSVC: SmallString<MinMappedSize * 2>
    SmallString<4 * 4096 * 2> Data;
    if (Error E = sys::fs::readNativeFileToEOF(FD, Data, MinMappedSize))
      return std::move(E);
    return store(None, makeArrayRef(Data.data(), Data.size()));
  };

  // Check whether we can trust the size from stat.
  if (Status->type() != sys::fs::file_type::regular_file &&
      Status->type() != sys::fs::file_type::block_file)
    return readWithStream();

  if (Status->getSize() < MinMappedSize)
    return readWithStream();

  std::error_code EC;
  sys::fs::mapped_file_region Map(FD, sys::fs::mapped_file_region::readonly,
                                  Status->getSize(),
                                  /*offset=*/0, EC);
  if (EC)
    return errorCodeToError(EC);

  // If the file is guaranteed to be null-terminated, use it directly. Note
  // that the file size may have changed from ::stat if this file is volatile,
  // so we need to check for an actual null character at the end.
  ArrayRef<char> Data(Map.data(), Map.size());
  HashType ComputedHash =
      BuiltinObjectHasher<HasherT>::hashObject(*this, None, Data);
  if (!isAligned(Align(PageSize), Data.size()) && Data.end()[0] == 0)
    return storeFromNullTerminatedRegion(ComputedHash, std::move(Map));
  return storeImpl(ComputedHash, None, Data);
}

Expected<ObjectHandle> BuiltinCAS::store(ArrayRef<ObjectRef> Refs,
                                         ArrayRef<char> Data) {
  return storeImpl(BuiltinObjectHasher<HasherT>::hashObject(*this, Refs, Data),
                   Refs, Data);
}

uint64_t BuiltinCAS::readDataImpl(ObjectHandle Node, raw_ostream &OS,
                                  uint64_t Offset, uint64_t MaxBytes) const {
  ArrayRef<char> Data = getDataConst(Node);
  assert(Offset < Data.size() && "Expected valid offset");
  Data = Data.drop_front(Offset).take_front(MaxBytes);
  OS << toStringRef(Data);
  return Data.size();
}

Error BuiltinCAS::validate(const CASID &ID) {
  auto Handle = load(ID);
  if (!Handle)
    return Handle.takeError();

  if (!*Handle)
    return createUnknownObjectError(ID);

  auto Proxy = ObjectProxy::load(*this, **Handle);
  SmallVector<ObjectRef> Refs;
  if (auto E = Proxy.forEachReference([&](ObjectRef Ref) -> Error {
        Refs.push_back(Ref);
        return Error::success();
      }))
    return E;

  ArrayRef<char> Data(Proxy.getData().data(), Proxy.getData().size());
  auto Hash = BuiltinObjectHasher<HasherT>::hashObject(*this, Refs, Data);
  if (!ID.getHash().equals(Hash))
    return createCorruptObjectError(ID);

  return Error::success();
}
