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
#include "llvm/Support/Process.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::builtin;

static StringRef getCASIDPrefix() { return "llvmcas://"; }
void BuiltinCASContext::anchor() {}

Expected<HashType> BuiltinCASContext::parseID(StringRef Reference) {
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

  assert(Binary.size() == sizeof(HashType));
  HashType Digest;
  llvm::copy(Binary, Digest.data());
  return Digest;
}

Expected<CASID> BuiltinCAS::parseID(StringRef Reference) {
  Expected<HashType> Digest = BuiltinCASContext::parseID(Reference);
  if (!Digest)
    return Digest.takeError();

  return CASID::create(&getContext(), toStringRef(*Digest));
}

void BuiltinCASContext::printID(ArrayRef<uint8_t> Digest, raw_ostream &OS) {
  SmallString<64> Hash;
  toHex(Digest, /*LowerCase=*/true, Hash);
  OS << getCASIDPrefix() << Hash;
}

void BuiltinCASContext::printIDImpl(raw_ostream &OS, const CASID &ID) const {
  BuiltinCASContext::printID(ID.getHash(), OS);
}

const BuiltinCASContext &BuiltinCASContext::getDefaultContext() {
  static BuiltinCASContext DefaultContext;
  return DefaultContext;
}

Expected<ObjectRef> BuiltinCAS::store(ArrayRef<ObjectRef> Refs,
                                      ArrayRef<char> Data) {
  return storeImpl(BuiltinObjectHasher<HasherT>::hashObject(*this, Refs, Data),
                   Refs, Data);
}

Error BuiltinCAS::validate(const CASID &ID) {
  auto Ref = getReference(ID);
  if (!Ref)
    return createUnknownObjectError(ID);

  auto Handle = load(*Ref);
  if (!Handle)
    return Handle.takeError();

  auto Proxy = ObjectProxy::load(*this, *Ref, *Handle);
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
