//===- BuiltinCAS.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CAS_BUILTINCAS_H
#define LLVM_LIB_CAS_BUILTINCAS_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Error.h"
#include <cstddef>

namespace llvm {
namespace cas {
namespace builtin {

/// Current hash type for the internal CAS.
///
/// FIXME: This should be configurable via an enum to allow configuring the hash
/// function. The enum should be sent into \a createInMemoryCAS() and \a
/// createOnDiskCAS().
///
/// This is important (at least) for future-proofing, when we want to make new
/// CAS instances use BLAKE7, but still know how to read/write BLAKE3.
///
/// Even just for BLAKE3, it would be useful to have these values:
///
///     BLAKE3     => 32B hash from BLAKE3
///     BLAKE3_16B => 16B hash from BLAKE3 (truncated)
///
/// ... where BLAKE3_16 uses \a TruncatedBLAKE3<16>.
///
/// Motivation for a truncated hash is that it's cheaper to store. It's not
/// clear if we always (or ever) need the full 32B, and for an ephemeral
/// in-memory CAS, we almost certainly don't need it.
///
/// Note that the cost is linear in the number of objects for the builtin CAS,
/// since we're using internal offsets and/or pointers as an optimization.
///
/// However, it's possible we'll want to hook up a local builtin CAS to, e.g.,
/// a distributed generic hash map to use as an ActionCache. In that scenario,
/// the transitive closure of the structured objects that are the results of
/// the cached actions would need to be serialized into the map, something
/// like:
///
///     "action:<schema>:<key>" -> "0123"
///     "object:<schema>:0123"  -> "3,4567,89AB,CDEF,9,some data"
///     "object:<schema>:4567"  -> ...
///     "object:<schema>:89AB"  -> ...
///     "object:<schema>:CDEF"  -> ...
///
/// These references would be full cost.
using HasherT = BLAKE3;
using HashType = decltype(HasherT::hash(std::declval<ArrayRef<uint8_t> &>()));

class BuiltinCASContext : public CASContext {
  void printIDImpl(raw_ostream &OS, const CASID &ID) const final;
  void anchor() override;

public:
  /// Get the name of the hash for any table identifiers.
  ///
  /// FIXME: This should be configurable via an enum, with at the following values:
  ///
  ///     "BLAKE3"    => 32B hash from BLAKE3
  ///     "BLAKE3.16" => 16B hash from BLAKE3 (truncated)
  ///
  /// Enum can be sent into \a createInMemoryCAS() and \a createOnDiskCAS().
  static StringRef getHashName() { return "BLAKE3"; }
  StringRef getHashSchemaIdentifier() const final {
    static const std::string ID =
        ("llvm.cas.builtin.v2[" + getHashName() + "]").str();
    return ID;
  }

  static const BuiltinCASContext &getDefaultContext();

  BuiltinCASContext() = default;
};

class BuiltinCAS : public ObjectStore {
public:
  BuiltinCAS() : ObjectStore(BuiltinCASContext::getDefaultContext()) {}

  Expected<CASID> parseID(StringRef Reference) final;

  virtual Expected<CASID> parseIDImpl(ArrayRef<uint8_t> Hash) = 0;

  Expected<ObjectRef> store(ArrayRef<ObjectRef> Refs,
                            ArrayRef<char> Data) final;
  virtual Expected<ObjectRef> storeImpl(ArrayRef<uint8_t> ComputedHash,
                                        ArrayRef<ObjectRef> Refs,
                                        ArrayRef<char> Data) = 0;

  Expected<ObjectRef>
  storeFromOpenFileImpl(sys::fs::file_t FD,
                        Optional<sys::fs::file_status> Status) override;
  virtual Expected<ObjectRef>
  storeFromNullTerminatedRegion(ArrayRef<uint8_t> ComputedHash,
                                sys::fs::mapped_file_region Map) {
    return storeImpl(ComputedHash, None, makeArrayRef(Map.data(), Map.size()));
  }

  /// Both builtin CAS implementations provide lifetime for free, so this can
  /// be const, and readData() and getDataSize() can be implemented on top of
  /// it.
  virtual ArrayRef<char> getDataConst(ObjectHandle Node) const = 0;

  ArrayRef<char> getData(ObjectHandle Node,
                         bool RequiresNullTerminator) const final {
    // BuiltinCAS Objects are always null terminated.
    return getDataConst(Node);
  }
  uint64_t getDataSize(ObjectHandle Node) const final {
    return getDataConst(Node).size();
  }

  Error createUnknownObjectError(const CASID &ID) const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "unknown object '" + ID.toString() + "'");
  }

  Error createCorruptObjectError(const CASID &ID) const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "corrupt object '" + ID.toString() + "'");
  }

  Error createCorruptStorageError() const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "corrupt storage");
  }

  Error validate(const CASID &ID) final;
};

// FIXME: Proxy not portable. Maybe also error-prone?
constexpr StringLiteral DefaultDirProxy = "/^llvm::cas::builtin::default";
constexpr StringLiteral DefaultDir = "llvm.cas.builtin.default";

} // end namespace builtin
} // end namespace cas
} // end namespace llvm

#endif // LLVM_LIB_CAS_BUILTINCAS_H
