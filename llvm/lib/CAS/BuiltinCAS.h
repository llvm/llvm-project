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
#include "llvm/CAS/CASDB.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Error.h"
#include <cstddef>

namespace llvm {
namespace cas {
namespace builtin {

/// Dereference the inner value in \p E, adding an Optional to the outside.
/// Useful for stripping an inner Optional in return chaining.
///
/// \code
/// Expected<Optional<SomeType>> f1(...);
///
/// Expected<SomeType> f2(...) {
///   if (Optional<Expected<NoneType>> E = transpose(f1()))
///     return std::move(*E);
///
///   // Deal with None...
/// }
/// \endcode
///
/// FIXME: Needs tests. Should be moved to Error.h.
template <class T>
inline Optional<Expected<T>> transpose(Expected<Optional<T>> E) {
  if (!E)
    return Expected<T>(E.takeError());
  if (*E)
    return Expected<T>(std::move(**E));
  return None;
}

/// Dereference the inner value in \p E, generating an error on failure.
///
/// \code
/// Expected<Optional<SomeType>> f1(...);
///
/// Expected<SomeType> f2(...) {
///   if (Optional<Expected<NoneType>> E = dereferenceValue(f1()))
///     return std::move(*E);
///
///   // Deal with None...
/// }
/// \endcode
///
/// FIXME: Needs tests. Should be moved to Error.h.
template <class T>
inline Expected<T> dereferenceValue(Expected<Optional<T>> E,
                                    function_ref<Error()> OnNone) {
  if (Optional<Expected<T>> MaybeExpected = transpose(std::move(E)))
    return std::move(*MaybeExpected);
  return OnNone();
}

/// If \p E and \c *E, move \c **E into \p Sink.
///
/// Enables expected and optional chaining in one statement:
///
/// \code
/// Expected<Optional<Type1>> f1(...);
///
/// Expected<Optional<Type2>> f2(...) {
///   SomeType V;
///   if (Optional<Expected<NoneType>> E = moveValueInto(f1(), V))
///     return std::move(*E);
///
///   // Deal with value...
/// }
/// \endcode
///
/// FIXME: Needs tests. Should be moved to Error.h.
template <class T, class SinkT>
inline Optional<Expected<NoneType>>
moveValueInto(Expected<Optional<T>> ExpectedOptional, SinkT &Sink) {
  if (!ExpectedOptional)
    return Expected<NoneType>(ExpectedOptional.takeError());
  if (!*ExpectedOptional)
    return Expected<NoneType>(None);
  Sink = std::move(*ExpectedOptional);
  return None;
}

/// FIXME: Should we switch to using ArrayRef<uint8_t>, or add a new
/// name to arrayRefFromStringRef that works with 'char'?
inline ArrayRef<char> toArrayRef(StringRef Data) {
  return arrayRefFromStringRef<char>(Data);
}

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
/// Note that the cost is linear in the number of objects for the builtin CAS
/// and embedded action cache, since we're using internal offsets and/or
/// pointers as an optimization.
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

class BuiltinCAS : public CASDB {
  void printIDImpl(raw_ostream &OS, const CASID &ID) const final;

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

  Expected<CASID> parseID(StringRef Reference) final;

  virtual Expected<CASID> parseIDImpl(ArrayRef<uint8_t> Hash) = 0;

  Expected<ObjectHandle> store(ArrayRef<ObjectRef> Refs,
                               ArrayRef<char> Data) final;
  virtual Expected<ObjectHandle> storeImpl(ArrayRef<uint8_t> ComputedHash,
                                           ArrayRef<ObjectRef> Refs,
                                           ArrayRef<char> Data) = 0;

  Expected<ObjectHandle>
  storeFromOpenFileImpl(sys::fs::file_t FD,
                        Optional<sys::fs::file_status> Status) override;
  virtual Expected<ObjectHandle>
  storeFromNullTerminatedRegion(ArrayRef<uint8_t> ComputedHash,
                                sys::fs::mapped_file_region Map) {
    return storeImpl(ComputedHash, None, makeArrayRef(Map.data(), Map.size()));
  }

  /// Both builtin CAS implementations provide lifetime for free, so this can
  /// be const, and readData() and getDataSize() can be implemented on top of
  /// it.
  virtual ArrayRef<char> getDataConst(ObjectHandle Node) const = 0;

  ArrayRef<char> getDataImpl(ObjectHandle Node, bool NullTerminate) final {
    return getDataConst(Node);
  }
  uint64_t getDataSize(ObjectHandle Node) const final {
    return getDataConst(Node).size();
  }
  uint64_t readDataImpl(ObjectHandle Node, raw_ostream &OS, uint64_t Offset,
                        uint64_t MaxBytes) const final;

  Error createUnknownObjectError(CASID ID) const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "unknown object '" + ID.toString() + "'");
  }

  Error createCorruptObjectError(CASID ID) const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "corrupt object '" + ID.toString() + "'");
  }

  Error createCorruptStorageError() const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "corrupt storage");
  }

  /// FIXME: This should not use Error.
  Error createResultCacheMissError(CASID Input) const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "no result for '" + Input.toString() + "'");
  }

  Error createResultCachePoisonedError(CASID Input, CASID Output,
                                       CASID ExistingOutput) const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "cache poisoned for '" + Input.toString() +
                                 "' (new='" + Output.toString() +
                                 "' vs. existing '" +
                                 ExistingOutput.toString() + "')");
  }

  Error createResultCacheCorruptError(CASID Input) const {
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "result cache corrupt for '" + Input.toString() +
                                 "'");
  }

  Error validate(const CASID &ID) final;
};

} // end namespace builtin
} // end namespace cas
} // end namespace llvm

#endif // LLVM_LIB_CAS_BUILTINCAS_H
