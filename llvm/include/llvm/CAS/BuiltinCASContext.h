//===- BuiltinCASContext.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_BUILTINCASCONTEXT_H
#define LLVM_CAS_BUILTINCASCONTEXT_H

#include "llvm/CAS/CASID.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Error.h"

namespace llvm::cas::builtin {

/// Current hash type for the builtin CAS.
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

/// CASContext for LLVM builtin CAS using BLAKE3 hash type.
class BuiltinCASContext : public CASContext {
  void printIDImpl(raw_ostream &OS, const CASID &ID) const final;
  void anchor() override;

public:
  /// Get the name of the hash for any table identifiers.
  ///
  /// FIXME: This should be configurable via an enum, with at the following
  /// values:
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

  static Expected<HashType> parseID(StringRef PrintedDigest);
  static void printID(ArrayRef<uint8_t> Digest, raw_ostream &OS);
};

} // namespace llvm::cas::builtin

#endif // LLVM_CAS_BUILTINCASCONTEXT_H
