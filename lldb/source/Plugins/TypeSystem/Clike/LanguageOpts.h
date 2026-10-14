//===-- LanguageOpts.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_LANGUAGEOPTS_H
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_LANGUAGEOPTS_H

#include "clang/Basic/TargetInfo.h"

#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"

#include "lldb/lldb-enumerations.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace lldb_private {
namespace clike {

/// Describes the target/language configuration that determines properties of
/// the type system, such as builtin type sizes and encodings.
class LanguageOpts {
public:
  /// Sizes (in bytes) of the target-dependent builtin types.
  struct BuiltinSizes {
    uint32_t bool_size = 0;
    uint32_t short_size = 0;
    uint32_t int_size = 0;
    uint32_t long_size = 0;
    uint32_t long_long_size = 0;
    uint32_t wchar_size = 0;
    uint32_t char16_size = 0;
    uint32_t char32_size = 0;
    uint32_t float_size = 0;
    uint32_t double_size = 0;
    uint32_t long_double_size = 0;
    uint32_t pointer_size = 0;
  };

  /// The language options for \p triple, or an error if Clang cannot describe
  /// the triple.
  static llvm::Expected<LanguageOpts> Create(llvm::Triple triple);

  const llvm::Triple &GetTriple() const { return m_triple; }
  const BuiltinSizes &GetBuiltinSizes() const { return m_builtin_sizes; }

  /// The floating point semantics for a float of the given storage size.
  const llvm::fltSemantics &
  GetFloatTypeSemantics(const size_t byte_size,
                        const lldb::Format format) const;

  /// The storage size (in bytes) of a `_BitInt(bits)` on this target.
  ///
  /// Returns std::nullopt if the bit width is invalid for the target.
  std::optional<uint64_t> GetBitIntByteSize(unsigned bits) const;

private:
  /// The target's float formats.
  struct FloatSemantics {
    const llvm::fltSemantics *half = nullptr;
    const llvm::fltSemantics *single = nullptr;
    const llvm::fltSemantics *double_ = nullptr;
    const llvm::fltSemantics *long_double = nullptr;
    const llvm::fltSemantics *float128 = nullptr;
  };

  LanguageOpts(llvm::Triple triple, std::unique_ptr<clang::TargetInfo> target,
               BuiltinSizes sizes, FloatSemantics semantics);

  llvm::Triple m_triple;
  std::unique_ptr<clang::TargetInfo> m_target;
  BuiltinSizes m_builtin_sizes;
  FloatSemantics m_semantics;
};

} // namespace clike
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_LANGUAGEOPTS_H
