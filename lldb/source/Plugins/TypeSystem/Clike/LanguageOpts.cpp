//===-- LanguageOpts.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LanguageOpts.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorExtras.h"
#include "llvm/Support/MathExtras.h"

#include <memory>
#include <utility>

using namespace lldb_private;
using namespace lldb_private::clike;

/// Build a TargetInfo for \p triple, or null if Clang doesn't know the target.
static std::unique_ptr<clang::TargetInfo>
CreateTargetInfo(clang::TargetOptions target_opts) {
  clang::DiagnosticOptions diag_opts;
  clang::DiagnosticsEngine diags(
      llvm::makeIntrusiveRefCnt<clang::DiagnosticIDs>(), diag_opts,
      new clang::IgnoringDiagConsumer());

  return std::unique_ptr<clang::TargetInfo>(
      clang::TargetInfo::CreateTargetInfo(diags, target_opts));
}

LanguageOpts::LanguageOpts(llvm::Triple triple,
                           std::unique_ptr<clang::TargetInfo> target,
                           BuiltinSizes sizes, FloatSemantics semantics)
    : m_triple(std::move(triple)), m_target(std::move(target)),
      m_builtin_sizes(sizes), m_semantics(semantics) {}

llvm::Expected<LanguageOpts> LanguageOpts::Create(llvm::Triple triple) {
  clang::TargetOptions target_opts;
  target_opts.Triple = triple.str();
  std::unique_ptr<clang::TargetInfo> target = CreateTargetInfo(target_opts);
  if (!target)
    return llvm::createStringErrorV("Unknown Clang target: '{0}'",
                                    triple.str());

  auto bytes = [](unsigned bits) -> uint32_t { return bits / 8; };
  BuiltinSizes sizes;
  sizes.bool_size = bytes(target->getBoolWidth());
  sizes.short_size = bytes(target->getShortWidth());
  sizes.int_size = bytes(target->getIntWidth());
  sizes.long_size = bytes(target->getLongWidth());
  sizes.long_long_size = bytes(target->getLongLongWidth());
  sizes.wchar_size = bytes(target->getWCharWidth());
  sizes.char16_size = bytes(target->getChar16Width());
  sizes.char32_size = bytes(target->getChar32Width());
  sizes.float_size = bytes(target->getFloatWidth());
  sizes.double_size = bytes(target->getDoubleWidth());
  sizes.long_double_size = bytes(target->getLongDoubleWidth());
  sizes.pointer_size = bytes(target->getPointerWidth(clang::LangAS::Default));

  FloatSemantics semantics;
  semantics.half = &target->getHalfFormat();
  semantics.single = &target->getFloatFormat();
  semantics.double_ = &target->getDoubleFormat();
  semantics.long_double = &target->getLongDoubleFormat();
  semantics.float128 = &target->getFloat128Format();

  return LanguageOpts(std::move(triple), std::move(target), sizes, semantics);
}

const llvm::fltSemantics &
LanguageOpts::GetFloatTypeSemantics(const size_t byte_size,
                                    const lldb::Format format) const {
  const size_t bit_size = byte_size * 8;
  if (byte_size == m_builtin_sizes.float_size)
    return *m_semantics.single;
  if (byte_size == m_builtin_sizes.double_size)
    return *m_semantics.double_;
  if (format == lldb::eFormatFloat128 && bit_size == 128)
    return *m_semantics.float128;
  if (byte_size == m_builtin_sizes.long_double_size ||
      bit_size == llvm::APFloat::semanticsSizeInBits(*m_semantics.long_double))
    return *m_semantics.long_double;
  if (bit_size == 16)
    return *m_semantics.half;
  if (bit_size == 128)
    return *m_semantics.float128;
  return llvm::APFloat::Bogus();
}

std::optional<uint64_t> LanguageOpts::GetBitIntByteSize(unsigned bits) const {
  if (bits == 0)
    return std::nullopt;

  if (bits > m_target->getMaxBitIntWidth())
    return std::nullopt;

  // clang lays out a `_BitInt` by rounding its width up to its ABI alignment.
  unsigned align_bits = m_target->getBitIntAlign(bits);
  uint64_t size_bits = llvm::alignTo(bits, align_bits);
  return size_bits / 8;
}
