//===- TemplateExtras.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_TEMPLATEEXTRAS_H
#define MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_TEMPLATEEXTRAS_H

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace sparse_tensor {
namespace ir_detail {

//===----------------------------------------------------------------------===//
// These two templates are like `AsmPrinter::{,detect_}has_print_method`,
// except they detect print methods taking `raw_ostream` (not `AsmPrinter`).
template <typename T>
using has_print_method =
    decltype(std::declval<T>().print(std::declval<llvm::raw_ostream &>()));
template <typename T>
using detect_has_print_method = llvm::is_detected<has_print_method, T>;
template <typename T, typename R = void>
using enable_if_has_print_method =
    std::enable_if_t<detect_has_print_method<T>::value, R>;

/// Generic template for defining `operator<<` overloads which delegate
/// to `T::print(raw_ostream&) const`.  Note that there's already another
/// generic template which defines `operator<<(AsmPrinterT&, T const&)`
/// via delegating to `operator<<(raw_ostream&, T const&)`.
template <typename T>
inline enable_if_has_print_method<T, llvm::raw_ostream &>
operator<<(llvm::raw_ostream &os, T const &t) {
  t.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
/// Convert an enum to its underlying type.  This template is designed
/// to avoid introducing implicit conversions to other integral types,
/// and is a backport of C++23 `std::to_underlying`.
template <typename Enum>
constexpr std::underlying_type_t<Enum> to_underlying(Enum e) noexcept {
  return static_cast<std::underlying_type_t<Enum>>(e);
}

//===----------------------------------------------------------------------===//
template <typename T>
static constexpr bool IsZeroCostAbstraction =
    // These two predicates license the compiler to make several optimizations;
    // some of which are explicitly documented by the C++ standard:
    // <https://en.cppreference.com/w/cpp/types/is_trivially_copyable#Notes>
    // <https://en.cppreference.com/w/cpp/types/is_trivially_destructible#Notes>
    // However, some key optimizations aren't mentioned by the standard; e.g.,
    // that trivially-copyable enables passing-by-value, and the conjunction
    // of trivially-copyable and trivially-destructible enables passing those
    // values in registers rather than on the stack (cf.,
    // <https://www.agner.org/optimize/calling_conventions.pdf>).
    std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T> &&
    // This one helps ensure ABI compatibility (e.g., padding and alignment):
    // <https://en.cppreference.com/w/cpp/types/is_standard_layout#Notes>
    // <https://en.cppreference.com/w/cpp/language/classes#Standard-layout_class>
    // In particular, the standard mentions that passing/returning a `struct`
    // by value can sometimes introduce ABI overhead compared to using
    // `enum class`; so this assertion is attempting to avoid that.
    // <https://en.cppreference.com/w/cpp/language/enum#enum_relaxed_init_cpp17>
    std::is_standard_layout_v<T> &&
    // These two are what SmallVector uses to determine whether it can
    // use memcpy.  The commentary there mentions that it's intended to be
    // an approximation of `is_trivially_copyable`, so this may be redundant
    // with the above, but we include it just to make sure.
    llvm::is_trivially_copy_constructible<T>::value &&
    llvm::is_trivially_move_constructible<T>::value;

//===----------------------------------------------------------------------===//

} // namespace ir_detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_TEMPLATEEXTRAS_H
