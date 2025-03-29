//===- FormatCommon.h - Formatters for common LLVM types --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FORMATCOMMON_H
#define LLVM_SUPPORT_FORMATCOMMON_H

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadicDetails.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
enum class AlignStyle { Left, Center, Right };

/// Helper class to format to a \p Width wide field, with alignment \p Where
/// within that field.
struct FmtAlign {
  support::detail::format_adapter &Adapter;
  AlignStyle Where;
  unsigned Width;
  char Fill;

  FmtAlign(support::detail::format_adapter &Adapter, AlignStyle Where,
           unsigned Width, char Fill = ' ')
      : Adapter(Adapter), Where(Where), Width(Width), Fill(Fill) {}

  void format(raw_ostream &S, StringRef Options) {
    // If we don't need to align, we can format straight into the underlying
    // stream.  Otherwise we have to go through an intermediate stream first
    // in order to calculate how long the output is so we can align it.
    // TODO: Make the format method return the number of bytes written, that
    // way we can also skip the intermediate stream for left-aligned output.
    if (Width == 0) {
      Adapter.format(S, Options);
      return;
    }
    SmallString<64> Item;
    raw_svector_ostream Stream(Item);

    Adapter.format(Stream, Options);
    if (Width <= Item.size()) {
      S << Item;
      return;
    }

    unsigned PadAmount = Width - static_cast<unsigned>(Item.size());
    switch (Where) {
    case AlignStyle::Left:
      S << Item;
      fill(S, PadAmount);
      break;
    case AlignStyle::Center: {
      unsigned X = PadAmount / 2;
      fill(S, X);
      S << Item;
      fill(S, PadAmount - X);
      break;
    }
    default:
      fill(S, PadAmount);
      S << Item;
      break;
    }
  }

private:
  void fill(llvm::raw_ostream &S, unsigned Count) {
    for (unsigned I = 0; I < Count; ++I)
      S << Fill;
  }
};
}

#endif
