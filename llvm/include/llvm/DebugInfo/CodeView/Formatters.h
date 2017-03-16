//===- Formatters.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_FORMATTERS_H
#define LLVM_DEBUGINFO_CODEVIEW_FORMATTERS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatAdapters.h"

namespace llvm {
namespace codeview {
namespace detail {
class GuidAdapter final : public llvm::FormatAdapter<ArrayRef<uint8_t>> {
  ArrayRef<uint8_t> Guid;

public:
  explicit GuidAdapter(ArrayRef<uint8_t> Guid);
  explicit GuidAdapter(StringRef Guid);
  void format(llvm::raw_ostream &Stream, StringRef Style);
};
}

inline detail::GuidAdapter fmt_guid(StringRef Item) {
  return detail::GuidAdapter(Item);
}

inline detail::GuidAdapter fmt_guid(ArrayRef<uint8_t> Item) {
  return detail::GuidAdapter(Item);
}
}
}

#endif
