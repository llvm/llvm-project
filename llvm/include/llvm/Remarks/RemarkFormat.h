//===-- llvm/Remarks/RemarkFormat.h - The format of remarks -----*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities to deal with the format of remarks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMARKS_REMARK_FORMAT_H
#define LLVM_REMARKS_REMARK_FORMAT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace remarks {

constexpr StringRef Magic("REMARKS", 7);

/// The format used for serializing/deserializing remarks.
enum class Format { Unknown, YAML };

/// Parse and validate a string for the remark format.
Expected<Format> parseFormat(StringRef FormatStr);

} // end namespace remarks
} // end namespace llvm

#endif /* LLVM_REMARKS_REMARK_FORMAT_H */
