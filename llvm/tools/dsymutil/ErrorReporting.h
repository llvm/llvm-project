//===- ErrorReporting.h - dsymutil error reporting  -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_DSYMUTIL_ERRORREPORTING_H
#define LLVM_TOOLS_DSYMUTIL_ERRORREPORTING_H
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace dsymutil {

inline raw_ostream &error_ostream() {
  return WithColor(errs(), HighlightColor::Error).get() << "error: ";
}

inline raw_ostream &warn_ostream() {
  return WithColor(errs(), HighlightColor::Warning).get() << "warning: ";
}

inline raw_ostream &note_ostream() {
  return WithColor(errs(), HighlightColor::Note).get() << "note: ";
}

} // namespace dsymutil
} // end namespace llvm

#endif // LLVM_TOOLS_DSYMUTIL_ERRORREPORTING_H
