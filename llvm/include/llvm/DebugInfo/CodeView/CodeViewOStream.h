//===- CodeViewOStream.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_CODEVIEWOSTREAM_H
#define LLVM_DEBUGINFO_CODEVIEW_CODEVIEWOSTREAM_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"

namespace llvm {
namespace codeview {

template <typename Writer> class CodeViewOStream {
private:
  CodeViewOStream(const CodeViewOStream &) = delete;
  CodeViewOStream &operator=(const CodeViewOStream &) = delete;

public:
  typedef typename Writer::LabelType LabelType;

public:
  explicit CodeViewOStream(Writer &W);

private:
  uint64_t size() const { return W.tell(); }

private:
  Writer &W;
};
}
}

#endif
