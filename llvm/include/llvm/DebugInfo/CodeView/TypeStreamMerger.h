//===- TypeStreamMerger.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPESTREAMMERGER_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPESTREAMMERGER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

class TypeIndex;
class TypeServerHandler;
class TypeTableBuilder;

/// Merges one type stream into another. Returns true on success.
Error mergeTypeStreams(TypeTableBuilder &DestIdStream,
                       TypeTableBuilder &DestTypeStream,
                       SmallVectorImpl<TypeIndex> &SourceToDest,
                       TypeServerHandler *Handler, const CVTypeArray &Types);

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPESTREAMMERGER_H
