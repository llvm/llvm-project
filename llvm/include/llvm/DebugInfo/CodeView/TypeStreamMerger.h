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

/// \brief Merge one set of type records into another.  This method assumes
/// that all records are type records, and there are no Id records present.
///
/// \param Dest The table to store the re-written type records into.
///
/// \param SourceToDest A vector, indexed by the TypeIndex in the source
/// type stream, that contains the index of the corresponding type record
/// in the destination stream.
///
/// \param Handler (optional) If non-null, an interface that gets invoked
/// to handle type server records.
///
/// \param Types The collection of types to merge in.
///
/// \returns Error::success() if the operation succeeded, otherwise an
/// appropriate error code.
Error mergeTypeRecords(TypeTableBuilder &Dest,
                       SmallVectorImpl<TypeIndex> &SourceToDest,
                       TypeServerHandler *Handler, const CVTypeArray &Types);

/// \brief Merge one set of id records into another.  This method assumes
/// that all records are id records, and there are no Type records present.
/// However, since Id records can refer back to Type records, this method
/// assumes that the referenced type records have also been merged into
/// another type stream (for example using the above method), and accepts
/// the mapping from source to dest for that stream so that it can re-write
/// the type record mappings accordingly.
///
/// \param Dest The table to store the re-written id records into.
///
/// \param Types The mapping to use for the type records that these id
/// records refer to.
///
/// \param SourceToDest A vector, indexed by the TypeIndex in the source
/// id stream, that contains the index of the corresponding id record
/// in the destination stream.
///
/// \param Ids The collection of id records to merge in.
///
/// \returns Error::success() if the operation succeeded, otherwise an
/// appropriate error code.
Error mergeIdRecords(TypeTableBuilder &Dest, ArrayRef<TypeIndex> Types,
                     SmallVectorImpl<TypeIndex> &SourceToDest,
  const CVTypeArray &Ids);

/// \brief Merge a unified set of type and id records, splitting them into
/// separate output streams.
///
/// \param DestIds The table to store the re-written id records into.
///
/// \param DestTypes the table to store the re-written type records into.
///
/// \param SourceToDest A vector, indexed by the TypeIndex in the source
/// id stream, that contains the index of the corresponding id record
/// in the destination stream.
///
/// \param Handler (optional) If non-null, an interface that gets invoked
/// to handle type server records.
///
/// \param IdsAndTypes The collection of id records to merge in.
///
/// \returns Error::success() if the operation succeeded, otherwise an
/// appropriate error code.
Error mergeTypeAndIdRecords(TypeTableBuilder &DestIds,
                            TypeTableBuilder &DestTypes,
                            SmallVectorImpl<TypeIndex> &SourceToDest,
                            TypeServerHandler *Handler,
  const CVTypeArray &IdsAndTypes);

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_TYPESTREAMMERGER_H
