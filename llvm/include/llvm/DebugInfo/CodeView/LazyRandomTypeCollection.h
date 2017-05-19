//===- LazyRandomTypeCollection.h ---------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_LAZYRANDOMTYPECOLLECTION_H
#define LLVM_DEBUGINFO_CODEVIEW_LAZYRANDOMTYPECOLLECTION_H

#include "llvm/DebugInfo/CodeView/TypeCollection.h"
#include "llvm/DebugInfo/CodeView/TypeDatabase.h"
#include "llvm/DebugInfo/CodeView/TypeDatabaseVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

class TypeDatabase;
class TypeVisitorCallbacks;

/// \brief Provides amortized O(1) random access to a CodeView type stream.
/// Normally to access a type from a type stream, you must know its byte
/// offset into the type stream, because type records are variable-lengthed.
/// However, this is not the way we prefer to access them.  For example, given
/// a symbol record one of the fields may be the TypeIndex of the symbol's
/// type record.  Or given a type record such as an array type, there might
/// be a TypeIndex for the element type.  Sequential access is perfect when
/// we're just dumping every entry, but it's very poor for real world usage.
///
/// Type streams in PDBs contain an additional field which is a list of pairs
/// containing indices and their corresponding offsets, roughly every ~8KB of
/// record data.  This general idea need not be confined to PDBs though.  By
/// supplying such an array, the producer of a type stream can allow the
/// consumer much better access time, because the consumer can find the nearest
/// index in this array, and do a linear scan forward only from there.
///
/// LazyRandomTypeCollection implements this algorithm, but additionally goes
/// one step further by caching offsets of every record that has been visited at
/// least once.  This way, even repeated visits of the same record will never
/// require more than one linear scan.  For a type stream of N elements divided
/// into M chunks of roughly equal size, this yields a worst case lookup time
/// of O(N/M) and an amortized time of O(1).
class LazyRandomTypeCollection : public TypeCollection {
  typedef FixedStreamArray<TypeIndexOffset> PartialOffsetArray;

public:
  explicit LazyRandomTypeCollection(uint32_t RecordCountHint);
  LazyRandomTypeCollection(StringRef Data, uint32_t RecordCountHint);
  LazyRandomTypeCollection(ArrayRef<uint8_t> Data, uint32_t RecordCountHint);
  LazyRandomTypeCollection(const CVTypeArray &Types, uint32_t RecordCountHint,
                           PartialOffsetArray PartialOffsets);
  LazyRandomTypeCollection(const CVTypeArray &Types, uint32_t RecordCountHint);

  void reset(ArrayRef<uint8_t> Data);
  void reset(StringRef Data);

  CVType getType(TypeIndex Index) override;
  StringRef getTypeName(TypeIndex Index) override;
  bool contains(TypeIndex Index) override;
  uint32_t size() override;
  uint32_t capacity() override;
  Optional<TypeIndex> getFirst() override;
  Optional<TypeIndex> getNext(TypeIndex Prev) override;

private:
  const TypeDatabase &database() const { return Database; }
  Error ensureTypeExists(TypeIndex Index);

  Error visitRangeForType(TypeIndex TI);
  Error fullScanForType(TypeIndex TI);
  Error visitRange(TypeIndex Begin, uint32_t BeginOffset, TypeIndex End);
  Error visitOneRecord(TypeIndex TI, uint32_t Offset, CVType &Record);

  /// Visited records get automatically added to the type database.
  TypeDatabase Database;

  /// The type array to allow random access visitation of.
  CVTypeArray Types;

  /// The database visitor which adds new records to the database.
  TypeDatabaseVisitor DatabaseVisitor;

  /// A vector mapping type indices to type offset.  For every record that has
  /// been visited, contains the absolute offset of that record in the record
  /// array.
  std::vector<uint32_t> KnownOffsets;

  /// An array of index offsets for the given type stream, allowing log(N)
  /// lookups of a type record by index.  Similar to KnownOffsets but only
  /// contains offsets for some type indices, some of which may not have
  /// ever been visited.
  PartialOffsetArray PartialOffsets;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_LAZYRANDOMTYPECOLLECTION_H
