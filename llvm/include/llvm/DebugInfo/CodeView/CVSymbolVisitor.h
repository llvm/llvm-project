//===- CVSymbolVisitor.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_CVSYMBOLVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_CVSYMBOLVISITOR_H

#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorDelegate.h"
#include "llvm/Support/ErrorOr.h"

namespace llvm {
namespace codeview {

template <typename Derived> class CVSymbolVisitor {
public:
  CVSymbolVisitor(SymbolVisitorDelegate *Delegate) : Delegate(Delegate) {}

  bool hadError() const { return HadError; }

  template <typename T>
  bool consumeObject(ArrayRef<uint8_t> &Data, const T *&Res) {
    if (Data.size() < sizeof(*Res)) {
      HadError = true;
      return false;
    }
    Res = reinterpret_cast<const T *>(Data.data());
    Data = Data.drop_front(sizeof(*Res));
    return true;
  }

/// Actions to take on known symbols. By default, they do nothing. Visit methods
/// for member records take the FieldData by non-const reference and are
/// expected to consume the trailing bytes used by the field.
/// FIXME: Make the visitor interpret the trailing bytes so that clients don't
/// need to.
#define SYMBOL_RECORD(EnumName, EnumVal, Name)                                 \
  void visit##Name(SymbolRecordKind Kind, Name &Record) {}
#define SYMBOL_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "CVSymbolTypes.def"

  void visitSymbolRecord(const CVRecord<SymbolKind> &Record) {
    ArrayRef<uint8_t> Data = Record.content();
    auto *DerivedThis = static_cast<Derived *>(this);
    DerivedThis->visitSymbolBegin(Record.Type, Data);
    uint32_t RecordOffset = Delegate ? Delegate->getRecordOffset(Data) : 0;
    switch (Record.Type) {
    default:
      DerivedThis->visitUnknownSymbol(Record.Type, Data);
      break;
#define SYMBOL_RECORD(EnumName, EnumVal, Name)                                 \
  case EnumName: {                                                             \
    SymbolRecordKind RK = static_cast<SymbolRecordKind>(EnumName);             \
    auto ExpectedResult = Name::deserialize(RK, RecordOffset, Data);           \
    if (!ExpectedResult) {                                                     \
      consumeError(ExpectedResult.takeError());                                \
      return parseError();                                                     \
    }                                                                          \
    DerivedThis->visit##Name(Record.Type, *ExpectedResult);                    \
    break;                                                                     \
  }
#define SYMBOL_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)                \
  SYMBOL_RECORD(EnumVal, EnumVal, AliasName)
#include "CVSymbolTypes.def"
    }
    DerivedThis->visitSymbolEnd(Record.Type, Record.content());
  }

  /// Visits the symbol records in Data. Sets the error flag on parse failures.
  void visitSymbolStream(const CVSymbolArray &Symbols) {
    for (const auto &I : Symbols) {
      visitSymbolRecord(I);
      if (hadError())
        break;
    }
  }

  /// Action to take on unknown symbols. By default, they are ignored.
  void visitUnknownSymbol(SymbolKind Kind, ArrayRef<uint8_t> Data) {}

  /// Paired begin/end actions for all symbols. Receives all record data,
  /// including the fixed-length record prefix.
  void visitSymbolBegin(SymbolKind Leaf, ArrayRef<uint8_t> RecordData) {}
  void visitSymbolEnd(SymbolKind Leaf, ArrayRef<uint8_t> OriginalSymData) {}

  /// Helper for returning from a void function when the stream is corrupted.
  void parseError() { HadError = true; }

private:
  SymbolVisitorDelegate *Delegate;
  /// Whether a symbol stream parsing error was encountered.
  bool HadError = false;
};

} // end namespace codeview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_CVSYMBOLVISITOR_H
