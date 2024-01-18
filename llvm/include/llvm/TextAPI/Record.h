//===- llvm/TextAPI/Record.h - TAPI Record ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implements the TAPI Record Types.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_RECORD_H
#define LLVM_TEXTAPI_RECORD_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/TextAPI/Symbol.h"
#include <string>

namespace llvm {
namespace MachO {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

class RecordsSlice;

// Defines a list of linkage types.
enum class RecordLinkage : uint8_t {
  // Unknown linkage.
  Unknown = 0,

  // Local, hidden or private extern linkage.
  Internal = 1,

  // Undefined linkage, it represents usage of external interface.
  Undefined = 2,

  // Re-exported linkage, record is defined in external interface.
  Rexported = 3,

  // Exported linkage.
  Exported = 4,
};

/// Define Record. They represent API's in binaries that could be linkable
/// symbols.
class Record {
public:
  Record() = default;
  Record(StringRef Name, RecordLinkage Linkage, SymbolFlags Flags)
      : Name(Name), Linkage(Linkage), Flags(Flags) {}

  bool isWeakDefined() const {
    return (Flags & SymbolFlags::WeakDefined) == SymbolFlags::WeakDefined;
  }

  bool isWeakReferenced() const {
    return (Flags & SymbolFlags::WeakReferenced) == SymbolFlags::WeakReferenced;
  }

  bool isThreadLocalValue() const {
    return (Flags & SymbolFlags::ThreadLocalValue) ==
           SymbolFlags::ThreadLocalValue;
  }

  bool isData() const {
    return (Flags & SymbolFlags::Data) == SymbolFlags::Data;
  }

  bool isText() const {
    return (Flags & SymbolFlags::Text) == SymbolFlags::Text;
  }

  bool isInternal() const { return Linkage == RecordLinkage::Internal; }
  bool isUndefined() const { return Linkage == RecordLinkage::Undefined; }
  bool isExported() const { return Linkage >= RecordLinkage::Rexported; }
  bool isRexported() const { return Linkage == RecordLinkage::Rexported; }

  StringRef getName() const { return Name; }

protected:
  StringRef Name;
  RecordLinkage Linkage;
  SymbolFlags Flags;

  friend class RecordsSlice;
};

// Defines broadly non-objc records, categorized as variables or functions.
class GlobalRecord : public Record {
public:
  enum class Kind : uint8_t {
    Unknown = 0,
    Variable = 1,
    Function = 2,
  };

  GlobalRecord(StringRef Name, RecordLinkage Linkage, SymbolFlags Flags,
               Kind GV)
      : Record({Name, Linkage, Flags}), GV(GV) {}

  bool isFunction() const { return GV == Kind::Function; }
  bool isVariable() const { return GV == Kind::Variable; }

private:
  Kind GV;
};

// Define Objective-C instance variable records.
class ObjCIVarRecord : public Record {
public:
  ObjCIVarRecord(StringRef Name, RecordLinkage Linkage)
      : Record({Name, Linkage, SymbolFlags::Data}) {}

  static std::string createScopedName(StringRef SuperClass, StringRef IVar) {
    return (SuperClass + "." + IVar).str();
  }
};

template <typename V, typename K = StringRef,
          typename std::enable_if<std::is_base_of<Record, V>::value>::type * =
              nullptr>
using RecordMap = llvm::MapVector<K, std::unique_ptr<V>>;

// Defines Objective-C record types that have assigned methods, properties,
// instance variable (ivars) and protocols.
class ObjCContainerRecord : public Record {
public:
  ObjCContainerRecord(StringRef Name, RecordLinkage Linkage)
      : Record({Name, Linkage, SymbolFlags::Data}) {}

  ObjCIVarRecord *addObjCIVar(StringRef IVar, RecordLinkage Linkage);
  ObjCIVarRecord *findObjCIVar(StringRef IVar) const;

private:
  RecordMap<ObjCIVarRecord> IVars;
};

// Define Objective-C category types. They don't generate linkable symbols, but
// they have assigned ivars that do.
class ObjCCategoryRecord : public ObjCContainerRecord {
public:
  ObjCCategoryRecord(StringRef ClassToExtend, StringRef Name)
      : ObjCContainerRecord(Name, RecordLinkage::Unknown),
        ClassToExtend(ClassToExtend) {}

private:
  StringRef ClassToExtend;
};

// Define Objective-C Interfaces or class types.
class ObjCInterfaceRecord : public ObjCContainerRecord {
public:
  ObjCInterfaceRecord(StringRef Name, RecordLinkage Linkage,
                      bool HasEHType = false)
      : ObjCContainerRecord(Name, Linkage), HasEHType(HasEHType) {}

  bool hasExceptionAttribute() const { return HasEHType; }
  bool addObjCCategory(ObjCCategoryRecord *Record);

private:
  bool HasEHType;
  // Non-owning containers of categories that extend the class.
  llvm::MapVector<StringRef, ObjCCategoryRecord *> Categories;
};

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_RECORD_H
