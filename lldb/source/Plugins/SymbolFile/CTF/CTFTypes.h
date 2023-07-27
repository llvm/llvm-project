//===-- CTFTypes.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_CTF_CTFTYPES_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_CTF_CTFTYPES_H

#include "lldb/lldb-types.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

struct CTFType {
  enum Kind : uint32_t {
    eUnknown = 0,
    eInteger = 1,
    eFloat = 2,
    ePointer = 3,
    eArray = 4,
    eFunction = 5,
    eStruct = 6,
    eUnion = 7,
    eEnum = 8,
    eForward = 9,
    eTypedef = 10,
    eVolatile = 11,
    eConst = 12,
    eRestrict = 13,
    eSlice = 14,
  };

  Kind kind;
  lldb::user_id_t uid;
  llvm::StringRef name;

  CTFType(Kind kind, lldb::user_id_t uid, llvm::StringRef name)
      : kind(kind), uid(uid), name(name) {}
};

struct CTFInteger : public CTFType {
  CTFInteger(lldb::user_id_t uid, llvm::StringRef name, uint32_t bits,
             uint32_t encoding)
      : CTFType(eInteger, uid, name), bits(bits), encoding(encoding) {}

  uint32_t bits;
  uint32_t encoding;
};

struct CTFModifier : public CTFType {
protected:
  CTFModifier(Kind kind, lldb::user_id_t uid, uint32_t type)
      : CTFType(kind, uid, ""), type(type) {}

public:
  uint32_t type;
};

struct CTFPointer : public CTFModifier {
  CTFPointer(lldb::user_id_t uid, uint32_t type)
      : CTFModifier(ePointer, uid, type) {}
};

struct CTFConst : public CTFModifier {
  CTFConst(lldb::user_id_t uid, uint32_t type)
      : CTFModifier(eConst, uid, type) {}
};

struct CTFVolatile : public CTFModifier {
  CTFVolatile(lldb::user_id_t uid, uint32_t type)
      : CTFModifier(eVolatile, uid, type) {}
};

struct CTFRestrict : public CTFModifier {
  CTFRestrict(lldb::user_id_t uid, uint32_t type)
      : CTFModifier(eRestrict, uid, type) {}
};

struct CTFTypedef : public CTFType {
  CTFTypedef(lldb::user_id_t uid, llvm::StringRef name, uint32_t type)
      : CTFType(eTypedef, uid, name), type(type) {}

  uint32_t type;
};

struct CTFArray : public CTFType {
  CTFArray(lldb::user_id_t uid, llvm::StringRef name, uint32_t type,
           uint32_t index, uint32_t nelems)
      : CTFType(eArray, uid, name), type(type), index(index), nelems(nelems) {}

  uint32_t type;
  uint32_t index;
  uint32_t nelems;
};

struct CTFEnum : public CTFType {
  struct Value {
    Value(llvm::StringRef name, uint32_t value) : name(name), value(value){};
    llvm::StringRef name;
    uint32_t value;
  };

  CTFEnum(lldb::user_id_t uid, llvm::StringRef name, uint32_t nelems,
          uint32_t size, std::vector<Value> values)
      : CTFType(eEnum, uid, name), nelems(nelems), size(size),
        values(std::move(values)) {
    assert(this->values.size() == nelems);
  }

  uint32_t nelems;
  uint32_t size;
  std::vector<Value> values;
};

struct CTFFunction : public CTFType {
  CTFFunction(lldb::user_id_t uid, llvm::StringRef name, uint32_t nargs,
              uint32_t return_type, std::vector<uint32_t> args, bool variadic)
      : CTFType(eFunction, uid, name), nargs(nargs), return_type(return_type),
        args(std::move(args)), variadic(variadic) {}

  uint32_t nargs;
  uint32_t return_type;

  std::vector<uint32_t> args;
  bool variadic = false;
};

struct CTFRecord : public CTFType {
public:
  struct Field {
    Field(llvm::StringRef name, uint32_t type, uint16_t offset,
          uint16_t padding)
        : name(name), type(type), offset(offset), padding(padding) {}

    llvm::StringRef name;
    uint32_t type;
    uint16_t offset;
    uint16_t padding;
  };

  CTFRecord(Kind kind, lldb::user_id_t uid, llvm::StringRef name,
            uint32_t nfields, uint32_t size, std::vector<Field> fields)
      : CTFType(kind, uid, name), nfields(nfields), size(size),
        fields(std::move(fields)) {}

  uint32_t nfields;
  uint32_t size;
  std::vector<Field> fields;
};

struct CTFStruct : public CTFRecord {
  CTFStruct(lldb::user_id_t uid, llvm::StringRef name, uint32_t nfields,
            uint32_t size, std::vector<Field> fields)
      : CTFRecord(eStruct, uid, name, nfields, size, std::move(fields)){};
};

struct CTFUnion : public CTFRecord {
  CTFUnion(lldb::user_id_t uid, llvm::StringRef name, uint32_t nfields,
           uint32_t size, std::vector<Field> fields)
      : CTFRecord(eUnion, uid, name, nfields, size, std::move(fields)){};
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_CTF_CTFTYPES_H
