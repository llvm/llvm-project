//===- PdbYAML.h ---------------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_CODEVIEWYAML_H
#define LLVM_TOOLS_LLVMPDBDUMP_CODEVIEWYAML_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace codeview {
namespace yaml {
class YamlTypeDumperCallbacks : public TypeVisitorCallbacks {
public:
  YamlTypeDumperCallbacks(llvm::yaml::IO &IO) : YamlIO(IO) {}

  virtual Expected<TypeLeafKind>
  visitTypeBegin(const CVRecord<TypeLeafKind> &Record) override;

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  Error visitKnownRecord(const CVRecord<TypeLeafKind> &CVR,                    \
                         Name##Record &Record) override {                      \
    visitKnownRecordImpl(#Name, CVR, Record);                                  \
    return Error::success();                                                   \
  }
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  TYPE_RECORD(EnumName, EnumVal, Name)
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

private:
  template <typename T>
  void visitKnownRecordImpl(const char *Name, const CVType &Type, T &Record) {
    YamlIO.mapRequired(Name, Record);
  }

  void visitKnownRecordImpl(const char *Name, const CVType &Type,
                            FieldListRecord &FieldList);

  llvm::yaml::IO &YamlIO;
};
}
}
}

namespace llvm {
namespace yaml {
template <> struct MappingTraits<codeview::MemberPointerInfo> {
  static void mapping(IO &IO, codeview::MemberPointerInfo &Obj);
};

template <> struct MappingTraits<codeview::CVType> {
  static void mapping(IO &IO, codeview::CVType &Obj);
};

template <> struct ScalarEnumerationTraits<codeview::TypeLeafKind> {
  static void enumeration(IO &io, codeview::TypeLeafKind &Value);
};

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  template <> struct MappingTraits<codeview::Name##Record> {                   \
    static void mapping(IO &IO, codeview::Name##Record &Obj);                  \
  };
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  TYPE_RECORD(EnumName, EnumVal, Name)
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"
}
}

#endif
