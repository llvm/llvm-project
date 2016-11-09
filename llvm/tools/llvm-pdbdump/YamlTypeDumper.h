//===- YamlTypeDumper.h --------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_YAMLTYPEDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_YAMLTYPEDUMPER_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace pdb {
namespace yaml {
struct SerializationContext;
}
}
namespace codeview {
namespace yaml {
class YamlTypeDumperCallbacks : public TypeVisitorCallbacks {
public:
  YamlTypeDumperCallbacks(llvm::yaml::IO &IO,
                          llvm::pdb::yaml::SerializationContext &Context)
      : YamlIO(IO), Context(Context) {}

  virtual Error visitTypeBegin(CVType &Record) override;
  virtual Error visitMemberBegin(CVMemberRecord &Record) override;

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  Error visitKnownRecord(CVRecord<TypeLeafKind> &CVR, Name##Record &Record)    \
      override {                                                               \
    visitKnownRecordImpl(#Name, CVR, Record);                                  \
    return Error::success();                                                   \
  }
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownMember(CVMemberRecord &CVR, Name##Record &Record) override { \
    visitKnownMemberImpl(#Name, Record);                                       \
    return Error::success();                                                   \
  }
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

private:
  template <typename T> void visitKnownMemberImpl(const char *Name, T &Record) {
    YamlIO.mapRequired(Name, Record);
  }

  template <typename T>
  void visitKnownRecordImpl(const char *Name, CVType &Type, T &Record) {
    YamlIO.mapRequired(Name, Record);
  }

  void visitKnownRecordImpl(const char *Name, CVType &CVR,
                            FieldListRecord &FieldList);

  llvm::yaml::IO &YamlIO;
  llvm::pdb::yaml::SerializationContext &Context;
};
}
}
namespace pdb {
namespace yaml {
struct SerializationContext;
}
}
}

namespace llvm {
namespace yaml {

template <> struct ScalarTraits<APSInt> {
  static void output(const APSInt &S, void *, llvm::raw_ostream &OS);
  static StringRef input(StringRef Scalar, void *Ctx, APSInt &S);
  static bool mustQuote(StringRef Scalar);
};

template <> struct ScalarTraits<codeview::TypeIndex> {
  static void output(const codeview::TypeIndex &S, void *,
                     llvm::raw_ostream &OS);
  static StringRef input(StringRef Scalar, void *Ctx, codeview::TypeIndex &S);
  static bool mustQuote(StringRef Scalar);
};

template <> struct MappingTraits<codeview::MemberPointerInfo> {
  static void mapping(IO &IO, codeview::MemberPointerInfo &Obj);
};

template <>
struct MappingContextTraits<codeview::CVType, pdb::yaml::SerializationContext> {
  static void mapping(IO &IO, codeview::CVType &Obj,
                      pdb::yaml::SerializationContext &Context);
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
