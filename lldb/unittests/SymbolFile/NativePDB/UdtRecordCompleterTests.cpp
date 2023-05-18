//===-- UdtRecordCompleterTests.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/NativePDB/UdtRecordCompleter.h"
#include "llvm/ADT/StringExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private::npdb;
using namespace llvm;

namespace {
using Member = UdtRecordCompleter::Member;
using MemberUP = std::unique_ptr<Member>;
using Record = UdtRecordCompleter::Record;

class WrappedMember {
public:
  WrappedMember(const Member &obj) : m_obj(obj) {}

private:
  const Member &m_obj;

  friend bool operator==(const WrappedMember &lhs, const WrappedMember &rhs) {
    return lhs.m_obj.kind == rhs.m_obj.kind &&
           lhs.m_obj.name == rhs.m_obj.name &&
           lhs.m_obj.bit_offset == rhs.m_obj.bit_offset &&
           lhs.m_obj.bit_size == rhs.m_obj.bit_size &&
           lhs.m_obj.base_offset == rhs.m_obj.base_offset &&
           std::equal(lhs.m_obj.fields.begin(), lhs.m_obj.fields.end(),
                      rhs.m_obj.fields.begin(), rhs.m_obj.fields.end(),
                      [](const MemberUP &lhs, const MemberUP &rhs) {
                        return WrappedMember(*lhs) == WrappedMember(*rhs);
                      });
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const WrappedMember &w) {
    os << llvm::formatv("Member{.kind={0}, .name=\"{1}\", .bit_offset={2}, "
                        ".bit_size={3}, .base_offset={4}, .fields=[",
                        w.m_obj.kind, w.m_obj.name, w.m_obj.bit_offset,
                        w.m_obj.bit_size, w.m_obj.base_offset);
    llvm::ListSeparator sep;
    for (auto &f : w.m_obj.fields)
      os << sep << WrappedMember(*f);
    return os << "]}";
  }
};

class WrappedRecord {
public:
  WrappedRecord(const Record &obj) : m_obj(obj) {}

private:
  const Record &m_obj;

  friend bool operator==(const WrappedRecord &lhs, const WrappedRecord &rhs) {
    return lhs.m_obj.start_offset == rhs.m_obj.start_offset &&
           std::equal(
               lhs.m_obj.record.fields.begin(), lhs.m_obj.record.fields.end(),
               rhs.m_obj.record.fields.begin(), rhs.m_obj.record.fields.end(),
               [](const MemberUP &lhs, const MemberUP &rhs) {
                 return WrappedMember(*lhs) == WrappedMember(*rhs);
               });
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const WrappedRecord &w) {
    os << llvm::formatv("Record{.start_offset={0}, .record.fields=[",
                        w.m_obj.start_offset);
    llvm::ListSeparator sep;
    for (const MemberUP &f : w.m_obj.record.fields)
      os << sep << WrappedMember(*f);
    return os << "]}";
  }
};

class UdtRecordCompleterRecordTests : public testing::Test {
protected:
  Record record;

public:
  void SetKind(Member::Kind kind) { record.record.kind = kind; }
  void CollectMember(StringRef name, uint64_t byte_offset, uint64_t byte_size) {
    record.CollectMember(name, byte_offset * 8, byte_size * 8,
                         clang::QualType(), lldb::eAccessPublic, 0);
  }
  void ConstructRecord() { record.ConstructRecord(); }
};
Member *AddField(Member *member, StringRef name, uint64_t byte_offset,
                 uint64_t byte_size, Member::Kind kind,
                 uint64_t base_offset = 0) {
  auto field =
      std::make_unique<Member>(name, byte_offset * 8, byte_size * 8,
                               clang::QualType(), lldb::eAccessPublic, 0);
  field->kind = kind;
  field->base_offset = base_offset;
  member->fields.push_back(std::move(field));
  return member->fields.back().get();
}
} // namespace

TEST_F(UdtRecordCompleterRecordTests, TestAnonymousUnionInStruct) {
  SetKind(Member::Kind::Struct);
  CollectMember("m1", 0, 4);
  CollectMember("m2", 0, 4);
  CollectMember("m3", 0, 1);
  CollectMember("m4", 0, 8);
  ConstructRecord();

  // struct {
  //   union {
  //       m1;
  //       m2;
  //       m3;
  //       m4;
  //   };
  // };
  Record record;
  record.start_offset = 0;
  Member *u = AddField(&record.record, "", 0, 0, Member::Union);
  AddField(u, "m1", 0, 4, Member::Field);
  AddField(u, "m2", 0, 4, Member::Field);
  AddField(u, "m3", 0, 1, Member::Field);
  AddField(u, "m4", 0, 8, Member::Field);
  EXPECT_EQ(WrappedRecord(this->record), WrappedRecord(record));
}

TEST_F(UdtRecordCompleterRecordTests, TestAnonymousUnionInUnion) {
  SetKind(Member::Kind::Union);
  CollectMember("m1", 0, 4);
  CollectMember("m2", 0, 4);
  CollectMember("m3", 0, 1);
  CollectMember("m4", 0, 8);
  ConstructRecord();

  // union {
  //   m1;
  //   m2;
  //   m3;
  //   m4;
  // };
  Record record;
  record.start_offset = 0;
  AddField(&record.record, "m1", 0, 4, Member::Field);
  AddField(&record.record, "m2", 0, 4, Member::Field);
  AddField(&record.record, "m3", 0, 1, Member::Field);
  AddField(&record.record, "m4", 0, 8, Member::Field);
  EXPECT_EQ(WrappedRecord(this->record), WrappedRecord(record));
}

TEST_F(UdtRecordCompleterRecordTests, TestAnonymousStructInUnion) {
  SetKind(Member::Kind::Union);
  CollectMember("m1", 0, 4);
  CollectMember("m2", 4, 4);
  CollectMember("m3", 8, 1);
  ConstructRecord();

  // union {
  //   struct {
  //     m1;
  //     m2;
  //     m3;
  //   };
  // };
  Record record;
  record.start_offset = 0;
  Member *s = AddField(&record.record, "", 0, 0, Member::Kind::Struct);
  AddField(s, "m1", 0, 4, Member::Field);
  AddField(s, "m2", 4, 4, Member::Field);
  AddField(s, "m3", 8, 1, Member::Field);
  EXPECT_EQ(WrappedRecord(this->record), WrappedRecord(record));
}

TEST_F(UdtRecordCompleterRecordTests, TestNestedUnionStructInStruct) {
  SetKind(Member::Kind::Struct);
  CollectMember("m1", 0, 4);
  CollectMember("m2", 0, 2);
  CollectMember("m3", 0, 2);
  CollectMember("m4", 2, 4);
  CollectMember("m5", 3, 2);
  ConstructRecord();

  // struct {
  //   union {
  //       m1;
  //       struct {
  //           m2;
  //           m5;
  //       };
  //       struct {
  //           m3;
  //           m4;
  //       };
  //   };
  // };
  Record record;
  record.start_offset = 0;
  Member *u = AddField(&record.record, "", 0, 0, Member::Union);
  AddField(u, "m1", 0, 4, Member::Field);
  Member *s1 = AddField(u, "", 0, 0, Member::Struct);
  Member *s2 = AddField(u, "", 0, 0, Member::Struct);
  AddField(s1, "m2", 0, 2, Member::Field);
  AddField(s1, "m5", 3, 2, Member::Field);
  AddField(s2, "m3", 0, 2, Member::Field);
  AddField(s2, "m4", 2, 4, Member::Field);
  EXPECT_EQ(WrappedRecord(this->record), WrappedRecord(record));
}

TEST_F(UdtRecordCompleterRecordTests, TestNestedUnionStructInUnion) {
  SetKind(Member::Kind::Union);
  CollectMember("m1", 0, 4);
  CollectMember("m2", 0, 2);
  CollectMember("m3", 0, 2);
  CollectMember("m4", 2, 4);
  CollectMember("m5", 3, 2);
  ConstructRecord();

  // union {
  //   m1;
  //   struct {
  //       m2;
  //       m5;
  //   };
  //   struct {
  //       m3;
  //       m4;
  //   };
  // };
  Record record;
  record.start_offset = 0;
  AddField(&record.record, "m1", 0, 4, Member::Field);
  Member *s1 = AddField(&record.record, "", 0, 0, Member::Struct);
  Member *s2 = AddField(&record.record, "", 0, 0, Member::Struct);
  AddField(s1, "m2", 0, 2, Member::Field);
  AddField(s1, "m5", 3, 2, Member::Field);
  AddField(s2, "m3", 0, 2, Member::Field);
  AddField(s2, "m4", 2, 4, Member::Field);
  EXPECT_EQ(WrappedRecord(this->record), WrappedRecord(record));
}
