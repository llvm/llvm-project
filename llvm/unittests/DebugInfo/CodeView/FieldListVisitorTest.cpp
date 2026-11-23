//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/ContinuationRecordBuilder.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::codeview;

TEST(FieldListVisitor, StreamVisit) {
  ContinuationRecordBuilder CRB;
  CRB.begin(ContinuationRecordKind::FieldList);

  DataMemberRecord D(MemberAttributes(MemberAccess::Public), TypeIndex::Int32(),
                     42, "foo");
  CRB.writeMemberType(D);
  BaseClassRecord B(MemberAccess::Private, TypeIndex::None(), 0);
  CRB.writeMemberType(B);

  struct Visitor : public TypeVisitorCallbacks {
    Visitor(DataMemberRecord &DataMember, BaseClassRecord &Base)
        : DataMember(DataMember), Base(Base) {}

    Error visitKnownMember(CVMemberRecord &CVM,
                           BaseClassRecord &Record) override {
      EXPECT_FALSE(SeenBase);
      SeenBase = true;
      EXPECT_EQ(Base.Kind, Record.Kind);
      EXPECT_EQ(Base.Attrs, Record.Attrs);
      EXPECT_EQ(Base.Offset, Record.Offset);
      EXPECT_EQ(Base.Type, Record.Type);
      return Error::success();
    }

    Error visitKnownMember(CVMemberRecord &CVM,
                           DataMemberRecord &Record) override {
      EXPECT_FALSE(SeenDataMember);
      SeenDataMember = true;
      EXPECT_EQ(DataMember.Kind, Record.Kind);
      EXPECT_EQ(DataMember.Attrs, Record.Attrs);
      EXPECT_EQ(DataMember.FieldOffset, Record.FieldOffset);
      EXPECT_EQ(DataMember.Type, Record.Type);
      EXPECT_EQ(DataMember.Name, Record.Name);
      return Error::success();
    }

    DataMemberRecord &DataMember;
    bool SeenDataMember = false;
    BaseClassRecord &Base;
    bool SeenBase = false;
  };

  std::vector<CVType> Records =
      CRB.end(TypeIndex(TypeIndex::FirstNonSimpleIndex));
  ASSERT_EQ(Records.size(), size_t(1));
  Visitor Vis(D, B);
  Error EC = visitMemberRecordStream(Records.front().data(), Vis);
  ASSERT_THAT_ERROR(std::move(EC), Succeeded());
  ASSERT_TRUE(Vis.SeenDataMember);
  ASSERT_TRUE(Vis.SeenBase);
}

TEST(FieldListVisitor, VisitBadRecord) {
  ContinuationRecordBuilder CRB;
  CRB.begin(ContinuationRecordKind::FieldList);

  DataMemberRecord D(MemberAttributes(MemberAccess::Public), TypeIndex::Int32(),
                     42, "foo");
  CRB.writeMemberType(D);

  struct Visitor : public TypeVisitorCallbacks {
    Error visitKnownMember(CVMemberRecord &CVM,
                           DataMemberRecord &Record) override {
      EXPECT_TRUE(false); // Should never get here.
      return Error::success();
    }
  };

  std::vector<CVType> Records =
      CRB.end(TypeIndex(TypeIndex::FirstNonSimpleIndex));
  ASSERT_EQ(Records.size(), size_t(1));
  // Remove some data from the back to get the member deserializer to error out.
  Records.front().RecordData = Records.front().RecordData.drop_back(4);
  Visitor Vis;
  Error EC = visitMemberRecordStream(Records.front().data(), Vis);
  ASSERT_THAT_ERROR(std::move(EC), Failed());
}

TEST(FieldListVisitor, VisitEarlyExit) {
  ContinuationRecordBuilder CRB;
  CRB.begin(ContinuationRecordKind::FieldList);

  DataMemberRecord D(MemberAttributes(MemberAccess::Public), TypeIndex::Int32(),
                     42, "foo");
  CRB.writeMemberType(D);
  BaseClassRecord B(MemberAccess::Private, TypeIndex::None(), 0);
  CRB.writeMemberType(B);

  struct Visitor : public TypeVisitorCallbacks {
    Visitor(DataMemberRecord &DataMember) : DataMember(DataMember) {}

    Error visitKnownMember(CVMemberRecord &CVM,
                           BaseClassRecord &Record) override {
      EXPECT_FALSE(true); // Should never get here.
      return Error::success();
    }

    Error visitKnownMember(CVMemberRecord &CVM,
                           DataMemberRecord &Record) override {
      EXPECT_FALSE(SeenDataMember);
      SeenDataMember = true;
      EXPECT_EQ(DataMember.Kind, Record.Kind);
      EXPECT_EQ(DataMember.Attrs, Record.Attrs);
      EXPECT_EQ(DataMember.FieldOffset, Record.FieldOffset);
      EXPECT_EQ(DataMember.Type, Record.Type);
      EXPECT_EQ(DataMember.Name, Record.Name);
      return createStringError("stop it");
    }

    DataMemberRecord &DataMember;
    bool SeenDataMember = false;
  };

  std::vector<CVType> Records =
      CRB.end(TypeIndex(TypeIndex::FirstNonSimpleIndex));
  ASSERT_EQ(Records.size(), size_t(1));
  Visitor Vis(D);
  Error EC = visitMemberRecordStream(Records.front().data(), Vis);
  ASSERT_THAT_ERROR(std::move(EC), FailedWithMessage("stop it"));
  ASSERT_TRUE(Vis.SeenDataMember);
}
