//===- llvm/unittest/DebugInfo/CodeView/TypeHashingTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeHashing.h"
#include "llvm/DebugInfo/CodeView/AppendingTypeTableBuilder.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::codeview;

static TypeIndex createPointerRecord(AppendingTypeTableBuilder &Builder,
                                     TypeIndex TI) {
  PointerRecord PR(TypeRecordKind::Pointer);
  PR.setAttrs(PointerKind::Near32, PointerMode::Pointer, PointerOptions::None,
              4);
  PR.ReferentType = TI;
  return Builder.writeLeafType(PR);
}

static TypeIndex createArgListRecord(AppendingTypeTableBuilder &Builder,
                                     TypeIndex Q, TypeIndex R) {
  ArgListRecord AR(TypeRecordKind::ArgList);
  AR.ArgIndices.push_back(Q);
  AR.ArgIndices.push_back(R);
  return Builder.writeLeafType(AR);
}

static TypeIndex createProcedureRecord(AppendingTypeTableBuilder &Builder,
                                       uint32_t ParamCount, TypeIndex Return,
                                       TypeIndex ArgList) {
  ProcedureRecord PR(TypeRecordKind::Procedure);
  PR.ArgumentList = ArgList;
  PR.CallConv = CallingConvention::NearC;
  PR.Options = FunctionOptions::None;
  PR.ParameterCount = ParamCount;
  PR.ReturnType = Return;
  return Builder.writeLeafType(PR);
}

static ArrayRef<uint8_t> hash_of(ArrayRef<GloballyHashedType> Hashes,
                                 TypeIndex TI) {
  return Hashes[TI.toArrayIndex()].Hash;
}

static void verifyHashUniqueness(ArrayRef<GloballyHashedType> Hashes) {
  assert(!Hashes.empty());

  for (size_t I = 0; I < Hashes.size() - 1; ++I) {
    for (size_t J = I + 1; J < Hashes.size(); ++J) {
      EXPECT_NE(Hashes[I].Hash, Hashes[J].Hash);
    }
  }
}

TEST(TypeHashingTest, ContentHash) {
  SimpleTypeSerializer Serializer;

  TypeIndex CharStar(SimpleTypeKind::SignedCharacter,
                     SimpleTypeMode::NearPointer32);

  BumpPtrAllocator Alloc;
  AppendingTypeTableBuilder Ordering1(Alloc);
  AppendingTypeTableBuilder Ordering2(Alloc);

  TypeIndex CharP(SimpleTypeKind::SignedCharacter, SimpleTypeMode::NearPointer);
  TypeIndex IntP(SimpleTypeKind::Int32, SimpleTypeMode::NearPointer);
  TypeIndex DoubleP(SimpleTypeKind::Float64, SimpleTypeMode::NearPointer);

  // We're going to the same type sequence with two different orderings, and
  // then confirm all records are hashed the same.

  TypeIndex CharPP[2];
  TypeIndex IntPP[2];
  TypeIndex IntPPP[2];
  TypeIndex DoublePP[2];
  TypeIndex Args[2];
  TypeIndex Proc[2];

  // Ordering 1
  // ----------------------------------------
  // LF_POINTER             0x1000   {char**}
  //   Referent = char*
  // LF_POINTER             0x1001   {int**}
  //   Referent = int*
  // LF_POINTER             0x1002   {int***}
  //   Referent = 0x1001
  // LF_ARGLIST             0x1003   {(char**, int***)}
  //   Arg[0] = 0x1000
  //   Arg[1] = 0x1002
  // LF_PROCEDURE           0x1004   {int** func(char**, int***)}
  //   ArgList = 0x1003
  //   ReturnType = 0x1001
  std::vector<GloballyHashedType> Ordering1Hashes;
  CharPP[0] = createPointerRecord(Ordering1, CharP);
  IntPP[0] = createPointerRecord(Ordering1, IntP);
  IntPPP[0] = createPointerRecord(Ordering1, IntPP[0]);
  Args[0] = createArgListRecord(Ordering1, CharPP[0], IntPPP[0]);
  Proc[0] = createProcedureRecord(Ordering1, 2, IntPP[0], Args[0]);

  ASSERT_EQ(0x1000U, CharPP[0].getIndex());
  ASSERT_EQ(0x1001U, IntPP[0].getIndex());
  ASSERT_EQ(0x1002U, IntPPP[0].getIndex());
  ASSERT_EQ(0x1003U, Args[0].getIndex());
  ASSERT_EQ(0x1004U, Proc[0].getIndex());

  auto Hashes1 = GloballyHashedType::hashTypes(Ordering1.records());

  // Ordering 2
  // ----------------------------------------
  // LF_POINTER             0x1000   {int**}
  //   Referent = int*
  // LF_POINTER             0x1001   {int***}
  //   Referent = 0x1000
  // LF_POINTER             0x1002   {char**}
  //   Referent = char*
  // LF_POINTER             0x1003   {double**}
  //   Referent = double*
  // LF_ARGLIST             0x1004   {(char**, int***)}
  //   Arg[0] = 0x1002
  //   Arg[1] = 0x1001
  // LF_PROCEDURE           0x1005   {int** func(char**, int***)}
  //   ArgList = 0x1004
  //   ReturnType = 0x1000
  IntPP[1] = createPointerRecord(Ordering2, IntP);
  IntPPP[1] = createPointerRecord(Ordering2, IntPP[1]);
  CharPP[1] = createPointerRecord(Ordering2, CharP);
  DoublePP[1] = createPointerRecord(Ordering2, DoubleP);
  Args[1] = createArgListRecord(Ordering2, CharPP[1], IntPPP[1]);
  Proc[1] = createProcedureRecord(Ordering2, 2, IntPP[1], Args[1]);
  auto Hashes2 = GloballyHashedType::hashTypes(Ordering2.records());

  ASSERT_EQ(0x1000U, IntPP[1].getIndex());
  ASSERT_EQ(0x1001U, IntPPP[1].getIndex());
  ASSERT_EQ(0x1002U, CharPP[1].getIndex());
  ASSERT_EQ(0x1003U, DoublePP[1].getIndex());
  ASSERT_EQ(0x1004U, Args[1].getIndex());
  ASSERT_EQ(0x1005U, Proc[1].getIndex());

  // Sanity check to make sure all same-ordering hashes are different
  // from each other.
  verifyHashUniqueness(Hashes1);
  verifyHashUniqueness(Hashes2);

  EXPECT_EQ(hash_of(Hashes1, IntPP[0]), hash_of(Hashes2, IntPP[1]));
  EXPECT_EQ(hash_of(Hashes1, IntPPP[0]), hash_of(Hashes2, IntPPP[1]));
  EXPECT_EQ(hash_of(Hashes1, CharPP[0]), hash_of(Hashes2, CharPP[1]));
  EXPECT_EQ(hash_of(Hashes1, Args[0]), hash_of(Hashes2, Args[1]));
  EXPECT_EQ(hash_of(Hashes1, Proc[0]), hash_of(Hashes2, Proc[1]));
}
