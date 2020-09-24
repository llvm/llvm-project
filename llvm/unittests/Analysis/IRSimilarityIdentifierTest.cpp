//===- IRSimilarityIdentifierTest.cpp - IRSimilarityIdentifier unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for components for finding similarity such as the instruction mapper,
// suffix tree usage, and structural analysis.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Analysis/IRSimilarityIdentifier.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace IRSimilarity;

static std::unique_ptr<Module> makeLLVMModule(LLVMContext &Context,
                                              StringRef ModuleStr) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleStr, Err, Context);
  assert(M && "Bad LLVM IR?");
  return M;
}

void getVectors(Module &M, IRInstructionMapper &Mapper,
                std::vector<IRInstructionData *> &InstrList,
                std::vector<unsigned> &UnsignedVec) {
  for (Function &F : M)
    for (BasicBlock &BB : F)
      Mapper.convertToUnsignedVec(BB, InstrList, UnsignedVec);
}

// Checks that different opcodes are mapped to different values.
TEST(IRInstructionMapper, OpcodeDifferentiation) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = mul i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check that the size of the unsigned vector and the instruction list are the
  // same as a safety check.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  // Make sure that the unsigned vector is the expected size.
  ASSERT_TRUE(UnsignedVec.size() == 3);

  // Check whether the instructions are not mapped to the same value.
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that the same opcodes and types are mapped to the same values.
TEST(IRInstructionMapper, OpcodeTypeSimilarity) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);

  // Check whether the instructions are mapped to the same value.
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that the same opcode and different types are mapped to different
// values.
TEST(IRInstructionMapper, TypeDifferentiation) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b, i64 %c, i64 %d) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i64 %c, %d
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that different predicates map to different values.
TEST(IRInstructionMapper, PredicateDifferentiation) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = icmp sge i32 %b, %a
                             %1 = icmp slt i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that predicates with the same swapped predicate map to different
// values.
TEST(IRInstructionMapper, PredicateIsomorphism) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = icmp sgt i32 %a, %b
                             %1 = icmp slt i32 %b, %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that the same predicate maps to the same value.
TEST(IRInstructionMapper, PredicateSimilarity) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = icmp slt i32 %a, %b
                             %1 = icmp slt i32 %b, %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that the same predicate maps to the same value for floating point
// CmpInsts.
TEST(IRInstructionMapper, FPPredicateSimilarity) {
  StringRef ModuleString = R"(
                          define i32 @f(double %a, double %b) {
                          bb0:
                             %0 = fcmp olt double %a, %b
                             %1 = fcmp olt double %b, %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that the different predicate maps to a different value for floating
// point CmpInsts.
TEST(IRInstructionMapper, FPPredicatDifference) {
  StringRef ModuleString = R"(
                          define i32 @f(double %a, double %b) {
                          bb0:
                             %0 = fcmp olt double %a, %b
                             %1 = fcmp oge double %b, %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that the zexts that have the same type parameters map to the same
// unsigned integer.
TEST(IRInstructionMapper, ZextTypeSimilarity) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a) {
                          bb0:
                             %0 = zext i32  %a to i64
                             %1 = zext i32  %a to i64
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that the sexts that have the same type parameters map to the same
// unsigned integer.
TEST(IRInstructionMapper, SextTypeSimilarity) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a) {
                          bb0:
                             %0 = sext i32  %a to i64
                             %1 = sext i32  %a to i64
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that the zexts that have the different type parameters map to the
// different unsigned integers.
TEST(IRInstructionMapper, ZextTypeDifference) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i8 %b) {
                          bb0:
                             %0 = zext i32 %a to i64
                             %1 = zext i8 %b to i32
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that the sexts that have the different type parameters map to the
// different unsigned integers.
TEST(IRInstructionMapper, SextTypeDifference) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i8 %b) {
                          bb0:
                             %0 = sext i32 %a to i64
                             %1 = sext i8 %b to i32
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that loads that have the same type are mapped to the same unsigned
// integer.
TEST(IRInstructionMapper, LoadSimilarType) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             %0 = load i32, i32* %a
                             %1 = load i32, i32* %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that loads that have the different types are mapped to
// different unsigned integers.
TEST(IRInstructionMapper, LoadDifferentType) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i64* %b) {
                          bb0:
                             %0 = load i32, i32* %a
                             %1 = load i64, i64* %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that loads that have the different aligns are mapped to different
// unsigned integers.
TEST(IRInstructionMapper, LoadDifferentAlign) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             %0 = load i32, i32* %a, align 4
                             %1 = load i32, i32* %b, align 8
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that loads that have the different volatile settings are mapped to
// different unsigned integers.
TEST(IRInstructionMapper, LoadDifferentVolatile) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             %0 = load volatile i32, i32* %a
                             %1 = load i32, i32* %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that loads that have the same volatile settings are mapped to
// different unsigned integers.
TEST(IRInstructionMapper, LoadSameVolatile) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             %0 = load volatile i32, i32* %a
                             %1 = load volatile i32, i32* %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that loads that have the different atomicity settings are mapped to
// different unsigned integers.
TEST(IRInstructionMapper, LoadDifferentAtomic) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             %0 = load atomic i32, i32* %a unordered, align 4
                             %1 = load atomic i32, i32* %b monotonic, align 4
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that loads that have the same atomicity settings are mapped to
// different unsigned integers.
TEST(IRInstructionMapper, LoadSameAtomic) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             %0 = load atomic i32, i32* %a unordered, align 4
                             %1 = load atomic i32, i32* %b unordered, align 4
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that stores that have the same type are mapped to the same unsigned
// integer.
TEST(IRInstructionMapper, StoreSimilarType) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             store i32 1, i32* %a
                             store i32 2, i32* %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that stores that have the different types are mapped to
// different unsigned integers.
TEST(IRInstructionMapper, StoreDifferentType) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i64* %b) {
                          bb0:
                             store i32 1, i32* %a
                             store i64 1, i64* %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that stores that have the different aligns are mapped to different
// unsigned integers.
TEST(IRInstructionMapper, StoreDifferentAlign) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             store i32 1, i32* %a, align 4
                             store i32 1, i32* %b, align 8
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that stores that have the different volatile settings are mapped to
// different unsigned integers.
TEST(IRInstructionMapper, StoreDifferentVolatile) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             store volatile i32 1, i32* %a
                             store i32 1, i32* %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// Checks that stores that have the same volatile settings are mapped to
// different unsigned integers.
TEST(IRInstructionMapper, StoreSameVolatile) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             store volatile i32 1, i32* %a
                             store volatile i32 1, i32* %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that loads that have the same atomicity settings are mapped to
// different unsigned integers.
TEST(IRInstructionMapper, StoreSameAtomic) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             store atomic i32 1, i32* %a unordered, align 4
                             store atomic i32 1, i32* %b unordered, align 4
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
}

// Checks that loads that have the different atomicity settings are mapped to
// different unsigned integers.
TEST(IRInstructionMapper, StoreDifferentAtomic) {
  StringRef ModuleString = R"(
                          define i32 @f(i32* %a, i32* %b) {
                          bb0:
                             store atomic i32 1, i32* %a unordered, align 4
                             store atomic i32 1, i32* %b monotonic, align 4
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());
  ASSERT_TRUE(UnsignedVec.size() == 3);
  ASSERT_TRUE(UnsignedVec[0] != UnsignedVec[1]);
}

// In most cases, the illegal instructions we are collecting don't require any
// sort of setup.  In these cases, we can just only have illegal instructions,
// and the mapper will create 0 length vectors, and we can check that.

// In cases where we have legal instructions needed to set up the illegal
// instruction, to check illegal instructions are assigned unsigned integers
// from the maximum value decreasing to 0, it will be greater than a legal
// instruction that comes after.  So to check that we have an illegal
// instruction, we place a legal instruction after an illegal instruction, and
// check that the illegal unsigned integer is greater than the unsigned integer
// of the legal instruction.

// Checks that the branch is mapped to be illegal since there is extra checking
// needed to ensure that a branch in one region is branching to an isomorphic
// location in a different region.
TEST(IRInstructionMapper, BranchIllegal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = icmp slt i32 %a, %b
                             br i1 %0, label %bb0, label %bb1
                          bb1:
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that a PHINode is mapped to be illegal since there is extra checking
// needed to ensure that a branch in one region is bin an isomorphic
// location in a different region.
TEST(IRInstructionMapper, PhiIllegal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = phi i1 [ 0, %bb0 ], [ %0, %bb1 ]
                             ret i32 0
                          bb1:
                             ret i32 1
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that an alloca instruction is mapped to be illegal.
TEST(IRInstructionMapper, AllocaIllegal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = alloca i32
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that an getelementptr instruction is mapped to be illegal.  There is
// extra checking required for the parameters if a getelementptr has more than
// two operands.
TEST(IRInstructionMapper, GetElementPtrIllegal) {
  StringRef ModuleString = R"(
    %struct.RT = type { i8, [10 x [20 x i32]], i8 }
    %struct.ST = type { i32, double, %struct.RT }
    define i32 @f(%struct.ST* %s, i32 %a, i32 %b) {
    bb0:
       %0 = getelementptr inbounds %struct.ST, %struct.ST* %s, i64 1
       ret i32 0
    })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that a call instruction is mapped to be illegal.  We have to perform
// extra checks to ensure that both the name and function type are the same.
TEST(IRInstructionMapper, CallIllegal) {
  StringRef ModuleString = R"(
                          declare i32 @f1(i32, i32)
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = call i32 @f1(i32 %a, i32 %b)
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that an invoke instruction is mapped to be illegal. Invoke
// instructions are considered to be illegal because of the change in the
// control flow that is currently not recognized.
TEST(IRInstructionMapper, InvokeIllegal) {
  StringRef ModuleString = R"(
                          define i32 @f(i8 *%gep1, i32 %b) {
                          then:                       
                            invoke i32 undef(i8* undef)
                               to label %invoke unwind label %lpad

                          invoke:
                            unreachable

                          lpad:
                            landingpad { i8*, i32 }
                               catch i8* null
                            unreachable
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that an callbr instructions are considered to be illegal.  Callbr
// instructions are considered to be illegal because of the change in the
// control flow that is currently not recognized.
TEST(IRInstructionMapper, CallBrInstIllegal) {
  StringRef ModuleString = R"(
  define void @test() {
    fail:
      ret void
  }

  define i32 @f(i32 %a, i32 %b) {
      bb0:
        callbr void asm "xorl $0, $0; jmp ${1:l}", "r,X,~{dirflag},~{fpsr},~{flags}"(i32 %a, i8* blockaddress(@test, %fail)) to label %normal [label %fail]
      fail:
        ret i32 0
      normal:
        ret i32 0
  })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that an debuginfo intrinsics are mapped to be invisible.  Since they
// do not semantically change the program, they can be recognized as similar.
TEST(IRInstructionMapper, DebugInfoInvisible) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          then:
                            %0 = add i32 %a, %b                    
                            call void @llvm.dbg.value(metadata !0)
                            %1 = add i32 %a, %b     
                            ret i32 0
                          }

                          declare void @llvm.dbg.value(metadata)
                          !0 = distinct !{!"test\00", i32 10})";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
}

// The following are all exception handling intrinsics.  We do not currently
// handle these instruction because they are very context dependent.

// Checks that an eh.typeid.for intrinsic is mapped to be illegal.
TEST(IRInstructionMapper, ExceptionHandlingTypeIdIllegal) {
  StringRef ModuleString = R"(
    @_ZTIi = external constant i8*
    define i32 @f() {
    then:
      %0 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
      ret i32 0
    }

    declare i32 @llvm.eh.typeid.for(i8*))";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that an eh.exceptioncode intrinsic is mapped to be illegal.
TEST(IRInstructionMapper, ExceptionHandlingExceptionCodeIllegal) {
  StringRef ModuleString = R"(
    define i32 @f(i32 %a, i32 %b) {
    entry:
      %0 = catchswitch within none [label %__except] unwind to caller

    __except:
      %1 = catchpad within %0 [i8* null]
      catchret from %1 to label %__except

    then:
      %2 = call i32 @llvm.eh.exceptioncode(token %1)
      ret i32 0
    }

    declare i32 @llvm.eh.exceptioncode(token))";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that an eh.unwind intrinsic is mapped to be illegal.
TEST(IRInstructionMapper, ExceptionHandlingUnwindIllegal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          entry:
                            call void @llvm.eh.unwind.init()
                            ret i32 0
                          }

                          declare void @llvm.eh.unwind.init())";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that an eh.exceptionpointer intrinsic is mapped to be illegal.
TEST(IRInstructionMapper, ExceptionHandlingExceptionPointerIllegal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          entry:
                            %0 = call i8* @llvm.eh.exceptionpointer.p0i8(i32 0)
                            ret i32 0
                          }

                          declare i8* @llvm.eh.exceptionpointer.p0i8(i32))";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that a catchpad instruction is mapped to an illegal value.
TEST(IRInstructionMapper, CatchpadIllegal) {
  StringRef ModuleString = R"(
    declare void @llvm.donothing() nounwind readnone

    define void @function() personality i8 3 {
      entry:
        invoke void @llvm.donothing() to label %normal unwind label %exception
      exception:
        %cs1 = catchswitch within none [label %catchpad1] unwind to caller
      catchpad1:
        catchpad within %cs1 []
        br label %normal
      normal:
        ret void
  })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// Checks that a cleanuppad instruction is mapped to an illegal value.
TEST(IRInstructionMapper, CleanuppadIllegal) {
  StringRef ModuleString = R"(
    declare void @llvm.donothing() nounwind readnone

    define void @function() personality i8 3 {
      entry:
        invoke void @llvm.donothing() to label %normal unwind label %exception
      exception:
        %cs1 = catchswitch within none [label %catchpad1] unwind to caller
      catchpad1:
        %clean = cleanuppad within none []
        br label %normal
      normal:
        ret void
  })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(0));
}

// The following three instructions are memory transfer and setting based, which
// are considered illegal since is extra checking needed to handle the address
// space checking.

// Checks that a memset instruction is mapped to an illegal value.
TEST(IRInstructionMapper, MemSetIllegal) {
  StringRef ModuleString = R"(
  declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1)

  define i64 @function(i64 %x, i64 %z, i64 %n) {
  entry:
    %pool = alloca [59 x i64], align 4
    %tmp = bitcast [59 x i64]* %pool to i8*
    call void @llvm.memset.p0i8.i64(i8* nonnull %tmp, i8 0, i64 236, i32 4, i1 false)
    %cmp3 = icmp eq i64 %n, 0
    %a = add i64 %x, %z
    %c = add i64 %x, %z
    ret i64 0
  })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(6));
  ASSERT_TRUE(UnsignedVec[2] < UnsignedVec[1]);
}

// Checks that a memcpy instruction is mapped to an illegal value.
TEST(IRInstructionMapper, MemCpyIllegal) {
  StringRef ModuleString = R"(
  declare void @llvm.memcpy.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1)

  define i64 @function(i64 %x, i64 %z, i64 %n) {
  entry:
    %pool = alloca [59 x i64], align 4
    %tmp = bitcast [59 x i64]* %pool to i8*
    call void @llvm.memcpy.p0i8.i64(i8* nonnull %tmp, i8 0, i64 236, i32 4, i1 false)
    %cmp3 = icmp eq i64 %n, 0
    %a = add i64 %x, %z
    %c = add i64 %x, %z
    ret i64 0
  })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(6));
  ASSERT_TRUE(UnsignedVec[2] < UnsignedVec[1]);
}

// Checks that a memmove instruction is mapped to an illegal value.
TEST(IRInstructionMapper, MemMoveIllegal) {
  StringRef ModuleString = R"(
  declare void @llvm.memmove.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1)

  define i64 @function(i64 %x, i64 %z, i64 %n) {
  entry:
    %pool = alloca [59 x i64], align 4
    %tmp = bitcast [59 x i64]* %pool to i8*
    call void @llvm.memmove.p0i8.i64(i8* nonnull %tmp, i8 0, i64 236, i32 4, i1 false)
    %cmp3 = icmp eq i64 %n, 0
    %a = add i64 %x, %z
    %c = add i64 %x, %z
    ret i64 0
  })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(6));
  ASSERT_TRUE(UnsignedVec[2] < UnsignedVec[1]);
}

// Checks that a variable argument instructions are mapped to an illegal value.
// We exclude variable argument instructions since variable arguments
// requires extra checking of the argument list.
TEST(IRInstructionMapper, VarArgsIllegal) {
  StringRef ModuleString = R"(
  declare void @llvm.va_start(i8*)
  declare void @llvm.va_copy(i8*, i8*)
  declare void @llvm.va_end(i8*)

  define i32 @func1(i32 %a, double %b, i8* %v, ...) nounwind {
  entry:
    %a.addr = alloca i32, align 4
    %b.addr = alloca double, align 8
    %ap = alloca i8*, align 4
    %c = alloca i32, align 4
    store i32 %a, i32* %a.addr, align 4
    store double %b, double* %b.addr, align 8
    %ap1 = bitcast i8** %ap to i8*
    call void @llvm.va_start(i8* %ap1)
    store double %b, double* %b.addr, align 8
    store double %b, double* %b.addr, align 8
    %0 = va_arg i8** %ap, i32
    store double %b, double* %b.addr, align 8
    store double %b, double* %b.addr, align 8
    call void @llvm.va_copy(i8* %v, i8* %ap1)
    store double %b, double* %b.addr, align 8
    store double %b, double* %b.addr, align 8
    call void @llvm.va_end(i8* %ap1)
    store i32 %0, i32* %c, align 4
    %tmp = load i32, i32* %c, align 4
    ret i32 %tmp
  })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(16));
  ASSERT_TRUE(UnsignedVec[4] < UnsignedVec[3]);
  ASSERT_TRUE(UnsignedVec[7] < UnsignedVec[6]);
  ASSERT_TRUE(UnsignedVec[10] < UnsignedVec[9]);
  ASSERT_TRUE(UnsignedVec[13] < UnsignedVec[12]);
}

// Check the length of adding two illegal instructions one after th other.  We
// should find that only one element is added for each illegal range.
TEST(IRInstructionMapper, RepeatedIllegalLength) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = mul i32 %a, %b
                             %2 = call i32 @f(i32 %a, i32 %b)
                             %3 = call i32 @f(i32 %a, i32 %b)
                             %4 = add i32 %a, %b
                             %5 = mul i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check that the size of the unsigned vector and the instruction list are the
  // same as a safety check.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  // Make sure that the unsigned vector is the expected size.
  ASSERT_TRUE(UnsignedVec.size() == 6);
}

// A helper function that accepts an instruction list from a module made up of
// two blocks of two legal instructions and terminator, and checks them for
// instruction similarity.
static bool longSimCandCompare(std::vector<IRInstructionData *> &InstrList) {
  std::vector<IRInstructionData *>::iterator Start, End;

  Start = InstrList.begin();
  End = InstrList.begin();

  std::advance(End, 1);
  IRSimilarityCandidate Cand1(0, 2, *Start, *End);

  Start = InstrList.begin();
  End = InstrList.begin();

  std::advance(Start, 3);
  std::advance(End, 4);
  IRSimilarityCandidate Cand2(3, 2, *Start, *End);
  return IRSimilarityCandidate::isSimilar(Cand1, Cand2);
}

// Checks that two adds with commuted operands are considered to be the same
// instructions.
TEST(IRSimilarityCandidate, CheckIdenticalInstructions) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(3));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  std::vector<IRInstructionData *>::iterator Start, End;
  Start = InstrList.begin();
  End = InstrList.begin();
  std::advance(End, 1);
  IRSimilarityCandidate Cand1(0, 2, *Start, *End);
  IRSimilarityCandidate Cand2(0, 2, *Start, *End);

  ASSERT_TRUE(IRSimilarityCandidate::isSimilar(Cand1, Cand2));
}

// Checks that IRSimilarityCandidates wrapping these two regions of instructions
// are able to differentiate between instructions that have different opcodes.
TEST(IRSimilarityCandidate, CheckRegionsDifferentInstruction) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             ret i32 0
                          bb1:
                             %2 = sub i32 %a, %b
                             %3 = add i32 %b, %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(6));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_FALSE(longSimCandCompare(InstrList));
}

// Checks that IRSimilarityCandidates wrapping these two regions of instructions
// are able to differentiate between instructions that have different types.
TEST(IRSimilarityCandidate, CheckRegionsDifferentTypes) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b, i64 %c, i64 %d) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             ret i32 0
                          bb1:
                             %2 = add i64 %c, %d
                             %3 = add i64 %d, %c
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(6));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_FALSE(longSimCandCompare(InstrList));
}

// Check that debug instructions do not impact similarity. They are marked as
// invisible.
TEST(IRSimilarityCandidate, IdenticalWithDebug) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             call void @llvm.dbg.value(metadata !0)
                             %1 = add i32 %b, %a
                             ret i32 0
                          bb1:
                             %2 = add i32 %a, %b
                             call void @llvm.dbg.value(metadata !1)
                             %3 = add i32 %b, %a
                             ret i32 0
                          bb2:
                             %4 = add i32 %a, %b
                             %5 = add i32 %b, %a
                             ret i32 0       
                          }

                          declare void @llvm.dbg.value(metadata)
                          !0 = distinct !{!"test\00", i32 10}
                          !1 = distinct !{!"test\00", i32 11})";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(9));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_TRUE(longSimCandCompare(InstrList));
}

// Checks that IRSimilarityCandidates that include illegal instructions, are not
// considered to be the same set of instructions.  In these sets of instructions
// the allocas are illegal.
TEST(IRSimilarityCandidate, IllegalInCandidate) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %a, %b
                             %2 = alloca i32
                             ret i32 0
                          bb1:
                             %3 = add i32 %a, %b
                             %4 = add i32 %a, %b
                             %5 = alloca i32
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(6));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  std::vector<IRInstructionData *>::iterator Start, End;

  Start = InstrList.begin();
  End = InstrList.begin();

  std::advance(End, 2);
  IRSimilarityCandidate Cand1(0, 3, *Start, *End);

  Start = InstrList.begin();
  End = InstrList.begin();

  std::advance(Start, 3);
  std::advance(End, 5);
  IRSimilarityCandidate Cand2(3, 3, *Start, *End);
  ASSERT_FALSE(IRSimilarityCandidate::isSimilar(Cand1, Cand2));
}
