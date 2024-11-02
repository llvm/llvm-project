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

#include "llvm/Analysis/IRSimilarityIdentifier.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

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

void getSimilarities(
    Module &M,
    std::vector<std::vector<IRSimilarityCandidate>> &SimilarityCandidates) {
  // In order to keep the size of the tests from becoming too large, we do not
  // recognize similarity for branches unless explicitly needed.
  IRSimilarityIdentifier Identifier(/*EnableBranchMatching = */false);
  SimilarityCandidates = Identifier.findSimilarity(M);
}

// Checks that different opcodes are mapped to different values
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

// Checks that predicates where that can be considered the same when the
// operands are swapped, i.e. greater than to less than are mapped to the same
// unsigned integer.
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
  ASSERT_TRUE(UnsignedVec[0] == UnsignedVec[1]);
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

// Checks that the branch is mapped to legal when the option is set.
TEST(IRInstructionMapper, BranchLegal) {
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
  Mapper.InstClassifier.EnableBranches = true;
  Mapper.initializeForBBs(*M);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_TRUE(UnsignedVec[1] > UnsignedVec[0]);
  ASSERT_TRUE(UnsignedVec[1] < UnsignedVec[2]);
}

// Checks that a PHINode is mapped to be legal.
TEST(IRInstructionMapper, PhiLegal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = phi i1 [ 0, %bb0 ], [ %0, %bb1 ]
                             %1 = add i32 %a, %b
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
  Mapper.InstClassifier.EnableBranches = true;
  Mapper.initializeForBBs(*M);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
}

// Checks that a PHINode is mapped to be legal.
TEST(IRInstructionMapper, PhiIllegal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = phi i1 [ 0, %bb0 ], [ %0, %bb1 ]
                             %1 = add i32 %a, %b
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
  Mapper.initializeForBBs(*M);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(1));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
}

// Checks that an getelementptr instruction is mapped to be legal.  And that
// the operands in getelementpointer instructions are the exact same after the
// first element operand, which only requires the same type.
TEST(IRInstructionMapper, GetElementPtrSameEndOperands) {
  StringRef ModuleString = R"(
    %struct.RT = type { i8, [10 x [20 x i32]], i8 }
    %struct.ST = type { i32, double, %struct.RT }
    define i32 @f(%struct.ST* %s, i64 %a, i64 %b) {
    bb0:
       %0 = getelementptr inbounds %struct.ST, %struct.ST* %s, i64 %a, i32 0
       %1 = getelementptr inbounds %struct.ST, %struct.ST* %s, i64 %b, i32 0
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_EQ(UnsignedVec[0], UnsignedVec[1]);
}

// Check that when the operands in getelementpointer instructions are not the
// exact same after the first element operand, the instructions are mapped to
// different values.
TEST(IRInstructionMapper, GetElementPtrDifferentEndOperands) {
  StringRef ModuleString = R"(
    %struct.RT = type { i8, [10 x [20 x i32]], i8 }
    %struct.ST = type { i32, double, %struct.RT }
    define i32 @f(%struct.ST* %s, i64 %a, i64 %b) {
    bb0:
       %0 = getelementptr inbounds %struct.ST, %struct.ST* %s, i64 %a, i32 0
       %1 = getelementptr inbounds %struct.ST, %struct.ST* %s, i64 %b, i32 2
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_NE(UnsignedVec[0], UnsignedVec[1]);
}

// Check that when the operands in getelementpointer instructions are not the
// same initial base type, each instruction is mapped to a different value.
TEST(IRInstructionMapper, GetElementPtrDifferentBaseType) {
  StringRef ModuleString = R"(
    %struct.RT = type { i8, [10 x [20 x i32]], i8 }
    %struct.ST = type { i32, double, %struct.RT }
    define i32 @f(%struct.ST* %s, %struct.RT* %r, i64 %a, i64 %b) {
    bb0:
       %0 = getelementptr inbounds %struct.ST, %struct.ST* %s, i64 %a
       %1 = getelementptr inbounds %struct.RT, %struct.RT* %r, i64 %b
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_NE(UnsignedVec[0], UnsignedVec[1]);
}

// Check that when the operands in getelementpointer instructions do not have
// the same inbounds modifier, they are not counted as the same.
TEST(IRInstructionMapper, GetElementPtrDifferentInBounds) {
  StringRef ModuleString = R"(
    %struct.RT = type { i8, [10 x [20 x i32]], i8 }
    %struct.ST = type { i32, double, %struct.RT }
    define i32 @f(%struct.ST* %s, %struct.RT* %r, i64 %a, i64 %b) {
    bb0:
       %0 = getelementptr inbounds %struct.ST, %struct.ST* %s, i64 %a, i32 0
       %1 = getelementptr %struct.ST, %struct.ST* %s, i64 %b, i32 0
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_NE(UnsignedVec[0], UnsignedVec[1]);
}

// Checks that indirect call instructions are mapped to be illegal when it is
// specified to disallow them.
TEST(IRInstructionMapper, CallsIllegalIndirect) {
  StringRef ModuleString = R"(
                          define i32 @f(void()* %func) {
                          bb0:
                             call void %func()
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  Mapper.InstClassifier.EnableIndirectCalls = false;
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(1));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
}

// Checks that indirect call instructions are mapped to be legal when it is not
// specified to disallow them.
TEST(IRInstructionMapper, CallsLegalIndirect) {
  StringRef ModuleString = R"(
                          define i32 @f(void()* %func) {
                          bb0:
                             call void %func()
                             call void %func()
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  Mapper.InstClassifier.EnableIndirectCalls = true;
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
}

// Checks that a call instruction is mapped to be legal.  Here we check that
// a call with the same name, and same types are mapped to the same
// value.
TEST(IRInstructionMapper, CallsSameTypeSameName) {
  StringRef ModuleString = R"(
                          declare i32 @f1(i32, i32)
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = call i32 @f1(i32 %a, i32 %b)
                             %1 = call i32 @f1(i32 %a, i32 %b)
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_EQ(UnsignedVec[0], UnsignedVec[1]);
}

// Here we check that a calls with different names, but the same arguments types
// are mapped to different value when specified that the name must match.
TEST(IRInstructionMapper, CallsSameArgTypeDifferentNameDisallowed) {
  StringRef ModuleString = R"(
                          declare i32 @f1(i32, i32)
                          declare i32 @f2(i32, i32)
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = call i32 @f1(i32 %a, i32 %b)
                             %1 = call i32 @f2(i32 %a, i32 %b)
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  Mapper.EnableMatchCallsByName = true;
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_NE(UnsignedVec[0], UnsignedVec[1]);
}

// Here we check that a calls with different names, but the same arguments types
// are mapped to the same value when it is not specifed that they must match.
TEST(IRInstructionMapper, CallsSameArgTypeDifferentName) {
  StringRef ModuleString = R"(
                          declare i32 @f1(i32, i32)
                          declare i32 @f2(i32, i32)
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = call i32 @f1(i32 %a, i32 %b)
                             %1 = call i32 @f2(i32 %a, i32 %b)
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  Mapper.EnableMatchCallsByName = false;
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_EQ(UnsignedVec[0], UnsignedVec[1]);
}

// Here we check that a calls with different names, and different arguments
// types are mapped to different value.
TEST(IRInstructionMapper, CallsDifferentArgTypeDifferentName) {
  StringRef ModuleString = R"(
                          declare i32 @f1(i32, i32)
                          declare i32 @f2(i32)
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = call i32 @f1(i32 %a, i32 %b)
                             %1 = call i32 @f2(i32 %a)
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_NE(UnsignedVec[0], UnsignedVec[1]);
}

// Here we check that calls with different names, and different return
// types are mapped to different value.
TEST(IRInstructionMapper, CallsDifferentReturnTypeDifferentName) {
  StringRef ModuleString = R"(
                          declare i64 @f1(i32, i32)
                          declare i32 @f2(i32, i32)
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = call i64 @f1(i32 %a, i32 %b)
                             %1 = call i32 @f2(i32 %a, i32 %b)
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_NE(UnsignedVec[0], UnsignedVec[1]);
}

// Here we check that calls with the same name, types, and parameters map to the
// same unsigned integer.
TEST(IRInstructionMapper, CallsSameParameters) {
  StringRef ModuleString = R"(
                          declare i32 @f1(i32, i32)
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = tail call fastcc i32 @f1(i32 %a, i32 %b)
                             %1 = tail call fastcc i32 @f1(i32 %a, i32 %b)
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_EQ(UnsignedVec[0], UnsignedVec[1]);
}

// Here we check that calls with different tail call settings are mapped to
// different values.
TEST(IRInstructionMapper, CallsDifferentTails) {
  StringRef ModuleString = R"(
                          declare i32 @f1(i32, i32)
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = tail call i32 @f1(i32 %a, i32 %b)
                             %1 = call i32 @f1(i32 %a, i32 %b)
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_NE(UnsignedVec[0], UnsignedVec[1]);
}

// Here we check that calls with different calling convention settings are
// mapped to different values.
TEST(IRInstructionMapper, CallsDifferentCallingConventions) {
  StringRef ModuleString = R"(
                          declare i32 @f1(i32, i32)
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = call fastcc i32 @f1(i32 %a, i32 %b)
                             %1 = call i32 @f1(i32 %a, i32 %b)
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(3));
  ASSERT_NE(UnsignedVec[0], UnsignedVec[1]);
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(1));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(1));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(1));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(1));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(1));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(1));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(1));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
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
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(1));
  ASSERT_GT(UnsignedVec[0], Mapper.IllegalInstrNumber);
}

// The following three instructions are memory transfer and setting based, which
// are considered illegal since is extra checking needed to handle the address
// space checking.

// Checks that a memset instruction is mapped to an illegal value when
// specified.
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
  Mapper.InstClassifier.EnableIntrinsics = false;
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(7));
  ASSERT_TRUE(UnsignedVec[2] < UnsignedVec[0]);
}

// Checks that a memcpy instruction is mapped to an illegal value  when
// specified.
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
  Mapper.InstClassifier.EnableIntrinsics = false;
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(7));
  ASSERT_GT(UnsignedVec[2], UnsignedVec[3]);
  ASSERT_LT(UnsignedVec[2], UnsignedVec[0]);
}

// Checks that a memmove instruction is mapped to an illegal value  when
// specified.
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
  Mapper.InstClassifier.EnableIntrinsics = false;
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(7));
  ASSERT_LT(UnsignedVec[2], UnsignedVec[0]);
}

// Checks that mem* instructions are mapped to an legal value when not
// specified, and that all the intrinsics are marked differently.
TEST(IRInstructionMapper, MemOpsLegal) {
  StringRef ModuleString = R"(
  declare void @llvm.memmove.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1)
  declare void @llvm.memcpy.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1)
  declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1)

  define i64 @function(i64 %x, i64 %z, i64 %n) {
  entry:
    %pool = alloca [59 x i64], align 4
    %tmp = bitcast [59 x i64]* %pool to i8*
    call void @llvm.memmove.p0i8.i64(i8* nonnull %tmp, i8 0, i64 236, i32 4, i1 false)
    call void @llvm.memcpy.p0i8.i64(i8* nonnull %tmp, i8 0, i64 236, i32 4, i1 false)
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
  Mapper.InstClassifier.EnableIntrinsics = true;
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(9));
  ASSERT_LT(UnsignedVec[2], UnsignedVec[3]);
  ASSERT_LT(UnsignedVec[3], UnsignedVec[4]);
  ASSERT_LT(UnsignedVec[4], UnsignedVec[5]);
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
  Mapper.InstClassifier.EnableIntrinsics = false;
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  ASSERT_EQ(InstrList.size(), UnsignedVec.size());
  ASSERT_EQ(UnsignedVec.size(), static_cast<unsigned>(17));
  ASSERT_TRUE(UnsignedVec[7] < UnsignedVec[0]);
  ASSERT_TRUE(UnsignedVec[13] < UnsignedVec[10]);
  ASSERT_TRUE(UnsignedVec[16] < UnsignedVec[13]);
}

// Check the length of adding two illegal instructions one after th other.  We
// should find that only one element is added for each illegal range.
TEST(IRInstructionMapper, RepeatedIllegalLength) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = mul i32 %a, %b
                             %2 = alloca i32
                             %3 = alloca i32
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
static bool longSimCandCompare(std::vector<IRInstructionData *> &InstrList,
                               bool Structure = false, unsigned Length = 2,
                               unsigned StartIdxOne = 0,
                               unsigned StartIdxTwo = 3) {
  std::vector<IRInstructionData *>::iterator Start, End;

  Start = InstrList.begin();
  End = InstrList.begin();

  std::advance(End, StartIdxOne + Length - 1);
  IRSimilarityCandidate Cand1(StartIdxOne, Length, *Start, *End);

  Start = InstrList.begin();
  End = InstrList.begin();

  std::advance(Start, StartIdxTwo);
  std::advance(End, StartIdxTwo + Length - 1);
  IRSimilarityCandidate Cand2(StartIdxTwo, Length, *Start, *End);
  if (Structure)
    return IRSimilarityCandidate::compareStructure(Cand1, Cand2);
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

// Checks that comparison instructions are found to be similar instructions
// when the operands are flipped and the predicate is also swapped.
TEST(IRSimilarityCandidate, PredicateIsomorphism) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = icmp sgt i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = icmp slt i32 %a, %b
                             %3 = add i32 %a, %b
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

  ASSERT_TRUE(InstrList.size() > 5);
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

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

// Checks that different structure, in this case, where we introduce a new
// needed input in one region, is recognized as different.
TEST(IRSimilarityCandidate, DifferentStructure) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             ret i32 0
                          bb1:
                             %2 = add i32 %a, %b
                             %3 = add i32 %b, %0
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

  ASSERT_FALSE(longSimCandCompare(InstrList, true));
}

// Checks that comparison instructions are found to have the same structure
// when the operands are flipped and the predicate is also swapped.
TEST(IRSimilarityCandidate, PredicateIsomorphismStructure) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = icmp sgt i32 %a, %b
                             %1 = add i32 %a, %b
                             br label %bb1
                          bb1:
                             %2 = icmp slt i32 %b, %a
                             %3 = add i32 %a, %b
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

  ASSERT_TRUE(InstrList.size() > 5);
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_TRUE(longSimCandCompare(InstrList, true));
}

// Checks that different predicates are counted as diferent.
TEST(IRSimilarityCandidate, PredicateDifference) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = icmp sge i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = icmp slt i32 %b, %a
                             %3 = add i32 %a, %b
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

  ASSERT_TRUE(InstrList.size() > 5);
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_FALSE(longSimCandCompare(InstrList));
}

// Checks that the same structure is recognized between two candidates. The
// items %a and %b are used in the same way in both sets of instructions.
TEST(IRSimilarityCandidate, SameStructure) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = sub i32 %b, %a
                             ret i32 0
                          bb1:
                             %2 = add i32 %a, %b
                             %3 = sub i32 %b, %a
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

  ASSERT_TRUE(longSimCandCompare(InstrList, true));
}

// Checks that the canonical numbering between two candidates matches the found
// mapping between two candidates.
TEST(IRSimilarityCandidate, CanonicalNumbering) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = sub i32 %b, %a
                             ret i32 0
                          bb1:
                             %2 = add i32 %a, %b
                             %3 = sub i32 %b, %a
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
  ASSERT_EQ(InstrList.size(), UnsignedVec.size());

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
  DenseMap<unsigned, DenseSet<unsigned>> Mapping1;
  DenseMap<unsigned, DenseSet<unsigned>> Mapping2;
  ASSERT_TRUE(IRSimilarityCandidate::compareStructure(Cand1, Cand2, Mapping1,
                                                      Mapping2));
  IRSimilarityCandidate::createCanonicalMappingFor(Cand1);
  Cand2.createCanonicalRelationFrom(Cand1, Mapping1, Mapping2);

  for (std::pair<unsigned, DenseSet<unsigned>> &P : Mapping2) {
    unsigned Source = P.first;

    ASSERT_TRUE(Cand2.getCanonicalNum(Source).hasValue());
    unsigned Canon = *Cand2.getCanonicalNum(Source);
    ASSERT_TRUE(Cand1.fromCanonicalNum(Canon).hasValue());
    unsigned Dest = *Cand1.fromCanonicalNum(Canon);

    DenseSet<unsigned>::iterator It = P.second.find(Dest);
    ASSERT_NE(It, P.second.end());
  }
}

// Checks that the same structure is recognized between two candidates. While
// the input names are reversed, they still perform the same overall operation.
TEST(IRSimilarityCandidate, DifferentNameSameStructure) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             ret i32 0
                          bb1:
                             %2 = add i32 %b, %a
                             %3 = add i32 %a, %b
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

  ASSERT_TRUE(longSimCandCompare(InstrList, true));
}

// Checks that the same structure is recognized between two candidates when
// the branches target other blocks inside the same region, the relative
// distance between the blocks must be the same.
TEST(IRSimilarityCandidate, SameBranchStructureInternal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %b, %a
                             %3 = add i32 %a, %b
                             ret i32 0
                          }
                          
                          define i32 @f2(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %b, %a
                             %3 = add i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  Mapper.InstClassifier.EnableBranches = true;
  Mapper.initializeForBBs(*M);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(12));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_TRUE(longSimCandCompare(InstrList, true, 5, 0, 6));
}

// Checks that the different structure is recognized between two candidates,
// when the branches target other blocks inside the same region, the relative
// distance between the blocks must be the same.
TEST(IRSimilarityCandidate, DifferentBranchStructureInternal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb2
                          bb1:
                             %2 = add i32 %b, %a
                             %3 = add i32 %a, %b
                             br label %bb2
                          bb2:
                             %4 = add i32 %b, %a
                             %5 = add i32 %a, %b
                             ret i32 0
                          }
                          
                          define i32 @f2(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %b, %a
                             %3 = add i32 %a, %b
                             br label %bb2
                          bb2:
                             %4 = add i32 %b, %a
                             %5 = add i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  Mapper.InstClassifier.EnableBranches = true;
  Mapper.initializeForBBs(*M);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(18));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_FALSE(longSimCandCompare(InstrList, true, 6, 0, 9));
}

// Checks that the same structure is recognized between two candidates, when
// the branches target other blocks outside region, the relative distance
// does not need to be the same.
TEST(IRSimilarityCandidate, SameBranchStructureOutside) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %b, %a
                             %3 = add i32 %a, %b
                             ret i32 0
                          }
                          
                          define i32 @f2(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %b, %a
                             %3 = add i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  Mapper.InstClassifier.EnableBranches = true;
  Mapper.initializeForBBs(*M);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(12));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_TRUE(longSimCandCompare(InstrList, true, 3, 0, 6));
}

// Checks that the same structure is recognized between two candidates, when
// the branches target other blocks outside region, the relative distance
// does not need to be the same.
TEST(IRSimilarityCandidate, DifferentBranchStructureOutside) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %b, %a
                             %3 = add i32 %a, %b
                             ret i32 0
                          }
                          
                          define i32 @f2(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb2
                          bb1:
                             %2 = add i32 %b, %a
                             %3 = add i32 %a, %b
                             br label %bb2
                          bb2:
                             %4 = add i32 %b, %a
                             %5 = add i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  Mapper.InstClassifier.EnableBranches = true;
  Mapper.initializeForBBs(*M);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(15));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_TRUE(longSimCandCompare(InstrList, true, 3, 0, 6));
}

// Checks that the same structure is recognized between two candidates,
// when the phi predecessor are other blocks inside the same region,
// the relative distance between the blocks must be the same.
TEST(IRSimilarityCandidate, SamePHIStructureInternal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             br label %bb2
                          bb1:
                             br label %bb2
                          bb2:
                             %0 = phi i32 [ %a, %bb0 ], [ %b, %bb1 ] 
                             %1 = add i32 %b, %a
                             %2 = add i32 %a, %b
                             ret i32 0
                          }
                          
                          define i32 @f2(i32 %a, i32 %b) {
                          bb0:
                             br label %bb2
                          bb1:
                             br label %bb2
                          bb2:
                             %0 = phi i32 [ %a, %bb0 ], [ %b, %bb1 ]
                             %1 = add i32 %b, %a
                             %2 = add i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  Mapper.InstClassifier.EnableBranches = true;
  Mapper.initializeForBBs(*M);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(12));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_TRUE(longSimCandCompare(InstrList, true, 4, 0, 6));
}

// Checks that the different structure is recognized between two candidates,
// when the phi predecessor are other blocks inside the same region,
// the relative distance between the blocks must be the same.
TEST(IRSimilarityCandidate, DifferentPHIStructureInternal) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             br label %bb2
                          bb1:
                             br label %bb2
                          bb3:
                             br label %bb2
                          bb2:
                             %0 = phi i32 [ %a, %bb0 ], [ %b, %bb1 ] 
                             %1 = add i32 %b, %a
                             %2 = add i32 %a, %b
                             ret i32 0
                          }
                          
                          define i32 @f2(i32 %a, i32 %b) {
                          bb0:
                             br label %bb2
                          bb1:
                             br label %bb2
                          bb3:
                             br label %bb2
                          bb2:
                             %0 = phi i32 [ %a, %bb0 ], [ %b, %bb3 ] 
                             %1 = add i32 %b, %a
                             %2 = add i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<IRInstructionData *> InstrList;
  std::vector<unsigned> UnsignedVec;

  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;
  SpecificBumpPtrAllocator<IRInstructionDataList> IDLAllocator;
  IRInstructionMapper Mapper(&InstDataAllocator, &IDLAllocator);
  Mapper.InstClassifier.EnableBranches = true;
  Mapper.initializeForBBs(*M);
  getVectors(*M, Mapper, InstrList, UnsignedVec);

  // Check to make sure that we have a long enough region.
  ASSERT_EQ(InstrList.size(), static_cast<unsigned>(14));
  // Check that the instructions were added correctly to both vectors.
  ASSERT_TRUE(InstrList.size() == UnsignedVec.size());

  ASSERT_FALSE(longSimCandCompare(InstrList, true, 5, 0, 7));
}

// Checks that two sets of identical instructions are found to be the same.
// Both sequences of adds have the same operand ordering, and the same
// instructions, making them strcturally equivalent.
TEST(IRSimilarityIdentifier, IdentitySimilarity) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = sub i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %a, %b
                             %3 = sub i32 %b, %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.size() == 1);
  for (std::vector<IRSimilarityCandidate> &Cands : SimilarityCandidates) {
    ASSERT_TRUE(Cands.size() == 2);
    unsigned InstIdx = 0;
    for (IRSimilarityCandidate &Cand : Cands) {
      ASSERT_TRUE(Cand.getStartIdx() == InstIdx);
      InstIdx += 3;
    }
  }
}

// Checks that incorrect sequences are not found as similar.  In this case,
// we have different sequences of instructions.
TEST(IRSimilarityIdentifier, InstructionDifference) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b, i32 %c, i32 %d) {
                          bb0:
                             %0 = sub i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %c, %d
                             %3 = sub i32 %d, %c
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.empty());
}

// This test checks to see whether we can detect similarity for commutative
// instructions where the operands have been reversed.
TEST(IRSimilarityIdentifier, CommutativeSimilarity) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %a, %b
                             %3 = add i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.size() == 1);
  for (std::vector<IRSimilarityCandidate> &Cands : SimilarityCandidates) {
    ASSERT_TRUE(Cands.size() == 2);
    unsigned InstIdx = 0;
    for (IRSimilarityCandidate &Cand : Cands) {
      ASSERT_TRUE(Cand.getStartIdx() == InstIdx);
      InstIdx += 3;
    }
  }
}

// This test ensures that when the first instruction in a sequence is
// a commutative instruction with the same value (mcomm_inst_same_val), but the
// corresponding instruction (comm_inst_diff_val) is not, we mark the regions
// and not similar.
TEST(IRSimilarityIdentifier, CommutativeSameValueFirstMisMatch) {
  StringRef ModuleString = R"(
                          define void @v_1_0(i64 %v_33) {
                            entry:
                              %comm_inst_same_val = mul i64 undef, undef
                              %add = add i64 %comm_inst_same_val, %v_33
                              %comm_inst_diff_val = mul i64 0, undef
                              %mul.i = add i64 %comm_inst_diff_val, %comm_inst_diff_val
                              unreachable
                            })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.size() == 0);
}

// This test makes sure that intrinsic functions that are marked commutative
// are still treated as non-commutative since they are function calls.
TEST(IRSimilarityIdentifier, IntrinsicCommutative) {
  // If treated as commutative, we will fail to find a valid mapping, causing
  // an assertion error.
  StringRef ModuleString = R"(
  define void @foo() {
    entry:
      %0 = call i16 @llvm.smul.fix.i16(i16 16384, i16 16384, i32 15)
      store i16 %0, i16* undef, align 1
      %1 = icmp eq i16 undef, 8192
      call void @bar()
      %2 = call i16 @llvm.smul.fix.i16(i16 -16384, i16 16384, i32 15)
      store i16 %2, i16* undef, align 1
      %3 = icmp eq i16 undef, -8192
      call void @bar()
      %4 = call i16 @llvm.smul.fix.i16(i16 -16384, i16 -16384, i32 15)
      ret void
  }

  declare void @bar()

  ; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
  declare i16 @llvm.smul.fix.i16(i16, i16, i32 immarg))";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.size() == 0);
}

// This test checks to see whether we can detect different structure in
// commutative instructions.  In this case, the second operand in the second
// add is different.
TEST(IRSimilarityIdentifier, NoCommutativeSimilarity) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = add i32 %1, %b
                             br label %bb1
                          bb1:
                             %2 = add i32 %a, %b
                             %3 = add i32 %2, %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.size() == 0);
}

// Check that we are not finding similarity in non commutative
// instructions.  That is, while the instruction and operands used are the same
// in the two subtraction sequences, they are in a different order, and cannot
// be counted as the same since a subtraction is not commutative.
TEST(IRSimilarityIdentifier, NonCommutativeDifference) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = sub i32 %a, %b
                             %1 = sub i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = sub i32 %a, %b
                             %3 = sub i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.empty());
}

// Check that we find similarity despite changing the register names.
TEST(IRSimilarityIdentifier, MappingSimilarity) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b, i32 %c, i32 %d) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = sub i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %c, %d
                             %3 = sub i32 %d, %c
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.size() == 1);
  for (std::vector<IRSimilarityCandidate> &Cands : SimilarityCandidates) {
    ASSERT_TRUE(Cands.size() == 2);
    unsigned InstIdx = 0;
    for (IRSimilarityCandidate &Cand : Cands) {
      ASSERT_TRUE(Cand.getStartIdx() == InstIdx);
      InstIdx += 3;
    }
  }
}

// Check that we find instances of swapped predicate isomorphism.  That is,
// for predicates that can be flipped, e.g. greater than to less than,
// we can identify that instances of these different literal predicates, but are
// the same within a single swap can be found.
TEST(IRSimilarityIdentifier, PredicateIsomorphism) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 %a, %b
                             %1 = icmp sgt i32 %b, %a
                             br label %bb1
                          bb1:
                             %2 = add i32 %a, %b
                             %3 = icmp slt i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.size() == 1);
  for (std::vector<IRSimilarityCandidate> &Cands : SimilarityCandidates) {
    ASSERT_TRUE(Cands.size() == 2);
    unsigned InstIdx = 0;
    for (IRSimilarityCandidate &Cand : Cands) {
      ASSERT_TRUE(Cand.getStartIdx() == InstIdx);
      InstIdx += 3;
    }
  }
}

// Checks that constants are detected as the same operand in each use in the
// sequences of instructions.  Also checks that we can find structural
// equivalence using constants.  In this case the 1 has the same use pattern as
// %a.
TEST(IRSimilarityIdentifier, ConstantMappingSimilarity) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 1, %b
                             %1 = icmp sgt i32 %b, 1
                             br label %bb1
                          bb1:
                             %2 = add i32 %a, %b
                             %3 = icmp sgt i32 %b, %a
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.size() == 1);
  for (std::vector<IRSimilarityCandidate> &Cands : SimilarityCandidates) {
    ASSERT_TRUE(Cands.size() == 2);
    unsigned InstIdx = 0;
    for (IRSimilarityCandidate &Cand : Cands) {
      ASSERT_TRUE(Cand.getStartIdx() == InstIdx);
      InstIdx += 3;
    }
  }
}

// Check that constants are uniquely identified. i.e. two different constants
// are not considered the same.  This means that this should not find any
// structural similarity.
TEST(IRSimilarityIdentifier, ConstantMappingDifference) {
  StringRef ModuleString = R"(
                          define i32 @f(i32 %a, i32 %b) {
                          bb0:
                             %0 = add i32 1, %b
                             %1 = icmp sgt i32 %b, 2
                             br label %bb1
                          bb1:
                             %2 = add i32 %a, %b
                             %3 = icmp slt i32 %a, %b
                             ret i32 0
                          })";
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);

  std::vector<std::vector<IRSimilarityCandidate>> SimilarityCandidates;
  getSimilarities(*M, SimilarityCandidates);

  ASSERT_TRUE(SimilarityCandidates.empty());
}
