//
// Created by tanmay on 6/13/22.
//

#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
// --------------- fp32 Atomic Condition Tests ---------------
// ---------------------------------------------------------------------------

TEST(AtomicConditionFP32, FP32UnaryOperations) {

}

TEST(AtomicConditionFP32, FP32BinaryOperations) {
  fACCreate(8);

  fACfp32BinaryDriver("AddTest1Op1", 2.0, "AddTest1Op2", 4.0, 14, 1);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("AddTest1Op1", 2.0, "AddTest1Op2", 4.0, 14, 1, 0.33333333));
  fACfp32BinaryDriver("AddTest2Op1", 2.0, "AddTest2Op2", 4.0, 14, 2);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("AddTest2Op1", 2.0, "AddTest2Op2", 4.0, 14, 2, 0.66666666));


  fACfp32BinaryDriver("SubTest1Op1", 2.0, "SubTest1Op2", 4.0, 16, 1);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("SubTest1Op1", 2.0, "SubTest1Op2", 4.0, 16, 1, 1.0));
  fACfp32BinaryDriver("SubTest2Op1", 2.0, "SubTest2Op2", 4.0, 16, 2);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("SubTest2Op1", 2.0, "SubTest2Op2", 4.0, 16, 2, 2.0));

  fACfp32BinaryDriver("MulTest1Op1", 2.0, "MulTest1Op2", 4.0, 18, 1);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("MulTest1Op1", 2.0, "MulTest1Op2", 4.0, 18, 1, 1.0));
  fACfp32BinaryDriver("MulTest2Op1", 2.0, "MulTest2Op2", 4.0, 18, 2);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("MulTest2Op1", 2.0, "MulTest2Op2", 4.0, 18, 2, 1.0));

  fACfp32BinaryDriver("DivTest1Op1", 2.0, "DivTest1Op2", 4.0, 21, 1);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("DivTest1Op1", 2.0, "DivTest1Op2", 4.0, 21, 1, 1.0));
  fACfp32BinaryDriver("DivTest2Op1", 2.0, "DivTest2Op2", 4.0, 21, 2);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("DivTest2Op1", 2.0, "DivTest2Op2", 4.0, 21, 2, 1.0));



  delete StorageTable;
}


// ---------------------------------------------------------------------------
// --------------- fp64 Atomic Condition Tests ---------------
// ---------------------------------------------------------------------------

TEST(AtomicConditionFP64, FP64UnaryOperations) {

}

TEST(AtomicConditionFP64, FP64BinaryOperations) {
  fACCreate(8);

  fACfp64BinaryDriver("AddTest1Op1", 2.0, "AddTest1Op2", 4.0, 14, 1);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("AddTest1Op1", 2.0, "AddTest1Op2", 4.0, 14, 1, 0.33333333333333333));
  fACfp64BinaryDriver("AddTest2Op1", 2.0, "AddTest2Op2", 4.0, 14, 2);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("AddTest2Op1", 2.0, "AddTest2Op2", 4.0, 14, 2, 0.66666666666666666));

  fACfp64BinaryDriver("SubTest1Op1", 2.0, "SubTest1Op2", 4.0, 16,1);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("SubTest1Op1", 2.0, "SubTest1Op2", 4.0, 16,1, 1.0));
  fACfp64BinaryDriver("SubTest2Op1", 2.0, "SubTest2Op2", 4.0, 16,2);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("SubTest2Op1", 2.0, "SubTest2Op2", 4.0, 16,2, 2.0));

  fACfp64BinaryDriver("MulTest1Op1", 2.0, "MulTest1Op2", 4.0, 18, 1);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("MulTest1Op1", 2.0, "MulTest1Op2", 4.0, 18, 1, 1.0));
  fACfp64BinaryDriver("MulTest2Op1", 2.0, "MulTest2Op2", 4.0, 18, 2);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("MulTest2Op1", 2.0, "MulTest2Op2", 4.0, 18, 2, 1.0));

  fACfp64BinaryDriver("DivTest1Op1", 2.0, "DivTest1Op2", 4.0, 21, 1);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("DivTest1Op1", 2.0, "DivTest1Op2", 4.0, 21, 1, 1.0));
  fACfp64BinaryDriver("DivTest2Op1", 2.0, "DivTest2Op2", 4.0, 21, 2);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("DivTest2Op1", 2.0, "DivTest2Op2", 4.0, 21, 2, 1.0));

  EXPECT_EQ(int(StorageTable->FP64ACItems.size()), 8);

  delete StorageTable;
}