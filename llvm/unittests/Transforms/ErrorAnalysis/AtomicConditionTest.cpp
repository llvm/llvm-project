//
// Created by tanmay on 6/13/22.
//

#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
// --------------- fp32 Atomic Condition Tests ---------------
// ---------------------------------------------------------------------------

TEST(AtomicConditionFP32, FP32UnaryOperations) {
  fACCreate(12);

  fACfp32UnaryDriver("SinTest", 1.57, Operation::Sin);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("SinTest", 1.57, "", 0, Operation::Sin, 1, 0.00125015108));
  fACfp32UnaryDriver("CosTest", 1.57, Operation::Cos);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("CosTest", 1.57, "", 0, Operation::Cos, 1, 1971.681885));
  fACfp32UnaryDriver("TanTest", 1.57, Operation::Tan);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("TanTest", 1.57, "", 0, Operation::Tan, 1, 1971.683105));
  fACfp32UnaryDriver("ArcSinTest", 0.5, Operation::ArcSin);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("ArcSinTest", 0.5, "", 0, Operation::ArcSin, 1, 1.10265779));
  fACfp32UnaryDriver("ArcCosTest", 0.5, Operation::ArcCos);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("ArcCosTest", 0.5, "", 0, Operation::ArcCos, 1, 0.551328897));
  fACfp32UnaryDriver("ArcTanTest", 0.5, Operation::ArcTan);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("ArcTanTest", 0.5, "", 0, Operation::ArcTan, 1, 0.700625896));
  fACfp32UnaryDriver("SinhTest", 1.57, Operation::Sinh);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("SinhTest", 1.57, "", 0, Operation::Sinh, 1, 1.71205664));
  fACfp32UnaryDriver("CoshTest", 1.57, Operation::Cosh);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("CoshTest", 1.57, "", 0, Operation::Cosh, 1, 1.43973053));
  fACfp32UnaryDriver("TanhTest", 1.57, Operation::Tanh);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("TanhTest", 1.57, "", 0, Operation::Tanh, 1, 0.272326142));
  fACfp32UnaryDriver("ExpTest", -2, Operation::Exp);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("ExpTest", -2, "", 0, Operation::Exp, 1, 2));
  fACfp32UnaryDriver("LogTest", 10, Operation::Log);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("LogTest", 10, "", 0, Operation::Log, 1, 0.434294462));
  fACfp32UnaryDriver("SqrtTest", 2, Operation::Sqrt);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("SqrtTest", 2, "", 0, Operation::Sqrt, 1, 0.5));

  delete StorageTable;
}

TEST(AtomicConditionFP32, FP32BinaryOperations) {
  fACCreate(8);

  fACfp32BinaryDriver("AddTest1Op1", 2.0, "AddTest1Op2", 4.0, Operation::Add, 1);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("AddTest1Op1", 2.0, "AddTest1Op2", 4.0, Operation::Add, 1, 0.33333333));
  fACfp32BinaryDriver("AddTest2Op1", 2.0, "AddTest2Op2", 4.0, Operation::Add, 2);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("AddTest2Op1", 2.0, "AddTest2Op2", 4.0, Operation::Add, 2, 0.66666666));

  fACfp32BinaryDriver("SubTest1Op1", 2.0, "SubTest1Op2", 4.0, Operation::Sub, 1);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("SubTest1Op1", 2.0, "SubTest1Op2", 4.0, Operation::Sub, 1, 1.0));
  fACfp32BinaryDriver("SubTest2Op1", 2.0, "SubTest2Op2", 4.0, Operation::Sub, 2);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("SubTest2Op1", 2.0, "SubTest2Op2", 4.0, Operation::Sub, 2, 2.0));

  fACfp32BinaryDriver("MulTest1Op1", 2.0, "MulTest1Op2", 4.0, Operation::Mul, 1);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("MulTest1Op1", 2.0, "MulTest1Op2", 4.0, Operation::Mul, 1, 1.0));
  fACfp32BinaryDriver("MulTest2Op1", 2.0, "MulTest2Op2", 4.0, Operation::Mul, 2);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("MulTest2Op1", 2.0, "MulTest2Op2", 4.0, Operation::Mul, 2, 1.0));

  fACfp32BinaryDriver("DivTest1Op1", 2.0, "DivTest1Op2", 4.0, Operation::Div, 1);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("DivTest1Op1", 2.0, "DivTest1Op2", 4.0, Operation::Div, 1, 1.0));
  fACfp32BinaryDriver("DivTest2Op1", 2.0, "DivTest2Op2", 4.0, Operation::Div, 2);
  EXPECT_EQ(StorageTable->FP32ACItems.back(), ACItem<float>("DivTest2Op1", 2.0, "DivTest2Op2", 4.0, Operation::Div, 2, 1.0));

  EXPECT_EQ(int(StorageTable->FP32ACItems.size()), 8);

  delete StorageTable;
}


// ---------------------------------------------------------------------------
// --------------- fp64 Atomic Condition Tests ---------------
// ---------------------------------------------------------------------------

TEST(AtomicConditionFP64, FP64UnaryOperations) {
  fACCreate(12);

  fACfp64UnaryDriver("SinTest", 1.57, Operation::Sin);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("SinTest", 1.57, "", 0, Operation::Sin, 1, 0.0012502333322604124));
  fACfp64UnaryDriver("CosTest", 1.57, Operation::Cos);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("CosTest", 1.57, "", 0, Operation::Cos, 1, 1971.55197865624));
  fACfp64UnaryDriver("TanTest", 1.57, Operation::Tan);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("TanTest", 1.57, "", 0, Operation::Tan, 1, 1971.553228889572));
  fACfp64UnaryDriver("ArcSinTest", 0.5, Operation::ArcSin);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("ArcSinTest", 0.5, "", 0, Operation::ArcSin, 1, 1.1026577908435842));
  fACfp64UnaryDriver("ArcCosTest", 0.5, ArcCos);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("ArcCosTest", 0.5, "", 0, Operation::ArcCos, 1, 0.55132889542179209));
  fACfp64UnaryDriver("ArcTanTest", 0.5, ArcTan);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("ArcTanTest", 0.5, "", 0, Operation::ArcTan, 1, 0.70062590232742616));
  fACfp64UnaryDriver("SinhTest", 1.57, Operation::Sinh);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("SinhTest", 1.57, "", 0, Operation::Sinh, 1, 1.7120565921822388));
  fACfp64UnaryDriver("CoshTest", 1.57, Operation::Cosh);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("CoshTest", 1.57, "", 0, Operation::Cosh, 1, 1.4397304453926751));
  fACfp64UnaryDriver("TanhTest", 1.57, Operation::Tanh);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("TanhTest", 1.57, "", 0, Operation::Tanh, 1, 0.27232614678956374));
  fACfp64UnaryDriver("ExpTest", -2, Operation::Exp);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("ExpTest", -2, "", 0, Operation::Exp, 1, 2.0));
  fACfp64UnaryDriver("LogTest", 10, Operation::Log);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("LogTest", 10, "", 0, Operation::Log, 1, 0.43429448190325176));
  fACfp64UnaryDriver("SqrtTest", 2, Operation::Sqrt);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("SqrtTest", 2, "", 0, Operation::Sqrt, 1, 0.5));
  fACfp64UnaryDriver("TruncToFloatTest", 1.57, Operation::TruncToFloat);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("TruncToFloatTest", 1.57, "", 0, Operation::TruncToFloat, 1, 1.0));

  delete StorageTable;
}

TEST(AtomicConditionFP64, FP64BinaryOperations) {
  fACCreate(8);

  fACfp64BinaryDriver("AddTest1Op1", 2.0, "AddTest1Op2", 4.0, Operation::Add, 1);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("AddTest1Op1", 2.0, "AddTest1Op2", 4.0, Operation::Add, 1, 0.33333333333333333));
  fACfp64BinaryDriver("AddTest2Op1", 2.0, "AddTest2Op2", 4.0, Operation::Add, 2);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("AddTest2Op1", 2.0, "AddTest2Op2", 4.0, Operation::Add, 2, 0.66666666666666666));

  fACfp64BinaryDriver("SubTest1Op1", 2.0, "SubTest1Op2", 4.0, Operation::Sub,1);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("SubTest1Op1", 2.0, "SubTest1Op2", 4.0, Operation::Sub,1, 1.0));
  fACfp64BinaryDriver("SubTest2Op1", 2.0, "SubTest2Op2", 4.0, Operation::Sub,2);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("SubTest2Op1", 2.0, "SubTest2Op2", 4.0, Operation::Sub,2, 2.0));

  fACfp64BinaryDriver("MulTest1Op1", 2.0, "MulTest1Op2", 4.0, Operation::Mul, 1);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("MulTest1Op1", 2.0, "MulTest1Op2", 4.0, Operation::Mul, 1, 1.0));
  fACfp64BinaryDriver("MulTest2Op1", 2.0, "MulTest2Op2", 4.0, Operation::Mul, 2);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("MulTest2Op1", 2.0, "MulTest2Op2", 4.0, Operation::Mul, 2, 1.0));

  fACfp64BinaryDriver("DivTest1Op1", 2.0, "DivTest1Op2", 4.0, Operation::Div, 1);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("DivTest1Op1", 2.0, "DivTest1Op2", 4.0, Operation::Div, 1, 1.0));
  fACfp64BinaryDriver("DivTest2Op1", 2.0, "DivTest2Op2", 4.0, Operation::Div, 2);
  EXPECT_EQ(StorageTable->FP64ACItems.back(), ACItem<double>("DivTest2Op1", 2.0, "DivTest2Op2", 4.0, Operation::Div, 2, 1.0));

  EXPECT_EQ(int(StorageTable->FP64ACItems.size()), 8);

  delete StorageTable;
}