//===- HLSLSemanticSignatureMetadataTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/SemanticSignatures.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::hlsl;

namespace {

class HLSLSemanticSignatureMetadataTest : public testing::Test {
protected:
  LLVMContext Ctx;

  Metadata *getI32(uint32_t Val) {
    return ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(Ctx), Val));
  }

  Metadata *getI8(uint8_t Val) {
    return ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt8Ty(Ctx), Val));
  }

  Metadata *getStr(StringRef Val) { return MDString::get(Ctx, Val); }

  MDNode *getIndices(ArrayRef<uint32_t> Indices) {
    SmallVector<Metadata *> Ops;
    for (uint32_t I : Indices)
      Ops.push_back(getI32(I));
    return MDNode::get(Ctx, Ops);
  }

  // Assemble a raw signature element node from the example in the spec
  MDNode *getElement(uint32_t SigId, StringRef Name, uint32_t CompType,
                     uint32_t SemanticKind, ArrayRef<uint32_t> Indices,
                     uint32_t InterpMode, uint32_t Rows, uint8_t Cols,
                     uint32_t StartRow, uint8_t StartCol, uint8_t UsageMask,
                     uint8_t DynIndexMask, uint32_t GSStream) {
    return MDNode::get(
        Ctx, {getI32(SigId), getStr(Name), getI32(CompType),
              getI32(SemanticKind), getIndices(Indices), getI32(InterpMode),
              getI32(Rows), getI8(Cols), getI32(StartRow), getI8(StartCol),
              getI8(UsageMask), getI8(DynIndexMask), getI32(GSStream)});
  }

  // Read back an integer operand
  uint64_t getIntOp(const MDNode *N, unsigned I) {
    return mdconst::extract<ConstantInt>(N->getOperand(I))->getZExtValue();
  }

  // Read back a string operand
  StringRef getStrOp(const MDNode *N, unsigned I) {
    return cast<MDString>(N->getOperand(I))->getString();
  }
};

//===----------------------------------------------------------------------===//
// Success cases
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignatureMetadataTest, StructHelpers) {
  SemanticSignatureElement Elem;
  EXPECT_FALSE(Elem.isAllocated());

  Elem.Cols = 4;
  Elem.StartRow = 0;
  Elem.StartCol = 0;
  EXPECT_TRUE(Elem.isAllocated());
  EXPECT_EQ(Elem.getDeclaredMask(), 0xF);
}

TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElement) {
  MDNode *Node = getElement(/*SigId=*/1, "TEXCOORD", /*CompType=*/9,
                            /*SemanticKind=*/0, /*Indices=*/{0, 1},
                            /*InterpMode=*/0, /*Rows=*/2, /*Cols=*/4,
                            /*StartRow=*/1, /*StartCol=*/0, /*UsageMask=*/0,
                            /*DynIndexMask=*/0, /*GSStream=*/0);

  Expected<SemanticSignatureElement> Elem =
      SemanticSignatureElement::fromMetadata(Node);
  ASSERT_THAT_EXPECTED(Elem, Succeeded());

  EXPECT_EQ(Elem->SigId, 1u);
  EXPECT_EQ(Elem->SemanticName, "TEXCOORD");
  EXPECT_EQ(Elem->CompType, dxil::ElementType::F32);
  EXPECT_EQ(Elem->SemanticKind, dxbc::PSV::SemanticKind::Arbitrary);
  EXPECT_THAT(Elem->SemanticIndices, testing::ElementsAre(0u, 1u));
  EXPECT_EQ(Elem->InterpMode, dxbc::PSV::InterpolationMode::Undefined);
  EXPECT_EQ(Elem->Rows, 2u);
  EXPECT_EQ(Elem->Cols, 4u);
  EXPECT_EQ(Elem->StartRow, 1u);
  EXPECT_EQ(Elem->StartCol, 0u);
  EXPECT_EQ(Elem->UsageMask, 0u);
  EXPECT_EQ(Elem->DynIndexMask, 0u);
  EXPECT_EQ(Elem->GSStream, 0u);
}

// SV_Target output with a non-zero usage/dynamic-index mask and semantic index
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementSystemValue) {
  MDNode *Node = getElement(/*SigId=*/1, "SV_Target", /*CompType=*/9,
                            /*SemanticKind=*/16, /*Indices=*/{1},
                            /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/4,
                            /*StartRow=*/1, /*StartCol=*/0, /*UsageMask=*/0x7,
                            /*DynIndexMask=*/0x1, /*GSStream=*/0);

  Expected<SemanticSignatureElement> Elem =
      SemanticSignatureElement::fromMetadata(Node);
  ASSERT_THAT_EXPECTED(Elem, Succeeded());

  EXPECT_EQ(Elem->SemanticName, "SV_Target");
  EXPECT_EQ(Elem->SemanticKind, dxbc::PSV::SemanticKind::Target);
  EXPECT_THAT(Elem->SemanticIndices, testing::ElementsAre(1u));
  EXPECT_EQ(Elem->UsageMask, 0x7u);
  EXPECT_EQ(Elem->DynIndexMask, 0x1u);
}

// An unallocated element uses the row/col sentinels
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementUnallocated) {
  MDNode *Node = getElement(/*SigId=*/0, "POSITION", /*CompType=*/9,
                            /*SemanticKind=*/0, /*Indices=*/{0},
                            /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/4,
                            /*StartRow=*/UnallocatedRow, /*StartCol=*/UnallocatedCol,
                            /*UsageMask=*/0, /*DynIndexMask=*/0, /*GSStream=*/0);

  Expected<SemanticSignatureElement> Elem =
      SemanticSignatureElement::fromMetadata(Node);
  ASSERT_THAT_EXPECTED(Elem, Succeeded());

  EXPECT_EQ(Elem->StartRow, UnallocatedRow);
  EXPECT_EQ(Elem->StartCol, UnallocatedCol);
  EXPECT_FALSE(Elem->isAllocated());
}

// Every component type value maps onto the matching dxil::ElementType
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementComponentTypes) {
  for (dxil::ElementType CompType :
       {dxil::ElementType::I32, dxil::ElementType::U32,
        dxil::ElementType::F16, dxil::ElementType::F32,
        dxil::ElementType::F64, dxil::ElementType::I16}) {
    MDNode *Node = getElement(
        /*SigId=*/0, "A", static_cast<uint32_t>(CompType), /*SemanticKind=*/0,
        /*Indices=*/{0}, /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1,
        /*StartRow=*/0, /*StartCol=*/0, /*UsageMask=*/0, /*DynIndexMask=*/0,
        /*GSStream=*/0);

    Expected<SemanticSignatureElement> Elem =
        SemanticSignatureElement::fromMetadata(Node);
    ASSERT_THAT_EXPECTED(Elem, Succeeded());
    EXPECT_EQ(Elem->CompType, CompType);
  }
}

// Every interpolation mode value maps onto the matching enumerator
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementInterpModes) {
  for (dxbc::PSV::InterpolationMode Mode :
       {dxbc::PSV::InterpolationMode::Constant,
        dxbc::PSV::InterpolationMode::Linear,
        dxbc::PSV::InterpolationMode::LinearCentroid,
        dxbc::PSV::InterpolationMode::LinearNoperspective,
        dxbc::PSV::InterpolationMode::LinearSample}) {
    MDNode *Node = getElement(
        /*SigId=*/0, "A", /*CompType=*/9, /*SemanticKind=*/0, /*Indices=*/{0},
        static_cast<uint32_t>(Mode), /*Rows=*/1, /*Cols=*/1, /*StartRow=*/0,
        /*StartCol=*/0, /*UsageMask=*/0, /*DynIndexMask=*/0, /*GSStream=*/0);

    Expected<SemanticSignatureElement> Elem =
        SemanticSignatureElement::fromMetadata(Node);
    ASSERT_THAT_EXPECTED(Elem, Succeeded());
    EXPECT_EQ(Elem->InterpMode, Mode);
  }
}

// A column-offset allocation drives the derived declared/usage masks
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementDerivedMasks) {
  MDNode *Node = getElement(/*SigId=*/0, "SV_Position", /*CompType=*/9,
                            /*SemanticKind=*/3, /*Indices=*/{0},
                            /*InterpMode=*/4, /*Rows=*/1, /*Cols=*/2,
                            /*StartRow=*/0, /*StartCol=*/1, /*UsageMask=*/0x2,
                            /*DynIndexMask=*/0, /*GSStream=*/0);

  Expected<SemanticSignatureElement> Elem =
      SemanticSignatureElement::fromMetadata(Node);
  ASSERT_THAT_EXPECTED(Elem, Succeeded());

  EXPECT_EQ(Elem->SemanticKind, dxbc::PSV::SemanticKind::Position);
  EXPECT_TRUE(Elem->isAllocated());
  // ((1 << 2) - 1) << 1 == 0b0110
  EXPECT_EQ(Elem->getDeclaredMask(), 0x6);
  EXPECT_EQ(Elem->getAlwaysReadsMask(), 0x2);
  // ~0x2 & 0x6 == 0x4
  EXPECT_EQ(Elem->getNeverWritesMask(), 0x4);
}

// A geometry shader output carries a non-zero stream index
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementGSStream) {
  MDNode *Node = getElement(/*SigId=*/0, "A", /*CompType=*/9,
                            /*SemanticKind=*/0, /*Indices=*/{0},
                            /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1,
                            /*StartRow=*/0, /*StartCol=*/0, /*UsageMask=*/0,
                            /*DynIndexMask=*/0, /*GSStream=*/3);

  Expected<SemanticSignatureElement> Elem =
      SemanticSignatureElement::fromMetadata(Node);
  ASSERT_THAT_EXPECTED(Elem, Succeeded());
  EXPECT_EQ(Elem->GSStream, 3u);
}

//===----------------------------------------------------------------------===//
// Error cases
//===----------------------------------------------------------------------===//

// A null node is not a valid signature element
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementNull) {
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(nullptr),
                       Failed());
}

// A node with the wrong number of operands is rejected
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementWrongOperandCount) {
  MDNode *Node = MDNode::get(Ctx, {getI32(0), getStr("A")});
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(Node), Failed());
}

// A node with an operand of the wrong type is rejected
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementWrongOperandType) {
  // Operand 0 (SigId) should be an integer, not a string
  MDNode *Node = getElement(/*SigId=*/0, "A", /*CompType=*/9,
                            /*SemanticKind=*/0, /*Indices=*/{0},
                            /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1,
                            /*StartRow=*/0, /*StartCol=*/0, /*UsageMask=*/0,
                            /*DynIndexMask=*/0, /*GSStream=*/0);
  SmallVector<Metadata *> Ops(Node->op_begin(), Node->op_end());
  Ops[0] = getStr("not an int");
  EXPECT_THAT_EXPECTED(
      SemanticSignatureElement::fromMetadata(MDNode::get(Ctx, Ops)), Failed());
}

//===--------------------------------------------------------------------===//
// Per-operand range/enum validation
//===--------------------------------------------------------------------===//

// A component type outside dxil::ElementType is rejected
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementInvalidCompType) {
  MDNode *Node = getElement(/*SigId=*/0, "A", /*CompType=*/100,
                            /*SemanticKind=*/0, /*Indices=*/{0},
                            /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1,
                            /*StartRow=*/0, /*StartCol=*/0, /*UsageMask=*/0,
                            /*DynIndexMask=*/0, /*GSStream=*/0);
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(Node), Failed());
}

// A semantic kind outside dxbc::PSV::SemanticKind is rejected
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementInvalidSemanticKind) {
  MDNode *Node = getElement(/*SigId=*/0, "A", /*CompType=*/9,
                            /*SemanticKind=*/100, /*Indices=*/{0},
                            /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1,
                            /*StartRow=*/0, /*StartCol=*/0, /*UsageMask=*/0,
                            /*DynIndexMask=*/0, /*GSStream=*/0);
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(Node), Failed());
}

// An interpolation mode outside dxbc::PSV::InterpolationMode is rejected
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementInvalidInterpMode) {
  MDNode *Node = getElement(/*SigId=*/0, "A", /*CompType=*/9,
                            /*SemanticKind=*/0, /*Indices=*/{0},
                            /*InterpMode=*/100, /*Rows=*/1, /*Cols=*/1,
                            /*StartRow=*/0, /*StartCol=*/0, /*UsageMask=*/0,
                            /*DynIndexMask=*/0, /*GSStream=*/0);
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(Node), Failed());
}

// The number of components per row must be within 1-4
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementInvalidCols) {
  for (uint8_t Cols : {uint8_t(0), uint8_t(5)}) {
    MDNode *Node = getElement(/*SigId=*/0, "A", /*CompType=*/9,
                              /*SemanticKind=*/0, /*Indices=*/{0},
                              /*InterpMode=*/0, /*Rows=*/1, Cols,
                              /*StartRow=*/0, /*StartCol=*/0, /*UsageMask=*/0,
                              /*DynIndexMask=*/0, /*GSStream=*/0);
    EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(Node),
                         Failed());
  }
}

// A start column must be within 0-3 or the unallocated sentinel
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementInvalidStartCol) {
  MDNode *Node = getElement(/*SigId=*/0, "A", /*CompType=*/9,
                            /*SemanticKind=*/0, /*Indices=*/{0},
                            /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1,
                            /*StartRow=*/0, /*StartCol=*/4, /*UsageMask=*/0,
                            /*DynIndexMask=*/0, /*GSStream=*/0);
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(Node), Failed());
}

// The row/col sentinels must be set together
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementInvalidSentinelPair) {
  MDNode *RowOnly = getElement(
      /*SigId=*/0, "A", /*CompType=*/9, /*SemanticKind=*/0, /*Indices=*/{0},
      /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1, /*StartRow=*/UnallocatedRow,
      /*StartCol=*/0, /*UsageMask=*/0, /*DynIndexMask=*/0, /*GSStream=*/0);
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(RowOnly),
                       Failed());

  MDNode *ColOnly = getElement(
      /*SigId=*/0, "A", /*CompType=*/9, /*SemanticKind=*/0, /*Indices=*/{0},
      /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1, /*StartRow=*/0,
      /*StartCol=*/UnallocatedCol, /*UsageMask=*/0, /*DynIndexMask=*/0,
      /*GSStream=*/0);
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(ColOnly),
                       Failed());
}

// The usage and dynamic-index masks are 4-bit values
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementInvalidMasks) {
  MDNode *BadUsage = getElement(
      /*SigId=*/0, "A", /*CompType=*/9, /*SemanticKind=*/0, /*Indices=*/{0},
      /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1, /*StartRow=*/0, /*StartCol=*/0,
      /*UsageMask=*/0x10, /*DynIndexMask=*/0, /*GSStream=*/0);
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(BadUsage),
                       Failed());

  MDNode *BadDyn = getElement(
      /*SigId=*/0, "A", /*CompType=*/9, /*SemanticKind=*/0, /*Indices=*/{0},
      /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1, /*StartRow=*/0, /*StartCol=*/0,
      /*UsageMask=*/0, /*DynIndexMask=*/0x10, /*GSStream=*/0);
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(BadDyn),
                       Failed());
}

// A geometry shader output stream index must be within 0-3
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementInvalidGSStream) {
  MDNode *Node = getElement(/*SigId=*/0, "A", /*CompType=*/9,
                            /*SemanticKind=*/0, /*Indices=*/{0},
                            /*InterpMode=*/0, /*Rows=*/1, /*Cols=*/1,
                            /*StartRow=*/0, /*StartCol=*/0, /*UsageMask=*/0,
                            /*DynIndexMask=*/0, /*GSStream=*/4);
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(Node), Failed());
}

// The number of semantic indices must equal the number of rows
TEST_F(HLSLSemanticSignatureMetadataTest, MetadataToElementIndicesRowMismatch) {
  MDNode *Node = getElement(/*SigId=*/0, "A", /*CompType=*/9,
                            /*SemanticKind=*/0, /*Indices=*/{0},
                            /*InterpMode=*/0, /*Rows=*/2, /*Cols=*/1,
                            /*StartRow=*/0, /*StartCol=*/0, /*UsageMask=*/0,
                            /*DynIndexMask=*/0, /*GSStream=*/0);
  EXPECT_THAT_EXPECTED(SemanticSignatureElement::fromMetadata(Node), Failed());
}

//===--------------------------------------------------------------------===//
// struct -> metadata
//===--------------------------------------------------------------------===//

// A fully populated element emits all 13 operands in order
TEST_F(HLSLSemanticSignatureMetadataTest, ElementToMetadata) {
  SemanticSignatureElement Elem;
  Elem.SigId = 1;
  Elem.SemanticName = "TEXCOORD";
  Elem.CompType = dxil::ElementType::F32;
  Elem.SemanticKind = dxbc::PSV::SemanticKind::Arbitrary;
  Elem.SemanticIndices = {0, 1};
  Elem.InterpMode = dxbc::PSV::InterpolationMode::Undefined;
  Elem.Rows = 2;
  Elem.Cols = 4;
  Elem.StartRow = 1;
  Elem.StartCol = 0;
  Elem.UsageMask = 0;
  Elem.DynIndexMask = 0;
  Elem.GSStream = 0;

  MDNode *Node = Elem.toMetadata(Ctx);
  ASSERT_EQ(Node->getNumOperands(), 13u);
  EXPECT_EQ(getIntOp(Node, 0), 1u);
  EXPECT_EQ(getStrOp(Node, 1), "TEXCOORD");
  EXPECT_EQ(getIntOp(Node, 2), 9u);
  EXPECT_EQ(getIntOp(Node, 3), 0u);
  EXPECT_EQ(Node->getOperand(4).get(), getIndices({0, 1}));
  EXPECT_EQ(getIntOp(Node, 5), 0u);
  EXPECT_EQ(getIntOp(Node, 6), 2u);
  EXPECT_EQ(getIntOp(Node, 7), 4u);
  EXPECT_EQ(getIntOp(Node, 8), 1u);
  EXPECT_EQ(getIntOp(Node, 9), 0u);
  EXPECT_EQ(getIntOp(Node, 10), 0u);
  EXPECT_EQ(getIntOp(Node, 11), 0u);
  EXPECT_EQ(getIntOp(Node, 12), 0u);
}

// System value, non-zero masks and a non-zero stream index are emitted
TEST_F(HLSLSemanticSignatureMetadataTest, ElementToMetadataSystemValue) {
  SemanticSignatureElement Elem;
  Elem.SigId = 1;
  Elem.SemanticName = "SV_Target";
  Elem.CompType = dxil::ElementType::F32;
  Elem.SemanticKind = dxbc::PSV::SemanticKind::Target;
  Elem.SemanticIndices = {1};
  Elem.Rows = 1;
  Elem.Cols = 4;
  Elem.StartRow = 1;
  Elem.StartCol = 0;
  Elem.UsageMask = 0x7;
  Elem.DynIndexMask = 0x1;
  Elem.GSStream = 2;

  MDNode *Node = Elem.toMetadata(Ctx);
  ASSERT_EQ(Node->getNumOperands(), 13u);
  EXPECT_EQ(getStrOp(Node, 1), "SV_Target");
  EXPECT_EQ(getIntOp(Node, 3), 16u);
  EXPECT_EQ(Node->getOperand(4).get(), getIndices({1}));
  EXPECT_EQ(getIntOp(Node, 10), 0x7u);
  EXPECT_EQ(getIntOp(Node, 11), 0x1u);
  EXPECT_EQ(getIntOp(Node, 12), 2u);
}

// An unallocated element emits the row/col sentinels
TEST_F(HLSLSemanticSignatureMetadataTest, ElementToMetadataUnallocated) {
  SemanticSignatureElement Elem;
  Elem.SemanticName = "POSITION";
  Elem.CompType = dxil::ElementType::F32;
  Elem.SemanticIndices = {0};

  MDNode *Node = Elem.toMetadata(Ctx);
  ASSERT_EQ(Node->getNumOperands(), 13u);
  EXPECT_EQ(getIntOp(Node, 8), UnallocatedRow);
  EXPECT_EQ(getIntOp(Node, 9), UnallocatedCol);
}

// Emitting then parsing yields an equivalent element
TEST_F(HLSLSemanticSignatureMetadataTest, ElementRoundTrip) {
  SemanticSignatureElement Elem;
  Elem.SigId = 2;
  Elem.SemanticName = "TEXCOORD";
  Elem.CompType = dxil::ElementType::F32;
  Elem.SemanticKind = dxbc::PSV::SemanticKind::Arbitrary;
  Elem.SemanticIndices = {1};
  Elem.InterpMode = dxbc::PSV::InterpolationMode::LinearNoperspective;
  Elem.Rows = 1;
  Elem.Cols = 4;
  Elem.StartRow = 2;
  Elem.StartCol = 0;
  Elem.UsageMask = 0x7;
  Elem.DynIndexMask = 0;
  Elem.GSStream = 0;

  Expected<SemanticSignatureElement> Parsed =
      SemanticSignatureElement::fromMetadata(Elem.toMetadata(Ctx));
  ASSERT_THAT_EXPECTED(Parsed, Succeeded());
  EXPECT_EQ(Parsed->SigId, Elem.SigId);
  EXPECT_EQ(Parsed->SemanticName, Elem.SemanticName);
  EXPECT_EQ(Parsed->CompType, Elem.CompType);
  EXPECT_EQ(Parsed->SemanticKind, Elem.SemanticKind);
  EXPECT_THAT(Parsed->SemanticIndices,
              testing::ElementsAreArray(Elem.SemanticIndices));
  EXPECT_EQ(Parsed->InterpMode, Elem.InterpMode);
  EXPECT_EQ(Parsed->Rows, Elem.Rows);
  EXPECT_EQ(Parsed->Cols, Elem.Cols);
  EXPECT_EQ(Parsed->StartRow, Elem.StartRow);
  EXPECT_EQ(Parsed->StartCol, Elem.StartCol);
  EXPECT_EQ(Parsed->UsageMask, Elem.UsageMask);
  EXPECT_EQ(Parsed->DynIndexMask, Elem.DynIndexMask);
  EXPECT_EQ(Parsed->GSStream, Elem.GSStream);
}

} // namespace
