//===- AMDGPUTargetInfoTest.cpp - AMDGPU ABI unit tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

namespace {

// RecordFlags' bitmask operators are declared in namespace llvm, so combining
// two of them needs that namespace visible.
using namespace llvm;

using ABIType = llvm::abi::Type;
using llvm::abi::ABICompatInfo;
using llvm::abi::ArgInfo;
using llvm::abi::createAMDGPUTargetInfo;
using llvm::abi::FieldInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::RecordFlags;
using llvm::abi::RequiredArgs;
using llvm::abi::StructPacking;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;

class AMDGPUTargetInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const ABIType *I8;
  const ABIType *I16;
  const ABIType *I32;
  const ABIType *F32;
  const ABIType *Void;
  /// An empty class: a record with no fields, one byte wide, register-passable.
  const ABIType *Empty;

  AMDGPUTargetInfoTest()
      : TB(Alloc), I8(TB.getIntegerType(8, llvm::Align(1), /*Signed=*/true)),
        I16(TB.getIntegerType(16, llvm::Align(2), /*Signed=*/true)),
        I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        F32(TB.getFloatType(llvm::APFloat::IEEEsingle(), llvm::Align(4))),
        Void(TB.getVoidType()),
        Empty(TB.getRecordType({}, llvm::TypeSize::getFixed(8), llvm::Align(1),
                               llvm::Align(1), StructPacking::Default, {}, {},
                               RecordFlags::CanPassInRegisters)) {}

  std::unique_ptr<TargetInfo> target() const {
    return createAMDGPUTargetInfo(const_cast<TypeBuilder &>(TB));
  }

  /// A target configured like a HIP compilation: generic scalar-pointer kernel
  /// arguments are coerced to the global address space.
  std::unique_ptr<TargetInfo> hipTarget() const {
    return createAMDGPUTargetInfo(const_cast<TypeBuilder &>(TB),
                                  /*CoerceGenericPtrArgToGlobal=*/true);
  }

  /// The classification a kernel argument gets on target \p TI.
  const ArgInfo &classifyKernelArg(const ABIType *ArgTy,
                                   std::unique_ptr<FunctionInfo> &FI,
                                   std::unique_ptr<TargetInfo> &TI) {
    FI = FunctionInfo::create(CallingConv::AMDGPU_KERNEL, Void, {ArgTy});
    TI->computeInfo(*FI);
    return FI->getArgInfo(0).Info;
  }

  /// A register-passable record with the given fields, size and alignment.
  const ABIType *recordOf(llvm::ArrayRef<FieldInfo> Fields, uint64_t SizeInBits,
                          llvm::Align Alignment) {
    return TB.getRecordType(Fields, llvm::TypeSize::getFixed(SizeInBits),
                            Alignment, Alignment, StructPacking::Default, {},
                            {}, RecordFlags::CanPassInRegisters);
  }

  /// The argument classification the target computes for a single parameter
  /// under calling convention \p CC.
  const ArgInfo &classifyArg(const ABIType *ArgTy,
                             std::unique_ptr<FunctionInfo> &FI,
                             std::unique_ptr<TargetInfo> &TI,
                             CallingConv::ID CC = CallingConv::C) {
    TI = target();
    FI = FunctionInfo::create(CC, Void, {ArgTy});
    TI->computeInfo(*FI);
    return FI->getArgInfo(0).Info;
  }

  /// The return classification the target computes for \p RetTy.
  const ArgInfo &classifyRet(const ABIType *RetTy,
                             std::unique_ptr<FunctionInfo> &FI,
                             std::unique_ptr<TargetInfo> &TI) {
    TI = target();
    FI = FunctionInfo::create(CallingConv::C, RetTy, {});
    TI->computeInfo(*FI);
    return FI->getReturnInfo();
  }
};

// A direct arg coerced to a pointer in address space \p AS.
static void expectDirectPointerAS(const ArgInfo &Info, unsigned AS) {
  ASSERT_TRUE(Info.isDirect());
  const auto *PT =
      llvm::dyn_cast_or_null<llvm::abi::PointerType>(Info.getCoerceToType());
  ASSERT_NE(PT, nullptr);
  EXPECT_EQ(PT->getAddrSpace(), AS);
}

static void expectUncoercedDirect(const ArgInfo &Info) {
  ASSERT_TRUE(Info.isDirect());
  EXPECT_EQ(Info.getCoerceToType(), nullptr);
}

static void expectDirectInteger(const ArgInfo &Info, unsigned Bits) {
  ASSERT_TRUE(Info.isDirect());
  const ABIType *Coerce = Info.getCoerceToType();
  ASSERT_NE(Coerce, nullptr);
  const auto *IT = llvm::dyn_cast<llvm::abi::IntegerType>(Coerce);
  ASSERT_NE(IT, nullptr);
  EXPECT_EQ(IT->getSizeInBits().getFixedValue(), Bits);
}

static void expectDirectFloat(const ArgInfo &Info,
                              const llvm::fltSemantics &Sem) {
  ASSERT_TRUE(Info.isDirect());
  const ABIType *Coerce = Info.getCoerceToType();
  ASSERT_NE(Coerce, nullptr);
  const auto *FT = llvm::dyn_cast<llvm::abi::FloatType>(Coerce);
  ASSERT_NE(FT, nullptr);
  EXPECT_EQ(FT->getSemantics(), &Sem);
}

// A <= 8-byte aggregate coerces to [2 x i32].
static void expectDirectI32Pair(const ArgInfo &Info) {
  ASSERT_TRUE(Info.isDirect());
  const auto *AT =
      llvm::dyn_cast_or_null<llvm::abi::ArrayType>(Info.getCoerceToType());
  ASSERT_NE(AT, nullptr);
  EXPECT_EQ(AT->getNumElements(), 2u);
  const auto *IT = llvm::dyn_cast<llvm::abi::IntegerType>(AT->getElementType());
  ASSERT_NE(IT, nullptr);
  EXPECT_EQ(IT->getSizeInBits().getFixedValue(), 32u);
}

static void expectIndirect(const ArgInfo &Info, llvm::Align ExpectedAlign,
                           bool ByVal, unsigned AddrSpace) {
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_EQ(Info.getIndirectAlign(), ExpectedAlign);
  EXPECT_EQ(Info.getIndirectByVal(), ByVal);
  EXPECT_EQ(Info.getIndirectAddrSpace(), AddrSpace);
}

// A 32-bit integer and a pointer-sized scalar pass directly in their own type.
TEST_F(AMDGPUTargetInfoTest, ScalarPassesDirect) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ArgInfo &Info = classifyArg(I32, FI, TI);
  expectUncoercedDirect(Info);
  // Ordinary direct args stay flattenable (the getDirect default).
  EXPECT_TRUE(Info.getCanBeFlattened());
  expectUncoercedDirect(classifyArg(F32, FI, TI));
}

// A sub-word integer is sign/zero extended to fill its register.
TEST_F(AMDGPUTargetInfoTest, PromotableIntegerExtends) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ArgInfo &Info = classifyArg(I8, FI, TI);
  ASSERT_TRUE(Info.isExtend());
  EXPECT_TRUE(Info.isSignExt());
}

// An oversized _BitInt (> 128 bits) is passed indirectly, byval, like classic
// DefaultABIInfo. The 128-bit boundary width stays direct.
TEST_F(AMDGPUTargetInfoTest, OversizedBitIntArgumentIsIndirect) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *BitInt256 = TB.getIntegerType(256, llvm::Align(16),
                                               /*Signed=*/true,
                                               /*IsBitInt=*/true);
  expectIndirect(classifyArg(BitInt256, FI, TI), llvm::Align(16),
                 /*ByVal=*/true, llvm::AMDGPUAS::PRIVATE_ADDRESS);

  const ABIType *BitInt128 = TB.getIntegerType(128, llvm::Align(16),
                                               /*Signed=*/true,
                                               /*IsBitInt=*/true);
  EXPECT_TRUE(classifyArg(BitInt128, FI, TI).isDirect());
}

// An oversized _BitInt return is indirect too. Classic DefaultABIInfo's
// getNaturalAlignIndirect defaults ByVal to true, so returns carry it too.
TEST_F(AMDGPUTargetInfoTest, OversizedBitIntReturnIsIndirect) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *BitInt256 = TB.getIntegerType(256, llvm::Align(16),
                                               /*Signed=*/true,
                                               /*IsBitInt=*/true);
  expectIndirect(classifyRet(BitInt256, FI, TI), llvm::Align(16),
                 /*ByVal=*/true, llvm::AMDGPUAS::PRIVATE_ADDRESS);
}

// Aggregates <= 16/32/64 bits pack into i16 / i32 / [2 x i32].
TEST_F(AMDGPUTargetInfoTest, SmallAggregatesPackIntoRegisters) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;

  const ABIType *S16 =
      recordOf({FieldInfo(I8, 0), FieldInfo(I8, 8)}, 16, llvm::Align(1));
  expectDirectInteger(classifyArg(S16, FI, TI), 16);

  const ABIType *S32 =
      recordOf({FieldInfo(I16, 0), FieldInfo(I16, 16)}, 32, llvm::Align(2));
  expectDirectInteger(classifyArg(S32, FI, TI), 32);

  const ABIType *S64 =
      recordOf({FieldInfo(I32, 0), FieldInfo(I32, 32)}, 64, llvm::Align(4));
  expectDirectI32Pair(classifyArg(S64, FI, TI));
}

// An empty struct is dropped from the argument list.
TEST_F(AMDGPUTargetInfoTest, EmptyAggregateIsIgnored) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  EXPECT_TRUE(classifyArg(Empty, FI, TI).isIgnore());
}

// A single-element struct is passed as its inner scalar.
TEST_F(AMDGPUTargetInfoTest, SingleElementStructUnwraps) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *Wrapper = recordOf({FieldInfo(F32, 0)}, 32, llvm::Align(4));
  expectDirectFloat(classifyArg(Wrapper, FI, TI), llvm::APFloat::IEEEsingle());
}

// A large aggregate that does not fit the 16-register budget is passed by
// reference in the private address space.
TEST_F(AMDGPUTargetInfoTest, OversizedAggregateIsIndirectPrivate) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  // Two i32[10] fields => 20 registers, above MaxNumRegsForArgsRet (16).
  const ABIType *ArrTy = TB.getArrayType(I32, /*NumElements=*/10,
                                         /*SizeInBits=*/320);
  const ABIType *Big = recordOf({FieldInfo(ArrTy, 0), FieldInfo(ArrTy, 320)},
                                640, llvm::Align(4));
  expectIndirect(classifyArg(Big, FI, TI), llvm::Align(4), /*ByVal=*/false,
                 llvm::AMDGPUAS::PRIVATE_ADDRESS);
}

// A record that cannot pass in registers (non-trivial C++ type) is passed
// indirectly in the private address space.
TEST_F(AMDGPUTargetInfoTest, NonTrivialRecordIsIndirectPrivate) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *CannotPass = TB.getRecordType(
      {FieldInfo(I32, 0)}, llvm::TypeSize::getFixed(32), llvm::Align(4),
      llvm::Align(4), StructPacking::Default, {}, {}, RecordFlags::IsCXXRecord);
  expectIndirect(classifyArg(CannotPass, FI, TI), llvm::Align(4),
                 /*ByVal=*/false, llvm::AMDGPUAS::PRIVATE_ADDRESS);
}

// A variadic argument bypasses register packing and passes through unchanged,
// kept intact rather than flattened into per-field wire arguments.
TEST_F(AMDGPUTargetInfoTest, VariadicArgumentPassesDirect) {
  std::unique_ptr<TargetInfo> TI = target();
  const ABIType *S16 =
      recordOf({FieldInfo(I8, 0), FieldInfo(I8, 8)}, 16, llvm::Align(1));
  // Zero declared parameters, so the sole argument is variadic.
  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(CallingConv::C, Void, {S16}, RequiredArgs(0));
  TI->computeInfo(*FI);
  expectUncoercedDirect(FI->getArgInfo(0).Info);
  EXPECT_FALSE(FI->getArgInfo(0).Info.getCanBeFlattened());
}

// Kernel aggregate arguments are passed by reference in the constant address
// space (the kernarg segment), never byval.
TEST_F(AMDGPUTargetInfoTest, KernelAggregateIsIndirectConstant) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *S =
      recordOf({FieldInfo(I32, 0), FieldInfo(I32, 32)}, 64, llvm::Align(4));
  expectIndirect(classifyArg(S, FI, TI, CallingConv::AMDGPU_KERNEL),
                 llvm::Align(4), /*ByVal=*/false,
                 llvm::AMDGPUAS::CONSTANT_ADDRESS);
}

// Kernel direct arguments are passed directly and kept intact.
TEST_F(AMDGPUTargetInfoTest, KernelScalarIsDirect) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ArgInfo &Info = classifyArg(I32, FI, TI, CallingConv::AMDGPU_KERNEL);
  expectDirectInteger(Info, 32);
  EXPECT_FALSE(Info.getCanBeFlattened());
}

// Under HIP a generic scalar-pointer kernel argument is coerced to a global
// pointer and passed directly, kept intact.
TEST_F(AMDGPUTargetInfoTest, KernelGenericPointerCoercesToGlobalUnderHIP) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI = hipTarget();
  const ABIType *GenericPtr =
      TB.getPointerType(64, llvm::Align(8), llvm::AMDGPUAS::FLAT_ADDRESS);
  const ArgInfo &Info = classifyKernelArg(GenericPtr, FI, TI);
  expectDirectPointerAS(Info, llvm::AMDGPUAS::GLOBAL_ADDRESS);
  EXPECT_FALSE(Info.getCanBeFlattened());
}

// A pointer already in the global address space is left untouched under HIP.
TEST_F(AMDGPUTargetInfoTest, KernelGlobalPointerUnchangedUnderHIP) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI = hipTarget();
  const ABIType *GlobalPtr =
      TB.getPointerType(64, llvm::Align(8), llvm::AMDGPUAS::GLOBAL_ADDRESS);
  expectDirectPointerAS(classifyKernelArg(GlobalPtr, FI, TI),
                        llvm::AMDGPUAS::GLOBAL_ADDRESS);
}

// Without the HIP rule a generic pointer kernel argument stays generic.
TEST_F(AMDGPUTargetInfoTest, KernelGenericPointerUnchangedByDefault) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *GenericPtr =
      TB.getPointerType(64, llvm::Align(8), llvm::AMDGPUAS::FLAT_ADDRESS);
  expectDirectPointerAS(
      classifyArg(GenericPtr, FI, TI, CallingConv::AMDGPU_KERNEL),
      llvm::AMDGPUAS::FLAT_ADDRESS);
}

// A void return is ignored.
TEST_F(AMDGPUTargetInfoTest, VoidReturnIsIgnored) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  EXPECT_TRUE(classifyRet(Void, FI, TI).isIgnore());
}

// A <= 8-byte aggregate return packs into [2 x i32].
TEST_F(AMDGPUTargetInfoTest, SmallAggregateReturnPacks) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *S64 =
      recordOf({FieldInfo(I32, 0), FieldInfo(I32, 32)}, 64, llvm::Align(4));
  expectDirectI32Pair(classifyRet(S64, FI, TI));
}

// A record that cannot pass in registers is returned indirectly (sret-style,
// ByVal=false) via the target-independent return rule.
TEST_F(AMDGPUTargetInfoTest, NonTrivialRecordReturnIsIndirect) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *CannotPass = TB.getRecordType(
      {FieldInfo(I32, 0)}, llvm::TypeSize::getFixed(32), llvm::Align(4),
      llvm::Align(4), StructPacking::Default, {}, {}, RecordFlags::IsCXXRecord);
  const ArgInfo &Info = classifyRet(CannotPass, FI, TI);
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_FALSE(Info.getIndirectByVal());
}

} // namespace
