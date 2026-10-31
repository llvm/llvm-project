//===- SPIRVTargetInfoTest.cpp - SPIR-V ABI unit tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/SPIRVAddrSpace.h"
#include "gtest/gtest.h"

namespace {

using namespace llvm;

using ABIType = llvm::abi::Type;
using llvm::abi::ArgInfo;
using llvm::abi::createSPIRVTargetInfo;
using llvm::abi::FieldInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::RecordFlags;
using llvm::abi::StructPacking;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;

class SPIRVTargetInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const ABIType *I32;
  const ABIType *Void;
  /// A trivially copyable record.
  const ABIType *Trivial;
  /// A C++ record that cannot be passed in registers (e.g. a non-trivial copy
  /// constructor).
  const ABIType *NonTrivial;

  SPIRVTargetInfoTest()
      : TB(Alloc), I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        Void(TB.getVoidType()),
        Trivial(TB.getRecordType({FieldInfo(I32, 0), FieldInfo(I32, 32)},
                                 llvm::TypeSize::getFixed(64), llvm::Align(4),
                                 llvm::Align(4), StructPacking::Default, {}, {},
                                 RecordFlags::CanPassInRegisters)),
        NonTrivial(TB.getRecordType(
            {FieldInfo(I32, 0)}, llvm::TypeSize::getFixed(32), llvm::Align(4),
            llvm::Align(4), StructPacking::Default, {}, {},
            RecordFlags::IsCXXRecord)) {}

  /// A target configured like a device compilation whose default address
  /// space is \p DefaultAS.
  std::unique_ptr<TargetInfo> deviceTarget(unsigned DefaultAS) {
    return createSPIRVTargetInfo(TB, DefaultAS,
                                 /*KernelPassAggregatesIndirect=*/true);
  }

  /// A target configured like a non-device compilation (e.g. OpenCL) whose
  /// default address space is \p DefaultAS.
  std::unique_ptr<TargetInfo> nonDeviceTarget(unsigned DefaultAS) {
    return createSPIRVTargetInfo(TB, DefaultAS);
  }

  /// The classification of a single parameter under calling convention \p CC.
  const ArgInfo &classifyArg(const ABIType *ArgTy,
                             std::unique_ptr<TargetInfo> Target,
                             CallingConv::ID CC,
                             std::unique_ptr<FunctionInfo> &FI,
                             std::unique_ptr<TargetInfo> &TI) {
    TI = std::move(Target);
    FI = FunctionInfo::create(CC, Void, {ArgTy});
    TI->computeInfo(*FI);
    return FI->getArgInfo(0).Info;
  }

  /// The return classification of \p RetTy on \p Target.
  const ArgInfo &classifyRet(const ABIType *RetTy,
                             std::unique_ptr<TargetInfo> Target,
                             std::unique_ptr<FunctionInfo> &FI,
                             std::unique_ptr<TargetInfo> &TI) {
    TI = std::move(Target);
    FI = FunctionInfo::create(CallingConv::SPIR_FUNC, RetTy, {});
    TI->computeInfo(*FI);
    return FI->getReturnInfo();
  }

  const ABIType *pointerIn(unsigned AS, uint64_t Size = 64) {
    return TB.getPointerType(Size, llvm::Align(Size / 8), AS,
                             /*HasAddressSpaceQualifier=*/AS != 0);
  }
};

static void expectIndirect(const ArgInfo &Info, bool ByVal) {
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_EQ(Info.getIndirectAlign(), llvm::Align(4));
  EXPECT_EQ(Info.getIndirectByVal(), ByVal);
  EXPECT_EQ(Info.getIndirectAddrSpace(), SPIRVAS::Function);
}

// A direct arg coerced to a CrossWorkgroup pointer of \p Size bits.
static void expectCrossWorkgroupPointer(const ArgInfo &Info, uint64_t Size) {
  ASSERT_TRUE(Info.isDirect());
  EXPECT_FALSE(Info.getCanBeFlattened());
  const auto *PT =
      llvm::dyn_cast_or_null<llvm::abi::PointerType>(Info.getCoerceToType());
  ASSERT_NE(PT, nullptr);
  EXPECT_EQ(PT->getAddrSpace(), SPIRVAS::CrossWorkgroup);
  EXPECT_EQ(PT->getSizeInBits().getFixedValue(), Size);
  EXPECT_EQ(PT->getAlignment(), llvm::Align(Size / 8));
}

static void expectUncoercedDirect(const ArgInfo &Info) {
  ASSERT_TRUE(Info.isDirect());
  EXPECT_EQ(Info.getCoerceToType(), nullptr);
}

// Outside kernels the default classification applies: aggregates are passed
// byval and returned indirectly.
TEST_F(SPIRVTargetInfoTest, NonKernelUsesDefaultClassification) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  expectIndirect(classifyArg(Trivial, deviceTarget(SPIRVAS::Generic),
                             CallingConv::SPIR_FUNC, FI, TI),
                 /*ByVal=*/true);
  expectIndirect(classifyArg(NonTrivial, deviceTarget(SPIRVAS::Generic),
                             CallingConv::SPIR_FUNC, FI, TI),
                 /*ByVal=*/false);

  TI = deviceTarget(SPIRVAS::Generic);
  FI = FunctionInfo::create(CallingConv::SPIR_FUNC, Trivial, {});
  TI->computeInfo(*FI);
  ASSERT_TRUE(FI->getReturnInfo().isIndirect());
  EXPECT_EQ(FI->getReturnInfo().getIndirectAddrSpace(), SPIRVAS::Function);
}

// A kernel aggregate is copied byval for device compilations, even when the
// record could not otherwise be passed by value.
TEST_F(SPIRVTargetInfoTest, KernelAggregateIsByValOnDevice) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  expectIndirect(classifyArg(NonTrivial, deviceTarget(SPIRVAS::Generic),
                             CallingConv::SPIR_KERNEL, FI, TI),
                 /*ByVal=*/true);
  expectIndirect(classifyArg(NonTrivial, nonDeviceTarget(SPIRVAS::Function),
                             CallingConv::SPIR_KERNEL, FI, TI),
                 /*ByVal=*/false);
}

// A default address space kernel pointer is passed as a CrossWorkgroup
// pointer, whichever address space the language's default maps to, for device
// and non-device compilations alike.
TEST_F(SPIRVTargetInfoTest, KernelDefaultPointerCoercesToCrossWorkgroup) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  for (unsigned DefaultAS : {SPIRVAS::Generic, SPIRVAS::Function}) {
    for (uint64_t Size : {64, 32}) {
      const ArgInfo &Info =
          classifyArg(pointerIn(DefaultAS, Size), deviceTarget(DefaultAS),
                      CallingConv::SPIR_KERNEL, FI, TI);
      expectCrossWorkgroupPointer(Info, Size);
    }
  }
  expectCrossWorkgroupPointer(classifyArg(pointerIn(SPIRVAS::Function),
                                          nonDeviceTarget(SPIRVAS::Function),
                                          CallingConv::SPIR_KERNEL, FI, TI),
                              64);
}

// Only default address space pointers in kernels are coerced.
TEST_F(SPIRVTargetInfoTest, OtherPointersAreNotCoerced) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  expectUncoercedDirect(classifyArg(pointerIn(SPIRVAS::Workgroup),
                                    deviceTarget(SPIRVAS::Generic),
                                    CallingConv::SPIR_KERNEL, FI, TI));
  expectUncoercedDirect(classifyArg(pointerIn(SPIRVAS::Generic),
                                    deviceTarget(SPIRVAS::Generic),
                                    CallingConv::SPIR_FUNC, FI, TI));
}

// An atomic pointer kernel argument is coerced like the pointer itself.
TEST_F(SPIRVTargetInfoTest, KernelAtomicPointerCoercesToCrossWorkgroup) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  const ABIType *AtomicPtr =
      TB.getAtomicType(pointerIn(SPIRVAS::Generic), 64, llvm::Align(8));
  expectCrossWorkgroupPointer(classifyArg(AtomicPtr,
                                          deviceTarget(SPIRVAS::Generic),
                                          CallingConv::SPIR_KERNEL, FI, TI),
                              64);
}

// A record that cannot be copied is returned through an sret pointer in the
// default address space.
TEST_F(SPIRVTargetInfoTest, NonTrivialReturnUsesDefaultAddrSpace) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  for (unsigned DefaultAS : {SPIRVAS::Generic, SPIRVAS::Function}) {
    const ArgInfo &Info =
        classifyRet(NonTrivial, createSPIRVTargetInfo(TB, DefaultAS), FI, TI);
    ASSERT_TRUE(Info.isIndirect());
    EXPECT_FALSE(Info.getIndirectByVal());
    EXPECT_EQ(Info.getIndirectAddrSpace(), DefaultAS);
  }
  // Other aggregates are still returned in the alloca address space.
  const ArgInfo &Info =
      classifyRet(Trivial, deviceTarget(SPIRVAS::Generic), FI, TI);
  ASSERT_TRUE(Info.isIndirect());
  EXPECT_EQ(Info.getIndirectAddrSpace(), SPIRVAS::Function);
}

// Without a 128-bit integer type, a _BitInt wider than 64 bits is passed
// indirectly.
TEST_F(SPIRVTargetInfoTest, BitIntWiderThan64IsIndirect) {
  std::unique_ptr<FunctionInfo> FI;
  std::unique_ptr<TargetInfo> TI;
  auto BitInt = [&](unsigned Bits) {
    return TB.getIntegerType(Bits, llvm::Align(8), /*Signed=*/true,
                             /*IsBitInt=*/true);
  };
  EXPECT_TRUE(classifyArg(BitInt(64),
                          createSPIRVTargetInfo(TB, SPIRVAS::Function),
                          CallingConv::SPIR_FUNC, FI, TI)
                  .isDirect());
  EXPECT_TRUE(classifyArg(BitInt(65),
                          createSPIRVTargetInfo(TB, SPIRVAS::Function),
                          CallingConv::SPIR_FUNC, FI, TI)
                  .isIndirect());
}

} // namespace
