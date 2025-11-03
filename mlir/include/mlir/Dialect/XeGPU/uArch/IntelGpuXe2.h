//===--- IntelGpuXe2.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Xe2 uArch definition. Xe2 is the second generation of Intel Xe GPUs.
// This file defines the uArch details for Xe2 and its derived architectures.
// This includes Ponte Vecchio (PVC) and Battlemage (BMG) architectures.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE2_H
#define MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE2_H

#include "mlir/Dialect/XeGPU/uArch/uArchBase.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include <map>
#include <string>

using namespace mlir;
using namespace mlir::xegpu::uArch;

namespace mlir {
namespace xegpu {
namespace uArch {

struct Xe2Plus : public uArch {
  Xe2Plus(StringRef archName, StringRef archDescription,
          llvm::ArrayRef<const Instruction *> instructionRegistry,
          const XeCoreInfo &xeCore)
      : uArch(archName, archDescription, instructionRegistry), xeCore(xeCore) {}
  int getSubgroupSize() const override { return 16; }
  unsigned getGeneralPackedFormatBitSize() const override { return 32; }

protected:
  XeCoreInfo xeCore;
};

//===----------------------------------------------------------------------===//
// uArch instructions
//===----------------------------------------------------------------------===//
struct Subgroup2DBlockStoreInstruction : public Instruction {
  Subgroup2DBlockStoreInstruction()
      : Instruction(InstructionKind::Subgroup2DBlockStore,
                    InstructionScope::Subgroup) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == InstructionKind::Subgroup2DBlockStore;
  }
  // Source :
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html#_add_a_new_section_5_2_x_cl_intel_subgroup_2d_block_io
  std::optional<
      std::tuple<llvm::ArrayRef<int>, llvm::ArrayRef<int>, llvm::ArrayRef<int>>>
  getBlockWidthHeightCount(Type elemTy) const {
    const static int kHeight[] = {1, 2, 4, 8};
    const static int kWidth16[] = {16};
    const static int kWidth32[] = {16};
    const static int kCount[] = {1};
    const int elemByteSize = elemTy.getIntOrFloatBitWidth() / 8;
    if (elemByteSize == 1)
      return std::make_tuple(llvm::ArrayRef<int>(kWidth32),
                             llvm::ArrayRef<int>(kHeight),
                             llvm::ArrayRef<int>(kCount));
    else if (elemByteSize == 2 || elemByteSize == 4)
      return std::make_tuple(llvm::ArrayRef<int>(kWidth16),
                             llvm::ArrayRef<int>(kHeight),
                             llvm::ArrayRef<int>(kCount));
    return std::nullopt;
  }

  int32_t getPackedFormatBitSize() const { return 16; }
};

struct Subgroup2DBlockLoadInstruction : public Instruction {
  Subgroup2DBlockLoadInstruction()
      : Instruction(InstructionKind::Subgroup2DBlockLoad,
                    InstructionScope::Subgroup) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == InstructionKind::Subgroup2DBlockLoad;
  }

  // Source :
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html#_add_a_new_section_5_2_x_cl_intel_subgroup_2d_block_io
  std::optional<
      std::tuple<llvm::ArrayRef<int>, llvm::ArrayRef<int>, llvm::ArrayRef<int>>>
  getBlockWidthHeightCount(Type elemTy, bool hasTransform, bool hasTranspose,
                           bool upConv = false) const {
    static const int kHeightAtLeast1[] = {1, 2, 4, 8, 16, 32};
    static const int kHeightAtLeast8[] = {8, 16, 32};
    static const int kHeightAtLeast16[] = {16, 32};
    static const int kHeightAtLeast32[] = {32};

    static const int kWidth32[] = {32};
    static const int kWidth16[] = {16};
    static const int kWidth8[] = {8};

    static const int32_t kCount1[] = {1};
    static const int32_t kCount2[] = {1, 2};
    static const int32_t kCount4[] = {1, 2, 4};
    static const int32_t kCount4Only[] = {4};
    // (elemBytes, transform, transpose, upConvert)
    using Key = std::tuple<int, uint8_t, uint8_t, uint8_t>;
    // (widths, heights, counts)
    using Value = std::tuple<llvm::ArrayRef<int32_t>, llvm::ArrayRef<int32_t>,
                             llvm::ArrayRef<int32_t>>;
    static const llvm::DenseMap<Key, Value> kMap = {
        {{1, false, false, false}, {kWidth32, kHeightAtLeast1, kCount2}},
        {{1, false, false, true}, {kWidth16, kHeightAtLeast8, kCount4Only}},
        {{2, false, false, false}, {kWidth16, kHeightAtLeast1, kCount2}},
        {{4, false, false, false}, {kWidth16, kHeightAtLeast1, kCount1}},
        // Block Loads with Transform:
        {{1, true, false, false}, {kWidth16, kHeightAtLeast32, kCount4}},
        {{2, true, false, false}, {kWidth16, kHeightAtLeast16, kCount2}},
        // Block Loads with Transpose:
        {{4, false, true, false}, {kWidth8, kHeightAtLeast16, kCount1}},
    };
    const int elemByteSize = elemTy.getIntOrFloatBitWidth() / 8;
    auto it = kMap.find({elemByteSize, hasTransform, hasTranspose, upConv});
    if (it != kMap.end())
      return it->second;
    return std::nullopt;
  }

  int32_t getPackedFormatBitSize() const { return 16; }
};

struct Subgroup2DBlockPrefetchInstruction : public Instruction {
  Subgroup2DBlockPrefetchInstruction()
      : Instruction(InstructionKind::Subgroup2DBlockPrefetch,
                    InstructionScope::Subgroup) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == InstructionKind::Subgroup2DBlockPrefetch;
  }
  // Source :
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_buffer_prefetch.html#_add_a_new_section_6_15_x_sub_group_prefetch_functions
  std::optional<
      std::tuple<llvm::ArrayRef<int>, llvm::ArrayRef<int>, llvm::ArrayRef<int>>>
  getBlockWidthHeightCount(Type elemTy) const {
    static const int kHeightAtLeast1[] = {1, 2, 4, 8, 16, 32};

    static const int kWidth32[] = {32};
    static const int kWidth16[] = {16};

    static const int32_t kCount1[] = {1};
    static const int32_t kCount2[] = {1, 2};
    // elemBytes
    using Key = int;
    // (widths, heights, counts)
    using Value = std::tuple<llvm::ArrayRef<int32_t>, llvm::ArrayRef<int32_t>,
                             llvm::ArrayRef<int32_t>>;
    static const llvm::DenseMap<Key, Value> kMap = {
        {1, {kWidth32, kHeightAtLeast1, kCount2}},
        {2, {kWidth16, kHeightAtLeast1, kCount2}},
        {4, {kWidth16, kHeightAtLeast1, kCount1}},
    };
    const int elemByteSize = elemTy.getIntOrFloatBitWidth() / 8;
    auto it = kMap.find(elemByteSize);
    if (it != kMap.end())
      return it->second;
    return std::nullopt;
  }
  int32_t getPackedFormatBitSize() const { return 16; }
};

struct SubgroupMatrixMultiplyAcc : public Instruction,
                                   public MMAInstructionInterface {
  SubgroupMatrixMultiplyAcc(unsigned packedFormatBitSizeA,
                            unsigned packedFormatBitSizeB)
      : Instruction(InstructionKind::SubgroupMatrixMultiplyAcc,
                    InstructionScope::Subgroup),
        packedFormatBitSizeA(packedFormatBitSizeA),
        packedFormatBitSizeB(packedFormatBitSizeB) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() ==
           InstructionKind::SubgroupMatrixMultiplyAcc;
  }
  // Source:
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html

  // Override all virtuals from MatrixOpInterface
  virtual llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16>
  getSupportedShapes(Type dataType, MMAOpndKind matrixType) override;
  virtual llvm::SmallVector<Type, 8>
  getSupportedTypes(MLIRContext &context, MMAOpndKind matrixType) override;
  virtual bool
  checkSupportedShapesAndTypes(std::pair<uint32_t, uint32_t> AShape,
                               std::pair<uint32_t, uint32_t> BShape,
                               std::pair<uint32_t, uint32_t> CShape,
                               std::pair<uint32_t, uint32_t> DShape, Type AType,
                               Type BType, Type CType, Type DType) override;
  virtual bool checkSupportedTypes(Type AType, Type BType, Type CType,
                                   Type DType) override;
  virtual bool validate(std::pair<uint32_t, uint32_t> AShape,
                        std::pair<uint32_t, uint32_t> BShape,
                        std::pair<uint32_t, uint32_t> CShape,
                        std::pair<uint32_t, uint32_t> DShape, Type AType,
                        Type BType, Type CType, Type DType) override;
  virtual llvm::SmallVector<uint32_t, 8>
  getSupportedM(Type type) const override;
  virtual llvm::SmallVector<uint32_t, 8>
  getSupportedK(Type type) const override;
  virtual llvm::SmallVector<uint32_t, 8>
  getSupportedN(Type type) const override;

  unsigned getPackedFormatBitSizeA() const { return packedFormatBitSizeA; }
  unsigned getPackedFormatBitSizeB() const { return packedFormatBitSizeB; }

protected:
  const unsigned packedFormatBitSizeA;
  const unsigned packedFormatBitSizeB;
};

//===----------------------------------------------------------------------===//
// uArch instances
//===----------------------------------------------------------------------===//

struct PVCuArch final : public Xe2Plus {
  static llvm::ArrayRef<const Instruction *> getInstructionRegistryArr() {
    static const SubgroupMatrixMultiplyAcc dpasInst{16, 32};
    static const Subgroup2DBlockLoadInstruction loadNdInst;
    static const Subgroup2DBlockStoreInstruction storeNdInst;
    static const Subgroup2DBlockPrefetchInstruction prefetchNdInst;
    static const Instruction *arr[] = {&dpasInst, &loadNdInst, &storeNdInst,
                                       &prefetchNdInst};
    return arr;
  }

  PVCuArch()
      : Xe2Plus("pvc",                        // archName
                "Ponte Vecchio Architecture", // archDescription
                getInstructionRegistryArr(),
                XeCoreInfo(8, SharedMemory(512 * 1024, 4), 8, 8) // xeCore
        ) {}
  static const uArch *getInstance() {
    static const PVCuArch instance;
    return reinterpret_cast<const uArch *>(&instance);
  }
};

struct BMGuArch : public Xe2Plus {
  static llvm::ArrayRef<const Instruction *> getInstructionRegistryArr() {
    static const SubgroupMatrixMultiplyAcc dpasInst{16, 32};
    static const Subgroup2DBlockLoadInstruction loadNdInst;
    static const Subgroup2DBlockStoreInstruction storeNdInst;
    static const Subgroup2DBlockPrefetchInstruction prefetchNdInst;
    static const Instruction *arr[] = {&dpasInst, &loadNdInst, &storeNdInst,
                                       &prefetchNdInst};
    return arr;
  }

  BMGuArch()
      : Xe2Plus("bmg",                     // archName
                "Battlemage Architecture", // archDescription
                getInstructionRegistryArr(),
                XeCoreInfo(8, SharedMemory(256 * 1024, 4), 8, 8) // xeCore
        ) {}
  static const uArch *getInstance() {
    static const BMGuArch instance;
    return reinterpret_cast<const uArch *>(&instance);
  }
};

inline const uArch *getUArch(llvm::StringRef archName) {
  if (archName.equals_insensitive("pvc"))
    return PVCuArch::getInstance();
  else if (archName.equals_insensitive("bmg"))
    return BMGuArch::getInstance();
  else
    llvm_unreachable("No matching uArch found");

  return nullptr;
}

} // namespace uArch
} // namespace xegpu
} // namespace mlir

//===----------------------------------------------------------------------===//
// Instruction implementations
//===----------------------------------------------------------------------===//

inline llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16>
SubgroupMatrixMultiplyAcc::getSupportedShapes(Type dataType,
                                              MMAOpndKind matrixType) {
  auto combineVectors = [](const llvm::SmallVector<uint32_t, 8> &a,
                           const llvm::SmallVector<uint32_t, 8> &b)
      -> llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16> {
    llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16> result;
    for (unsigned x : a) {
      for (unsigned y : b) {
        result.emplace_back(x, y);
      }
    }
    return result;
  };

  auto M = getSupportedM(dataType);
  auto K = getSupportedK(dataType);
  auto N = getSupportedN(dataType);
  llvm::SmallVector<std::pair<unsigned, unsigned>, 16> resultMatrix;

  switch (matrixType) {
  case MMAOpndKind::MatrixA:
    resultMatrix = combineVectors(M, K);
    break;
  case MMAOpndKind::MatrixB:
    resultMatrix = combineVectors(K, N);
    break;
  case MMAOpndKind::MatrixC:
    resultMatrix = combineVectors(M, N);
    break;
  case MMAOpndKind::MatrixD:
    resultMatrix = combineVectors(M, N);
    break;
  }
  return resultMatrix;
}

inline llvm::SmallVector<Type, 8>
SubgroupMatrixMultiplyAcc::getSupportedTypes(MLIRContext &context,
                                             MMAOpndKind matrixType) {
  Type bf16Type = BFloat16Type::get(&context);
  Type f16Type = Float16Type::get(&context);
  Type tf32Type = FloatTF32Type::get(&context);
  Type f32Type = Float32Type::get(&context);

  switch (matrixType) {
  case MMAOpndKind::MatrixA:
    return {bf16Type, f16Type, tf32Type};
  case MMAOpndKind::MatrixB:
    return {bf16Type, f16Type, tf32Type};
  case MMAOpndKind::MatrixC:
    return {bf16Type, f16Type, f32Type};
  case MMAOpndKind::MatrixD:
    return {bf16Type, f16Type, f32Type};
  }
  return {};
}

inline bool SubgroupMatrixMultiplyAcc::checkSupportedTypes(Type AType,
                                                           Type BType,
                                                           Type CType,
                                                           Type DType) {
  if (AType.isF16() || BType.isF16()) {
    if (AType != BType || (CType && (!CType.isF32() && !CType.isF16())) ||
        (!DType.isF32() && !DType.isF16())) {
      LDBG() << "Unsupported dpas combinations of Dst, Acc, A and B matrices.";
      return false;
    }
  } else if (AType.isBF16() || BType.isBF16()) {
    if (AType != BType || (CType && (!CType.isF32() && !CType.isBF16())) ||
        (!DType.isF32() && !DType.isBF16())) {
      LDBG() << "Unsupported dpas combinations of Dst, Acc, A and B matrices.";
      return false;
    }
  } else if (AType.isTF32() || BType.isTF32()) {
    if (AType != BType || (CType && (!CType.isF32() && !DType.isF32())) ||
        (!DType.isF32())) {
      LDBG() << "Unsupported dpas combinations of Dst, Acc, A and B matrices.";
      return false;
    }
  } else if (!(AType.isInteger(2) || AType.isInteger(4) ||
               AType.isInteger(8)) &&
             !(BType.isInteger(2) || BType.isInteger(4) ||
               BType.isInteger(8))) {
    LDBG() << "Unsupported dpas combinations of Dst, Acc, A and B matrices.";
    return false;
  }

  return true;
}

inline bool SubgroupMatrixMultiplyAcc::checkSupportedShapesAndTypes(
    std::pair<uint32_t, uint32_t> AShape, std::pair<uint32_t, uint32_t> BShape,
    std::pair<uint32_t, uint32_t> CShape, std::pair<uint32_t, uint32_t> DShape,
    Type AType, Type BType, Type CType, Type DType) {
  auto supportedAShapes = getSupportedShapes(AType, MMAOpndKind::MatrixA);
  auto supportedBShapes = getSupportedShapes(BType, MMAOpndKind::MatrixB);
  auto supportedCShapes = getSupportedShapes(CType, MMAOpndKind::MatrixC);
  auto supportedDShapes = getSupportedShapes(DType, MMAOpndKind::MatrixD);
  return llvm::is_contained(supportedAShapes, AShape) &&
         llvm::is_contained(supportedBShapes, BShape) &&
         llvm::is_contained(supportedCShapes, CShape) &&
         llvm::is_contained(supportedDShapes, DShape) &&
         checkSupportedTypes(AType, BType, CType, DType);
}

inline bool SubgroupMatrixMultiplyAcc::validate(
    std::pair<uint32_t, uint32_t> AShape, std::pair<uint32_t, uint32_t> BShape,
    std::pair<uint32_t, uint32_t> CShape, std::pair<uint32_t, uint32_t> DShape,
    Type AType, Type BType, Type CType, Type DType) {
  return checkSupportedShapesAndTypes(AShape, BShape, CShape, DShape, AType,
                                      BType, CType, DType);
}

inline llvm::SmallVector<uint32_t, 8>
SubgroupMatrixMultiplyAcc::getSupportedM(Type type) const {
  return {1, 2, 3, 4, 5, 6, 7, 8};
}

inline llvm::SmallVector<uint32_t, 8>
SubgroupMatrixMultiplyAcc::getSupportedK(Type type) const {
  // assert if data type is not int or float type
  assert(type.isIntOrFloat() && "Matrix type must be int or float");
  auto bitWidth = type.getIntOrFloatBitWidth();
  uint32_t kSize = 0;
  switch (bitWidth) {
  case 2:
    kSize = 64;
    break;
  case 4:
    kSize = 64;
    break;
  case 8:
    kSize = 32;
    break;
  case 16:
    kSize = 16;
    break;
  case 32:
    kSize = 8;
    break;
  default:
    llvm_unreachable("Invalid int or float");
  }
  return {kSize};
}

inline llvm::SmallVector<uint32_t, 8>
SubgroupMatrixMultiplyAcc::getSupportedN(Type type) const {
  return {16};
}

#endif // MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE2_H
