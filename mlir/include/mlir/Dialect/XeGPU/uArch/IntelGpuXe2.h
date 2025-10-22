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
  unsigned getPackedFormatBitSize() const override { return 16; }
  unsigned getPackedFormatBitSizeGatherScatter() const override { return 32; }

protected:
  XeCoreInfo xeCore;
};

//===----------------------------------------------------------------------===//
// uArch instructions
//===----------------------------------------------------------------------===//
struct StoreNdInstruction : public Instruction {
  StoreNdInstruction()
      : Instruction(InstructionKind::STORE_ND, InstructionScope::Subgroup) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == InstructionKind::STORE_ND;
  }
  // Source :
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html#_add_a_new_section_6_13_x_sub_group_read_and_write_functions
  // Reads 1, 2, 4, or 8 uints of data for each work item in the sub-group from
  // the specified pointer
  llvm::ArrayRef<int> getSortedLaneVectorLengths() const {
    const static int sortedLaneVectorLengths[] = {1, 2, 4, 8};
    return sortedLaneVectorLengths;
  }
};

struct LoadNdInstruction : public Instruction {
  LoadNdInstruction()
      : Instruction(InstructionKind::LOAD_ND, InstructionScope::Subgroup) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == InstructionKind::LOAD_ND;
  }
  // Source :
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html#_add_a_new_section_6_13_x_sub_group_read_and_write_functions
  // Writes 1, 2, 4, or 8 uints of data for each work item in the sub-group to
  // the specified pointer.
  llvm::ArrayRef<int> getSortedLaneVectorLengths() const {
    const static int sortedLaneVectorLengths[] = {1, 2, 4, 8};
    return sortedLaneVectorLengths;
  }
};

struct PrefetchNdInstruction : public Instruction {
  PrefetchNdInstruction()
      : Instruction(InstructionKind::PREFETCH_ND, InstructionScope::Subgroup) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == InstructionKind::PREFETCH_ND;
  }
  // Source :
  // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_buffer_prefetch.html#_add_a_new_section_6_15_x_sub_group_prefetch_functions
  llvm::ArrayRef<int> getSortedLaneVectorLengths(int elementBitwidth) const {
    const static int sortedNarrowTypesLengths[] = {1, 2, 4, 8, 16};
    const static int sortedWideTypesLengths[] = {1, 2, 4, 8};
    switch (elementBitwidth) {
    case 8:
    case 16:
      return sortedNarrowTypesLengths;
    case 32:
    case 64:
      return sortedWideTypesLengths;
    default:
      llvm_unreachable("Unsupported element bitwidth");
    }
  }
};

struct DPASInstruction : public Instruction, public MMAInstructionInterface {
  DPASInstruction(unsigned packedFormatBitSizeA, unsigned packedFormatBitSizeB,
                  unsigned packedFormatBitSizeC)
      : Instruction(InstructionKind::DPAS, InstructionScope::Subgroup),
        packedFormatBitSizeA(packedFormatBitSizeA),
        packedFormatBitSizeB(packedFormatBitSizeB),
        packedFormatBitSizeC(packedFormatBitSizeC) {}
  static bool classof(const Instruction *B) {
    return B->getInstructionKind() == InstructionKind::DPAS;
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
  unsigned getPackedFormatBitSizeC() const { return packedFormatBitSizeC; }

protected:
  const unsigned packedFormatBitSizeA;
  const unsigned packedFormatBitSizeB;
  const unsigned packedFormatBitSizeC;
};

//===----------------------------------------------------------------------===//
// uArch instances
//===----------------------------------------------------------------------===//

struct PVCuArch final : public Xe2Plus {
  static llvm::ArrayRef<const Instruction *> getInstructionRegistryArr() {
    static const DPASInstruction dpasInst{16, 32, 32};
    static const StoreNdInstruction loadNdInst;
    static const StoreNdInstruction storeNdInst;
    static const PrefetchNdInstruction prefetchNdInst;
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
    static const DPASInstruction dpasInst{16, 32, 32};
    static const StoreNdInstruction loadNdInst;
    static const StoreNdInstruction storeNdInst;
    static const PrefetchNdInstruction prefetchNdInst;
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

  return nullptr;
}

} // namespace uArch
} // namespace xegpu
} // namespace mlir

//===----------------------------------------------------------------------===//
// Instruction implementations
//===----------------------------------------------------------------------===//

inline llvm::SmallVector<std::pair<uint32_t, uint32_t>, 16>
DPASInstruction::getSupportedShapes(Type dataType, MMAOpndKind matrixType) {
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
DPASInstruction::getSupportedTypes(MLIRContext &context,
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

inline bool DPASInstruction::checkSupportedTypes(Type AType, Type BType,
                                                 Type CType, Type DType) {
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

inline bool DPASInstruction::checkSupportedShapesAndTypes(
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

inline bool DPASInstruction::validate(std::pair<uint32_t, uint32_t> AShape,
                                      std::pair<uint32_t, uint32_t> BShape,
                                      std::pair<uint32_t, uint32_t> CShape,
                                      std::pair<uint32_t, uint32_t> DShape,
                                      Type AType, Type BType, Type CType,
                                      Type DType) {
  return checkSupportedShapesAndTypes(AShape, BShape, CShape, DShape, AType,
                                      BType, CType, DType);
}

inline llvm::SmallVector<uint32_t, 8>
DPASInstruction::getSupportedM(Type type) const {
  return {1, 2, 3, 4, 5, 6, 7, 8};
}

inline llvm::SmallVector<uint32_t, 8>
DPASInstruction::getSupportedK(Type type) const {
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
DPASInstruction::getSupportedN(Type type) const {
  return {16};
}

#endif // MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE2_H
