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
#include "llvm/Support/DebugLog.h"
#include <map>
#include <string>
#include <vector>

#define DEBUG_TYPE "xegpu-uarch"

using namespace mlir;
using namespace mlir::xegpu::uArch;

namespace mlir {
namespace xegpu {
namespace uArch {
struct XeCoreInfo {
  uint32_t num_threads;
  SharedMemory shared_memory;
  uint32_t num_vector_units;
  uint32_t num_matrix_units;

  // Constructor
  XeCoreInfo(uint32_t num_threads, const SharedMemory &shared_memory,
             uint32_t num_vector_units, uint32_t num_matrix_units)
      : num_threads(num_threads), shared_memory(shared_memory),
        num_vector_units(num_vector_units), num_matrix_units(num_matrix_units) {
  }
};

struct Xe2Plus : public uArch {
  XeCoreInfo xe_core;
  Xe2Plus(
      const std::string &archName, const std::string &archDescription,
      const XeCoreInfo &xeCore,
      const std::map<RegisterFileType, RegisterFileInfo> &regInfo = {},
      const std::vector<CacheInfo> &cacheInfo = {},
      const std::map<std::string, std::shared_ptr<Instruction>> &instrs = {})
      : uArch(archName, archDescription, regInfo, cacheInfo, instrs),
        xe_core(xeCore) {}
};

// struct to represent DPAS instruction
struct DPASInstruction : public Instruction, public MMAInstructionInterface {
  DPASInstruction()
      : Instruction("dpas",                   // name
                    "Dot Product Accumulate") // description
  {}

  // Override all virtuals from MatrixOpInterface
  virtual std::vector<std::pair<uint32_t, uint32_t>>
  getSupportedShapes(Type dataType, MMAOpndKind matrixType) override;
  virtual std::vector<Type> getSupportedTypes(MLIRContext &context,
                                              MMAOpndKind matrixType) override;
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
  virtual std::vector<uint32_t> getSupportedM(Type type) override;
  virtual std::vector<uint32_t> getSupportedK(Type type) override;
  virtual std::vector<uint32_t> getSupportedN(Type type) override;
};

struct PVCuArch : public Xe2Plus {
  // Maintaines ownership of the instructions owned by PVUarch
  std::vector<std::shared_ptr<Instruction>> owned_instructions;
  PVCuArch()
      : Xe2Plus("pvc",                        // archName
                "Ponte Vecchio Architecture", // archDescription
                XeCoreInfo(8, SharedMemory(512 * 1024, 4), 8, 8), // xeCore
                {/* register_file_info */}, // Optional: empty
                {/* cache_info */},         // Optional: empty
                {/* instructions */}        // Optional: empty
        ) {
    // Intialize register file info
    // GRF
    this->registerFileInfo.emplace(
        RegisterFileType::GRF,
        RegisterFileInfo(
            64 * 1024,                                          // size in bits
            {RegisterFileMode::Small, RegisterFileMode::Large}, // GRF modes
            {128, 256} // registers per thread per mode
            ));
    // Initialize cache info
    // L1 cache, XeCore level
    this->cacheInfo.push_back(
        CacheInfo(512 * 1024, 64, CacheHierarchyLevel::L1));
    // L2 cache, XeStack level
    this->cacheInfo.push_back(
        CacheInfo(512 * 1024, 64, CacheHierarchyLevel::L2));

    // Add the instructions
    auto dpas = std::make_shared<DPASInstruction>();
    instructions.emplace(dpas->getName(), dpas);
    // instructions[dpas->name] = dpas.get();
    owned_instructions.push_back(dpas);
  }
};

struct BMGuArch : public Xe2Plus {
  // Maintaines ownership of the instructions owned by PVUarch
  std::vector<std::shared_ptr<Instruction>> owned_instructions;
  BMGuArch()
      : Xe2Plus("bmg",                     // archName
                "Battlemage Architecture", // archDescription
                XeCoreInfo(8, SharedMemory(256 * 1024, 4), 8, 8), // xeCore
                {/* register_file_info */}, // Optional: empty
                {/* cache_info */},         // Optional: empty
                {/* instructions */}        // Optional: empty)
        ) {
    // Intialize register file info
    // GRF
    this->registerFileInfo[RegisterFileType::GRF] = RegisterFileInfo(
        64 * 1024,                                          // size in bits
        {RegisterFileMode::Small, RegisterFileMode::Large}, // GRF modes
        {128, 256} // registers per thread per mode
    );
    // Initialize cache info
    // L1 cache, XeCore level
    this->cacheInfo.push_back(
        CacheInfo(256 * 1024, 64, CacheHierarchyLevel::L1));
    // L2 cache, XeStack level
    this->cacheInfo.push_back(
        CacheInfo(18 * 1024 * 1024, 256, CacheHierarchyLevel::L2));

    // Add the instructions
    auto dpas = std::make_shared<DPASInstruction>();
    instructions.emplace(dpas->getName(), dpas);
    // instructions[dpas->name] = dpas.get();
    owned_instructions.push_back(dpas);
  }
};
} // namespace uArch
} // namespace xegpu
} // namespace mlir

inline std::vector<std::pair<uint32_t, uint32_t>>
DPASInstruction::getSupportedShapes(Type dataType, MMAOpndKind matrixType) {
  auto combineVectors = [](const std::vector<uint32_t> &a,
                           const std::vector<uint32_t> &b)
      -> std::vector<std::pair<uint32_t, uint32_t>> {
    std::vector<std::pair<uint32_t, uint32_t>> result;
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
  std::vector<std::pair<unsigned, unsigned>> resultMatrix;

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

inline std::vector<Type>
DPASInstruction::getSupportedTypes(MLIRContext &context,
                                   MMAOpndKind matrixType) {
  Type bf16Type = BFloat16Type::get(&context);
  Type f16Type = Float16Type::get(&context);
  Type tf32Type = FloatTF32Type::get(&context);
  Type f32Type = Float32Type::get(&context);

  switch (matrixType) {
  case MMAOpndKind::MatrixA:
    return {bf16Type, f16Type, tf32Type};
    break;
  case MMAOpndKind::MatrixB:
    return {bf16Type, f16Type, tf32Type};
    break;
  case MMAOpndKind::MatrixC:
    return {bf16Type, f16Type, f32Type};
    break;
  case MMAOpndKind::MatrixD:
    return {bf16Type, f16Type, f32Type};
    break;
  }
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

inline std::vector<uint32_t> DPASInstruction::getSupportedM(Type type) {
  return {1, 2, 3, 4, 5, 6, 7, 8};
}

inline std::vector<uint32_t> DPASInstruction::getSupportedK(Type type) {
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

inline std::vector<uint32_t> DPASInstruction::getSupportedN(Type type) {
  return {16};
}

#endif // MLIR_DIALECT_XEGPU_UARCH_INTELGPUXE2H
