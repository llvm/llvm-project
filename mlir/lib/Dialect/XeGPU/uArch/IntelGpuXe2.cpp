#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/YAMLTraits.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace mlir::xegpu::uArch;
using namespace mlir::xegpu::uArch::Xe2Plus;

namespace mlir {
namespace xegpu {
namespace uArch {
namespace Xe2Plus {

std::vector<std::pair<uint32_t, uint32_t>>
DPASInstruction::getSupportedShapes(mlir::Type dataType,
                                    MMAOpndEnum matrixType) {
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
  case MMAOpndEnum::MatrixA:
    resultMatrix = combineVectors(M, K);
    break;
  case MMAOpndEnum::MatrixB:
    resultMatrix = combineVectors(K, N);
    break;
  case MMAOpndEnum::MatrixC:
    resultMatrix = combineVectors(M, N);
    break;
  case MMAOpndEnum::MatrixD:
    resultMatrix = combineVectors(M, N);
    break;
  }
  return resultMatrix;
}

std::vector<mlir::Type>
DPASInstruction::getSupportedTypes(MLIRContext &context,
                                   MMAOpndEnum matrixType) {
  mlir::Type bf16Type = mlir::BFloat16Type::get(&context);
  mlir::Type f16Type = mlir::Float16Type::get(&context);
  mlir::Type tf32Type = mlir::FloatTF32Type::get(&context);
  mlir::Type f32Type = mlir::Float32Type::get(&context);

  switch (matrixType) {
  case MMAOpndEnum::MatrixA:
    return {bf16Type, f16Type, tf32Type};
    break;
  case MMAOpndEnum::MatrixB:
    return {bf16Type, f16Type, tf32Type};
    break;
  case MMAOpndEnum::MatrixC:
    return {bf16Type, f16Type, f32Type};
    break;
  case MMAOpndEnum::MatrixD:
    return {bf16Type, f16Type, f32Type};
    break;
  }
}

bool DPASInstruction::checkSupportedTypes(mlir::Type AType, mlir::Type BType,
                                          mlir::Type CType, mlir::Type DType) {
  if (AType.isF16() || BType.isF16()) {
    if (AType != BType || (CType && (!CType.isF32() && !CType.isF16())) ||
        (!DType.isF32() && !DType.isF16())) {
      llvm::errs()
          << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
          << "Supported types are:\n"
          << "  Dst    |   Acc   |   A   |  B  \n"
          << " f, hf   |  f, hf  |   hf  |  hf \n"
          << "AType: " << AType << " BType: " << BType << " CType: " << CType
          << " DType: " << DType;
      return false;
    }
  } else if (AType.isBF16() || BType.isBF16()) {
    if (AType != BType || (CType && (!CType.isF32() && !CType.isBF16())) ||
        (!DType.isF32() && !DType.isBF16())) {
      llvm::errs()
          << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
          << "Supported types are:\n"
          << "  Dst    |   Acc   |   A   |  B  \n"
          << " f, bf   |  f, bf  |   bf  |  bf \n"
          << "AType: " << AType << " BType: " << BType << " CType: " << CType
          << " DType: " << DType;
      return false;
    }
  } else if (AType.isTF32() || BType.isTF32()) {
    if (AType != BType || (CType && (!CType.isF32() && !DType.isF32())) ||
        (!DType.isF32())) {
      llvm::errs()
          << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
          << "Supported types are:\n"
          << "  Dst    |   Acc   |   A    |   B  \n"
          << "   f     |    f    |  tf32  |  tf32 \n"
          << "AType: " << AType << " BType: " << BType << " CType: " << CType
          << " DType: " << DType;
      return false;
    }
  } else if (!(AType.isInteger(2) || AType.isInteger(4) ||
               AType.isInteger(8)) &&
             !(BType.isInteger(2) || BType.isInteger(4) ||
               BType.isInteger(8))) {
    llvm::errs()
        << "Unsupported dpas combinations of Dst, Acc, A and B matrices, "
        << "Supported types are:\n"
        << "  Dst     |   Acc    |         A           |         B          "
           " \n"
        << " ud, d    |  ud,d    |  ub,b,u4,s4,u2,s2   |  ub,b,u4,s4,u2,s2  "
        << "AType: " << AType << " BType: " << BType << " CType: " << CType
        << " DType: " << DType;
    return false;
  }

  return true;
}

bool DPASInstruction::checkSupportedShapesAndTypes(
    std::pair<uint32_t, uint32_t> AShape, std::pair<uint32_t, uint32_t> BShape,
    std::pair<uint32_t, uint32_t> CShape, std::pair<uint32_t, uint32_t> DShape,
    mlir::Type AType, mlir::Type BType, mlir::Type CType, mlir::Type DType) {
  auto supportedAShapes = getSupportedShapes(AType, MMAOpndEnum::MatrixA);
  auto supportedBShapes = getSupportedShapes(BType, MMAOpndEnum::MatrixB);
  auto supportedCShapes = getSupportedShapes(CType, MMAOpndEnum::MatrixC);
  auto supportedDShapes = getSupportedShapes(DType, MMAOpndEnum::MatrixD);
  return llvm::is_contained(supportedAShapes, AShape) &&
         llvm::is_contained(supportedBShapes, BShape) &&
         llvm::is_contained(supportedCShapes, CShape) &&
         llvm::is_contained(supportedDShapes, DShape) &&
         checkSupportedTypes(AType, BType, CType, DType);
}

bool DPASInstruction::validate(std::pair<uint32_t, uint32_t> AShape,
                               std::pair<uint32_t, uint32_t> BShape,
                               std::pair<uint32_t, uint32_t> CShape,
                               std::pair<uint32_t, uint32_t> DShape,
                               mlir::Type AType, mlir::Type BType,
                               mlir::Type CType, mlir::Type DType) {
  return checkSupportedShapesAndTypes(AShape, BShape, CShape, DShape, AType,
                                      BType, CType, DType);
}

std::vector<uint32_t> DPASInstruction::getSupportedM(mlir::Type type) {
  return {1, 2, 3, 4, 5, 6, 7, 8};
}

std::vector<uint32_t> DPASInstruction::getSupportedK(mlir::Type type) {
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

std::vector<uint32_t> DPASInstruction::getSupportedN(mlir::Type type) {
  return {16};
}

} // namespace Xe2Plus
} // namespace uArch
} // namespace xegpu
} // namespace mlir
