//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_PASSDETAIL_H_
#define CONVERSION_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class AffineDialect;
class StandardOpsDialect;

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace acc {
class OpenACCDialect;
} // end namespace acc

namespace complex {
class ComplexDialect;
} // end namespace complex

namespace gpu {
class GPUDialect;
class GPUModuleOp;
} // end namespace gpu

namespace LLVM {
class LLVMDialect;
} // end namespace LLVM

namespace NVVM {
class NVVMDialect;
} // end namespace NVVM

namespace math {
class MathDialect;
} // end namespace math

namespace memref {
class MemRefDialect;
} // end namespace memref

namespace omp {
class OpenMPDialect;
} // end namespace omp

namespace pdl_interp {
class PDLInterpDialect;
} // end namespace pdl_interp

namespace ROCDL {
class ROCDLDialect;
} // end namespace ROCDL

namespace scf {
class SCFDialect;
} // end namespace scf

namespace spirv {
class SPIRVDialect;
} // end namespace spirv

namespace tensor {
class TensorDialect;
} // end namespace tensor

namespace tosa {
class TosaDialect;
} // end namespace tosa

namespace vector {
class VectorDialect;
} // end namespace vector

namespace arm_neon {
class ArmNeonDialect;
} // end namespace arm_neon

#define GEN_PASS_CLASSES
#include "mlir/Conversion/Passes.h.inc"

} // end namespace mlir

#endif // CONVERSION_PASSDETAIL_H_
