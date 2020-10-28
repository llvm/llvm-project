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

namespace gpu {
class GPUDialect;
class GPUModuleOp;
} // end namespace gpu

namespace LLVM {
class LLVMDialect;
class LLVMAVX512Dialect;
} // end namespace LLVM

namespace NVVM {
class NVVMDialect;
} // end namespace NVVM

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

namespace vector {
class VectorDialect;
} // end namespace vector

#define GEN_PASS_CLASSES
#include "mlir/Conversion/Passes.h.inc"

} // end namespace mlir

#endif // CONVERSION_PASSDETAIL_H_
