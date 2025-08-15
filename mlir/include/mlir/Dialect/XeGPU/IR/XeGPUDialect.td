//===- XeGPUDialect.td - XeGPU dialect definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_IR_XEGPUDIALECT_TD
#define MLIR_DIALECT_XEGPU_IR_XEGPUDIALECT_TD

include "mlir/IR/OpBase.td"

def XeGPU_Dialect : Dialect {
    let name = "xegpu";
    let cppNamespace = "::mlir::xegpu";
    let summary = "The XeGPU dialect that models Intel GPU's ISA";
    let description = [{
      The XeGPU dialect closely models a subset of the Xe GPU's ISA, providing an
      abstraction to support high-performance GEMM code generation. It serves as a
      bridge dialect in the MLIR gradual lowering process, working with MLIR memref
      and vector types, and complements the Arith, Math, Vector, and Memref dialects.
      XeGPU operations are introduced for special Xe instructions not modeled by the
      LLVM/SPIR-V dialect, such as DPAS and 2D block load and store.

      It supports a tile-based programming model, decomposing the GEMM kernel into
      large predefined tile sizes at the subgroup and workgroup levels. XeGPU allows
      the high-level GEMM algorithm to be easily expressed. Underneath, it uses
      target-specific recipes and hardware features to achieve optimal performance
      on specific hardware. By decomposing GEMM at submatrix granularity and mapping it
      to registers, it naturally supports optimizations like fusing with neighboring
      operations.
    }];

    let dependentDialects = ["arith::ArithDialect"];

    let useDefaultTypePrinterParser = true;
    let useDefaultAttributePrinterParser = true;

    let extraClassDeclaration = [{
      /// Checks if the given shape can be evenly distributed based on the layout
      /// and data factors provided by the LayoutAttr.
      static bool isEvenlyDistributable(llvm::ArrayRef<int64_t> shape, xegpu::LayoutAttr attr);

      /// drops/slices the shape in the specified dims, and return the rest. e.g.,
      /// for shape = [32, 64, 8], dims = [0, 2], it will return [64]
      template<typename T, typename U>
      static llvm::SmallVector<T> slice(llvm::ArrayRef<T> shape, llvm::ArrayRef<U> dims) {
        llvm::SmallVector<T> result;
        for (auto [i, v]: llvm::enumerate(shape)) {
          if (!llvm::is_contained(dims, i))
            result.push_back(v);
        }
        return result;
      }
    }];
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUDIALECT_TD
