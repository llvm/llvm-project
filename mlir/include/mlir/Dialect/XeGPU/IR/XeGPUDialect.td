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
      The XeGPU dialect models Intel Xe ISA semantics but works at vector and
      TensorDesc data type. It provides 1:1 mappings to match Xe instructions 
      like DPAS and 2D block load. The matrix size being processed at this level
      exactly matches the hardware instructions or the intrinsic supported by
      the lower-level GPU compiler.
    }];

    // let useDefaultTypePrinterParser = true;
    // let useDefaultAttributePrinterParser = true;
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUDIALECT_TD
