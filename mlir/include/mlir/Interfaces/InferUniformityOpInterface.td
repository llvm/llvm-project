//===- InferUniformityOpInterface.td - Uniformity ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface through which operations participate in uniformity
// (divergence) analysis on SIMT targets.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERUNIFORMITYOPINTERFACE
#define MLIR_INTERFACES_INFERUNIFORMITYOPINTERFACE

include "mlir/IR/OpBase.td"

def InferUniformityOpInterface : OpInterface<"InferUniformityOpInterface"> {
  let description = [{
    Allows operations to participate in uniformity analysis, which determines
    for each SSA value the widest group of threads (subgroup, workgroup,
    cluster, the whole launch) within which every thread observes the same
    value.

    Operations that read thread identity (`gpu.thread_id`), communicate across
    lanes (`gpu.subgroup_broadcast`, `gpu.shuffle`) or define values for a
    region executed by a subset of threads (`gpu.warp_execute_on_lane_0`)
    implement this interface to describe how uniformity flows through them.
    Operations that compute a function of their operands only need not
    implement it: the analysis joins the uniformity of the operands.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
      Infer the uniformity of the values defined by this operation, given the
      uniformity of its operands, by calling `setUniformity` for each of them.

      The values that may be set are the results of the operation and the
      entry block arguments of its regions. A value that is not set keeps the
      uniformity computed by the analysis: for a region entry block argument
      or a result that region control flow forwards a value to, the join of
      the forwarded values; for any other value, divergent.

      For an operation implementing `RegionBranchOpInterface`, setting any
      successor input of a region successor (a result, or an entry block
      argument of a region) makes the operation authoritative for all
      successor inputs of that successor: the analysis then does not join the
      forwarded values into them. This is how an operation whose results are
      not the values yielded from its regions, such as a broadcast, describes
      itself.

      Operands whose uniformity is not known yet are passed as uninitialized;
      `Uniformity::join` treats them as neutral.
    }],
    /*retTy=*/"void",
    /*methodName=*/"inferUniformity",
    /*args=*/(ins "::llvm::ArrayRef<::mlir::Uniformity>":$operandUniformity,
                  "::mlir::SetUniformityFn":$setUniformity)>,

    InterfaceMethod<[{
      Returns true if the regions of this operation are the entry of a SIMT
      launch: the threads executing them are not the threads executing the
      operation itself, so the control flow enclosing the operation does not
      narrow the uniformity of their execution. `gpu.launch` returns true.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isLaunchBoundary",
    /*args=*/(ins),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{ return false; }]>
  ];
}

#endif // MLIR_INTERFACES_INFERUNIFORMITYOPINTERFACE
