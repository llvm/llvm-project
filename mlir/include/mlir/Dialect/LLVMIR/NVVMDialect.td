//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the NVVM IR dialect.
///
//===----------------------------------------------------------------------===//

#ifndef NVVMIR_DIALECT
#define NVVMIR_DIALECT

include "mlir/IR/DialectBase.td"

def NVVM_Dialect : Dialect {
  let name = "nvvm";
  let cppNamespace = "::mlir::NVVM";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "nvvm.kernel"; }
    /// Get the name of the attribute used to annotate max threads required
    /// per CTA for kernel functions.
    static StringRef getMaxntidAttrName() { return "nvvm.maxntid"; }
    /// Get the name of the metadata names for each dimension
    static StringRef getMaxntidXName() { return "maxntidx"; }
    static StringRef getMaxntidYName() { return "maxntidy"; }
    static StringRef getMaxntidZName() { return "maxntidz"; }

    /// Get the name of the attribute used to annotate exact threads required
    /// per CTA for kernel functions.
    static StringRef getReqntidAttrName() { return "nvvm.reqntid"; }
    /// Get the name of the metadata names for each dimension
    static StringRef getReqntidXName() { return "reqntidx"; }
    static StringRef getReqntidYName() { return "reqntidy"; }
    static StringRef getReqntidZName() { return "reqntidz"; }

    /// Get the name of the attribute used to annotate exact CTAs required
    /// per cluster for kernel functions.
    static StringRef getClusterDimAttrName() { return "nvvm.cluster_dim"; }
    /// Get the name of the metadata names for each dimension
    static StringRef getClusterDimXName() { return "cluster_dim_x"; }
    static StringRef getClusterDimYName() { return "cluster_dim_y"; }
    static StringRef getClusterDimZName() { return "cluster_dim_z"; }

    /// Get the name of the attribute used to annotate maximum number of
    /// CTAs per cluster for kernel functions.
    static StringRef getClusterMaxBlocksAttrName() {  return "nvvm.cluster_max_blocks"; }

    /// Get the name of the attribute used to annotate min CTA required
    /// per SM for kernel functions.
    static StringRef getMinctasmAttrName() { return "nvvm.minctasm"; }

    /// Get the name of the attribute used to annotate max number of
    /// registers that can be allocated per thread.
    static StringRef getMaxnregAttrName() { return "nvvm.maxnreg"; }

    /// Get the name of the attribute used to annotate kernel arguments that
    /// are grid constants.
    static StringRef getGridConstantAttrName() { return "nvvm.grid_constant"; }

    /// Get the name of the attribute used to annotate the `.blocksareclusters`
    /// PTX directive for kernel functions.
    /// This attribute implies that the grid launch configuration for the
    /// corresponding kernel function is specifying the number of clusters
    /// instead of the number of thread blocks. This attribute is only
    /// allowed for kernel functions and requires nvvm.reqntid and
    /// nvvm.cluster_dim attributes.
    static StringRef getBlocksAreClustersAttrName() { return "nvvm.blocksareclusters"; }

    /// Get the name of the attribute used to annotate managed global variables.
    static StringRef getManagedAttrName() { return "nvvm.managed"; }

    /// Verify an attribute from this dialect on the argument at 'argIndex' for
    /// the region at 'regionIndex' on the given operation. Returns failure if
    /// the verification failed, success otherwise. This hook may optionally be
    /// invoked from any operation containing a region.
    LogicalResult verifyRegionArgAttribute(Operation *op,
                                           unsigned regionIndex,
                                           unsigned argIndex,
                                           NamedAttribute argAttr) override;
  }];

  let useDefaultAttributePrinterParser = 1;
}

#endif // NVVMIR_DIALECT
