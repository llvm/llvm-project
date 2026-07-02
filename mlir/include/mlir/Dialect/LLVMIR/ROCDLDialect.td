//===-- ROCDLDialect.td - ROCDL IR dialect definition -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCDLIR_DIALECT
#define ROCDLIR_DIALECT

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// ROCDL dialect definitions
//===----------------------------------------------------------------------===//

def ROCDL_Dialect : Dialect {
  let name = "rocdl";
  let cppNamespace = "::mlir::ROCDL";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let summary = "Dialect for wrapping LLVM AMDGPU backend intrinsics and attributes";
  let hasOperationAttrVerify = 1;

  let description = [{
    The ROCDL dialect, like the other platform-specific LLVM dialects, serves
    as the location of wrappers around the AMD-specific intrinsics and attributes
    in LLVM.

    This dialect, like other GPU lowering targets, also contains the infrastructure
    used by the built-in compilation/offloading framework to compile AMD-specific
    LLVM IR into binaries.

    # Dialect inclusion criteria and guidelines

    The operations in this dialect are 1:1 wrappers around their corresponding
    LLVM intrinsics. Operations that do not correspond to intrinsics should not
    be placed in this dialect.

    The definition of a ROCDL op should match its LLVM counterpart. If the
    argument and result types are fixed, they should be specified as type
    constraints, including by overriding the default variadic type on LLVM
    intrinsics by doing a `let results` in the operation definition.

    As an exception, if an `immarg` intrinsic argument corresponds to a backend
    enum, it should be represented in MLIR by an attribute that models that
    enum.

    LLVM attributes do not need to be replicated exactly if it wouldn't be
    easy to do so, but pure operations and ones that read/write memory should
    be annotated as such.

    While LLVM intrinsics currently don't allow constraining the values an
    `any_type` can take, it is acceptable (but not required) to impose such
    constraints if they are known.

    When an LLVM intrinsic uses an `immarg`, this corresponds to an attribute
    in MLIR.

    Human-readable assembly formats (those that, for example, explicitly indicate
    parameter names) may be used, and are encouraged for intrinsics that have
    complex argument schemes and don't have any higher-level wrapper (such as
    in the `amdgpu` dialect).

    While not all existing operations follow this convention, new operations should
    generally provide argument and result types except in cases where they are
    clearly redundant (such as with operations like `rocdl.fmed3`, which doesn't
    need to reiterate the single type at issue multiple times). This convention
    enhances the readability of low-level IR and prevents programmers from needing
    to find non-local type information.

    Dialect-defined discardable attributes (any attribute starting with `rocdl.`
    that has special handling) need to correspond to AMD-specific attributes, metadata,
    or other entities (such as calling conventions) in LLVM, or be needed for
    GPU compilation management. Outside of the compilation infrastructure,
    dialect-specific enums or attributes are extremely unlikely to be needed
    and should be avoided.

    Operation documentation should specify when the operation was introduced
    (if relevant) and include usage examples. Operations should have
    parser/printer tests in `mlir/test/Dialect/LLVMIR/rocdl.mlir` and
    lowering tests in `mlir/test/Target/LLVMIR/rocdl.mlir`.

    # General documentation (What does this op do?)

    While rocdl ops sometimes carry their own documentation, there is no
    expectation that such documentation will exist (or be kept up to date).

    Since ROCDL operations correspond to LLVM intrinsics, the semantics and
    behavior of these operations can be determined by investigating the
    documentation for the corresponding intrinsic. This documentation
    can be found in
    - `llvm/docs/AMDGPUUsage.rst` and
    - The comments of `llvm/include/llvm/IR/IntrinsicsAMDGPU.td`, which
      is where details of the meaning of certain bitfields or of how an
      intrinsic corresponds to hardware instructions are most likely to
      be found.

    Since many intrinsics are themselves minimal wrappers around hardware
    instructions, these documentation sources often do not repeat hardware
    documentation. If an intrinsic appears undocumented, information about
    its behavior will often be available in published ISA descriptions or
    (sometimes known as shader programming guides).

    If an operation doesn't provide usage examples, it is likely that they
    can be found in `mlir/test/Dialect/LLVMIR/rocdl.mlir` (op syntax and
    verification) or `mlir/test/Target/LLVMIR/rocdl.mlir` (translation
    to LLVM IR).
  }];

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "rocdl.kernel"; }
    static constexpr ::llvm::StringLiteral getFlatWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("rocdl.flat_work_group_size");
    }
    static constexpr ::llvm::StringLiteral getReqdWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("rocdl.reqd_work_group_size");
    }
    /// MLIR's gpu-related infrastructure effectively assume uniform workgroup
    /// sizes, so this attribute defaults to "true" on `rocdl.kernel` functions.
    /// It is provided here to allow overriding this assumption.
    static constexpr ::llvm::StringLiteral getUniformWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("rocdl.uniform_work_group_size");
    }
    /// Get the LLVM module flag key for the AMDGPU buffer OOB mode.
    static constexpr ::llvm::StringLiteral
    getModuleFlagKeyBufferOOBModeName() {
      return ::llvm::StringLiteral("amdgpu.buffer.oob.mode");
    }
    /// Get the LLVM module flag key for the AMDGPU tbuffer OOB mode.
    static constexpr ::llvm::StringLiteral
    getModuleFlagKeyTBufferOOBModeName() {
      return ::llvm::StringLiteral("amdgpu.tbuffer.oob.mode");
    }
    /// The address space value that represents global memory.
    static constexpr unsigned kGlobalMemoryAddressSpace = 1;
    /// The address space value that represents shared memory.
    static constexpr unsigned kSharedMemoryAddressSpace = 3;
    /// The address space value that represents constant memory.
    static constexpr unsigned kConstantMemoryAddressSpace = 4;
    /// The address space value that represents private memory.
    static constexpr unsigned kPrivateMemoryAddressSpace = 5;
  }];

  let discardableAttrs = (ins
     "::mlir::UnitAttr":$kernel,
     "::mlir::DenseI32ArrayAttr":$reqd_work_group_size,
     "::mlir::StringAttr":$flat_work_group_size,
     "::mlir::IntegerAttr":$max_flat_work_group_size,
     "::mlir::IntegerAttr":$waves_per_eu,
     "::mlir::BoolAttr":$unsafe_fp_atomics,
     // Correspond to LLVM metadata of the same name
     "::mlir::UnitAttr":$last_use,
     "::mlir::UnitAttr":$no_remote_memory,
     "::mlir::UnitAttr":$no_fine_grained_memory,
     "::mlir::UnitAttr":$ignore_denormal_mode
  );

  let useDefaultAttributePrinterParser = 1;
}

#endif // ROCDLIR_DIALECT
