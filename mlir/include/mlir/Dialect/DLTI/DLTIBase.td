//===- DLTIBase.td - Target information dialect base defs ---*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DLTI_BASE
#define DLTI_BASE

include "mlir/IR/OpBase.td"

def DLTI_Dialect : Dialect {
  let name = "dlti";
  let cppNamespace = "::mlir";
  let hasOperationAttrVerify = 1;

  let description = [{
    The Data Layout and Target Information (DLTI) dialect is intended to hold
    attributes and other components pertaining to descriptions of in-memory
    data layout and compilation targets.
  }];

  let extraClassDeclaration = [{
    // Top-level attribute name for arbitrary description.
    constexpr const static ::llvm::StringRef
    kMapAttrName = "dlti.map";

    // Top-level attribute name for data layout description.
    constexpr const static ::llvm::StringRef
    kDataLayoutAttrName = "dlti.dl_spec";

    // Top-level attribute name for target system description.
    constexpr const static ::llvm::StringRef
    kTargetSystemDescAttrName = "dlti.target_system_spec";

    // Top-level attribute name for target device description.
    constexpr const static ::llvm::StringRef
    kTargetDeviceDescAttrName = "dlti.target_device_spec";

    // Constants used in entries.
    constexpr const static ::llvm::StringRef
    kDataLayoutEndiannessKey = "dlti.endianness";

    constexpr const static ::llvm::StringRef
    kDataLayoutEndiannessBig = "big";

    constexpr const static ::llvm::StringRef
    kDataLayoutEndiannessLittle = "little";

    constexpr const static ::llvm::StringRef
    kDataLayoutAllocaMemorySpaceKey = "dlti.alloca_memory_space";
    
    constexpr const static ::llvm::StringRef
    kDataLayoutProgramMemorySpaceKey = "dlti.program_memory_space";

    constexpr const static ::llvm::StringRef
    kDataLayoutGlobalMemorySpaceKey = "dlti.global_memory_space";

    constexpr const static ::llvm::StringRef
    kDataLayoutStackAlignmentKey = "dlti.stack_alignment";
  }];

  let useDefaultAttributePrinterParser = 1;
}

def HasDefaultDLTIDataLayout : NativeOpTrait<"HasDefaultDLTIDataLayout"> {
  let cppNamespace = "::mlir";
}

#endif // DLTI_BASE
