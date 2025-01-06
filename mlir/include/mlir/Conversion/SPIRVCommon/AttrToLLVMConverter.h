//===- AttrToLLVMConverter.h - SPIR-V attributes conversion to LLVM - C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_SPIRVCOMMON_ATTRTOLLVMCONVERTER_H_
#define MLIR_CONVERSION_SPIRVCOMMON_ATTRTOLLVMCONVERTER_H_

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"

namespace mlir {
unsigned storageClassToAddressSpace(spirv::ClientAPI clientAPI,
                                    spirv::StorageClass storageClass);
} // namespace mlir

#endif // MLIR_CONVERSION_SPIRVCOMMON_ATTRTOLLVMCONVERTER_H_
