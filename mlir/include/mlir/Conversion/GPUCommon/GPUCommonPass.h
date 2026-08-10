//===- GPUCommonPass.h - MLIR GPU runtime support -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_
#define MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_

#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {

class LLVMTypeConverter;
class Location;
class ModuleOp;
class Operation;
class RewritePatternSet;
class TypeConverter;

class Pass;

namespace gpu {
enum class AddressSpace : uint32_t;
class GPUModuleOp;
} // namespace gpu

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

#define GEN_PASS_DECL_GPUTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

using LoweringCallback = std::function<std::unique_ptr<llvm::Module>(
    Operation *, llvm::LLVMContext &, StringRef)>;

struct FunctionCallBuilder {
  FunctionCallBuilder(StringRef functionName, Type returnType,
                      ArrayRef<Type> argumentTypes)
      : functionName(functionName),
        functionType(LLVM::LLVMFunctionType::get(returnType, argumentTypes)) {}
  LLVM::CallOp create(Location loc, OpBuilder &builder,
                      ArrayRef<Value> arguments) const;

  StringRef functionName;
  LLVM::LLVMFunctionType functionType;
};

/// Collect a set of patterns to convert from the GPU dialect to LLVM and
/// populate converter for gpu types.
void populateGpuToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool kernelBarePtrCallConv = false,
    bool kernelIntersperseSizeCallConv = false);

/// A function that maps a MemorySpace enum to a target-specific integer value.
using MemorySpaceMapping = std::function<unsigned(gpu::AddressSpace)>;

/// Populates memory space attribute conversion rules for lowering
/// gpu.address_space to integer values.
void populateGpuMemorySpaceAttributeConversions(
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping);
} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_
