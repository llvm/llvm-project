//===-- TosaToSPIRVTosa.h - TOSA to SPIR-V Graph/TOSA patterns --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides pass and patterns to lower TOSA IR to SPIR-V Graph/TOSA
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TOSATOSPIRVTOSA_TOSATOSPIRVTOSA_H
#define MLIR_CONVERSION_TOSATOSPIRVTOSA_TOSATOSPIRVTOSA_H

#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

#define GEN_PASS_DECL_TOSATOSPIRVTOSAMARKGRAPHCONSTANTS
#define GEN_PASS_DECL_TOSATOSPIRVTOSA
#include "mlir/Conversion/Passes.h.inc"

namespace tosa {

// Allows users to specify descriptor sets and binding ids on the source
// function inputs and outputs. Use a source-side GraphARM attribute because
// `spirv.interface_var_abi` is verified by the SPIR-V dialect before this
// conversion runs, and result attrs are only accepted on `spirv.ARM.Graph`.
constexpr llvm::StringLiteral graphARMInterfaceVarABIAttrName =
    "grapharm.interface_var_abi";

// Marks a `tosa.const` or `tosa.const_shape` as a SPIR-V Graph constant.
// The conversion pass lowers marked constants to `spirv.ARM.GraphConstant`.
constexpr llvm::StringLiteral graphARMGraphConstantIdAttrName =
    "grapharm.graph_constant_id";

std::unique_ptr<Pass> createTosaToSPIRVTosaMarkGraphConstants();
std::unique_ptr<Pass> createTosaToSPIRVTosa();

spirv::VerCapExtAttr getDefaultVerCapExtAttr(MLIRContext *context);

spirv::TargetEnvAttr constructTargetEnvAttrWithCapExtDefaults(
    MLIRContext *context, spirv::ResourceLimitsAttr limits = {},
    spirv::ClientAPI clientAPI = spirv::ClientAPI::Unknown,
    spirv::Vendor vendorID = spirv::Vendor::Unknown,
    spirv::DeviceType deviceType = spirv::DeviceType::Unknown,
    uint32_t deviceID = spirv::TargetEnvAttr::kUnknownDeviceID);

void populateTosaToSPIRVTosaConversionPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns,
    spirv::TargetEnvAttr targetAttr);
void populateTosaToSPIRVTosaOpsConversionPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns);

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOSPIRVTOSA_TOSATOSPIRVTOSA_H
