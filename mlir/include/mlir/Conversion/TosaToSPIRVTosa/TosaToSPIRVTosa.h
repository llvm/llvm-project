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

namespace mlir {

#define GEN_PASS_DECL_TOSATOSPIRVTOSA
#include "mlir/Conversion/Passes.h.inc"

namespace tosa {

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

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOSPIRVTOSA_TOSATOSPIRVTOSA_H
