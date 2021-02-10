//===- TranslateRegistration.cpp - hooks to mlir-translate ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation from SPIR-V binary module to MLIR SPIR-V
// ModuleOp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVModule.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/SPIRV/Deserialization.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Deserialization registration
//===----------------------------------------------------------------------===//

// Deserializes the SPIR-V binary module stored in the file named as
// `inputFilename` and returns a module containing the SPIR-V module.
static OwningModuleRef deserializeModule(const llvm::MemoryBuffer *input,
                                         MLIRContext *context) {
  context->loadDialect<spirv::SPIRVDialect>();

  // Make sure the input stream can be treated as a stream of SPIR-V words
  auto start = input->getBufferStart();
  auto size = input->getBufferSize();
  if (size % sizeof(uint32_t) != 0) {
    emitError(UnknownLoc::get(context))
        << "SPIR-V binary module must contain integral number of 32-bit words";
    return {};
  }

  auto binary = llvm::makeArrayRef(reinterpret_cast<const uint32_t *>(start),
                                   size / sizeof(uint32_t));

  spirv::OwningSPIRVModuleRef spirvModule = spirv::deserialize(binary, context);
  if (!spirvModule)
    return {};

  OwningModuleRef module(ModuleOp::create(FileLineColLoc::get(
      input->getBufferIdentifier(), /*line=*/0, /*column=*/0, context)));
  module->getBody()->push_front(spirvModule.release());

  return module;
}

namespace mlir {
void registerFromSPIRVTranslation() {
  TranslateToMLIRRegistration fromBinary(
      "deserialize-spirv",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        assert(sourceMgr.getNumBuffers() == 1 && "expected one buffer");
        return deserializeModule(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), context);
      });
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// Serialization registration
//===----------------------------------------------------------------------===//

static LogicalResult serializeModule(ModuleOp module, raw_ostream &output) {
  if (!module)
    return failure();

  SmallVector<uint32_t, 0> binary;

  SmallVector<spirv::ModuleOp, 1> spirvModules;
  module.walk([&](spirv::ModuleOp op) { spirvModules.push_back(op); });

  if (spirvModules.empty())
    return module.emitError("found no 'spv.module' op");

  if (spirvModules.size() != 1)
    return module.emitError("found more than one 'spv.module' op");

  if (failed(
          spirv::serialize(spirvModules[0], binary, /*emitDebuginfo=*/false)))
    return failure();

  output.write(reinterpret_cast<char *>(binary.data()),
               binary.size() * sizeof(uint32_t));

  return mlir::success();
}

namespace mlir {
void registerToSPIRVTranslation() {
  TranslateFromMLIRRegistration toBinary(
      "serialize-spirv",
      [](ModuleOp module, raw_ostream &output) {
        return serializeModule(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<spirv::SPIRVDialect>();
      });
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// Round-trip registration
//===----------------------------------------------------------------------===//

static LogicalResult roundTripModule(ModuleOp srcModule, bool emitDebugInfo,
                                     raw_ostream &output) {
  SmallVector<uint32_t, 0> binary;
  MLIRContext *context = srcModule.getContext();
  auto spirvModules = srcModule.getOps<spirv::ModuleOp>();

  if (spirvModules.begin() == spirvModules.end())
    return srcModule.emitError("found no 'spv.module' op");

  if (std::next(spirvModules.begin()) != spirvModules.end())
    return srcModule.emitError("found more than one 'spv.module' op");

  if (failed(spirv::serialize(*spirvModules.begin(), binary, emitDebugInfo)))
    return failure();

  MLIRContext deserializationContext(context->getDialectRegistry());
  // TODO: we should only load the required dialects instead of all dialects.
  deserializationContext.loadAllAvailableDialects();
  // Then deserialize to get back a SPIR-V module.
  spirv::OwningSPIRVModuleRef spirvModule =
      spirv::deserialize(binary, &deserializationContext);
  if (!spirvModule)
    return failure();

  // Wrap around in a new MLIR module.
  OwningModuleRef dstModule(ModuleOp::create(FileLineColLoc::get(
      /*filename=*/"", /*line=*/0, /*column=*/0, &deserializationContext)));
  dstModule->getBody()->push_front(spirvModule.release());
  dstModule->print(output);

  return mlir::success();
}

namespace mlir {
void registerTestRoundtripSPIRV() {
  TranslateFromMLIRRegistration roundtrip(
      "test-spirv-roundtrip",
      [](ModuleOp module, raw_ostream &output) {
        return roundTripModule(module, /*emitDebugInfo=*/false, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<spirv::SPIRVDialect>();
      });
}

void registerTestRoundtripDebugSPIRV() {
  TranslateFromMLIRRegistration roundtrip(
      "test-spirv-roundtrip-debug",
      [](ModuleOp module, raw_ostream &output) {
        return roundTripModule(module, /*emitDebugInfo=*/true, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<spirv::SPIRVDialect>();
      });
}
} // namespace mlir
