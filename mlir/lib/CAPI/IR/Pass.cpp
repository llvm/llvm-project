//===- Pass.cpp - C Interface for General Pass Management APIs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Pass.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/Pass/PassManager.h"
#include <optional>

using namespace mlir;

//===----------------------------------------------------------------------===//
// PassManager/OpPassManager APIs.
//===----------------------------------------------------------------------===//

MlirPassManager mlirPassManagerCreate(MlirContext ctx) {
  return wrap(new PassManager(unwrap(ctx)));
}

MlirPassManager mlirPassManagerCreateOnOperation(MlirContext ctx,
                                                 MlirStringRef anchorOp) {
  return wrap(new PassManager(unwrap(ctx), unwrap(anchorOp)));
}

void mlirPassManagerDestroy(MlirPassManager passManager) {
  delete unwrap(passManager);
}

MlirOpPassManager
mlirPassManagerGetAsOpPassManager(MlirPassManager passManager) {
  return wrap(static_cast<OpPassManager *>(unwrap(passManager)));
}

MlirLogicalResult mlirPassManagerRunOnOp(MlirPassManager passManager,
                                         MlirOperation op) {
  return wrap(unwrap(passManager)->run(unwrap(op)));
}

void mlirPassManagerEnableIRPrinting(MlirPassManager passManager,
                                     bool printBeforeAll, bool printAfterAll,
                                     bool printModuleScope,
                                     bool printAfterOnlyOnChange,
                                     bool printAfterOnlyOnFailure,
                                     MlirOpPrintingFlags flags,
                                     MlirStringRef treePrintingPath) {
  auto shouldPrintBeforePass = [printBeforeAll](Pass *, Operation *) {
    return printBeforeAll;
  };
  auto shouldPrintAfterPass = [printAfterAll](Pass *, Operation *) {
    return printAfterAll;
  };
  if (unwrap(treePrintingPath).empty())
    return unwrap(passManager)
        ->enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                           printModuleScope, printAfterOnlyOnChange,
                           printAfterOnlyOnFailure, /*out=*/llvm::errs(),
                           *unwrap(flags));

  unwrap(passManager)
      ->enableIRPrintingToFileTree(shouldPrintBeforePass, shouldPrintAfterPass,
                                   printModuleScope, printAfterOnlyOnChange,
                                   printAfterOnlyOnFailure,
                                   unwrap(treePrintingPath), *unwrap(flags));
}

void mlirPassManagerEnableVerifier(MlirPassManager passManager, bool enable) {
  unwrap(passManager)->enableVerifier(enable);
}

void mlirPassManagerEnableTiming(MlirPassManager passManager) {
  unwrap(passManager)->enableTiming();
}

MlirOpPassManager mlirPassManagerGetNestedUnder(MlirPassManager passManager,
                                                MlirStringRef operationName) {
  return wrap(&unwrap(passManager)->nest(unwrap(operationName)));
}

MlirOpPassManager mlirOpPassManagerGetNestedUnder(MlirOpPassManager passManager,
                                                  MlirStringRef operationName) {
  return wrap(&unwrap(passManager)->nest(unwrap(operationName)));
}

void mlirPassManagerAddOwnedPass(MlirPassManager passManager, MlirPass pass) {
  unwrap(passManager)->addPass(std::unique_ptr<Pass>(unwrap(pass)));
}

void mlirOpPassManagerAddOwnedPass(MlirOpPassManager passManager,
                                   MlirPass pass) {
  unwrap(passManager)->addPass(std::unique_ptr<Pass>(unwrap(pass)));
}

MlirLogicalResult mlirOpPassManagerAddPipeline(MlirOpPassManager passManager,
                                               MlirStringRef pipelineElements,
                                               MlirStringCallback callback,
                                               void *userData) {
  detail::CallbackOstream stream(callback, userData);
  return wrap(parsePassPipeline(unwrap(pipelineElements), *unwrap(passManager),
                                stream));
}

void mlirPrintPassPipeline(MlirOpPassManager passManager,
                           MlirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(passManager)->printAsTextualPipeline(stream);
}

MlirLogicalResult mlirParsePassPipeline(MlirOpPassManager passManager,
                                        MlirStringRef pipeline,
                                        MlirStringCallback callback,
                                        void *userData) {
  detail::CallbackOstream stream(callback, userData);
  FailureOr<OpPassManager> pm = parsePassPipeline(unwrap(pipeline), stream);
  if (succeeded(pm))
    *unwrap(passManager) = std::move(*pm);
  return wrap(pm);
}

//===----------------------------------------------------------------------===//
// External Pass API.
//===----------------------------------------------------------------------===//

namespace mlir {
class ExternalPass;
} // namespace mlir
DEFINE_C_API_PTR_METHODS(MlirExternalPass, mlir::ExternalPass)

namespace mlir {
/// This pass class wraps external passes defined in other languages using the
/// MLIR C-interface
class ExternalPass : public Pass {
public:
  ExternalPass(TypeID passID, StringRef name, StringRef argument,
               StringRef description, std::optional<StringRef> opName,
               ArrayRef<MlirDialectHandle> dependentDialects,
               MlirExternalPassCallbacks callbacks, void *userData)
      : Pass(passID, opName), id(passID), name(name), argument(argument),
        description(description), dependentDialects(dependentDialects),
        callbacks(callbacks), userData(userData) {
    if (callbacks.construct)
      callbacks.construct(userData);
  }

  ~ExternalPass() override {
    if (callbacks.destruct)
      callbacks.destruct(userData);
  }

  StringRef getName() const override { return name; }
  StringRef getArgument() const override { return argument; }
  StringRef getDescription() const override { return description; }

  void getDependentDialects(DialectRegistry &registry) const override {
    MlirDialectRegistry cRegistry = wrap(&registry);
    for (MlirDialectHandle dialect : dependentDialects)
      mlirDialectHandleInsertDialect(dialect, cRegistry);
  }

  void signalPassFailure() { Pass::signalPassFailure(); }

protected:
  LogicalResult initialize(MLIRContext *ctx) override {
    if (callbacks.initialize)
      return unwrap(callbacks.initialize(wrap(ctx), userData));
    return success();
  }

  bool canScheduleOn(RegisteredOperationName opName) const override {
    if (std::optional<StringRef> specifiedOpName = getOpName())
      return opName.getStringRef() == specifiedOpName;
    return true;
  }

  void runOnOperation() override {
    callbacks.run(wrap(getOperation()), wrap(this), userData);
  }

  std::unique_ptr<Pass> clonePass() const override {
    void *clonedUserData = callbacks.clone(userData);
    return std::make_unique<ExternalPass>(id, name, argument, description,
                                          getOpName(), dependentDialects,
                                          callbacks, clonedUserData);
  }

private:
  TypeID id;
  std::string name;
  std::string argument;
  std::string description;
  std::vector<MlirDialectHandle> dependentDialects;
  MlirExternalPassCallbacks callbacks;
  void *userData;
};
} // namespace mlir

MlirPass mlirCreateExternalPass(MlirTypeID passID, MlirStringRef name,
                                MlirStringRef argument,
                                MlirStringRef description, MlirStringRef opName,
                                intptr_t nDependentDialects,
                                MlirDialectHandle *dependentDialects,
                                MlirExternalPassCallbacks callbacks,
                                void *userData) {
  return wrap(static_cast<mlir::Pass *>(new mlir::ExternalPass(
      unwrap(passID), unwrap(name), unwrap(argument), unwrap(description),
      opName.length > 0 ? std::optional<StringRef>(unwrap(opName))
                        : std::nullopt,
      {dependentDialects, static_cast<size_t>(nDependentDialects)}, callbacks,
      userData)));
}

void mlirExternalPassSignalFailure(MlirExternalPass pass) {
  unwrap(pass)->signalPassFailure();
}
