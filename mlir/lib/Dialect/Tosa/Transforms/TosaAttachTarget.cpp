//===- TosaAttachTarget.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attach target information to a TOSA module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TargetEnv.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tosa {

#define GEN_PASS_DEF_TOSAATTACHTARGET
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"

namespace {

class TosaAttachTarget
    : public tosa::impl::TosaAttachTargetBase<TosaAttachTarget> {
  using Base::Base;

public:
  void runOnOperation() override {
    llvm::SmallVector<Profile, 2> selectedProfiles;
    if (!profiles.empty()) {
      for (const std::string &prof : profiles) {
        std::optional<Profile> profSymbol = symbolizeProfile(prof);
        if (!profSymbol) {
          llvm::SmallVector<Profile> allProfiles = ProfileAttr::getAllValues();
          llvm::errs() << buildUnkownParameterErrorMessage(allProfiles,
                                                           "profile", prof);
          return signalPassFailure();
        }
        selectedProfiles.push_back(profSymbol.value());
      }
    }

    llvm::SmallVector<Extension, 10> selectedExtensions;
    if (!extensions.empty()) {
      for (const std::string &ext : extensions) {
        std::optional<Extension> extSymbol = symbolizeExtension(ext);
        if (!extSymbol) {
          llvm::SmallVector<Extension> allExtensions =
              ExtensionAttr::getAllValues();
          llvm::errs() << buildUnkownParameterErrorMessage(allExtensions,
                                                           "extension", ext);
          return signalPassFailure();
        }
        selectedExtensions.push_back(extSymbol.value());
      }
    }

    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    const auto targetEnvAttr = TargetEnvAttr::get(
        ctx, specificationVersion, level, selectedProfiles, selectedExtensions);
    mod->setAttr(TargetEnvAttr::name, targetEnvAttr);
  }

private:
  template <typename T>
  std::string buildUnkownParameterErrorMessage(llvm::SmallVector<T> &enumValues,
                                               std::string enumName,
                                               std::string unknownArgument) {
    std::string message;
    llvm::raw_string_ostream os(message);
    os << "Unknown TOSA " << enumName << " name passed in '" << unknownArgument
       << "', supported " << enumName << "s are: ";
    llvm::interleaveComma(enumValues, os);
    os << "\n";
    return message;
  }
};

} // namespace

} // namespace tosa
} // namespace mlir
