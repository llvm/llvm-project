//===-- Passes.td - ArmSME pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def EnableArmStreaming
    : Pass<"enable-arm-streaming", "mlir::func::FuncOp"> {
  let summary = "Enable Armv9 Streaming SVE mode";
  let description = [{
    Enables the Armv9 Streaming SVE mode [1] for func.func ops by annotating
    them with attributes. See options for more details.

    [1] https://developer.arm.com/documentation/ddi0616/aa
  }];
  let constructor = "mlir::arm_sme::createEnableArmStreamingPass()";
  let options = [
    Option<"mode", "mode", "mlir::arm_sme::ArmStreaming",
          /*default=*/"mlir::arm_sme::ArmStreaming::Default",
          "Select how streaming-mode is managed at the function-level.",
          [{::llvm::cl::values(
                clEnumValN(mlir::arm_sme::ArmStreaming::Default, "default",
						   "Streaming mode is part of the function interface "
						   "(ABI), caller manages PSTATE.SM on entry/exit."),
                clEnumValN(mlir::arm_sme::ArmStreaming::Locally, "locally",
						   "Streaming mode is internal to the function, callee "
						   "manages PSTATE.SM on entry/exit.")
          )}]>,
    Option<"enableZA", "enable-za", "bool", /*default=*/"false",
           "Enable ZA storage array.">,
  ];
  let dependentDialects = ["func::FuncDialect"];
}

def TileAllocation
    : Pass<"allocate-arm-sme-tiles", "mlir::func::FuncOp"> {
  let summary = "Allocate SME tiles";
  let description = [{
    This pass does tile allocation for SME "virtual tiles". It is run at the
    'func.func' op level, replacing 'arm_sme.get_tile_id' ops with (i32) tile
    ids. An error will be emitted when there's no tiles left.
  }];
  let constructor = "mlir::arm_sme::createTileAllocationPass()";
  let dependentDialects = ["func::FuncDialect"];
}

#endif // MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_TD
