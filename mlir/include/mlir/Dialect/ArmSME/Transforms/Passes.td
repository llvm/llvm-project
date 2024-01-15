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
include "mlir/IR/EnumAttr.td"

def ArmStreamingMode : I32EnumAttr<"ArmStreamingMode", "Armv9 Streaming SVE mode",
    [
      I32EnumAttrCase<"Disabled", 0, "disabled">,
      // Streaming: Streaming-mode is part of the function interface (ABI).
      I32EnumAttrCase<"Streaming", 1, "arm_streaming">,
      // StreamingLocally: PSTATE.SM is kept internal and the callee manages it
      // on entry/exit.
      I32EnumAttrCase<"StreamingLocally", 2, "arm_locally_streaming">,
      // StreamingCompatible: the function may be entered in either
      // non-streaming mode (PSTATE.SM=0) or in streaming mode (PSTATE.SM=1)
      I32EnumAttrCase<"StreamingCompatible", 3, "arm_streaming_compatible">,
    ]>{
  let cppNamespace = "mlir::arm_sme";
  let genSpecializedAttr = 0;
}

// https://arm-software.github.io/acle/main/acle.html#sme-attributes-relating-to-za
// See also the LLVM definitions: https://llvm.org/docs/AArch64SME.html
//
// Various frontends (e.g. Flang) that build on top of this may restrict or
// enforce how these attributes are used, both individually and in terms of
// combinations that are allowed.
//
// The MLIR interface here does not make any attempt to perform any checking,
// it is up to the higher level to ensure that these attributes are used in a
// way that both makes sense and is legal according to the Arm architecture.
def ArmZaMode : I32EnumAttr<"ArmZaMode", "Armv9 ZA storage mode",
    [
      I32EnumAttrCase<"Disabled", 0, "disabled">,
      // A function's ZA state is created on entry and destroyed on exit.
      I32EnumAttrCase<"NewZA", 1, "arm_new_za">,
      // A function that preserves ZA state.
      I32EnumAttrCase<"PreservesZA", 2, "arm_preserves_za">,
      // A function that uses ZA state as input and/or output
      I32EnumAttrCase<"SharedZA", 3, "arm_shared_za">,
    ]>{
  let cppNamespace = "mlir::arm_sme";
  let genSpecializedAttr = 0;
}

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
    Option<"streamingMode", "streaming-mode", "mlir::arm_sme::ArmStreamingMode",
          /*default=*/"mlir::arm_sme::ArmStreamingMode::Streaming",
          "Select how streaming-mode is managed at the function-level.",
          [{::llvm::cl::values(
                clEnumValN(mlir::arm_sme::ArmStreamingMode::Disabled,
                           "disabled", "Streaming mode is disabled."),
                clEnumValN(mlir::arm_sme::ArmStreamingMode::Streaming,
                           "streaming",
                           "Streaming mode is part of the function interface "
                           "(ABI), caller manages PSTATE.SM on entry/exit."),
                clEnumValN(mlir::arm_sme::ArmStreamingMode::StreamingLocally,
                           "streaming-locally",
                           "Streaming mode is internal to the function, callee "
                           "manages PSTATE.SM on entry/exit."),
                clEnumValN(mlir::arm_sme::ArmStreamingMode::StreamingCompatible,
                           "streaming-compatible",
                           "Function supports both streaming and non-streaming "
                           "modes.")
          )}]>,
    Option<"zaMode", "za-mode", "mlir::arm_sme::ArmZaMode",
           /*default=*/"mlir::arm_sme::ArmZaMode::Disabled",
           "Select how ZA-storage is managed at the function-level.",
           [{::llvm::cl::values(
                 clEnumValN(mlir::arm_sme::ArmZaMode::Disabled,
                            "disabled", "ZA storage is disabled."),
                 clEnumValN(mlir::arm_sme::ArmZaMode::NewZA,
                            "new-za",
                            "The function has ZA state. The ZA state is "
                            "created on entry and destroyed on exit."),
                 clEnumValN(mlir::arm_sme::ArmZaMode::PreservesZA,
                            "preserves-za",
                            "The function preserves ZA state. The ZA state is "
                            "saved on entry and restored on exit."),
                 clEnumValN(mlir::arm_sme::ArmZaMode::SharedZA,
                            "shared-za",
                            "The function uses ZA state. The ZA state may "
                            "be used for input and/or output.")
           )}]>,
    Option<"onlyIfRequiredByOps", "only-if-required-by-ops", "bool",
           /*default=*/"false",
           "Only apply the selected streaming/ZA modes if the function "
           " contains ops that require them.">
  ];
  let dependentDialects = ["func::FuncDialect"];
}

def TileAllocation
    : Pass<"allocate-arm-sme-tiles", "mlir::func::FuncOp"> {
  let summary = "Allocate SME tiles";
  let description = [{
    This pass does tile allocation for SME "virtual tiles". It is run at the
    'func.func' op level, and assigns tile IDs (via an attribute) to all ops
    that implement the `ArmSMETileOpInterface`. An error will be emitted when
    there's no tiles left.
  }];
  let constructor = "mlir::arm_sme::createTileAllocationPass()";
  let dependentDialects = ["func::FuncDialect"];
}

#endif // MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_TD
