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
      // A function with a Shared-ZA interfaces that takes ZA as input.
      I32EnumAttrCase<"InZA", 2, "arm_in_za">,
      // A function with a Shared-ZA interfaces that returns ZA as output.
      I32EnumAttrCase<"OutZA", 3, "arm_out_za">,
      // A function with a Shared-ZA interfaces that takes ZA as input and
      // returns ZA as output.
      I32EnumAttrCase<"InOutZA", 4, "arm_inout_za">,
      // A function with a Shared-ZA interface that does not read ZA and
      // returns with ZA unchanged.
      I32EnumAttrCase<"PreservesZA", 5, "arm_preserves_za">,
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
                 clEnumValN(mlir::arm_sme::ArmZaMode::InZA,
                            "in-za",
                            "The function uses ZA state. The ZA state may "
                            "be used for input."),
                 clEnumValN(mlir::arm_sme::ArmZaMode::OutZA,
                            "out-za",
                            "The function uses ZA state. The ZA state may "
                            "be used for output."),
                 clEnumValN(mlir::arm_sme::ArmZaMode::InOutZA,
                            "inout-za",
                            "The function uses ZA state. The ZA state may "
                            "be used for input and/or output."),
                 clEnumValN(mlir::arm_sme::ArmZaMode::PreservesZA,
                            "preserves-za",
                            "The function shares ZA state. The ZA state may "
                            "not be used for input and/or output and the "
                            "function must return with ZA unchanged")
           )}]>,
    Option<"ifRequiredByOps", "if-required-by-ops", "bool",
           /*default=*/"false",
           "Only apply the selected streaming/ZA modes if the function contains"
           " ops that implement the ArmSMETileOpInterface.">,
    Option<"ifScalableAndSupported", "if-scalable-and-supported",
           "bool", /*default=*/"false",
           "Only apply the selected streaming/ZA modes if the function contains"
           " supported scalable vector operations.">
  ];
  let dependentDialects = ["func::FuncDialect"];
}

def TestTileAllocation
    : Pass<"test-arm-sme-tile-allocation", "mlir::func::FuncOp"> {
  let summary = "Tests SME 'virtual tile' allocation";
  let description = [{
    This pass does tile allocation for SME "virtual tiles". It is run at the
    'func.func' op level, and assigns tile IDs (via an attribute) to all ops
    that implement the `ArmSMETileOpInterface`. Note: This pass is only intended
    to be used for testing, tile allocation is done as part of the ArmSME to
    LLVM conversion (`convert-arm-sme-to-llvm`).
  }];
  let options = [
    Option<"dumpTileLiveRanges", "dump-tile-live-ranges",
           "bool", /*default=*/"false",
           "Dump the live ranges of SME tiles (for debugging)">,
    Option<"preprocessOnly", "preprocess-only", "bool", /*default=*/"false",
           "Only preprocess IR so it is ready for tile allocation "
           "(but do not allocate any tiles)">
  ];
  let dependentDialects = ["func::FuncDialect", "arm_sme::ArmSMEDialect"];
}

def OuterProductFusion
    : Pass<"arm-sme-outer-product-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse 'arm_sme.outerproduct' operations into 2-way or 4-way widening variants";
  let description = [{
    This pass fuses 'arm_sme.outerproduct' operations that are chained via the
    accumulator into 2-way or 4-way ArmSME outer product operations.

    For example:
    ```mlir
    %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
    %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
    %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
    %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

    %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>, vector<[4]xf32>
    %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xf32>, vector<[4]xf32>
    ```

    Becomes:

    ```mlir
    %a_packed = "llvm.intr.experimental.vector.interleave2"(%a0, %a1) : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
    %b_packed = "llvm.intr.experimental.vector.interleave2"(%b0, %b1) : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
    %0 = arm_sme.fmopa_2way %a_packed, %b_packed : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
    ```

    For further information on the 2-way or 4-way widening ops see:
    https://mlir.llvm.org/docs/Dialects/ArmSME/#arm_smefmopa_2way-arm_smefmopa_2wayop
    https://mlir.llvm.org/docs/Dialects/ArmSME/#arm_smesmopa_4way-arm_smesmopa_4wayop
  }];
  let constructor = "mlir::arm_sme::createOuterProductFusionPass()";
  let dependentDialects = ["func::FuncDialect", "arm_sme::ArmSMEDialect", "LLVM::LLVMDialect"];
}

def VectorLegalization
  : Pass<"arm-sme-vector-legalization", "mlir::ModuleOp"> {
  let summary = "Legalize vectors for ArmSME";
  let description = [{
    This pass legalizes vector operations so that they can be lowered to ArmSME.
    This includes decomposing operations that operate on vector types larger
    than a single SME tile (e.g. `vector<[8]x[8]xf32>`) into multiple SME
    tile-sized operations, as well as rewrites needed to get operations into
    forms compatible with SME lowerings.

    Note: Decomposition is currently limited to vector types that are an exact
    multiple of SME tiles. That is scalable in two dimensions, with both the
    rows and columns divisible by the SVE vector length for the element type.
  }];
  let constructor = "mlir::arm_sme::createVectorLegalizationPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "arm_sme::ArmSMEDialect",
    "vector::VectorDialect",
    "arith::ArithDialect"
  ];
}

#endif // MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_TD
