// Define intrinsics written by hand

// VEL Intrinsic instructions.
let TargetPrefix = "ve" in {
  def int_ve_vl_svob : GCCBuiltin<"__builtin_ve_vl_svob">,
                       Intrinsic<[], [], [IntrHasSideEffects]>;
}

// Define intrinsics automatically generated
include "llvm/IR/IntrinsicsVEVL.gen.td"
