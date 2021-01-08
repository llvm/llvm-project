// Define intrinsics written by hand

// VEL Intrinsic instructions.
let TargetPrefix = "ve" in {
  def int_ve_vl_svob : GCCBuiltin<"__builtin_ve_vl_svob">,
                       Intrinsic<[], [], [IntrHasSideEffects]>;

  def int_ve_vl_pack_f32p : GCCBuiltin<"__builtin_ve_vl_pack_f32p">,
                            Intrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_ptr_ty],
                                      [IntrReadMem]>;
  def int_ve_vl_pack_f32a : GCCBuiltin<"__builtin_ve_vl_pack_f32a">,
                            Intrinsic<[llvm_i64_ty], [llvm_ptr_ty],
                                      [IntrReadMem]>;
}

// Define intrinsics automatically generated
include "llvm/IR/IntrinsicsVEVL.gen.td"
