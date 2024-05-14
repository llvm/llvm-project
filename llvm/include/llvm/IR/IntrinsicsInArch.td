
let TargetPrefix = "inarch" in {  // All intrinsics start with "llvm.inarch.".

def int_inarch_putpixel: ClangBuiltin<"inarchPutPixel">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrHasSideEffects]>;

def int_inarch_flush: ClangBuiltin<"inarchFlush">,
  Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

def int_inarch_rand: ClangBuiltin<"inarchRand">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrHasSideEffects]>;

} // TargetPrefix = "inarch"