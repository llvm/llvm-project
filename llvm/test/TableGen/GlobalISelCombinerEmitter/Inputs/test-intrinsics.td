// Dummy intrinsic definitions for TableGen.


def int_1in_1out : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty], []>;
def int_0in_1out : DefaultAttrsIntrinsic<[llvm_i32_ty], [], []>;

def int_sideeffects_1in_1out : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrHasSideEffects]>;

def int_convergent_1in_1out : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrConvergent]>;
def int_convergent_sideeffects_1in_1out : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrConvergent, IntrHasSideEffects]>;
