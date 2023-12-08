## llvm-objdump should decode instructions that are supported by extensions that are used in Tag_RISCV_arch attribute.
# RUN: llvm-mc -filetype=obj -triple riscv64 %s | \
# RUN:     llvm-objdump -d -M no-aliases - | \
# RUN:     FileCheck %s
	.attribute      5, "rv64gcv"
# CHECK-LABEL: <foo>:
foo:
# CHECK: vsetvli a3, a2, e8, m8, tu, mu
vsetvli a3, a2, e8, m8, tu, mu

# CHECK: fadd.s fs10, fs11, ft8
fadd.s f26, f27, f28

# CHECK: fld ft0, 12(a0)
fld f0, 12(a0)

# CHECK: fmul.d ft0, ft1, ft2, dyn
fmul.d ft0, ft1, ft2, dyn

# CHECK: vfsub.vv v8, v4, v20, v0.t
vfsub.vv v8, v4, v20, v0.t