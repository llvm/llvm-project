/// Check the interaction of `.option` directives with ymodeswy/ymodeswi.

// RUN: split-file %s %t
// RUN: llvm-mc --triple=riscv64 --mattr=+experimental,+experimental-zyhybrid,+c,+d \
// RUN:   --filetype=obj %t/rv64.s -o %t/rv64.o
// RUN: llvm-objdump -d --show-all-symbols %t/rv64.o | FileCheck %t/rv64.s
// RUN: llvm-mc --triple=riscv32 --mattr=+experimental,+experimental-zyhybrid,+c,+f \
// RUN:   --filetype=obj %t/rv32.s -o %t/rv32.o
// RUN: llvm-objdump -d --show-all-symbols %t/rv32.o | FileCheck %t/rv32.s
// RUN: not llvm-mc --triple=riscv32 --mattr=+experimental,+experimental-zyhybrid,+c,+d \
// RUN:   %t/invalid.s 2>&1 | FileCheck %t/invalid.s --check-prefixes=CHECK,RV32 --implicit-check-not=error:
// RUN: not llvm-mc --triple=riscv64 --mattr=+experimental,+experimental-zyhybrid,+c,+d --defsym=RV64=1 \
// RUN:   %t/invalid.s 2>&1 | FileCheck %t/invalid.s --check-prefixes=CHECK,RV64 --implicit-check-not=error:

//--- rv64.s
/// Start in RVI compatibility mode: C+D implies Zcd.
ymodeswi
// CHECK-LABEL: <$xrv64i2p1_f2p2_d2p2_c2p0_zicsr2p0_zyhybrid0p910_zca1p0_zcd1p0>:
// CHECK-NEXT:    ymodeswi

/// `.option pop` restores the mode from before `.option push`.
.option push
ymodeswy
// CHECK-NEXT:    ymodeswy
/// TODO: This `.option arch` should be implicit in `ymodeswy`.
.option arch, rv64y0p910_c_d_zyhybrid0p910
nop
// CHECK-EMPTY:
// CHECK-NEXT:  <$xrv64y0p910_f2p2_d2p2_c2p0_zicsr2p0_zyhybrid0p910_zca1p0>:
// CHECK-NEXT:    nop
.option pop

/// `.option norvc` and `.option rvc` in capability pointer mode must not imply
/// Zcd until switching back to RVI compatibility mode.
ymodeswy
// CHECK-EMPTY:
// CHECK-NEXT:  <$xrv64i2p1_f2p2_d2p2_c2p0_zicsr2p0_zyhybrid0p910_zca1p0_zcd1p0>:
// CHECK-NEXT:    ymodeswy
/// TODO: This `.option arch` should be implicit in `ymodeswy`.
.option arch, rv64y0p910_c_d_zyhybrid0p910
nop
// CHECK-EMPTY:
// CHECK-NEXT:  <$xrv64y0p910_f2p2_d2p2_c2p0_zicsr2p0_zyhybrid0p910_zca1p0>:
// CHECK-NEXT:    nop
.option norvc
nop
// CHECK-EMPTY:
// CHECK-NEXT:  <$xrv64y0p910_f2p2_d2p2_zicsr2p0_zyhybrid0p910>:
// CHECK-NEXT:    nop
.option rvc
ymodeswi
// CHECK-EMPTY:
// CHECK-NEXT:  <$xrv64y0p910_f2p2_d2p2_c2p0_zicsr2p0_zyhybrid0p910_zca1p0>:
// CHECK-NEXT:    ymodeswi
/// TODO: This `.option arch` should be implicit in `ymodeswi`.
.option arch, rv64i_c_d_zyhybrid0p910
nop
// CHECK-EMPTY:
// CHECK-NEXT:  <$xrv64i2p1_f2p2_d2p2_c2p0_zicsr2p0_zyhybrid0p910_zca1p0_zcd1p0>:
// CHECK-NEXT:    nop
// CHECK-EMPTY:

//--- rv32.s
/// Zcd is compatible with RV32Y, so only Zcf is omitted in capability mode.
.option arch, +d
ymodeswy
// CHECK-LABEL: <$xrv32i2p1_f2p2_d2p2_c2p0_zicsr2p0_zyhybrid0p910_zca1p0_zcd1p0_zcf1p0>:
// CHECK-NEXT:    ymodeswy
/// TODO: This `.option arch` should be implicit in `ymodeswy`.
.option arch, rv32y0p910_c_d_zyhybrid0p910
ymodeswi
// CHECK-EMPTY:
// CHECK-NEXT:  <$xrv32y0p910_f2p2_d2p2_c2p0_zicsr2p0_zyhybrid0p910_zca1p0_zcd1p0>:
// CHECK-NEXT:    ymodeswi
/// TODO: This `.option arch` should be implicit in `ymodeswi`.
.option arch, rv32i_c_d_zyhybrid0p910
nop
// CHECK-EMPTY:
// CHECK-NEXT:  <$xrv32i2p1_f2p2_d2p2_c2p0_zicsr2p0_zyhybrid0p910_zca1p0_zcd1p0_zcf1p0>:
// CHECK-NEXT:    nop
// CHECK-EMPTY:

//--- invalid.s
/// TODO: Support switching between rv*i and rv*y via `.option arch, +y/-y`.
.option arch, +y
// CHECK: :[[#@LINE-1]]:16: error: passing '+y'/'-y' is not supported yet; specify a full arch string instead
.option arch, -y
// CHECK: :[[#@LINE-1]]:16: error: passing '+y'/'-y' is not supported yet; specify a full arch string instead

/// Extensions that conflict with RVY cannot be enabled in capability pointer mode.
ymodeswy
/// TODO: This `.option arch` should be implicit in `ymodeswy`.
.ifdef RV64
.option arch, rv64y0p910_c_d_zyhybrid0p910
.option arch, +zcd
// RV64: :[[#@LINE-1]]:16: error: 'zcd' is incompatible with rv64y base
.option arch, +zcmp
// RV64: :[[#@LINE-1]]:16: error: 'zcmp' is incompatible with rv64y base
.option arch, +zcmt
// RV64: :[[#@LINE-1]]:16: error: 'zcmt' is incompatible with rv64y base
.else
.option arch, rv32y0p910_c_d_zyhybrid0p910
.option arch, +zcf
// RV32: :[[#@LINE-1]]:16: error: 'zcf' is incompatible with rv32y base
.option arch, +zclsd
// RV32: :[[#@LINE-1]]:16: error: 'zclsd' is incompatible with rv32y base
.endif
