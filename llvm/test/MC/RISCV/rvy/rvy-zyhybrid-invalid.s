// RUN: not llvm-mc --triple riscv32 --mattr=+experimental,+experimental-zyhybrid,+c,+d <%s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK --implicit-check-not=error:
// RUN: not llvm-mc --triple riscv64 --mattr=+experimental,+experimental-zyhybrid,+c,+d --defsym=RV64=1 <%s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK --implicit-check-not=error:

ymodew x0, a0, a0
// CHECK: :[[#@LINE-1]]:8: error: register must be a GPR excluding zero (x0)
ymodew f0, a0, a0
// CHECK: :[[#@LINE-1]]:8: error: register must be a GPR excluding zero (x0)
yadd f0, a0, a0
// CHECK: :[[#@LINE-1]]:6: error: register must be a GPR

.option arch, -zyhybrid
ymoder a0, a1
// CHECK: :[[#@LINE-1]]:1: error: instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility){{$}}
ymodew a0, a1, a2
// CHECK: :[[#@LINE-1]]:1: error: instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility){{$}}
ymodeswy
// CHECK: :[[#@LINE-1]]:1: error: instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility){{$}}
ymodeswi
// CHECK: :[[#@LINE-1]]:1: error: instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility){{$}}
