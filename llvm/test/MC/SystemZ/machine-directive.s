# RUN: not llvm-mc -triple=s390x-linux-gnu %s --filetype=asm 2>&1 | FileCheck %s
	
# CHECK: error: instruction requires: vector
# CHECK: vgbm   %v0, 1
# CHECK: ^
	
# CHECK-NOT: error:
# CHECK: .machine push
# CHECK: .machine z13
# CHECK: vgbm	%v0, 0
# CHECK: .machine zEC12
# CHECK: .machine pop
# CHECK: vgbm	%v0, 3
# CHECK: .machine pop

.machine push
.machine z13
.machine push
vgbm    %v0, 0
.machine zEC12
vgbm    %v0, 1
.machine pop
vgbm    %v0, 3
.machine pop
