# RUN: not llvm-mc -triple=s390x-linux-gnu %s --filetype=asm 2>&1 | FileCheck %s
	
# CHECK: error: instruction requires: vector
# CHECK: vgbm   %v0, 1
# CHECK: ^
	
# CHECK-NOT: error:
# CHECK: .machine z13
# CHECK: vgbm	%v0, 0
# CHECK: .machine zEC12
# CHECK: .machine z13
# CHECK: vgbm	%v0, 3

.machine z13
vgbm    %v0, 0
.machine zEC12
vgbm    %v0, 1
.machine z13
vgbm    %v0, 3

