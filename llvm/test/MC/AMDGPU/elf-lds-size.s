// RUN: not llvm-mc -triple=amdgpu6.00-- -defsym LDS_SIZE=65536 -filetype=null %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgpu9.00-- -defsym LDS_SIZE=65536 -filetype=null %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgpu9.50-- -defsym LDS_SIZE=163840 -filetype=null %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgpu10.30-- -mattr=-cumode -defsym LDS_SIZE=131072 -filetype=null %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgpu10.30-- -mattr=+cumode -defsym LDS_SIZE=65536 -filetype=null %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgpu12.50-- -mattr=-cumode -defsym LDS_SIZE=327680 -filetype=null %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgpu12.50-- -mattr=+cumode -defsym LDS_SIZE=327680 -filetype=null %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgpu13.10-- -mattr=-cumode -defsym LDS_SIZE=196608 -filetype=null %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgpu13.10-- -mattr=+cumode -defsym LDS_SIZE=98304 -filetype=null %s 2>&1 | FileCheck %s --implicit-check-not=error:

// LDS symbols can use the physical LDS available in the current mode, even
// when it is larger than the amount a single work-group can address.
.amdgpu_lds at_limit, LDS_SIZE
.amdgpu_lds over_limit, LDS_SIZE + 1
// CHECK: :[[@LINE-1]]:25: error: size is too large
