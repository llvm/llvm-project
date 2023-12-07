// RUN: llvm-mc -triple x86_64-unknown-unknown %s -o -      | FileCheck %s
	
// CHECK-NOT: .set var_xdata
var_xdata = %rcx

// CHECK: xorq %rcx, %rcx
xorq var_xdata, var_xdata

// CHECK: .data
// CHECK-NEXT: .byte 1	
.data 
.if var_xdata == %rax
  .byte 0
.elseif var_xdata == %rcx
  .byte 1
.else
  .byte 2	
.endif

// CHECK:      .byte 1
.if var_xdata != %rcx
  .byte 0
.elseif var_xdata != %rax
  .byte 1
.else
  .byte 2
.endif
