; Two kernels with the same opaque function type whose pointer parameters
; point to different types get function types of their own in AIR's typed
; bitcode: written with the first kernel's, the second one's i32 accesses
; through an i64 pointer made Metal's reader reject the library ("Unexpected
; bitcode file", Geekbench's normalize_color_histogram and composite).
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: llvm-bcanalyzer -dump %t.air | FileCheck %s

; The kernels' two function types (ptr, ptr, <3 x i32>): an i64 pointer
; pair and an i32 pointer pair.
; CHECK: <FUNCTION op0=0 op1=[[VOID:[0-9]+]] op2=[[P1:[0-9]+]] op3=[[P1]] op4=[[V:[0-9]+]]/>
; CHECK-NOT: <FUNCTION op0=0 op1=[[VOID]] op2=[[P1]]
; CHECK: <FUNCTION op0=0 op1=[[VOID]] op2=[[P2:[0-9]+]] op3=[[P2]] op4=[[V]]/>

define void @k1(ptr addrspace(1) %a, ptr addrspace(1) %out) "air-kernel" {
  %v = load i64, ptr addrspace(1) %a, align 8
  store i64 %v, ptr addrspace(1) %out, align 8
  ret void
}

define void @k2(ptr addrspace(1) %a, ptr addrspace(1) %out) "air-kernel" {
  %v = load i32, ptr addrspace(1) %a, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
