; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}


declare i8 @llvm.nvvm.ldu.global.i.i8.p1i8(i8 addrspace(1)* %ptr, i32 %align)
declare i32 @llvm.nvvm.ldu.global.i.i32.p1i32(i32 addrspace(1)* %ptr, i32 %align)
declare i8 @llvm.nvvm.ldg.global.i.i8.p1i8(i8 addrspace(1)* %ptr, i32 %align)
declare i32 @llvm.nvvm.ldg.global.i.i32.p1i32(i32 addrspace(1)* %ptr, i32 %align)


; CHECK: func0
define i8 @func0(i8 addrspace(1)* %ptr) {
; ldu.global.u8
  %val = tail call i8 @llvm.nvvm.ldu.global.i.i8.p1i8(i8 addrspace(1)* %ptr, i32 4)
  ret i8 %val
}

; CHECK: func1
define i32 @func1(i32 addrspace(1)* %ptr) {
; ldu.global.u32
  %val = tail call i32 @llvm.nvvm.ldu.global.i.i32.p1i32(i32 addrspace(1)* %ptr, i32 4)
  ret i32 %val
}

; CHECK: func2
define i8 @func2(i8 addrspace(1)* %ptr) {
; ld.global.nc.u8
  %val = tail call i8 @llvm.nvvm.ldg.global.i.i8.p1i8(i8 addrspace(1)* %ptr, i32 4)
  ret i8 %val
}

; CHECK: func3
define i32 @func3(i32 addrspace(1)* %ptr) {
; ld.global.nc.u32
  %val = tail call i32 @llvm.nvvm.ldg.global.i.i32.p1i32(i32 addrspace(1)* %ptr, i32 4)
  ret i32 %val
}
