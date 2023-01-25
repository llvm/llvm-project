; RUN: llc < %s -mtriple=thumbv6-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6-apple-darwin -regalloc=basic | FileCheck %s
; RUN: llc < %s -o %t -filetype=obj -mtriple=thumbv6-apple-darwin
; RUN: llvm-objdump --no-print-imm-hex --triple=thumbv6-apple-darwin -d %t | FileCheck %s

@__bar = external hidden global ptr
@__baz = external hidden global ptr

; rdar://8819685
define ptr @_foo() {
entry:
; CHECK-LABEL: __foo{{>?}}:

	%size = alloca i32, align 4
	%0 = load ptr, ptr @__bar, align 4
	%1 = icmp eq ptr %0, null
	br i1 %1, label %bb1, label %bb3
; CHECK: bne
		
bb1:
	store i32 1026, ptr %size, align 4
	%2 = alloca [1026 x i8], align 1
; CHECK: mov     [[R0:r[0-9]+]], sp
; CHECK: adds    {{r[0-9]+}}, [[R0]], {{r[0-9]+}}
	%3 = call i32 @_called_func(ptr %2, ptr %size) nounwind
	%4 = icmp eq i32 %3, 0
	br i1 %4, label %bb2, label %bb3
	
bb2:
	%5 = call ptr @strdup(ptr %2) nounwind
	store ptr %5, ptr @__baz, align 4
	br label %bb3
	
bb3:
	%.0 = phi ptr [ %0, %entry ], [ %5, %bb2 ], [ %2, %bb1 ]
; CHECK:      subs    r4, r7, #7
; CHECK-NEXT: subs    r4, #1
; CHECK-NEXT: mov     sp, r4
; CHECK-NEXT: pop     {r4, r6, r7, pc}
	ret ptr %.0
}

declare noalias ptr @strdup(ptr nocapture) nounwind
declare i32 @_called_func(ptr, ptr) nounwind

; Simple variable ending up *at* sp.
define void @test_simple_var() {
; CHECK-LABEL: test_simple_var{{>?}}:

  %addr32 = alloca i32

; CHECK: mov r0, sp
; CHECK-NOT: adds r0
; CHECK: bl
  call void @take_ptr(ptr %addr32)
  ret void
}

; Simple variable ending up at aligned offset from sp.
define void @test_local_var_addr_aligned() {
; CHECK-LABEL: test_local_var_addr_aligned{{>?}}:

  %addr1.32 = alloca i32
  %addr2.32 = alloca i32

; CHECK: add r0, sp, #{{[0-9]+}}
; CHECK: bl
  call void @take_ptr(ptr %addr1.32)

; CHECK: mov r0, sp
; CHECK-NOT: add r0
; CHECK: bl
  call void @take_ptr(ptr %addr2.32)

  ret void
}

; Simple variable ending up at aligned offset from sp.
define void @test_local_var_big_offset() {
; CHECK-LABEL: test_local_var_big_offset{{>?}}:
  %addr1.32 = alloca i32, i32 257
  %addr2.32 = alloca i32, i32 257

; CHECK: add [[RTMP:r[0-9]+]], sp, #1020
; CHECK: adds [[RTMP]], #8
; CHECK: bl
  call void @take_ptr(ptr %addr1.32)

  ret void
}

; Max range addressable with tADDrSPi
define void @test_local_var_offset_1020() {
; CHECK-LABEL: test_local_var_offset_1020
  %addr1 = alloca i8, i32 4
  %addr2 = alloca i8, i32 1020

; CHECK: add r0, sp, #1020
; CHECK-NEXT: bl
  call void @take_ptr(ptr %addr1)

  ret void
}

; Max range addressable with tADDrSPi + tADDi8 is 1275, however the automatic
; 4-byte aligning of objects on the stack combined with 8-byte stack alignment
; means that 1268 is the max offset we can use.
define void @test_local_var_offset_1268() {
; CHECK-LABEL: test_local_var_offset_1268
  %addr1 = alloca i8, i32 1
  %addr2 = alloca i8, i32 1268

; CHECK: add r0, sp, #1020
; CHECK: adds r0, #248
; CHECK-NEXT: bl
  call void @take_ptr(ptr %addr1)

  ret void
}

declare void @take_ptr(ptr)
