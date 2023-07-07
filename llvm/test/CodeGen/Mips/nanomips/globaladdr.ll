; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs -mgpopt < %s | FileCheck %s --check-prefix=CHECK-GP

@single = global i32 5

define i32 @load_global_val() {
; CHECK-LABEL: load_global_val
; CHECK: lwpc $a0, single
; CHECK-GP-LABEL: load_global_val
; CHECK-GP: lwpc $a0, single
  %r = load i32, i32* @single
  ret i32 %r
}

define i32* @load_global_addr() {
; CHECK-LABEL: load_global_addr
; CHECK: la $a0, single
; CHECK-GP-LABEL: load_global_addr
; CHECK-GP: la $a0, single
  ret i32* @single
}

define void @load_and_store_global(i32 %a) {
; CHECK-LABEL: load_and_store_global
; CHECK: la $a1, single
; CHECK-NOT: lwpc ${{[ast][0-7]}}, single
; CHECK-NOT: swpc ${{[ast][0-7]}}, single
  %1 = load i32, i32* @single
  %add = add nsw i32 %1, %a
  store i32 %add, i32* @single
  ret void
}

@array = global [3 x i32] zeroinitializer

define i32* @load_global_array_addr() {
; CHECK-LABEL: load_global_array_addr
; CHECK: la $a0, array
  ret i32* getelementptr ([3 x i32], [3 x i32]* @array, i32 0, i32 0)
}

define i32* @load_global_addr_with_offset() {
; CHECK-LABEL: load_global_addr_with_offset
; CHECK: la $a0, array+8
  ret i32* getelementptr ([3 x i32], [3 x i32]* @array, i32 0, i32 2)
}

define i32 @load_global_val_with_offset(i32 %a) {
; CHECK-LABEL: load_global_val_with_offset
; CHECK: lwpc $a1, array+8
  %g = load i32, i32* getelementptr ([3 x i32], [3 x i32]* @array, i32 0, i32 2)
  %add = add nsw i32 %g, %a
  ret i32 %add
}

define void @store_to_global_with_offset(i32 %a) {
; CHECK-LABEL: store_to_global_with_offset
; CHECK: swpc $a0, array+8
  store i32 %a, i32* getelementptr ([3 x i32], [3 x i32]* @array, i32 0, i32 2)
  ret void
}

declare void @abort()
%struct.pair = type { i32, i32 }
@gStruct = global %struct.pair { i32 97, i32 13 }

define i32 @global_used_in_multiple_bbs() {
; CHECK-LABEL: global_used_in_multiple_bbs
; CHECK: la $a0, gStruct
; CHECK-NOT: lwpc ${{[ast][0-7]}}, gStruct
entry:
  %0 = load i32, i32* getelementptr inbounds (%struct.pair, %struct.pair* @gStruct, i32 0, i32 0)
  %cmp.not = icmp eq i32 %0, 97
  br i1 %cmp.not, label %if.end, label %if.then

if.then:
  call void @abort()
  unreachable

if.end:
  %1 = load i32, i32* getelementptr inbounds (%struct.pair, %struct.pair* @gStruct, i32 0, i32 1)
  %cmp3.not = icmp eq i32 %1, 13
  br i1 %cmp3.not, label %if.end6, label %if.then5

if.then5:
  call void @abort()
  unreachable

if.end6:
  ret i32 0
}
