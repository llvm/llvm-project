; RUN: opt %s -rewrite-statepoints-for-gc -S 2>&1 | FileCheck %s

declare void @use_obj16(i16 addrspace(1)*)
declare void @use_obj32(i32 addrspace(1)*)
declare void @use_obj64(i64 addrspace(1)*)
declare void @do_safepoint()

define void @"test_gep_const"(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_gep_const
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %base, i32 15
  ; CHECK: getelementptr i32, i32 addrspace(1)* %base, i32 15
  %sp = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: %base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %sp, i32 7, i32 7)
  ; CHECK: bitcast i8 addrspace(1)* %base.relocated to i32 addrspace(1)*
  ; CHECK: getelementptr i32, i32 addrspace(1)* %base.relocated.casted, i32 15
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj32(i32 addrspace(1)* %ptr)
  ret void
}

define void @"test_gep_idx"(i32 addrspace(1)* %base, i32 %idx) gc "statepoint-example" {
; CHECK-LABEL: test_gep_idx
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %base, i32 %idx
  ; CHECK: getelementptr
  %sp = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: %base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %sp, i32 7, i32 7)
  ; CHECK: %base.relocated.casted = bitcast i8 addrspace(1)* %base.relocated to i32 addrspace(1)*
  ; CHECK: getelementptr i32, i32 addrspace(1)* %base.relocated.casted, i32 %idx
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj32(i32 addrspace(1)* %ptr)
  ret void
}

define void @"test_bitcast"(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_bitcast
entry:
  %ptr = bitcast i32 addrspace(1)* %base to i64 addrspace(1)*
  ; CHECK: bitcast i32 addrspace(1)* %base to i64 addrspace(1)*
  %sp = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: %base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %sp, i32 7, i32 7)
  ; CHECK: %base.relocated.casted = bitcast i8 addrspace(1)* %base.relocated to i32 addrspace(1)*
  ; CHECK: bitcast i32 addrspace(1)* %base.relocated.casted to i64 addrspace(1)*
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj64(i64 addrspace(1)* %ptr)
  ret void
}

define void @"test_bitcast_gep"(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_bitcast_gep
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  ; CHECK: getelementptr
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
  ; CHECK: bitcast
  %sp = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: gc.relocate
  ; CHECK: bitcast
  ; CHECK: getelementptr
  ; CHECK: bitcast
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  ret void
}

define void @"test_intersecting_chains"(i32 addrspace(1)* %base, i32 %idx) gc "statepoint-example" {
; CHECK-LABEL: test_intersecting_chains
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  ; CHECK: getelementptr
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
  ; CHECK: bitcast
  %ptr.cast2 = bitcast i32 addrspace(1)* %ptr.gep to i16 addrspace(1)*
  ; CHECK: bitcast
  %sp = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: getelementptr
  ; CHECK: bitcast
  ; CHECK: getelementptr
  ; CHECK: bitcast
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  call void @use_obj16(i16 addrspace(1)* %ptr.cast2)
  ret void
}

define void @"test_cost_threshold"(i32 addrspace(1)* %base, i32 %idx1, i32 %idx2, i32 %idx3) gc "statepoint-example" {
; CHECK-LABEL: test_cost_threshold
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  ; CHECK: getelementptr
  %ptr.gep2 = getelementptr i32, i32 addrspace(1)* %ptr.gep, i32 %idx1
  ; CHECK: getelementptr
  %ptr.gep3 = getelementptr i32, i32 addrspace(1)* %ptr.gep2, i32 %idx2
  ; CHECK: getelementptr
  %ptr.gep4 = getelementptr i32, i32 addrspace(1)* %ptr.gep3, i32 %idx3
  ; CHECK: getelementptr
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep4 to i64 addrspace(1)*
  ; CHECK: bitcast
  %sp = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: gc.relocate
  ; CHECK: bitcast
  ; CHECK: gc.relocate
  ; CHECK: bitcast
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  ret void
}

define void @"test_two_derived"(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_two_derived
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %base, i32 15
  %ptr2 = getelementptr i32, i32 addrspace(1)* %base, i32 12
  ; CHECK: getelementptr
  ; CHECK: getelementptr
  %sp = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: gc.relocate
  ; CHECK: bitcast
  ; CHECK: getelementptr
  ; CHECK: getelementptr
  call void @use_obj32(i32 addrspace(1)* %ptr)
  call void @use_obj32(i32 addrspace(1)* %ptr2)
  ret void
}

define void @"test_gep_smallint_array"([3 x i32] addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_gep_smallint_array
entry:
  %ptr = getelementptr [3 x i32], [3 x i32] addrspace(1)* %base, i32 0, i32 2
  ; CHECK: getelementptr
  %sp = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: gc.relocate
  ; CHECK: bitcast
  ; CHECK: getelementptr
  call void @use_obj32(i32 addrspace(1)* %ptr)
  ret void
}

declare i32 @fake_personality_function()

define void @"test_invoke"(i32 addrspace(1)* %base) gc "statepoint-example" personality i32 ()* @fake_personality_function {
; CHECK-LABEL: test_invoke
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  ; CHECK: getelementptr
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
  ; CHECK: bitcast
  %ptr.cast2 = bitcast i32 addrspace(1)* %ptr.gep to i16 addrspace(1)*
  ; CHECK: bitcast
  %sp = invoke token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
                to label %normal unwind label %exception

normal:
  ; CHECK-LABEL: normal:
  ; CHECK: gc.relocate
  ; CHECK: bitcast
  ; CHECK: getelementptr
  ; CHECK: bitcast
  ; CHECK: getelementptr
  ; CHECK: bitcast
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  call void @use_obj16(i16 addrspace(1)* %ptr.cast2)
  ret void

exception:
  ; CHECK-LABEL: exception:
  %landing_pad4 = landingpad token
          cleanup
  ; CHECK: gc.relocate
  ; CHECK: bitcast
  ; CHECK: getelementptr
  ; CHECK: bitcast
  ; CHECK: getelementptr
  ; CHECK: bitcast
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  call void @use_obj16(i16 addrspace(1)* %ptr.cast2)
  ret void
}

define void @"test_loop"(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_loop
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  ; CHECK: getelementptr
  br label %loop

loop:
  ; CHECK: phi i32 addrspace(1)* [ %ptr.gep, %entry ], [ %ptr.gep.remat, %loop ]
  ; CHECK: phi i32 addrspace(1)* [ %base, %entry ], [ %base.relocated.casted, %loop ]
  call void @use_obj32(i32 addrspace(1)* %ptr.gep)
  %sp = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: gc.relocate
  ; CHECK: bitcast
  ; CHECK: getelementptr
  br label %loop
}

define void @"test_too_long"(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_too_long
entry:
  %ptr.gep   = getelementptr i32, i32 addrspace(1)* %base, i32 15
  %ptr.gep1  = getelementptr i32, i32 addrspace(1)* %ptr.gep, i32 15
  %ptr.gep2  = getelementptr i32, i32 addrspace(1)* %ptr.gep1, i32 15
  %ptr.gep3  = getelementptr i32, i32 addrspace(1)* %ptr.gep2, i32 15
  %ptr.gep4  = getelementptr i32, i32 addrspace(1)* %ptr.gep3, i32 15
  %ptr.gep5  = getelementptr i32, i32 addrspace(1)* %ptr.gep4, i32 15
  %ptr.gep6  = getelementptr i32, i32 addrspace(1)* %ptr.gep5, i32 15
  %ptr.gep7  = getelementptr i32, i32 addrspace(1)* %ptr.gep6, i32 15
  %ptr.gep8  = getelementptr i32, i32 addrspace(1)* %ptr.gep7, i32 15
  %ptr.gep9  = getelementptr i32, i32 addrspace(1)* %ptr.gep8, i32 15
  %ptr.gep10 = getelementptr i32, i32 addrspace(1)* %ptr.gep9, i32 15
  %ptr.gep11 = getelementptr i32, i32 addrspace(1)* %ptr.gep10, i32 15
  %sp = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: gc.relocate
  ; CHECK: bitcast
  ; CHECK: gc.relocate
  ; CHECK: bitcast
  call void @use_obj32(i32 addrspace(1)* %ptr.gep11)
  ret void
}


declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
