; RUN: opt < %s -rewrite-statepoints-for-gc -S | FileCheck %s
; RUN: opt < %s -passes=rewrite-statepoints-for-gc -S | FileCheck %s


declare void @use_obj16(i16 addrspace(1)*) "gc-leaf-function"
declare void @use_obj32(i32 addrspace(1)*) "gc-leaf-function"
declare void @use_obj64(i64 addrspace(1)*) "gc-leaf-function"

declare void @do_safepoint()

define void @test_gep_const(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_gep_const
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %base, i32 15
; CHECK: getelementptr i32, i32 addrspace(1)* %base, i32 15
  call void @do_safepoint() [ "deopt"() ]
; CHECK: %base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 7, i32 7)
; CHECK: bitcast i8 addrspace(1)* %base.relocated to i32 addrspace(1)*
; CHECK: getelementptr i32, i32 addrspace(1)* %base.relocated.casted, i32 15
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj32(i32 addrspace(1)* %ptr)
  ret void
}

define void @test_gep_idx(i32 addrspace(1)* %base, i32 %idx) gc "statepoint-example" {
; CHECK-LABEL: test_gep_idx
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %base, i32 %idx
; CHECK: getelementptr
  call void @do_safepoint() [ "deopt"() ]
; CHECK: %base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 7, i32 7)
; CHECK: %base.relocated.casted = bitcast i8 addrspace(1)* %base.relocated to i32 addrspace(1)*
; CHECK: getelementptr i32, i32 addrspace(1)* %base.relocated.casted, i32 %idx
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj32(i32 addrspace(1)* %ptr)
  ret void
}

define void @test_bitcast(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_bitcast
entry:
  %ptr = bitcast i32 addrspace(1)* %base to i64 addrspace(1)*
; CHECK: bitcast i32 addrspace(1)* %base to i64 addrspace(1)*
  call void @do_safepoint() [ "deopt"() ]
; CHECK: %base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 7, i32 7)
; CHECK: %base.relocated.casted = bitcast i8 addrspace(1)* %base.relocated to i32 addrspace(1)*
; CHECK: bitcast i32 addrspace(1)* %base.relocated.casted to i64 addrspace(1)*
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj64(i64 addrspace(1)* %ptr)
  ret void
}

define void @test_bitcast_bitcast(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_bitcast_bitcast
entry:
  %ptr1 = bitcast i32 addrspace(1)* %base to i64 addrspace(1)*
  %ptr2 = bitcast i64 addrspace(1)* %ptr1 to i16 addrspace(1)*
; CHECK: bitcast i32 addrspace(1)* %base to i64 addrspace(1)*
; CHECK: bitcast i64 addrspace(1)* %ptr1 to i16 addrspace(1)*
  call void @do_safepoint() [ "deopt"() ]

; CHECK: %base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 7, i32 7)
; CHECK: %base.relocated.casted = bitcast i8 addrspace(1)* %base.relocated to i32 addrspace(1)*
; CHECK: bitcast i32 addrspace(1)* %base.relocated.casted to i64 addrspace(1)*
; CHECK: bitcast i64 addrspace(1)* %ptr1.remat to i16 addrspace(1)*
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj16(i16 addrspace(1)* %ptr2)
  ret void
}

define void @test_addrspacecast_addrspacecast(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_addrspacecast_addrspacecast
entry:
  %ptr1 = addrspacecast i32 addrspace(1)* %base to i32*
  %ptr2 = addrspacecast i32* %ptr1 to i32 addrspace(1)*
; CHECK: addrspacecast i32 addrspace(1)* %base to i32*
; CHECK: addrspacecast i32* %ptr1 to i32 addrspace(1)*
  call void @do_safepoint() [ "deopt"() ]

; CHECK: %ptr2.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 8, i32 7)
; CHECK: %ptr2.relocated.casted = bitcast i8 addrspace(1)* %ptr2.relocated to i32 addrspace(1)*
; CHECK: %base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 8, i32 8)
; CHECK: %base.relocated.casted = bitcast i8 addrspace(1)* %base.relocated to i32 addrspace(1)*
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj32(i32 addrspace(1)* %ptr2)
  ret void
}

define void @test_bitcast_gep(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_bitcast_gep
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
; CHECK: getelementptr
; CHECK: bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
  call void @do_safepoint() [ "deopt"() ]

; CHECK: gc.relocate
; CHECK: bitcast
; CHECK: getelementptr
; CHECK: bitcast
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  ret void
}

define void @test_intersecting_chains(i32 addrspace(1)* %base, i32 %idx) gc "statepoint-example" {
; CHECK-LABEL: test_intersecting_chains
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
; CHECK: getelementptr
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
; CHECK: bitcast
  %ptr.cast2 = bitcast i32 addrspace(1)* %ptr.gep to i16 addrspace(1)*
; CHECK: bitcast
  call void @do_safepoint() [ "deopt"() ]

; CHECK: getelementptr
; CHECK: bitcast
; CHECK: getelementptr
; CHECK: bitcast
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  call void @use_obj16(i16 addrspace(1)* %ptr.cast2)
  ret void
}

define void @test_cost_threshold(i32 addrspace(1)* %base, i32 %idx1, i32 %idx2, i32 %idx3) gc "statepoint-example" {
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
  call void @do_safepoint() [ "deopt"() ]

; CHECK: gc.relocate
; CHECK: bitcast
; CHECK: gc.relocate
; CHECK: bitcast
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  ret void
}

define void @test_two_derived(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_two_derived
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %base, i32 15
  %ptr2 = getelementptr i32, i32 addrspace(1)* %base, i32 12
; CHECK: getelementptr
; CHECK: getelementptr
  call void @do_safepoint() [ "deopt"() ]

; CHECK: gc.relocate
; CHECK: bitcast
; CHECK: getelementptr
; CHECK: getelementptr
  call void @use_obj32(i32 addrspace(1)* %ptr)
  call void @use_obj32(i32 addrspace(1)* %ptr2)
  ret void
}

define void @test_gep_smallint_array([3 x i32] addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_gep_smallint_array
entry:
  %ptr = getelementptr [3 x i32], [3 x i32] addrspace(1)* %base, i32 0, i32 2
; CHECK: getelementptr
  call void @do_safepoint() [ "deopt"() ]

; CHECK: gc.relocate
; CHECK: bitcast
; CHECK: getelementptr
  call void @use_obj32(i32 addrspace(1)* %ptr)
  ret void
}

declare i32 @fake_personality_function()

define void @test_invoke(i32 addrspace(1)* %base) gc "statepoint-example" personality i32 ()* @fake_personality_function {
; CHECK-LABEL: test_invoke
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
; CHECK: getelementptr
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
; CHECK: bitcast
  %ptr.cast2 = bitcast i32 addrspace(1)* %ptr.gep to i16 addrspace(1)*
; CHECK: bitcast
  invoke void @do_safepoint() [ "deopt"() ]
          to label %normal unwind label %exception

normal:
; CHECK: normal:
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
; CHECK: exception:
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

define void @test_loop(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_loop
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
; CHECK: getelementptr
  br label %loop

loop:                                             ; preds = %loop, %entry
; CHECK: phi i32 addrspace(1)* [ %ptr.gep, %entry ], [ %ptr.gep.remat, %loop ]
; CHECK: phi i32 addrspace(1)* [ %base, %entry ], [ %base.relocated.casted, %loop ]
  call void @use_obj32(i32 addrspace(1)* %ptr.gep)
  call void @do_safepoint() [ "deopt"() ]
; CHECK: gc.relocate
; CHECK: bitcast
; CHECK: getelementptr
  br label %loop
}

define void @test_too_long(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_too_long
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  %ptr.gep1 = getelementptr i32, i32 addrspace(1)* %ptr.gep, i32 15
  %ptr.gep2 = getelementptr i32, i32 addrspace(1)* %ptr.gep1, i32 15
  %ptr.gep3 = getelementptr i32, i32 addrspace(1)* %ptr.gep2, i32 15
  %ptr.gep4 = getelementptr i32, i32 addrspace(1)* %ptr.gep3, i32 15
  %ptr.gep5 = getelementptr i32, i32 addrspace(1)* %ptr.gep4, i32 15
  %ptr.gep6 = getelementptr i32, i32 addrspace(1)* %ptr.gep5, i32 15
  %ptr.gep7 = getelementptr i32, i32 addrspace(1)* %ptr.gep6, i32 15
  %ptr.gep8 = getelementptr i32, i32 addrspace(1)* %ptr.gep7, i32 15
  %ptr.gep9 = getelementptr i32, i32 addrspace(1)* %ptr.gep8, i32 15
  %ptr.gep10 = getelementptr i32, i32 addrspace(1)* %ptr.gep9, i32 15
  %ptr.gep11 = getelementptr i32, i32 addrspace(1)* %ptr.gep10, i32 15
  call void @do_safepoint() [ "deopt"() ]
; CHECK: gc.relocate
; CHECK: bitcast
; CHECK: gc.relocate
; CHECK: bitcast
  call void @use_obj32(i32 addrspace(1)* %ptr.gep11)
  ret void
}


declare i32 addrspace(1)* @new_instance() nounwind "gc-leaf-function"

; remat the gep in presence of base pointer which is a phi node.
; FIXME: We should remove the extra basephi.base as well.
define void @contains_basephi(i1 %cond) gc "statepoint-example" {
; CHECK-LABEL: contains_basephi
entry:
  %base1 = call i32 addrspace(1)* @new_instance()
  %base2 = call i32 addrspace(1)* @new_instance()
  br i1 %cond, label %here, label %there

here:
  br label %merge

there:
  br label %merge

merge:
  ; CHECK: %basephi.base = phi i32 addrspace(1)* [ %base1, %here ], [ %base2, %there ], !is_base_value !0
  ; CHECK: %basephi = phi i32 addrspace(1)* [ %base1, %here ], [ %base2, %there ]
  ; CHECK: %ptr.gep = getelementptr i32, i32 addrspace(1)* %basephi, i32 15
  ; CHECK: %statepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint
  ; CHECK: %basephi.base.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %statepoint_token, i32 7, i32 7) ; (%basephi.base, %basephi.base)
  ; CHECK: %basephi.base.relocated.casted = bitcast i8 addrspace(1)* %basephi.base.relocated to i32 addrspace(1)*
  ; CHECK: %ptr.gep.remat = getelementptr i32, i32 addrspace(1)* %basephi.base.relocated.casted, i32 15
  ; CHECK: call void @use_obj32(i32 addrspace(1)* %ptr.gep.remat)



  %basephi = phi i32 addrspace(1)* [ %base1, %here ], [ %base2, %there ]
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %basephi, i32 15
  call void @do_safepoint() ["deopt"() ]
  call void @use_obj32(i32 addrspace(1)* %ptr.gep)
  ret void
}


define void @test_intersecting_chains_with_phi(i1 %cond) gc "statepoint-example" {
; CHECK-LABEL: test_intersecting_chains_with_phi
entry:
  %base1 = call i32 addrspace(1)* @new_instance()
  %base2 = call i32 addrspace(1)* @new_instance()
  br i1 %cond, label %here, label %there

here:
  br label %merge

there:
  br label %merge

merge:
  %basephi = phi i32 addrspace(1)* [ %base1, %here ], [ %base2, %there ]
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %basephi, i32 15
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
  %ptr.cast2 = bitcast i32 addrspace(1)* %ptr.gep to i16 addrspace(1)*
  call void @do_safepoint() [ "deopt"() ]
  ; CHECK: statepoint
  ; CHECK: %ptr.gep.remat1 = getelementptr i32, i32 addrspace(1)* %basephi.base.relocated.casted, i32 15
  ; CHECK: %ptr.cast.remat = bitcast i32 addrspace(1)* %ptr.gep.remat1 to i64 addrspace(1)*
  ; CHECK: %ptr.gep.remat = getelementptr i32, i32 addrspace(1)* %basephi.base.relocated.casted, i32 15
  ; CHECK: %ptr.cast2.remat = bitcast i32 addrspace(1)* %ptr.gep.remat to i16 addrspace(1)*
  ; CHECK: call void @use_obj64(i64 addrspace(1)* %ptr.cast.remat)
  ; CHECK: call void @use_obj16(i16 addrspace(1)* %ptr.cast2.remat)
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  call void @use_obj16(i16 addrspace(1)* %ptr.cast2)
  ret void
}
