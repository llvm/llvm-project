; RUN: opt < %s -rewrite-statepoints-for-gc -rs4gc-use-deopt-bundles -S 2>&1 | FileCheck %s


declare void @use_obj16(i16 addrspace(1)*) "gc-leaf-function"
declare void @use_obj32(i32 addrspace(1)*) "gc-leaf-function"
declare void @use_obj64(i64 addrspace(1)*) "gc-leaf-function"

declare void @do_safepoint()

define void @test_gep_const(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_gep_const
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %base, i32 15
  call void @do_safepoint() [ "deopt"() ]
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj32(i32 addrspace(1)* %ptr)
  ret void
}

define void @test_gep_idx(i32 addrspace(1)* %base, i32 %idx) gc "statepoint-example" {
; CHECK-LABEL: test_gep_idx
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %base, i32 %idx
  call void @do_safepoint() [ "deopt"() ]
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj32(i32 addrspace(1)* %ptr)
  ret void
}

define void @test_bitcast(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_bitcast
entry:
  %ptr = bitcast i32 addrspace(1)* %base to i64 addrspace(1)*
  call void @do_safepoint() [ "deopt"() ]
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj64(i64 addrspace(1)* %ptr)
  ret void
}

define void @test_bitcast_gep(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_bitcast_gep
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
  call void @do_safepoint() [ "deopt"() ]
  call void @use_obj32(i32 addrspace(1)* %base)
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  ret void
}

define void @test_intersecting_chains(i32 addrspace(1)* %base, i32 %idx) gc "statepoint-example" {
; CHECK-LABEL: test_intersecting_chains
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
  %ptr.cast2 = bitcast i32 addrspace(1)* %ptr.gep to i16 addrspace(1)*
  call void @do_safepoint() [ "deopt"() ]
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  call void @use_obj16(i16 addrspace(1)* %ptr.cast2)
  ret void
}

define void @test_cost_threshold(i32 addrspace(1)* %base, i32 %idx1, i32 %idx2, i32 %idx3) gc "statepoint-example" {
; CHECK-LABEL: test_cost_threshold
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  %ptr.gep2 = getelementptr i32, i32 addrspace(1)* %ptr.gep, i32 %idx1
  %ptr.gep3 = getelementptr i32, i32 addrspace(1)* %ptr.gep2, i32 %idx2
  %ptr.gep4 = getelementptr i32, i32 addrspace(1)* %ptr.gep3, i32 %idx3
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep4 to i64 addrspace(1)*
  call void @do_safepoint() [ "deopt"() ]
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  ret void
}

define void @test_two_derived(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_two_derived
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %base, i32 15
  %ptr2 = getelementptr i32, i32 addrspace(1)* %base, i32 12
  call void @do_safepoint() [ "deopt"() ]
  call void @use_obj32(i32 addrspace(1)* %ptr)
  call void @use_obj32(i32 addrspace(1)* %ptr2)
  ret void
}

define void @test_gep_smallint_array([3 x i32] addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_gep_smallint_array
entry:
  %ptr = getelementptr [3 x i32], [3 x i32] addrspace(1)* %base, i32 0, i32 2
  call void @do_safepoint() [ "deopt"() ]
  call void @use_obj32(i32 addrspace(1)* %ptr)
  ret void
}

declare i32 @fake_personality_function()

define void @test_invoke(i32 addrspace(1)* %base) gc "statepoint-example" personality i32 ()* @fake_personality_function {
; CHECK-LABEL: test_invoke
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  %ptr.cast = bitcast i32 addrspace(1)* %ptr.gep to i64 addrspace(1)*
  %ptr.cast2 = bitcast i32 addrspace(1)* %ptr.gep to i16 addrspace(1)*
  invoke void @do_safepoint() [ "deopt"() ]
          to label %normal unwind label %exception

normal:                                           ; preds = %entry
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  call void @use_obj16(i16 addrspace(1)* %ptr.cast2)
  ret void

exception:                                        ; preds = %entry
  %landing_pad4 = landingpad token
          cleanup
  call void @use_obj64(i64 addrspace(1)* %ptr.cast)
  call void @use_obj16(i16 addrspace(1)* %ptr.cast2)
  ret void
}

define void @test_loop(i32 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: test_loop
entry:
  %ptr.gep = getelementptr i32, i32 addrspace(1)* %base, i32 15
  br label %loop

loop:                                             ; preds = %loop, %entry
  call void @use_obj32(i32 addrspace(1)* %ptr.gep)
  call void @do_safepoint() [ "deopt"() ]
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
  call void @use_obj32(i32 addrspace(1)* %ptr.gep11)
  ret void
}
