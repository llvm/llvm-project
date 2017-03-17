; RUN: opt < %s -rewrite-statepoints-for-gc -S | FileCheck  %s


define i64 addrspace(1)* @test(<2 x i64 addrspace(1)*> %vec, i32 %idx) gc "statepoint-example" {
; CHECK-LABEL: @test
; CHECK: extractelement
; CHECK: extractelement
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: ; (%base_ee, %obj)
; CHECK: gc.relocate
; CHECK-DAG: ; (%base_ee, %base_ee)
; Note that the second extractelement is actually redundant here.  A correct output would
; be to reuse the existing obj as a base since it is actually a base pointer.
entry:
  %obj = extractelement <2 x i64 addrspace(1)*> %vec, i32 %idx
  call void @do_safepoint() [ "deopt"() ]
  ret i64 addrspace(1)* %obj
}

define i64 addrspace(1)* @test2(<2 x i64 addrspace(1)*>* %ptr, i1 %cnd, i32 %idx1, i32 %idx2) gc "statepoint-example" {
; CHECK-LABEL: test2
entry:
  br i1 %cnd, label %taken, label %untaken

taken:                                            ; preds = %entry
  %obja = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge

untaken:                                          ; preds = %entry
  %objb = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge

merge:                                            ; preds = %untaken, %taken
  %vec = phi <2 x i64 addrspace(1)*> [ %obja, %taken ], [ %objb, %untaken ]
  br i1 %cnd, label %taken2, label %untaken2

taken2:                                           ; preds = %merge
  %obj0 = extractelement <2 x i64 addrspace(1)*> %vec, i32 %idx1
  br label %merge2

untaken2:                                         ; preds = %merge
  %obj1 = extractelement <2 x i64 addrspace(1)*> %vec, i32 %idx2
  br label %merge2

merge2:                                           ; preds = %untaken2, %taken2
; CHECK-LABEL: merge2:
; CHECK: %obj.base = phi i64 addrspace(1)*
; CHECK: %obj = phi i64 addrspace(1)*
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: ; (%obj.base, %obj)
; CHECK: gc.relocate
; CHECK-DAG: ; (%obj.base, %obj.base)
  %obj = phi i64 addrspace(1)* [ %obj0, %taken2 ], [ %obj1, %untaken2 ]
  call void @do_safepoint() [ "deopt"() ]
  ret i64 addrspace(1)* %obj
}

define i64 addrspace(1)* @test3(i64 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: test3
; CHECK: insertelement
; CHECK: extractelement
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: (%obj.base, %obj)
entry:
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %ptr, i32 0
  %obj = extractelement <2 x i64 addrspace(1)*> %vec, i32 0
  call void @do_safepoint() [ "deopt"() ]
  ret i64 addrspace(1)* %obj
}

define i64 addrspace(1)* @test4(i64 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: test4
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: ; (%obj.base, %obj)
; When we can optimize an extractelement from a known
; index and avoid introducing new base pointer instructions
entry:
  %derived = getelementptr i64, i64 addrspace(1)* %ptr, i64 16
  %veca = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %derived, i32 0
  %vec = insertelement <2 x i64 addrspace(1)*> %veca, i64 addrspace(1)* %ptr, i32 1
  %obj = extractelement <2 x i64 addrspace(1)*> %vec, i32 0
  call void @do_safepoint() [ "deopt"() ]
  ret i64 addrspace(1)* %obj
}

declare void @use(i64 addrspace(1)*) "gc-leaf-function"
declare void @use_vec(<4 x i64 addrspace(1)*>) "gc-leaf-function"

define void @test5(i1 %cnd, i64 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test5
; CHECK: gc.relocate
; CHECK-DAG: (%bdv.base, %bdv)
; When we fundementally have to duplicate
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %obj, i64 1
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep, i32 0
  %bdv = extractelement <2 x i64 addrspace(1)*> %vec, i32 0
  call void @do_safepoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  call void @use(i64 addrspace(1)* %bdv)
  ret void
}

define void @test6(i1 %cnd, i64 addrspace(1)* %obj, i64 %idx) gc "statepoint-example" {
; CHECK-LABEL: @test6
; CHECK: %gep = getelementptr i64, i64 addrspace(1)* %obj, i64 1
; CHECK: %vec.base = insertelement <2 x i64 addrspace(1)*> zeroinitializer, i64 addrspace(1)* %obj, i32 0, !is_base_value !0
; CHECK: %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep, i32 0
; CHECK: %bdv.base = extractelement <2 x i64 addrspace(1)*> %vec.base, i64 %idx, !is_base_value !0
; CHECK:  %bdv = extractelement <2 x i64 addrspace(1)*> %vec, i64 %idx
; CHECK: gc.statepoint
; CHECK: gc.relocate
; CHECK-DAG: (%bdv.base, %bdv)
; A more complicated example involving vector and scalar bases.
; This is derived from a failing test case when we didn't have correct
; insertelement handling.
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %obj, i64 1
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep, i32 0
  %bdv = extractelement <2 x i64 addrspace(1)*> %vec, i64 %idx
  call void @do_safepoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  call void @use(i64 addrspace(1)* %bdv)
  ret void
}

define i64 addrspace(1)* @test7(i1 %cnd, i64 addrspace(1)* %obj, i64 addrspace(1)* %obj2) gc "statepoint-example" {
; CHECK-LABEL: @test7
entry:
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %obj2, i32 0
  br label %merge1

merge1:                                           ; preds = %merge1, %entry
; CHECK-LABEL: merge1:
; CHECK: vec2.base
; CHECK: vec2
; CHECK: gep
; CHECK: vec3.base
; CHECK: vec3
  %vec2 = phi <2 x i64 addrspace(1)*> [ %vec, %entry ], [ %vec3, %merge1 ]
  %gep = getelementptr i64, i64 addrspace(1)* %obj2, i64 1
  %vec3 = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep, i32 0
  br i1 %cnd, label %merge1, label %next1

next1:                                            ; preds = %merge1
; CHECK-LABEL: next1:
; CHECK: bdv.base = 
; CHECK: bdv = 
  %bdv = extractelement <2 x i64 addrspace(1)*> %vec2, i32 0
  br label %merge

merge:                                            ; preds = %merge, %next1
; CHECK-LABEL: merge:
; CHECK: %objb.base
; CHECK: %objb
; CHECK: gc.statepoint
; CHECK: gc.relocate
; CHECK-DAG: (%objb.base, %objb)
  %objb = phi i64 addrspace(1)* [ %obj, %next1 ], [ %bdv, %merge ]
  br i1 %cnd, label %merge, label %next

next:                                             ; preds = %merge
  call void @do_safepoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i64 addrspace(1)* %objb
}

; identify base for shufflevector
define void @test8(i64 addrspace(1)* %obj, i64 %idx) gc "statepoint-example" {
; CHECK-LABEL: @test8
; CHECK: %gep = getelementptr i64, i64 addrspace(1)* %obj, i64 1
; CHECK: %gep2 = getelementptr i64, i64 addrspace(1)* %obj, i64 2
; CHECK: %vec1.base = insertelement <4 x i64 addrspace(1)*> zeroinitializer, i64 addrspace(1)* %obj, i32 0, !is_base_value !0
; CHECK: %vec1 = insertelement <4 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep, i32 0
; CHECK: %vec2.base = insertelement <4 x i64 addrspace(1)*> zeroinitializer, i64 addrspace(1)* %obj, i32 2, !is_base_value !0
; CHECK: %vec2 = insertelement <4 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep2, i32 2
; CHECK: %vec.base = shufflevector <4 x i64 addrspace(1)*> %vec1.base, <4 x i64 addrspace(1)*> %vec2.base, <2 x i32> <i32 0, i32 2>, !is_base_value !0
; CHECK: %vec = shufflevector <4 x i64 addrspace(1)*> %vec1, <4 x i64 addrspace(1)*> %vec2, <2 x i32> <i32 0, i32 2>
; CHECK: %bdv.base = extractelement <2 x i64 addrspace(1)*> %vec.base, i64 %idx, !is_base_value !0
; CHECK: %bdv = extractelement <2 x i64 addrspace(1)*> %vec, i64 %idx
; CHECK: gc.statepoint
; CHECK: gc.relocate
; CHECK-DAG: (%bdv.base, %bdv)
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %obj, i64 1
  %gep2 = getelementptr i64, i64 addrspace(1)* %obj, i64 2
  %vec1 = insertelement <4 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep, i32 0
  %vec2 = insertelement <4 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep2, i32 2
  %vec = shufflevector <4 x i64 addrspace(1)*> %vec1, <4 x i64 addrspace(1)*> %vec2, <2 x i32> <i32 0, i32 2>
  %bdv = extractelement <2 x i64 addrspace(1)*> %vec, i64 %idx
  call void @do_safepoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  call void @use(i64 addrspace(1)* %bdv)
  ret void
}

; Since the same 'base' vector is used in the shuffle operands, we do not need
; create a shufflevector base.
define void @test9(<4 x i64 addrspace(1)*> %vec1, i64 %idx) gc "statepoint-example" {
; CHECK-LABEL: @test9
; CHECK: %vec = shufflevector <4 x i64 addrspace(1)*> %vec1, <4 x i64 addrspace(1)*> %vec1, <2 x i32> <i32 0, i32 2>
; CHECK: %base_ee = extractelement <4 x i64 addrspace(1)*> %vec1, i64 %idx, !is_base_value !0
; CHECK: %bdv = extractelement <2 x i64 addrspace(1)*> %vec, i64 %idx
; CHECK: gc.statepoint
; CHECK: gc.relocate
; CHECK-DAG: (%base_ee, %bdv)
entry:
 ; shrinking vec1 into vec
  %vec = shufflevector <4 x i64 addrspace(1)*> %vec1, <4 x i64 addrspace(1)*> %vec1, <2 x i32> <i32 0, i32 2>
  %bdv = extractelement <2 x i64 addrspace(1)*> %vec, i64 %idx
  call void @do_safepoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  call void @use(i64 addrspace(1)* %bdv)
  ret void
}

; vector operand of shufflevector is a phi
define i64 addrspace(1)* @test10(i1 %cnd, i64 addrspace(1)* %obj, i64 addrspace(1)* %obj2) gc "statepoint-example" {
; CHECK-LABEL: @test10
entry:
  %vec1 = insertelement <4 x i64 addrspace(1)*> undef, i64 addrspace(1)* %obj, i32 0
  br i1 %cnd, label %here, label %merge

here:
  %vec2 = insertelement <4 x i64 addrspace(1)*> undef, i64 addrspace(1)* %obj2, i32 2
  br label %merge

merge:                                           ; preds = %merge, %entry, %here
; CHECK-LABEL: merge:
; CHECK: %vec.base = phi <4 x i64 addrspace(1)*> [ %vec1.base, %entry ], [ %vec2.base, %here ], [ %vec3.base, %merge ], !is_base_value !0
; CHECK: vec
; CHECK: vec3.base = shufflevector <4 x i64 addrspace(1)*> %vec.base, <4 x i64 addrspace(1)*> %vec.base
; CHECK: vec3
; CHECK: bdv.base
; CHECK: bdv
  %vec = phi <4 x i64 addrspace(1)*> [ %vec1, %entry ], [ %vec2, %here], [ %vec3, %merge]
  %vec3 = shufflevector <4 x i64 addrspace(1)*> %vec, <4 x i64 addrspace(1)*> %vec, <4 x i32> <i32 2, i32 0, i32 1, i32 3>
  %bdv = extractelement <4 x i64 addrspace(1)*> %vec3, i32 0
  br i1 %cnd, label %merge, label %next

next:
; CHECK-LABEL: next:
; CHECK: gc.statepoint
; CHECK: gc.relocate
; CHECK-DAG: (%bdv.base, %bdv)
  call void @do_safepoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i64 addrspace(1)* %bdv
}
declare void @do_safepoint()

define void @test11(<4 x i64 addrspace(1)*> %vec1) gc "statepoint-example" {
; CHECK-LABEL: @test11(
; CHECK: @llvm.experimental.gc.statepoint.p0f_isVoidf{{.*}}<4 x i64 addrspace(1)*> %vec1)
; CHECK: %vec1.relocated = call coldcc <4 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v4p1i8
; CHECK: %vec1.relocated.casted = bitcast <4 x i8 addrspace(1)*> %vec1.relocated to <4 x i64 addrspace(1)*>
; CHECK: %vec2.remat = getelementptr i64, <4 x i64 addrspace(1)*> %vec1.relocated.casted, i32 1024
; CHECK: call void @use_vec(<4 x i64 addrspace(1)*> %vec2.remat)
entry:
  %vec2 = getelementptr i64, <4 x i64 addrspace(1)*> %vec1, i32 1024
  call void @do_safepoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  call void @use_vec(<4 x i64 addrspace(1) *> %vec2)
  ret void
}
