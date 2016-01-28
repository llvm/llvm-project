; Test that we can correctly handle vectors of pointers in statepoint 
; rewriting.  Currently, we scalarize, but that's an implementation detail.
; RUN: opt < %s -rewrite-statepoints-for-gc -rs4gc-use-deopt-bundles -rs4gc-split-vector-values -S | FileCheck  %s

; A non-vector relocation for comparison

define i64 addrspace(1)* @test(i64 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: test
; CHECK: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: ret i64 addrspace(1)* %obj.relocated.casted
; A base vector from a argument
entry:
  call void @do_safepoint() [ "deopt"() ]
  ret i64 addrspace(1)* %obj
}

define <2 x i64 addrspace(1)*> @test2(<2 x i64 addrspace(1)*> %obj) gc "statepoint-example" {
; CHECK-LABEL: test2
; CHECK: extractelement
; CHECK-NEXT: extractelement
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %7
; A base vector from a load
entry:
  call void @do_safepoint() [ "deopt"() ]
  ret <2 x i64 addrspace(1)*> %obj
}

define <2 x i64 addrspace(1)*> @test3(<2 x i64 addrspace(1)*>* %ptr) gc "statepoint-example" {
; CHECK-LABEL: test3
; CHECK: load
; CHECK-NEXT: extractelement
; CHECK-NEXT: extractelement
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %7
; When a statepoint is an invoke rather than a call
entry:
  %obj = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  call void @do_safepoint() [ "deopt"() ]
  ret <2 x i64 addrspace(1)*> %obj
}

declare i32 @fake_personality_function()

define <2 x i64 addrspace(1)*> @test4(<2 x i64 addrspace(1)*>* %ptr) gc "statepoint-example" personality i32 ()* @fake_personality_function {
; CHECK-LABEL: test4
; CHECK: load
; CHECK-NEXT: extractelement
; CHECK-NEXT: extractelement
; CHECK-NEXT: gc.statepoint
entry:
  %obj = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  invoke void @do_safepoint() [ "deopt"() ]
          to label %normal_return unwind label %exceptional_return

normal_return:                                    ; preds = %entry
; CHECK-LABEL: normal_return:
; CHECK: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %7
  ret <2 x i64 addrspace(1)*> %obj

exceptional_return:                               ; preds = %entry
; CHECK-LABEL: exceptional_return:
; CHECK: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %13
; Can we handle an insert element with a constant offset?  This effectively
; tests both the equal and inequal case since we have to relocate both indices
; in the vector.
  %landing_pad4 = landingpad token
          cleanup
  ret <2 x i64 addrspace(1)*> %obj
}

define <2 x i64 addrspace(1)*> @test5(i64 addrspace(1)* %p) gc "statepoint-example" {
; CHECK-LABEL: test5
; CHECK: insertelement
; CHECK-NEXT: extractelement
; CHECK-NEXT: extractelement
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %7
; A base vector from a load
entry:
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %p, i32 0
  call void @do_safepoint() [ "deopt"() ]
  ret <2 x i64 addrspace(1)*> %vec
}

define <2 x i64 addrspace(1)*> @test6(i1 %cnd, <2 x i64 addrspace(1)*>* %ptr) gc "statepoint-example" {
; CHECK-LABEL: test6
entry:
  br i1 %cnd, label %taken, label %untaken

taken:                                            ; preds = %entry
  %obja = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge

untaken:                                          ; preds = %entry
  %objb = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge

merge:                                            ; preds = %untaken, %taken
; CHECK-LABEL: merge:
; CHECK-NEXT: = phi
; CHECK-NEXT: extractelement
; CHECK-NEXT: extractelement
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*>
  %obj = phi <2 x i64 addrspace(1)*> [ %obja, %taken ], [ %objb, %untaken ]
  call void @do_safepoint() [ "deopt"() ]
  ret <2 x i64 addrspace(1)*> %obj
}

declare void @do_safepoint()
