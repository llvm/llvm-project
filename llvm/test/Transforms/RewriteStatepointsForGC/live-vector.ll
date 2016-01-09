; Test that we can correctly handle vectors of pointers in statepoint 
; rewriting.  Currently, we scalarize, but that's an implementation detail.
; RUN: opt %s -rewrite-statepoints-for-gc -rs4gc-split-vector-values -S | FileCheck  %s

; A non-vector relocation for comparison
define i64 addrspace(1)* @test(i64 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: test
; CHECK: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: ret i64 addrspace(1)* %obj.relocated.casted
entry:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %obj
}

; A base vector from a argument
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
entry:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret <2 x i64 addrspace(1)*> %obj
}

; A base vector from a load
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
entry:
  %obj = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret <2 x i64 addrspace(1)*> %obj
}

declare i32 @fake_personality_function()

; When a statepoint is an invoke rather than a call
define <2 x i64 addrspace(1)*> @test4(<2 x i64 addrspace(1)*>* %ptr) gc "statepoint-example" personality i32 ()* @fake_personality_function {
; CHECK-LABEL: test4
; CHECK: load
; CHECK-NEXT: extractelement
; CHECK-NEXT: extractelement
; CHECK-NEXT: gc.statepoint
entry:
  %obj = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  invoke token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
          to label %normal_return unwind label %exceptional_return

; CHECK-LABEL: normal_return:
; CHECK: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %8
normal_return:                                    ; preds = %entry
  ret <2 x i64 addrspace(1)*> %obj

; CHECK-LABEL: exceptional_return:
; CHECK: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: ret <2 x i64 addrspace(1)*> %14
exceptional_return:                               ; preds = %entry
  %landing_pad4 = landingpad token
          cleanup
  ret <2 x i64 addrspace(1)*> %obj
}

; Can we handle an insert element with a constant offset?  This effectively
; tests both the equal and inequal case since we have to relocate both indices
; in the vector.
define <2 x i64 addrspace(1)*> @test5(i64 addrspace(1)* %p) 
     gc "statepoint-example" {
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
entry:
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %p, i32 0
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret <2 x i64 addrspace(1)*> %vec
}


; A base vector from a load
define <2 x i64 addrspace(1)*> @test6(i1 %cnd, <2 x i64 addrspace(1)*>* %ptr) 
    gc "statepoint-example" {
; CHECK-LABEL: test6
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
entry:
  br i1 %cnd, label %taken, label %untaken
taken:
  %obja = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge
untaken:
  %objb = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge

merge:
  %obj = phi <2 x i64 addrspace(1)*> [%obja, %taken], [%objb, %untaken]
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret <2 x i64 addrspace(1)*> %obj
}


declare void @do_safepoint()

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
