; RUN: opt %s -rewrite-statepoints-for-gc -S | FileCheck  %s

define i64 addrspace(1)* @test(<2 x i64 addrspace(1)*> %vec, i32 %idx) gc "statepoint-example" {
; CHECK-LABEL: @test
; CHECK: extractelement
; CHECK: extractelement
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: ; (%base_ee, %base_ee)
; CHECK: gc.relocate
; CHECK-DAG: ; (%base_ee, %obj)
; Note that the second extractelement is actually redundant here.  A correct output would
; be to reuse the existing obj as a base since it is actually a base pointer.
entry:
  %obj = extractelement <2 x i64 addrspace(1)*> %vec, i32 %idx
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)

  ret i64 addrspace(1)* %obj
}

define i64 addrspace(1)* @test2(<2 x i64 addrspace(1)*>* %ptr, i1 %cnd, i32 %idx1, i32 %idx2) 
    gc "statepoint-example" {
; CHECK-LABEL: test2
entry:
  br i1 %cnd, label %taken, label %untaken
taken:
  %obja = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge
untaken:
  %objb = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge
merge:
  %vec = phi <2 x i64 addrspace(1)*> [%obja, %taken], [%objb, %untaken]
  br i1 %cnd, label %taken2, label %untaken2
taken2:
  %obj0 = extractelement <2 x i64 addrspace(1)*> %vec, i32 %idx1
  br label %merge2
untaken2:
  %obj1 = extractelement <2 x i64 addrspace(1)*> %vec, i32 %idx2
  br label %merge2
merge2:
; CHECK-LABEL: merge2:
; CHECK-NEXT: %obj = phi i64 addrspace(1)*
; CHECK-NEXT: statepoint
; CHECK: gc.relocate
; CHECK-DAG: ; (%obj, %obj)
  %obj = phi i64 addrspace(1)* [%obj0, %taken2], [%obj1, %untaken2]
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %obj
}

define i64 addrspace(1)* @test3(i64 addrspace(1)* %ptr) 
    gc "statepoint-example" {
; CHECK-LABEL: test3
entry:
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %ptr, i32 0
  %obj = extractelement <2 x i64 addrspace(1)*> %vec, i32 0
; CHECK: insertelement
; CHECK: extractelement
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: (%obj, %obj)
   %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %obj
}
define i64 addrspace(1)* @test4(i64 addrspace(1)* %ptr) 
    gc "statepoint-example" {
; CHECK-LABEL: test4
entry:
  %derived = getelementptr i64, i64 addrspace(1)* %ptr, i64 16
  %veca = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %derived, i32 0
  %vec = insertelement <2 x i64 addrspace(1)*> %veca, i64 addrspace(1)* %ptr, i32 1
  %obj = extractelement <2 x i64 addrspace(1)*> %vec, i32 0
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: ; (%ptr, %obj)
; CHECK: gc.relocate
; CHECK-DAG: ; (%ptr, %ptr)
   %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %obj
}

declare void @use(i64 addrspace(1)*)

; When we can optimize an extractelement from a known
; index and avoid introducing new base pointer instructions
define void @test5(i1 %cnd, i64 addrspace(1)* %obj)
    gc "statepoint-example" {
; CHECK-LABEL: @test5
; CHECK: gc.relocate
; CHECK-DAG: (%obj, %bdv)
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %obj, i64 1
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep, i32 0
  %bdv = extractelement <2 x i64 addrspace(1)*> %vec, i32 0
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  call void @use(i64 addrspace(1)* %bdv)
  ret void
}

; When we fundementally have to duplicate
define void @test6(i1 %cnd, i64 addrspace(1)* %obj, i64 %idx)
    gc "statepoint-example" {
; CHECK-LABEL: @test6
; CHECK: %gep = getelementptr i64, i64 addrspace(1)* %obj, i64 1
; CHECK: %vec.base = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %obj, i32 0, !is_base_value !0
; CHECK: %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep, i32 0
; CHECK: %bdv.base = extractelement <2 x i64 addrspace(1)*> %vec.base, i64 %idx, !is_base_value !0
; CHECK:  %bdv = extractelement <2 x i64 addrspace(1)*> %vec, i64 %idx
; CHECK: gc.statepoint
; CHECK: gc.relocate
; CHECK-DAG: (%bdv.base, %bdv)
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %obj, i64 1
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %gep, i32 0
  %bdv = extractelement <2 x i64 addrspace(1)*> %vec, i64 %idx
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  call void @use(i64 addrspace(1)* %bdv)
  ret void
}

; A more complicated example involving vector and scalar bases.
; This is derived from a failing test case when we didn't have correct
; insertelement handling.
define i64 addrspace(1)* @test7(i1 %cnd, i64 addrspace(1)* %obj, 
                                i64 addrspace(1)* %obj2)
    gc "statepoint-example" {
; CHECK-LABEL: @test7
entry:
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %obj2, i32 0
  br label %merge1
merge1:
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
next1:
; CHECK-LABEL: next1:
; CHECK: bdv.base = 
; CHECK: bdv = 
  %bdv = extractelement <2 x i64 addrspace(1)*> %vec2, i32 0
  br label %merge
merge:
; CHECK-LABEL: merge:
; CHECK: %objb.base
; CHECK: %objb
; CHECK: gc.statepoint
; CHECK: gc.relocate
; CHECK-DAG: (%objb.base, %objb)

  %objb = phi i64 addrspace(1)* [ %obj, %next1 ], [ %bdv, %merge ]
  br i1 %cnd, label %merge, label %next
next:
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %objb
}


declare void @do_safepoint()

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
