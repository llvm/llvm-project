; A collection of liveness test cases to ensure we're reporting the
; correct live values at statepoints
; RUN: opt -rewrite-statepoints-for-gc -spp-rematerialization-threshold=0 -S < %s | FileCheck %s


; Tests to make sure we consider %obj live in both the taken and untaken 
; predeccessor of merge.
define i64 addrspace(1)* @test1(i1 %cmp, i64 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test1
entry:
  br i1 %cmp, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
; CHECK-NEXT: bitcast
; CHECK-NEXT: br label %merge
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  br label %merge

untaken:
; CHECK-LABEL: untaken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated1 = call coldcc i8 addrspace(1)*
; CHECK-NEXT: bitcast
; CHECK-NEXT: br label %merge
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  br label %merge

merge:
; CHECK-LABEL: merge:
; CHECK-NEXT: %.0 = phi i64 addrspace(1)* [ %obj.relocated.casted, %taken ], [ %obj.relocated1.casted, %untaken ]
; CHECK-NEXT: ret i64 addrspace(1)* %.0
  ret i64 addrspace(1)* %obj
}

; A local kill should not effect liveness in predecessor block
define i64 addrspace(1)* @test2(i1 %cmp, i64 addrspace(1)** %loc) gc "statepoint-example" {
; CHECK-LABEL: @test2
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT:  gc.statepoint
; CHECK-NEXT:  br
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  br i1 %cmp, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT:  %obj = load
; CHECK-NEXT:  gc.statepoint
; CHECK-NEXT:  gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT:  ret i64 addrspace(1)* %obj.relocated.casted

  %obj = load i64 addrspace(1)*, i64 addrspace(1)** %loc
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %obj

untaken:
  ret i64 addrspace(1)* null
}

; A local kill should effect values live from a successor phi.  Also, we
; should only propagate liveness from a phi to the appropriate predecessors.
define i64 addrspace(1)* @test3(i1 %cmp, i64 addrspace(1)** %loc) gc "statepoint-example" {
; CHECK-LABEL: @test3
entry:
  br i1 %cmp, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj = load
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
; CHECK-NEXT: bitcast
; CHECK-NEXT: br label %merge
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  %obj = load i64 addrspace(1)*, i64 addrspace(1)** %loc
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  br label %merge

untaken:
; CHECK-LABEL: taken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: br label %merge
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  br label %merge

merge:
  %phi = phi i64 addrspace(1)* [ %obj, %taken ], [ null, %untaken ]
  ret i64 addrspace(1)* %phi
}

; A base pointer must be live if it is needed at a later statepoint,
; even if the base pointer is otherwise unused.
define i64 addrspace(1)* @test4(i1 %cmp, i64 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test4
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT:  %derived = getelementptr
; CHECK-NEXT:  gc.statepoint
; CHECK-NEXT:  %derived.relocated =
; CHECK-NEXT:  bitcast 
; CHECK-NEXT:  %obj.relocated =
; CHECK-NEXT:  bitcast
; CHECK-NEXT:  gc.statepoint
; CHECK-NEXT:  %derived.relocated1 =
; CHECK-NEXT:  bitcast 
; Note: It's legal to relocate obj again, but not strictly needed
; CHECK-NEXT:  %obj.relocated2 =
; CHECK-NEXT:  bitcast
; CHECK-NEXT:  ret i64 addrspace(1)* %derived.relocated1.casted
; 
  %derived = getelementptr i64, i64 addrspace(1)* %obj, i64 8
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)

  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %derived
}

declare void @consume(...) readonly

; Make sure that a phi def visited during iteration is considered a kill.
; Also, liveness after base pointer analysis can change based on new uses,
; not just new defs.
define i64 addrspace(1)* @test5(i1 %cmp, i64 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test5
entry:
  br i1 %cmp, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
; CHECK-NEXT: bitcast
; CHECK-NEXT: br label %merge
  call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  br label %merge

untaken:
; CHECK-LABEL: untaken:
; CHECK-NEXT: br label %merge
  br label %merge

merge:
; CHECK-LABEL: merge:
; CHECK-NEXT: %.0 = phi i64 addrspace(1)*
; CHECK-NEXT: %obj2a = phi
; CHECK-NEXT: @consume
; CHECK-NEXT: br label %final
  %obj2a = phi i64 addrspace(1)* [ %obj, %taken ], [null, %untaken]
  call void (...) @consume(i64 addrspace(1)* %obj2a)
  br label %final
final:
; CHECK-LABEL: final:
; CHECK-NEXT: @consume
; CHECK-NEXT: ret i64 addrspace(1)* %.0
  call void (...) @consume(i64 addrspace(1)* %obj2a)
  ret i64 addrspace(1)* %obj
}

declare void @foo()

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
