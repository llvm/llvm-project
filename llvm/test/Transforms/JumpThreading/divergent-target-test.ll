; REQUIRES: amdgpu-registered-target && x86-registered-target
; RUN: opt < %s -mtriple=amdgcn -passes=jump-threading -S | FileCheck %s -check-prefixes=CHECK,DIVERGENT
; RUN: opt < %s -mtriple=x86_64 -passes=jump-threading -S | FileCheck %s -check-prefixes=CHECK,UNIFORM

; Here we assure that for the target with no branch divergence usual Jump Threading optimization performed
; For target with branch divergence - no optimization, so the IR is unchanged.

declare i32 @f1()
declare i32 @f2()
declare void @f3()

define i32 @test(i1 %cond) {
; CHECK: test
	br i1 %cond, label %T1, label %F1

; DIVERGENT:   T1
; UNIFORM-NOT: T1
T1:
	%v1 = call i32 @f1()
	br label %Merge
; DIVERGENT:   F1
; UNIFORM-NOT: F1
F1:
	%v2 = call i32 @f2()
	br label %Merge
; DIVERGENT:   Merge
; UNIFORM-NOT: Merge
Merge:
	%A = phi i1 [true, %T1], [false, %F1]
	%B = phi i32 [%v1, %T1], [%v2, %F1]
	br i1 %A, label %T2, label %F2

; DIVERGENT:   T2
T2:
; UNIFORM: T2:
; UNIFORM: %v1 = call i32 @f1()
; UNIFORM: call void @f3()
; UNIFORM: ret i32 %v1
	call void @f3()
	ret i32 %B
; DIVERGENT:   F2
F2:
; UNIFORM: F2:
; UNIFORM: %v2 = call i32 @f2()
; UNIFORM: ret i32 %v2
	ret i32 %B
}

; Check divergence check is skipped if there can't be divergence in
; the function.
define i32 @requires_single_lane_exec(i1 %cond) #0 {
; CHECK: requires_single_lane_exec
	br i1 %cond, label %T1, label %F1

; CHECK-NOT: T1
T1:
	%v1 = call i32 @f1()
	br label %Merge
; CHECK-NOT: F1
F1:
	%v2 = call i32 @f2()
	br label %Merge
; CHECK-NOT: Merge
Merge:
	%A = phi i1 [true, %T1], [false, %F1]
	%B = phi i32 [%v1, %T1], [%v2, %F1]
	br i1 %A, label %T2, label %F2

T2:
; CHECK: T2:
; CHECK: %v1 = call i32 @f1()
; CHECK: call void @f3()
; CHECK: ret i32 %v1
	call void @f3()
	ret i32 %B
F2:
; CHECK: F2:
; CHECK: %v2 = call i32 @f2()
; CHECK: ret i32 %v2
	ret i32 %B
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1" }
