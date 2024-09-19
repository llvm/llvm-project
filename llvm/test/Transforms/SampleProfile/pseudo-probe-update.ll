; RUN: opt < %s -passes='pseudo-probe,sample-profile,jump-threading,pseudo-probe-update' -sample-profile-file=%S/Inputs/pseudo-probe-update.prof -S  | FileCheck %s

declare i32 @f1()
declare i32 @f2()
declare void @f3()


;; This tests that the branch in 'merge' can be cloned up into T1.
define i32 @foo(i1 %cond, i1 %cond2) #0 {
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 1, i32 0, i64 -1)
	br i1 %cond, label %T1, label %F1
T1:
; CHECK: %v1 = call i32 @f1(), !prof ![[#PROF1:]]
	%v1 = call i32 @f1()
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 2, i32 0, i64 -1)
;; The distribution factor -8513881372706734080 stands for 53.85%, whic is from 7/6+7.
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 6, i32 0, i64 -8513881372706734080)
    %cond3 = icmp eq i32 %v1, 412
	br label %Merge
F1:
; CHECK: %v2 = call i32 @f2(), !prof ![[#PROF2:]]
	%v2 = call i32 @f2()
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 4, i32 0, i64 -1)
;; The distribution factor 8513881922462547968 stands for 46.25%, which is from 6/6+7.
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 6, i32 0, i64 8513881922462547968)
	br label %Merge
Merge:

	%A = phi i1 [%cond3, %T1], [%cond2, %F1]
	%B = phi i32 [%v1, %T1], [%v2, %F1]
	br i1 %A, label %T2, label %F2
T2:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 7, i32 0, i64 -1)
	call void @f3()
	ret i32 %B
F2:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 9, i32 0, i64 -1)
	ret i32 %B
}

; CHECK: ![[#PROF1]] = !{!"branch_weights", i32 7}
; CHECK: ![[#PROF2]] = !{!"branch_weights", i32 6}

attributes #0 = {"use-sample-profile"}
