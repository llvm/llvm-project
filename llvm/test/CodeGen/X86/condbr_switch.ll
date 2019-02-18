; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=sandybridge %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=ivybridge %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=haswell %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=broadwell %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=skylake %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=skx %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=NOTMERGE

@v1 = common dso_local local_unnamed_addr global i32 0, align 4
@v2 = common dso_local local_unnamed_addr global i32 0, align 4
@v3 = common dso_local local_unnamed_addr global i32 0, align 4
@v4 = common dso_local local_unnamed_addr global i32 0, align 4
@v5 = common dso_local local_unnamed_addr global i32 0, align 4
@v6 = common dso_local local_unnamed_addr global i32 0, align 4
@v7 = common dso_local local_unnamed_addr global i32 0, align 4
@v8 = common dso_local local_unnamed_addr global i32 0, align 4
@v9 = common dso_local local_unnamed_addr global i32 0, align 4
@v10 = common dso_local local_unnamed_addr global i32 0, align 4
@v11 = common dso_local local_unnamed_addr global i32 0, align 4
@v12 = common dso_local local_unnamed_addr global i32 0, align 4
@v13 = common dso_local local_unnamed_addr global i32 0, align 4
@v14 = common dso_local local_unnamed_addr global i32 0, align 4
@v15 = common dso_local local_unnamed_addr global i32 0, align 4

define dso_local i32 @fourcases(i32 %n) {
entry:
  switch i32 %n, label %return [
    i32 111, label %sw.bb
    i32 222, label %sw.bb1
    i32 3665, label %sw.bb2
    i32 4444, label %sw.bb4
  ]

sw.bb:
  %0 = load i32, i32* @v1, align 4
  br label %return

sw.bb1:
  %1 = load i32, i32* @v2, align 4
  %add = add nsw i32 %1, 12
  br label %return

sw.bb2:
  %2 = load i32, i32* @v3, align 4
  %add3 = add nsw i32 %2, 13
  br label %return

sw.bb4:
  %3 = load i32, i32* @v1, align 4
  %4 = load i32, i32* @v2, align 4
  %add5 = add nsw i32 %4, %3
  br label %return

return:
  %retval.0 = phi i32 [ %add5, %sw.bb4 ], [ %add3, %sw.bb2 ], [ %add, %sw.bb1 ], [ %0, %sw.bb ], [ 0, %entry ]
  ret i32 %retval.0
}
; MERGE-LABEL: fourcases
; MERGE: cmpl $3665
; MERGE-NEXT: jg
; MERGE-NEXT: jge
; NOTMERGE: cmpl $3664
; NOTMERGE-NEXT: jg

define dso_local i32 @fifteencases(i32) {
  switch i32 %0, label %32 [
    i32 -111, label %2
    i32 -13, label %4
    i32 25, label %6
    i32 37, label %8
    i32 89, label %10
    i32 111, label %12
    i32 213, label %14
    i32 271, label %16
    i32 283, label %18
    i32 325, label %20
    i32 327, label %22
    i32 429, label %24
    i32 500, label %26
    i32 603, label %28
    i32 605, label %30
  ]

; <label>:2
  %3 = load i32, i32* @v1, align 4
  br label %32

; <label>:4
  %5 = load i32, i32* @v2, align 4
  br label %32

; <label>:6
  %7 = load i32, i32* @v3, align 4
  br label %32

; <label>:8
  %9 = load i32, i32* @v4, align 4
  br label %32

; <label>:10
  %11 = load i32, i32* @v5, align 4
  br label %32

; <label>:12
  %13 = load i32, i32* @v6, align 4
  br label %32

; <label>:14
  %15 = load i32, i32* @v7, align 4
  br label %32

; <label>:16
  %17 = load i32, i32* @v8, align 4
  br label %32

; <label>:18
  %19 = load i32, i32* @v9, align 4
  br label %32

; <label>:20
  %21 = load i32, i32* @v10, align 4
  br label %32

; <label>:22
  %23 = load i32, i32* @v11, align 4
  br label %32

; <label>:24
  %25 = load i32, i32* @v12, align 4
  br label %32

; <label>:26
  %27 = load i32, i32* @v13, align 4
  br label %32

; <label>:28:
  %29 = load i32, i32* @v14, align 4
  br label %32

; <label>:30:
  %31 = load i32, i32* @v15, align 4
  br label %32

; <label>:32:
  %33 = phi i32 [ %31, %30 ], [ %29, %28 ], [ %27, %26 ], [ %25, %24 ], [ %23, %22 ], [ %21, %20 ], [ %19, %18 ], [ %17, %16 ], [ %15, %14 ], [ %13, %12 ], [ %11, %10 ], [ %9, %8 ], [ %7, %6 ], [ %5, %4 ], [ %3, %2 ], [ 0, %1 ]
  ret i32 %33
}
; MERGE-LABEL: fifteencases
; MERGE: cmpl $271
; MERGE-NEXT: jg
; MERGE-NEXT: jge
; MERGE: cmpl $37
; MERGE-NEXT: jg
; MERGE-NEXT: jge
; MERGE: cmpl $429
; MERGE-NEXT: jg
; MERGE-NEXT: jge
; MERGE: cmpl $325
; MERGE-NEXT: jg
; MERGE-NEXT: jge
; MERGE: cmpl $603
; MERGE-NEXT: jg
; MERGE-NEXT: jge
; NOTMERGE-LABEL: fifteencases
; NOTMERGE: cmpl $270
; NOTMERGE-NEXT: jle

