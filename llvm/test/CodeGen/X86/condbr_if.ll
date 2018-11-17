; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=sandybridge %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=ivybridge %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=haswell %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=broadwell %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=skylake %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu -mcpu=skx %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=MERGE
; RUN: llc -x86-condbr-folding=true -mtriple=x86_64-linux-gnu %s -o - -verify-machineinstrs | FileCheck %s --check-prefix=NOTMERGE

define i32 @length2_1(i32) {
  %2 = icmp slt i32 %0, 3
  br i1 %2, label %3, label %5

; <label>:3:
  %4 = tail call i32 (...) @f1()
  br label %13

; <label>:5:
  %6 = icmp slt i32 %0, 40
  br i1 %6, label %7, label %13

; <label>:7:
  %8 = icmp eq i32 %0, 3
  br i1 %8, label %9, label %11

; <label>:9:
  %10 = tail call i32 (...) @f2()
  br label %11

; <label>:11:
  %12 = tail call i32 (...) @f3() #2
  br label %13

; <label>:13:
  ret i32 0
}
; MERGE-LABEL: length2_1
; MERGE: cmpl $3
; MERGE-NEXT: jg
; MERGE-NEXT: jge
; NOTMERGE-LABEL: length2_1
; NOTMERGE: cmpl $2
; NOTMERGE-NEXT: jg

define i32 @length2_2(i32) {
  %2 = icmp sle i32 %0, 2
  br i1 %2, label %3, label %5

; <label>:3:
  %4 = tail call i32 (...) @f1()
  br label %13

; <label>:5:
  %6 = icmp slt i32 %0, 40
  br i1 %6, label %7, label %13

; <label>:7:
  %8 = icmp eq i32 %0, 3
  br i1 %8, label %9, label %11

; <label>:9:
  %10 = tail call i32 (...) @f2()
  br label %11

; <label>:11:
  %12 = tail call i32 (...) @f3() #2
  br label %13

; <label>:13:
  ret i32 0
}
; MERGE-LABEL: length2_2
; MERGE: cmpl $3
; MERGE-NEXT: jg
; MERGE-NEXT: jge
; NOTMERGE-LABEL: length2_2
; NOTMERGE: cmpl $2
; NOTMERGE-NEXT: jg

define i32 @length2_3(i32) {
  %2 = icmp sgt i32 %0, 3
  br i1 %2, label %3, label %5

; <label>:3:
  %4 = tail call i32 (...) @f1()
  br label %13

; <label>:5:
  %6 = icmp sgt i32 %0, -40
  br i1 %6, label %7, label %13

; <label>:7:
  %8 = icmp eq i32 %0, 3
  br i1 %8, label %9, label %11

; <label>:9:
  %10 = tail call i32 (...) @f2()
  br label %11

; <label>:11:
  %12 = tail call i32 (...) @f3() #2
  br label %13

; <label>:13:
  ret i32 0
}
; MERGE-LABEL: length2_3
; MERGE: cmpl $3
; MERGE-NEXT: jl
; MERGE-NEXT: jle
; NOTMERGE-LABEL: length2_3
; NOTMERGE: cmpl $4
; NOTMERGE-NEXT: jl

define i32 @length2_4(i32) {
  %2 = icmp sge i32 %0, 4
  br i1 %2, label %3, label %5

; <label>:3:
  %4 = tail call i32 (...) @f1()
  br label %13

; <label>:5:
  %6 = icmp sgt i32 %0, -40
  br i1 %6, label %7, label %13

; <label>:7:
  %8 = icmp eq i32 %0, 3
  br i1 %8, label %9, label %11

; <label>:9:
  %10 = tail call i32 (...) @f2()
  br label %11

; <label>:11:
  %12 = tail call i32 (...) @f3() #2
  br label %13

; <label>:13:
  ret i32 0
}
; MERGE-LABEL: length2_4
; MERGE: cmpl $3
; MERGE-NEXT: jl
; MERGE-NEXT: jle
; NOTMERGE-LABEL: length2_4
; NOTMERGE: cmpl $4
; NOTMERGE-NEXT: jl

declare i32 @f1(...)
declare i32 @f2(...)
declare i32 @f3(...)

define i32 @length1_1(i32) {
  %2 = icmp sgt i32 %0, 5
  br i1 %2, label %3, label %5

; <label>:3:
  %4 = tail call i32 (...) @f1()
  br label %9

; <label>:5:
  %6 = icmp eq i32 %0, 5
  br i1 %6, label %7, label %9

; <label>:7:
  %8 = tail call i32 (...) @f2()
  br label %9

; <label>:9:
  ret i32 0
}
; MERGE-LABEL: length1_1
; MERGE: cmpl $5
; MERGE-NEXT: jl
; MERGE-NEXT: jle
; NOTMERGE-LABEL: length1_1
; NOTMERGE: cmpl $6
; NOTMERGE-NEXT: jl
