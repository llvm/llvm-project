; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_100 -O3 | FileCheck %s --check-prefix=PIPELINE
; RUN: opt < %s -passes=gvn -enable-scalar-pre=false -S | llc -mtriple=nvptx64 -mcpu=sm_100 -O0 | FileCheck %s --check-prefix=NO-SCALAR-PRE
; RUN: opt < %s -passes=gvn -enable-scalar-pre=true -S | llc -mtriple=nvptx64 -mcpu=sm_100 -O0 | FileCheck %s --check-prefix=SCALAR-PRE

; Scalar PRE inserts a critical-edge computation and a PHI for the common add.
; That shape needs more NVPTX virtual registers than keeping the duplicated adds.

define void @kernel(ptr %arr, i8 %cond) {
; PIPELINE-LABEL: kernel(
; PIPELINE:      .reg .b32 {{%r<3>;}}
;
; NO-SCALAR-PRE-LABEL: kernel(
; NO-SCALAR-PRE:      .reg .b32 {{%r<4>;}}
;
; SCALAR-PRE-LABEL: kernel(
; SCALAR-PRE:      .reg .b32 {{%r<6>;}}
entry:
  %tobool.not = icmp eq i8 %cond, 0
  %tmp7.pre = load i32, ptr %arr, align 4
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  %add = add nsw i32 %tmp7.pre, 2
  %getElem = getelementptr inbounds nuw i8, ptr %arr, i64 8
  store i32 %add, ptr %getElem, align 4
  br label %if.end

if.end:
  %add8 = add nsw i32 %tmp7.pre, 2
  %getElem1 = getelementptr inbounds nuw i8, ptr %arr, i64 12
  store i32 %add8, ptr %getElem1, align 4
  ret void
}
