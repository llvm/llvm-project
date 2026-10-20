; RUN: opt %s -passes='loop-interchange' -pass-remarks=loop-interchange -disable-output 2>&1 | FileCheck --allow-empty %s

target triple = "aarch64-unknown-linux-gnu"

; CHECK-NOT: Computed dependence info, invoking the transform.

; For the below test, backedge count cannot be computed. 
; Computing backedge count requires only SCEV and should
; not require dependence info. 
;
; void bar(int m, int n) {
; for (unsigned int i = 0; i < m; ++i) {
;    for (unsigned int j = 0; j < m; ++j) {
;     // dummy code
;    }
;  }
;}

define void @bar(i32 %m, i32 %n)
{
entry:
  br label %outer.header

outer.header:
 %m_temp1 = phi i32 [%m, %entry], [%m_temp, %outer.latch]
 br label %inner.header


inner.header:
 %n_temp1 = phi i32 [%n, %outer.header], [%n_temp, %inner.latch]

 br label %body

body:
 ; dummy code

br label %inner.latch 

inner.latch:
%n_temp = add i32 %n_temp1, 1
%cmp2 = icmp eq i32 %n_temp, 1 
br i1 %cmp2, label %outer.latch, label %inner.header

outer.latch:
%m_temp = add i32 %n, 1
%cmp3 = icmp eq i32 %m_temp, 1 
br i1 %cmp3, label %exit, label %outer.header

exit:
ret void
}

