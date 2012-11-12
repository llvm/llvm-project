; RUN: llc < %s -march=x86 -disable-cgp-branch-opts    | FileCheck %s -check-prefix=32
; RUN: llc < %s -march=x86-64 -disable-cgp-branch-opts | FileCheck %s -check-prefix=64
; rdar://7573216
; PR6146

define i32 @t1(i32 %x) nounwind readnone ssp {
entry:
; 32: t1:
; 32: cmpl $1
; 32: sbbl

; 64: t1:
; 64: cmpl $1
; 64: sbbl
  %0 = icmp eq i32 %x, 0
  %iftmp.0.0 = select i1 %0, i32 -1, i32 0
  ret i32 %iftmp.0.0
}

define i32 @t2(i32 %x) nounwind readnone ssp {
entry:
; 32: t2:
; 32: cmpl $1
; 32: sbbl

; 64: t2:
; 64: cmpl $1
; 64: sbbl
  %0 = icmp eq i32 %x, 0
  %iftmp.0.0 = sext i1 %0 to i32
  ret i32 %iftmp.0.0
}

%struct.zbookmark = type { i64, i64 }
%struct.zstream = type { }

define i32 @t3() nounwind readonly {
entry:
; 32: t3:
; 32: cmpl $1
; 32: sbbl
; 32: cmpl
; 32: xorl

; 64: t3:
; 64: cmpl $1
; 64: sbbq
; 64: cmpq
; 64: xorl
  %not.tobool = icmp eq i32 undef, 0              ; <i1> [#uses=2]
  %cond = sext i1 %not.tobool to i32              ; <i32> [#uses=1]
  %conv = sext i1 %not.tobool to i64              ; <i64> [#uses=1]
  %add13 = add i64 0, %conv                       ; <i64> [#uses=1]
  %cmp = icmp ult i64 undef, %add13               ; <i1> [#uses=1]
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %xor27 = xor i32 undef, %cond                   ; <i32> [#uses=0]
  ret i32 0
}
