; RUN: llc < %s -march=x86 | FileCheck %s

define i32* @t() nounwind optsize ssp {
entry:
; CHECK-LABEL: t:
; CHECK: testl %eax, %eax
; CHECK: js
  %cmp = icmp slt i32 undef, 0                    ; <i1> [#uses=1]
  %outsearch.0 = select i1 %cmp, i1 false, i1 true ; <i1> [#uses=1]
  br i1 %outsearch.0, label %if.then27, label %if.else29

if.then27:                                        ; preds = %entry
  ret i32* undef

if.else29:                                        ; preds = %entry
  unreachable
}

