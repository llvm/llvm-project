; RUN: opt %s -dot-cfg -cfg-heat-colors -cfg-dot-filename-prefix=%t -disable-output
; RUN: FileCheck %s -input-file=%t.f.dot

; CHECK: color="#[[#%x,]]", style={{[a-z]+}}, fillcolor="#[[#%x,]]"
; CHECK: color="#[[#%x,]]", style={{[a-z]+}}, fillcolor="#[[#%x,]]"
; CHECK: color="#[[#%x,]]", style={{[a-z]+}}, fillcolor="#[[#%x,]]"

define void @f(i32) {
entry:
  %check = icmp sgt i32 %0, 0
  br i1 %check, label %if, label %exit, !prof !0

if:                     ; preds = %entry
  br label %exit
exit:                   ; preds = %entry, %if
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 200}
