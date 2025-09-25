; RUN: opt -passes='print<escape-analysis>' -disable-output %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: opt -passes='print<escape-analysis>' -escape-analysis-worklist-limit=1 -disable-output %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIMIT

; With the default limit, the passthrough chain is fully explored and remains local.
; With a tiny limit, CaptureTracking bails out conservatively and marks escape.
define void @phi_limit_bailout(i1 %c) {
; CHECK-DEFAULT-LABEL: Printing analysis 'Escape Analysis' for function 'phi_limit_bailout':
; CHECK-DEFAULT: a escapes: no
; CHECK-LIMIT-LABEL: Printing analysis 'Escape Analysis' for function 'phi_limit_bailout':
; CHECK-LIMIT: a escapes: yes
entry:
  %a = alloca i8, align 1
  br i1 %c, label %t, label %f

t:
  br label %merge

f:
  br label %merge

merge:
  %p = phi ptr [ %a, %t ], [ %a, %f ]
  ret void
}

