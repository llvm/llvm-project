; RUN: not llc -mtriple=x86_64-pc-windows-msvc %s -o /dev/null 2>&1 | FileCheck %s

; Win64 passes byval arguments indirectly, through a temporary in the caller's
; frame, which a guaranteed tail call would discard.

; CHECK: error: {{.*}}: cannot guarantee a tail call with arguments passed indirectly

declare tailcc void @h(i64, ptr byval([9 x i64]) align 8)

define tailcc void @f(i64 %a, ptr byval([9 x i64]) align 8 %p) {
  musttail call tailcc void @h(i64 %a, ptr byval([9 x i64]) align 8 %p)
  ret void
}
