; RUN: sed 's/UNWIND_MODE/2/' %s | not --crash llc -mtriple=x86_64-pc-windows-msvc 2>&1 | FileCheck %s --check-prefix=V2REQ
; RUN: sed 's/UNWIND_MODE/1/' %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=V2BEST

; A tail call that shrinks the stack argument area needs an exit that is only
; described by classic (V1) unwind info. Where V2 is required that is an error;
; where it is best-effort the function falls back to V1.

; V2REQ: Can't handle guaranteed tail calls that change the stack argument size with Windows x64 unwind v2 or v3 yet
; V2BEST-LABEL: shrink:
; V2BEST-NOT:   .seh_unwindv2start
; V2BEST:       addq $48, %rsp
; V2BEST-NEXT:  jmp h # TAILCALL

declare tailcc void @h(i64, i64, i64, i64, i64, i64, i64)

define tailcc void @shrink(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i64 %i, i64 %j, i64 %k, i64 %l) {
  musttail call tailcc void @h(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 UNWIND_MODE}
