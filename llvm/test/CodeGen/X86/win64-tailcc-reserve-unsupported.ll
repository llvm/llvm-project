; RUN: sed 's/UNWIND_MODE/2/' %s | not --crash llc -mtriple=x86_64-pc-windows-msvc 2>&1 | FileCheck %s --check-prefix=V2
; RUN: sed 's/UNWIND_MODE/3/' %s | not --crash llc -mtriple=x86_64-pc-windows-msvc 2>&1 | FileCheck %s --check-prefix=V3
; RUN: sed 's/UNWIND_MODE/0/' %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=V1
; RUN: sed 's/UNWIND_MODE/1/' %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=V2BEST

; The reserve is only described by classic (V1) unwind info, and is not handled
; in functions with EH funclets.

; V2: Can't handle guaranteed tail calls that change the stack argument size with Windows x64 unwind v2 or v3 yet
; V3: Can't handle guaranteed tail calls that change the stack argument size with Windows x64 unwind v2 or v3 yet
; V1: reserve:

; With best-effort V2 the function must fall back to V1 unwind info.
; The second function has pushes and a stack allocation, so its epilogue looks
; like a standard V2 epilogue.
; V2BEST-LABEL: reserve:
; V2BEST-NOT:   .seh_unwindv2start
; V2BEST:       .seh_endproc
; V2BEST-LABEL: reserve_fp:
; V2BEST-NOT:   .seh_unwindv2start
; V2BEST:       .seh_endproc

declare void @use(ptr)
declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define tailcc void @reserve(i64 %a, i64 %b) {
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}

define tailcc void @reserve_fp(i64 %a, i64 %b) "frame-pointer"="all" {
  %p = alloca [16 x i64]
  call void @use(ptr %p)
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"winx64-eh-unwind", i32 UNWIND_MODE}
