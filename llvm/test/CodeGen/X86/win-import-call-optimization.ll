; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s --check-prefix=CHECK
; RUN: llc --fast-isel -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s --check-prefix=CHECK
; RUN: llc --global-isel --global-isel-abort=2 -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s --check-prefix=CHECK
; RUN: llc -mtriple=x86_64-pc-windows-msvc -filetype=obj < %s | llvm-objdump -dr - | FileCheck %s --check-prefix=OBJ

define dso_local void @normal_call(ptr noundef readonly %func_ptr) local_unnamed_addr section "nc_sect" {
entry:
  call void @a()
  call void @a()
  call void %func_ptr()
  ret void
}
; CHECK-LABEL:  normal_call:
; CHECK:        .Limpcall0:
; CHECK-NEXT:     rex64
; CHECK-NEXT:     callq   __imp_a
; CHECK-NEXT:     nopl    8(%rax,%rax)
; CHECK-NEXT:   .Limpcall1:
; CHECK-NEXT:     rex64
; CHECK-NEXT:     callq   __imp_a
; CHECK-NEXT:     nopl    8(%rax,%rax)
; CHECK-NEXT:     movq    %rsi, %rax
; CHECK-NEXT:   .Limpcall2:
; CHECK-NEXT:     callq   *%rax
; CHECK-NEXT:     nopl    (%rax)
; CHECK-NEXT:     nop

define dso_local void @tail_call() local_unnamed_addr section "tc_sect" {
entry:
  tail call void @b()
  ret void
}
; CHECK-LABEL:  tail_call:
; CHECK:        .Limpcall3:
; CHECK-NEXT:     jmp __imp_b

define dso_local void @tail_call_fp(ptr noundef readonly %func_ptr) local_unnamed_addr section "tc_sect" {
entry:
  tail call void %func_ptr()
  ret void
}
; CHECK-LABEL:  tail_call_fp:
; CHECK:          movq    %rcx, %rax
; CHECK-NEXT:   .Limpcall4:
; CHECK-NEXT:     rex64 jmpq      *%rax

; The loader can overwrite the five-byte ICO NOP, so the WinEH NOP must
; remain a separate instruction before the end of the invoke's EH range.
define void @import_invoke() section "nc_sect" personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @a() to label %cont unwind label %ehcleanup

cont:
  ret void

ehcleanup:
  %cp = cleanuppad within none []
  call void @cleanup() [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}
; CHECK-LABEL:  import_invoke:
; CHECK:        [[IMPORT_INVOKE:\.Limpcall[0-9]+]]:
; CHECK-NEXT:     rex64
; CHECK-NEXT:     callq   __imp_a
; CHECK-NEXT:     nopl    8(%rax,%rax)
; CHECK-NEXT:     nop
; CHECK-NEXT:   .Ltmp{{[0-9]+}}:{{.*}}# EH_LABEL
; OBJ-LABEL:    <import_invoke>:
; OBJ:          {{^ *[[:xdigit:]]+:}} 48 e8 00 00 00 00 callq
; OBJ-NEXT:     IMAGE_REL_AMD64_REL32 __imp_a
; OBJ-NEXT:     {{^ *[[:xdigit:]]+:}} 0f 1f 44 00 08 nopl
; OBJ-NEXT:     {{^ *[[:xdigit:]]+:}} 90 nop{{$}}

declare dllimport void @a() local_unnamed_addr
declare dllimport void @b() local_unnamed_addr
declare void @cleanup()
declare i32 @__CxxFrameHandler3(...)

; CHECK:       .section   .retplne,"yi"
; CHECK-NEXT   .asciz  "RetpolineV1"
; CHECK-NEXT   .long   24
; CHECK-NEXT   .secnum tc_sect
; CHECK-NEXT   .long   3
; CHECK-NEXT   .secoffset      .Limpcall3
; CHECK-NEXT   .long   5
; CHECK-NEXT   .secoffset      .Limpcall4
; CHECK:       .long   40
; CHECK-NEXT:  .secnum nc_sect
; CHECK-NEXT:  .long   3
; CHECK-NEXT:  .secoffset      .Limpcall0
; CHECK-NEXT:  .long   3
; CHECK-NEXT:  .secoffset      .Limpcall1
; CHECK-NEXT:  .long   5
; CHECK-NEXT:  .secoffset      .Limpcall2
; CHECK-NEXT:  .long   3
; CHECK-NEXT:  .secoffset      [[IMPORT_INVOKE]]

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"import-call-optimization", i32 1}
