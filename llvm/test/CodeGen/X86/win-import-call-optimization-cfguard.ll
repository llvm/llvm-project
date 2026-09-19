; RUN: llc -mtriple=x86_64-pc-windows-msvc -o - %s | FileCheck %s --check-prefix ASM
; RUN: llc --fast-isel -mtriple=x86_64-pc-windows-msvc -o - %s | FileCheck %s --check-prefix ASM
; RUN: llc --global-isel --global-isel-abort=2 -mtriple=x86_64-pc-windows-msvc -o - %s | \
; RUN:  FileCheck %s --check-prefix ASM
; RUN: llc -mtriple=x86_64-pc-windows-msvc  --filetype=obj -o - %s | llvm-objdump - --disassemble \
; RUN:  | FileCheck %s --check-prefix OBJ

@global_func_ptr = external dso_local local_unnamed_addr global ptr, align 8
declare dllimport void @a() local_unnamed_addr
declare dllimport void @b() local_unnamed_addr
declare dso_local i32 @__C_specific_handler(...)

define dso_local void @normal_call(ptr noundef readonly %func_ptr) local_unnamed_addr section "nc_sect" {
entry:
  call void @a()
  call void @a()
  call void %func_ptr()
  %0 = load ptr, ptr @global_func_ptr, align 8
  call void %0()
  ret void
}
; ASM-LABEL:  normal_call:
; ASM:          movq    %rcx, %rsi
; ASM-NEXT:   .Limpcall0:
; ASM-NEXT:     rex64
; ASM-NEXT:     callq   *__imp_a(%rip)
; ASM-NEXT:     nopl    (%rax,%rax)
; ASM-NEXT:   .Limpcall1:
; ASM-NEXT:     rex64
; ASM-NEXT:     callq   *__imp_a(%rip)
; ASM-NEXT:     nopl    (%rax,%rax)
; ASM-NEXT:     movq    %rsi, %rax
; ASM-NEXT:   .Limpcall2:
; ASM-NEXT:     callq   *__guard_dispatch_icall_fptr(%rip)
; ASM-NEXT:     movq    global_func_ptr(%rip), %rax
; ASM-NEXT:   .Limpcall3:
; ASM-NEXT:     callq   *__guard_dispatch_icall_fptr(%rip)
; ASM-NEXT:     nop

define dso_local void @tail_call() local_unnamed_addr section "tc_sect" {
entry:
  tail call void @b()
  ret void
}
; ASM-LABEL:  tail_call:
; ASM:        .Limpcall4:
; ASM-NEXT:     rex64 jmpq *__imp_b(%rip)

define dso_local void @tail_call_fp(ptr noundef readonly %func_ptr) local_unnamed_addr section "tc_sect" {
entry:
  tail call void %func_ptr()
  ret void
}
; ASM-LABEL:  tail_call_fp:
; ASM:          movq    %rcx, %rax
; ASM-NEXT:   .Limpcall5:
; ASM-NEXT:     rex64 jmpq      *__guard_dispatch_icall_fptr(%rip)

define dso_local void @tail_call_global_fp(ptr noundef readonly %func_ptr) local_unnamed_addr section "tc_sect" {
entry:
  %0 = load ptr, ptr @global_func_ptr, align 8
  tail call void %0()
  ret void
}
; ASM-LABEL:  tail_call_global_fp:
; ASM:          movq    global_func_ptr(%rip), %rax
; ASM-NEXT:  .Limpcall6:
; ASM-NEXT:     rex64 jmpq      *__guard_dispatch_icall_fptr(%rip)

; Regression test: the call to the CFG Guard was being indirected via a register, which is not
; permitted when retpoline is enabled.
define dso_local i32 @might_call_global_func_ptr(ptr %0, ptr %1, i32 %2) {
3:
  %4 = icmp eq i32 %2, 0
  br i1 %4, label %5, label %8

5:                                               ; preds = %11
  %6 = load ptr, ptr @global_func_ptr, align 8
  %7 = tail call i32 %6(ptr noundef %1)
  br label %8

8:
  %9 = phi i32 [ %7, %5 ], [ -1, %3 ]
  ret i32 %9
}
; ASM-LABEL:  might_call_global_func_ptr:
; ASM:          movq    global_func_ptr(%rip), %rax
; ASM-NEXT:     movq    %rdx, %rcx
; ASM-NEXT:  .Limpcall7:
; ASM-NEXT:     rex64 jmpq      *__guard_dispatch_icall_fptr(%rip)

define dso_local void @invoke_many_args(ptr %0, ptr %1, ptr %2) personality ptr @__C_specific_handler {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  invoke void %0(ptr %1, ptr %2, ptr %4, ptr %5, ptr %6)
          to label %7 unwind label %8

7:
  ret void

8:
  %9 = cleanuppad within none []
  cleanupret from %9 unwind to caller
}
; ASM-LABEL:  invoke_many_args:
; ASM:        .Limpcall8:
; ASM-NEXT:     callq   *__guard_dispatch_icall_fptr(%rip)
; ASM-NEXT:     nop

; ASM-LABEL  .section   .retplne,"yi"
; ASM-NEXT   .asciz  "RetpolineV1"
; ASM-NEXT   .long   24
; ASM-NEXT   .secnum .text
; ASM-NEXT   .long   10
; ASM-NEXT   .secoffset      .Limpcall7
; ASM-NEXT   .long   9
; ASM-NEXT   .secoffset      .Limpcall8
; ASM-NEXT   .long   40
; ASM-NEXT   .secnum nc_sect
; ASM-NEXT   .long   3
; ASM-NEXT   .secoffset      .Limpcall0
; ASM-NEXT   .long   3
; ASM-NEXT   .secoffset      .Limpcall1
; ASM-NEXT   .long   9
; ASM-NEXT   .secoffset      .Limpcall2
; ASM-NEXT   .long   9
; ASM-NEXT   .secoffset      .Limpcall3
; ASM-NEXT   .long   32
; ASM-NEXT   .secnum tc_sect
; ASM-NEXT   .long   2
; ASM-NEXT   .secoffset      .Limpcall4
; ASM-NEXT   .long   10
; ASM-NEXT   .secoffset      .Limpcall5
; ASM-NEXT   .long   10
; ASM-NEXT   .secoffset      .Limpcall6

; The loader assumes an exact sequence of instructions/bytes at each marked site since it may
; replace the instruction(s) with new instruction(s), and the MSVC linker validates these at link
; time.

; Kind = 9 (IMAGE_RETPOLINE_AMD64_CFG_CALL)
; OBJ-LABEL:  <normal_call>:
; OBJ:        : ff 15 00 00 00 00             callq   *(%rip)

; Kind = 10 (IMAGE_RETPOLINE_AMD64_CFG_BR_REX)
; OBJ-LABEL:  <tc_sect>:
; OBJ:        : 48 ff 25 00 00 00 00          jmpq    *(%rip)

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"import-call-optimization", i32 1}
!1 = !{i32 2, !"cfguard", i32 2}
