; RUN: llc -mtriple=x86_64-pc-windows-msvc -o - %s | FileCheck %s --check-prefix ASM
; RUN: llc --fast-isel -mtriple=x86_64-pc-windows-msvc -o - %s | FileCheck %s --check-prefix ASM
; RUN: llc --global-isel --global-isel-abort=2 -mtriple=x86_64-pc-windows-msvc -o - %s | \
; RUN:  FileCheck %s --check-prefix ASM
; RUN: llc -mtriple=x86_64-pc-windows-msvc  --filetype=obj -o - %s | llvm-objdump - --disassemble \
; RUN:  | FileCheck %s --check-prefix OBJ

@global_func_ptr = external dso_local local_unnamed_addr global ptr, align 8

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
; ASM:        .Limpcall0:
; ASM-NEXT:     rex64
; ASM-NEXT:     callq   *__imp_a(%rip)
; ASM-NEXT:     nopl    (%rax,%rax)
; ASM-NEXT:   .Limpcall1:
; ASM-NEXT:     rex64
; ASM-NEXT:     callq   *__imp_a(%rip)
; ASM-NEXT:     nopl    (%rax,%rax)
; ASM-NEXT:     movq    %rsi, %rax
; ASM-NEXT:   .Limpcall2:
; ASM-NEXT:     callq   *%rax
; ASM-NEXT:     nopl    (%rax)
; ASM-NEXT:     movq global_func_ptr(%rip), %rax
; ASM-NEXT:   .Limpcall3:
; ASM-NEXT:     callq   *%rax
; ASM-NEXT:     nopl    (%rax)
; ASM-NEXT:     nop

define dso_local void @tail_call() local_unnamed_addr section "tc_sect" {
entry:
  tail call void @b()
  ret void
}
; ASM-LABEL:  tail_call:
; ASM:        .Limpcall4:
; ASM-NEXT:     rex64 jmpq      *__imp_b(%rip)
; ASM-NEXT:     int3
; ASM-NEXT:     int3
; ASM-NEXT:     int3
; ASM-NEXT:     int3
; ASM-NEXT:     int3

define dso_local void @tail_call_fp(ptr noundef readonly %func_ptr) local_unnamed_addr section "tc_sect" {
entry:
  tail call void %func_ptr()
  ret void
}
; ASM-LABEL:  tail_call_fp:
; ASM:          movq    %rcx, %rax
; ASM-NEXT:   .Limpcall5:
; ASM-NEXT:     rex64 jmpq      *%rax
; ASM-NEXT:     int3
; ASM-NEXT:     int3

define dso_local void @tail_call_global_fp(ptr noundef readonly %func_ptr) local_unnamed_addr section "tc_sect" {
entry:
  %0 = load ptr, ptr @global_func_ptr, align 8
  tail call void %0()
  ret void
}
; ASM-LABEL:  tail_call_global_fp:
; ASM:          movq    global_func_ptr(%rip), %rax
; ASM-NEXT:   .Limpcall6:
; ASM-NEXT:     rex64 jmpq      *%rax
; ASM-NEXT:     int3
; ASM-NEXT:     int3

; Regression test: conditional tail calls can't be encoded, so make sure they aren't emitted.
define void @might_call(i1 %4) local_unnamed_addr {
  br i1 %4, label %makecall, label %finish

makecall:
  tail call void @a()
  br label %finish

finish:
  ret void
}
; ASM-LABEL:  might_call:
; ASM:        .Limpcall7:
; ASM-NEXT:     rex64 jmpq      *__imp_a(%rip)
; ASM-NEXT:     int3
; ASM-NEXT:     int3
; ASM-NEXT:     int3
; ASM-NEXT:     int3
; ASM-NEXT:     int3

; Regression test: this particular sequence caused a cycle in DAG scheduling due
; to the requirement to use RAX for register-indirect calls. We now explicitly
; copy to RAX which breaks the cycle.
define dso_local i32 @not_scheduled_repro(ptr %0, ptr %1, ptr %2) local_unnamed_addr {
  %4 = load i64, ptr %0, align 8
  %5 = inttoptr i64 %4 to ptr
  %6 = tail call i64 %5(ptr noundef %1)
  store i64 %6, ptr %2, align 8
  ret i32 0
}
; ASM-LABEL:  not_scheduled_repro:
; ASM:          movq    (%rcx), %rax
; ASM-NEXT:     movq    %rdx, %rcx
; ASM-NEXT:   .Limpcall8:
; ASM-NEXT:     callq   *%rax
; ASM-NEXT:     nopl    (%rax)

define dso_local void @not_scheduled_repro_tc(ptr %0, ptr %1) local_unnamed_addr {
  %4 = load i64, ptr %0, align 8
  %5 = inttoptr i64 %4 to ptr
  tail call void %5(ptr noundef %1)
  ret void
}
; ASM-LABEL:  not_scheduled_repro_tc:
; ASM:          movq    (%rcx), %rax
; ASM-NEXT:     movq    %rdx, %rcx
; ASM-NEXT:   .Limpcall9:
; ASM-NEXT:     rex64 jmpq      *%rax
; ASM-NEXT:     int3
; ASM-NEXT:     int3

declare dllimport void @a() local_unnamed_addr
declare dllimport void @b() local_unnamed_addr

; ASM-LABEL  .section   .retplne,"yi"
; ASM-NEXT   .asciz  "RetpolineV1"
; ASM-NEXT   .long   32
; ASM-NEXT   .secnum tc_sect
; ASM-NEXT   .long   2
; ASM-NEXT   .secoffset      .Limpcall4
; ASM-NEXT   .long   6
; ASM-NEXT   .secoffset      .Limpcall5
; ASM-NEXT   .long   6
; ASM-NEXT   .secoffset      .Limpcall6
; ASM-NEXT   .long   40
; ASM-NEXT   .secnum nc_sect
; ASM-NEXT   .long   3
; ASM-NEXT   .secoffset      .Limpcall0
; ASM-NEXT   .long   3
; ASM-NEXT   .secoffset      .Limpcall1
; ASM-NEXT   .long   5
; ASM-NEXT   .secoffset      .Limpcall2
; ASM-NEXT   .long   5
; ASM-NEXT   .secoffset      .Limpcall3
; ASM-NEXT   .long   32
; ASM-NEXT   .secnum .text
; ASM-NEXT   .long   2
; ASM-NEXT   .secoffset      .Limpcall7
; ASM-NEXT   .long   5
; ASM-NEXT   .secoffset      .Limpcall8
; ASM-NEXT   .long   6
; ASM-NEXT   .secoffset      .Limpcall9

; The loader assumes an exact sequence of instructions/bytes at each marked site since it may
; replace the instruction(s) with new instruction(s), and the MSVC linker validates these at link
; time.

; Kind = 3 (IMAGE_RETPOLINE_AMD64_IMPORT_CALL)
; OBJ-LABEL:  <normal_call>:
; OBJ:        : 48 ff 15 00 00 00 00          callq   *(%rip)
; OBJ-NEXT:   : 0f 1f 44 00 00                nopl    (%rax,%rax)

; Kind = 5 (IMAGE_RETPOLINE_AMD64_INDIR_CALL)
; OBJ:        : ff d0                         callq   *%rax
; OBJ-NEXT:   : 0f 1f 00                      nopl    (%rax)

; Kind = 2 (IMAGE_RETPOLINE_AMD64_IMPORT_BR)
; OBJ-LABEL:  <tc_sect>:
; OBJ:        : 48 ff 25 00 00 00 00          jmpq    *(%rip)
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3

; Kind = 6 (IMAGE_RETPOLINE_AMD64_INDIR_BR)
; OBJ-LABEL:  <tail_call_fp>:
; OBJ:        : 48 ff e0                      jmpq    *%rax
; OBJ-NEXT:   : cc                            int3
; OBJ-NEXT:   : cc                            int3

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"import-call-optimization", i32 1}
