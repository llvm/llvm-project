; RUN: llc %s -o - -filetype=obj --mcpu=znver2 | llvm-objdump -dr  - | FileCheck %s
; RUN: llc %s -o - -filetype=asm --mcpu=znver2 | llvm-mc - -o - --mcpu=znver2 -filetype=obj -triple x86_64-unknown-linux-gnu | llvm-objdump -dr  - | FileCheck %s
; RUN: llc %s -o - -filetype=asm --mcpu=znver2 | FileCheck %s --check-prefix=ASM

;; Check that we produce a push, then an align-to-16-bytes p2align.
;
; ASM:        # %bb.0:
; ASM-NEXT:   pushq   %rax
; ASM-NEXT:   .cfi_def_cfa_offset 16
; ASM-NEXT:   .p2align 4{{$}}

;; When we assemble the file, either using the built-in asssembler or going
;; via a textual assembly file, we should get the same padding between the
;; initial push and the next block for alignment. It's a single 15 byte
;; nop.

; CHECK:        0:   50
; CHECK-NEXT:   66 66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00 nopw %cs:(%rax,%rax)

;; Note that we specify a CPU to ensure the same nop patterns are selected
;; between llvm-mc and llc, just in case defaults changed, which one isn't
;; important.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noreturn nounwind uwtable
define dso_local void @b() local_unnamed_addr {
entry:
  br label %for.cond

for.cond:
  tail call void (...) @a()
  br label %for.cond
}

declare void @a(...) local_unnamed_addr

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang"}
