; RUN: llc -mtriple=m68k -mattr="+reserve-a0" < %s | FileCheck --check-prefix=A0 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-a1" < %s | FileCheck --check-prefix=A1 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-a2" < %s | FileCheck --check-prefix=A2 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-a3" < %s | FileCheck --check-prefix=A3 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-a4" < %s | FileCheck --check-prefix=A4 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-a5" < %s | FileCheck --check-prefix=A5 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-a6" < %s | FileCheck --check-prefix=A6 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-d0" < %s | FileCheck --check-prefix=D0 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-d1" < %s | FileCheck --check-prefix=D1 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-d2" < %s | FileCheck --check-prefix=D2 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-d3" < %s | FileCheck --check-prefix=D3 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-d4" < %s | FileCheck --check-prefix=D4 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-d5" < %s | FileCheck --check-prefix=D5 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-d6" < %s | FileCheck --check-prefix=D6 %s
; RUN: llc -mtriple=m68k -mattr="+reserve-d7" < %s | FileCheck --check-prefix=D7 %s

; Used to exhaust all registers
;
; A better way to do this might be:
; ```
; @var = global [16 x i32] zeroinitializer
; ...
; %tmp = load load volatile [16 x i32], [16 x i32]* @var
; store volatile [16 x i32] %tmp, [16 x i32]* @var
; ```
; Which is copied from `test/CodeGen/RISCV/reserved-regs.ll`.
; But currently we have problem doing codegen for the above snippet
; (https://bugs.llvm.org/show_bug.cgi?id=50377).
define void @foo(ptr nocapture readonly %a, ptr nocapture readonly %b, ptr nocapture readonly %c, ptr nocapture readonly %d,
                 ptr nocapture readonly %a1, ptr nocapture readonly %b1, ptr nocapture readonly %c1, ptr nocapture readonly %d1,
                 ptr nocapture readonly %a2, ptr nocapture readonly %b2, ptr nocapture readonly %c2, ptr nocapture readonly %d2,
                 ptr nocapture readonly %a3, ptr nocapture readonly %b3, ptr nocapture readonly %c3, ptr nocapture readonly %d3) {
entry:
  %0 = load i32, ptr %a, align 4
  %1 = load i32, ptr %b, align 4
  %2 = load i32, ptr %c, align 4
  %3 = load i32, ptr %d, align 4
  %4 = load i32, ptr %a1, align 4
  %5 = load i32, ptr %b1, align 4
  %6 = load i32, ptr %c1, align 4
  %7 = load i32, ptr %d1, align 4
  %8 = load i32, ptr %a2, align 4
  %9 = load i32, ptr %b2, align 4
  %10 = load i32, ptr %c2, align 4
  %11 = load i32, ptr %d2, align 4
  %12 = load i32, ptr %a3, align 4
  %13 = load i32, ptr %b3, align 4
  %14 = load i32, ptr %c3, align 4
  %15 = load i32, ptr %d3, align 4
  ; A0-NOT: %a0
  ; A1-NOT: %a1
  ; A2-NOT: %a2
  ; A3-NOT: %a3
  ; A4-NOT: %a4
  ; A5-NOT: %a5
  ; A6-NOT: %a6
  ; D0-NOT: %d0
  ; D1-NOT: %d1
  ; D2-NOT: %d2
  ; D3-NOT: %d3
  ; D4-NOT: %d4
  ; D5-NOT: %d5
  ; D6-NOT: %d6
  ; D7-NOT: %d7
  tail call void @bar(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14, i32 %15)
  ret void
}

declare void @bar(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

