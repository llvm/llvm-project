; RUN: llc -O3 -mtriple=riscv64 -riscv-use-rematerializable-movimm \
; RUN:   -riscv-enable-test-foldasload \
; RUN:   < %s -o /dev/null

; This keeps a rematerializable immediate and a COPY alive through register
; coalescing. LiveRangeEdit::foldAsLoad() folds the immediate into the COPY,
; then removes the old COPY from the coalescer worklist.

target triple = "riscv64-pc-unknown-gnu"

define void @f() {
entry:
  br label %loop

loop:
  %i32 = phi i32 [ %and, %inc ], [ 3, %entry ]
  %i8 = phi i8 [ %tr, %inc ], [ 3, %entry ]
  %c = icmp eq i8 %i8, 0
  br i1 %c, label %exit, label %inc

exit:
  ret void

inc:
  %tr = trunc i32 %i32 to i8
  %and = and i32 %i32, 255
  br label %loop
}