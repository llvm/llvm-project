; RUN: opt < %s -indvars -replexitval=always -S | FileCheck %s --check-prefix=ALWAYS
; RUN: opt < %s -indvars -replexitval=never -S | FileCheck %s --check-prefix=NEVER
; RUN: opt < %s -indvars -replexitval=cheap -scev-cheap-expansion-budget=1 -S | FileCheck %s --check-prefix=CHEAP

; rewriteLoopExitValues() must rewrite all or none of a PHI's values from a given block.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@a = common global i8 0, align 1

define internal fastcc void @d(i8* %c) unnamed_addr #0 {
entry:
  %cmp = icmp ule i8* %c, getelementptr inbounds (i8, i8* @a, i64 65535)
  %add.ptr = getelementptr inbounds i8, i8* %c, i64 -65535
  br label %while.cond

while.cond:
  br i1 icmp ne (i8 0, i8 0), label %cont, label %while.end

cont:
  %a.mux = select i1 %cmp, i8* @a, i8* %add.ptr
  switch i64 0, label %while.cond [
    i64 -1, label %handler.pointer_overflow.i
    i64 0, label %handler.pointer_overflow.i
  ]

handler.pointer_overflow.i:
  %a.mux.lcssa4 = phi i8* [ %a.mux, %cont ], [ %a.mux, %cont ]
; ALWAYS: [ %scevgep, %cont ], [ %scevgep, %cont ]
; NEVER: [ %a.mux, %cont ], [ %a.mux, %cont ]
; In cheap mode, use either one as long as it's consistent.
; CHEAP: [ %[[VAL:.*]], %cont ], [ %[[VAL]], %cont ]
  %x5 = ptrtoint i8* %a.mux.lcssa4 to i64
  br label %while.end

while.end:
  ret void
}
