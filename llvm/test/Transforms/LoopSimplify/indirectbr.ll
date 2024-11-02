; RUN: opt < %s -passes=loop-simplify,lcssa -verify-loop-info -verify-dom-info -S \
; RUN:   | grep -F "indirectbr i8* %x, [label %L0, label %L1]" \
; RUN:   | count 6

; LoopSimplify should not try to transform loops when indirectbr is involved.

define void @entry(i8* %x) {
entry:
  indirectbr i8* %x, [ label %L0, label %L1 ]

L0:
  br label %L0

L1:
  ret void
}

define void @backedge(i8* %x) {
entry:
  br label %L0

L0:
  br label %L1

L1:
  indirectbr i8* %x, [ label %L0, label %L1 ]
}

define i64 @exit(i8* %x) {
entry:
  br label %L2

L2:
  %z = bitcast i64 0 to i64
  indirectbr i8* %x, [ label %L0, label %L1 ]

L0:
  br label %L2

L1:
  ret i64 %z
}

define i64 @criticalexit(i8* %x, i1 %a) {
entry:
  br i1 %a, label %L1, label %L2

L2:
  %z = bitcast i64 0 to i64
  indirectbr i8* %x, [ label %L0, label %L1 ]

L0:
  br label %L2

L1:
  %y = phi i64 [ %z, %L2 ], [ 1, %entry ]
  ret i64 %y
}

define i64 @exit_backedge(i8* %x) {
entry:
  br label %L0

L0:
  %z = bitcast i64 0 to i64
  indirectbr i8* %x, [ label %L0, label %L1 ]

L1:
  ret i64 %z
}

define i64 @criticalexit_backedge(i8* %x, i1 %a) {
entry:
  br i1 %a, label %L0, label %L1

L0:
  %z = bitcast i64 0 to i64
  indirectbr i8* %x, [ label %L0, label %L1 ]

L1:
  %y = phi i64 [ %z, %L0 ], [ 1, %entry ]
  ret i64 %y
}

define void @pr5502() nounwind {
entry:
  br label %while.cond

while.cond:
  br i1 undef, label %while.body, label %while.end

while.body:
  indirectbr i8* undef, [label %end_opcode, label %end_opcode]

end_opcode:
  br i1 false, label %end_opcode, label %while.cond

while.end:
  ret void
}
