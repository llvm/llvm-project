; RUN: opt -passes="module(print<block-freq>),function(loop(loop-idiom)),module(print<block-freq>)" -mtriple=x86_64 -mcpu=core-avx2 %s -disable-output 2>&1 | FileCheck --check-prefix=PROFILE %s

declare void @escape_inner(i8, i8, i8, i1, i8)
declare void @escape_outer(i8, i8, i8, i1, i8)

declare i8 @gen.i8()

; Most basic pattern; Note that iff the shift amount is offset, said offsetting
; must not cause an overflow, but `add nsw` is fine.
define i8 @p0(i8 %val, i8 %start, i8 %extraoffset) mustprogress {
entry:
  br label %loop

loop:
  %iv = phi i8 [ %start, %entry ], [ %iv.next, %loop ]
  %nbits = add nsw i8 %iv, %extraoffset
  %val.shifted = ashr i8 %val, %nbits
  %val.shifted.iszero = icmp eq i8 %val.shifted, 0
  %iv.next = add i8 %iv, 1

  call void @escape_inner(i8 %iv, i8 %nbits, i8 %val.shifted, i1 %val.shifted.iszero, i8 %iv.next)

  br i1 %val.shifted.iszero, label %end, label %loop, !prof !{!"branch_weights", i32 1, i32 1000 }

end:
  %iv.res = phi i8 [ %iv, %loop ]
  %nbits.res = phi i8 [ %nbits, %loop ]
  %val.shifted.res = phi i8 [ %val.shifted, %loop ]
  %val.shifted.iszero.res = phi i1 [ %val.shifted.iszero, %loop ]
  %iv.next.res = phi i8 [ %iv.next, %loop ]

  call void @escape_outer(i8 %iv.res, i8 %nbits.res, i8 %val.shifted.res, i1 %val.shifted.iszero.res, i8 %iv.next.res)

  ret i8 %iv.res
}

define i32 @p1(i32 %x, i32 %bit) {
entry:
  %bitmask = shl i32 1, %bit
  br label %loop

loop:
  %x.curr = phi i32 [ %x, %entry ], [ %x.next, %loop ]
  %x.curr.bitmasked = and i32 %x.curr, %bitmask
  %x.curr.isbitunset = icmp eq i32 %x.curr.bitmasked, 0
  %x.next = shl i32 %x.curr, 1
  br i1 %x.curr.isbitunset, label %loop, label %end, !prof !{!"branch_weights", i32 500, i32 1 }

end:
  ret i32 %x.curr
}

;
; PROFILE: Printing analysis results of BFI for function 'p0':
; PROFILE: block-frequency-info: p0
; PROFILE: - entry: float = 1.0,
; PROFILE:  - loop: float = 1001.0,
; PROFILE: - end: float = 1.0,
; PROFILE: block-frequency-info: p1
; PROFILE: - entry: float = 1.0, 
; PROFILE: - loop: float = 501.0,
; PROFILE: - end: float = 1.0,
; PROFILE: block-frequency-info: p0
; PROFILE: - entry: float = 1.0,
; PROFILE:  - loop: float = 1001.0,
; PROFILE: - end: float = 1.0,
; PROFILE: block-frequency-info: p1
; PROFILE: - entry: float = 1.0, 
; PROFILE: - loop: float = 501.0,
; PROFILE: - end: float = 1.0,
