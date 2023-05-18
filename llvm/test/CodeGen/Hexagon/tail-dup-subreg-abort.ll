; RUN: llc -march=hexagon -O2 -disable-cgp < %s
; REQUIRES: asserts
;
; Tail duplication can ignore subregister information on PHI nodes, and as
; a result, generate COPY instructions between registers of different classes.
; This could lead to HexagonInstrInfo::copyPhysReg aborting on an unhandled
; src/dst combination.
;
define i32 @foo(i32 %x, i64 %y) nounwind {
entry:
  %a = icmp slt i32 %x, 0
  %lo = trunc i64 %y to i32
  br i1 %a, label %next, label %tail
tail:
  br label %join
next:
  %c = icmp eq i32 %x, 0
  br i1 %c, label %b1, label %tail
b1:
  %t1 = lshr i64 %y, 32
  %hi = trunc i64 %t1 to i32
  br label %join
join:
  %val = phi i32 [ %hi, %b1 ], [ %lo, %tail ]
  ret i32 %val
}


