; RUN: llc < %s -mtriple=i386-apple-macosx   | FileCheck %s --check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-apple-macosx | FileCheck %s --check-prefix=X64

define i32 @t1(i32 %t, i32 %val) nounwind {
; X32: t1:
; X32-NOT: andl
; X32: shll

; X64: t1:
; X64-NOT: andl
; X64: shll
       %shamt = and i32 %t, 31
       %res = shl i32 %val, %shamt
       ret i32 %res
}

define i32 @t2(i32 %t, i32 %val) nounwind {
; X32: t2:
; X32-NOT: andl
; X32: shll

; X64: t2:
; X64-NOT: andl
; X64: shll
       %shamt = and i32 %t, 63
       %res = shl i32 %val, %shamt
       ret i32 %res
}

@X = internal global i16 0

define void @t3(i16 %t) nounwind {
; X32: t3:
; X32-NOT: andl
; X32: sarw

; X64: t3:
; X64-NOT: andl
; X64: sarw
       %shamt = and i16 %t, 31
       %tmp = load i16* @X
       %tmp1 = ashr i16 %tmp, %shamt
       store i16 %tmp1, i16* @X
       ret void
}

define i64 @t4(i64 %t, i64 %val) nounwind {
; X64: t4:
; X64-NOT: and
; X64: shrq
       %shamt = and i64 %t, 63
       %res = lshr i64 %val, %shamt
       ret i64 %res
}

define i64 @t5(i64 %t, i64 %val) nounwind {
; X64: t5:
; X64-NOT: and
; X64: shrq
       %shamt = and i64 %t, 191
       %res = lshr i64 %val, %shamt
       ret i64 %res
}


; rdar://11866926
define i64 @t6(i64 %key, i64* nocapture %val) nounwind {
entry:
; X64: t6:
; X64-NOT: movabsq
; X64: decq
; X64: andq
  %shr = lshr i64 %key, 3
  %0 = load i64* %val, align 8
  %sub = add i64 %0, 2305843009213693951
  %and = and i64 %sub, %shr
  ret i64 %and
}
