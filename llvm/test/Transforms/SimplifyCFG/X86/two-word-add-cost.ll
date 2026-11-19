; RUN: opt -mtriple=x86_64 -passes=simplifycfg -verify-each -S < %s | FileCheck %s

; A two-word addition should have the same speculation cost as add i128.
; With the default budget, both positive cases should fold to selects.
define { i64, i64, i64 } @add_carry(i64 %a.lo, i64 %a.hi, i64 %b.lo, i64 %b.hi, i64 %q, i1 %c) {
; CHECK-LABEL: define { i64, i64, i64 } @add_carry(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %q.dec = add i64 %q, -1
; CHECK-NEXT:    %s = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a.lo, i64 %b.lo)
; CHECK-NEXT:    %lo = extractvalue { i64, i1 } %s, 0
; CHECK-NEXT:    %carry = extractvalue { i64, i1 } %s, 1
; CHECK-NEXT:    %carry.ext = zext i1 %carry to i64
; CHECK-NEXT:    %hi.tmp = add i64 %a.hi, %b.hi
; CHECK-NEXT:    %hi = add i64 %hi.tmp, %carry.ext
; CHECK-NEXT:    %q.r = select i1 %c, i64 %q.dec, i64 %q
; CHECK-NEXT:    %lo.r = select i1 %c, i64 %lo, i64 %a.lo
; CHECK-NEXT:    %hi.r = select i1 %c, i64 %hi, i64 %a.hi
; CHECK-NEXT:    %r0 = insertvalue { i64, i64, i64 } poison, i64 %lo.r, 0
; CHECK-NEXT:    %r1 = insertvalue { i64, i64, i64 } %r0, i64 %hi.r, 1
; CHECK-NEXT:    %r2 = insertvalue { i64, i64, i64 } %r1, i64 %q.r, 2
; CHECK-NEXT:    ret { i64, i64, i64 } %r2
; CHECK-NEXT:  }
entry:
  br i1 %c, label %then, label %join

then:
  %q.dec = add i64 %q, -1
  %s = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a.lo, i64 %b.lo)
  %lo = extractvalue { i64, i1 } %s, 0
  %carry = extractvalue { i64, i1 } %s, 1
  %carry.ext = zext i1 %carry to i64
  %hi.tmp = add i64 %a.hi, %b.hi
  %hi = add i64 %hi.tmp, %carry.ext
  br label %join

join:
  %q.r = phi i64 [ %q.dec, %then ], [ %q, %entry ]
  %lo.r = phi i64 [ %lo, %then ], [ %a.lo, %entry ]
  %hi.r = phi i64 [ %hi, %then ], [ %a.hi, %entry ]
  %r0 = insertvalue { i64, i64, i64 } poison, i64 %lo.r, 0
  %r1 = insertvalue { i64, i64, i64 } %r0, i64 %hi.r, 1
  %r2 = insertvalue { i64, i64, i64 } %r1, i64 %q.r, 2
  ret { i64, i64, i64 } %r2
}

define { i128, i64 } @add_i128(i128 %a, i128 %b, i64 %q, i1 %c) {
; CHECK-LABEL: define { i128, i64 } @add_i128(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %q.dec = add i64 %q, -1
; CHECK-NEXT:    %s = add i128 %a, %b
; CHECK-NEXT:    %q.r = select i1 %c, i64 %q.dec, i64 %q
; CHECK-NEXT:    %s.r = select i1 %c, i128 %s, i128 %a
; CHECK-NEXT:    %r0 = insertvalue { i128, i64 } poison, i128 %s.r, 0
; CHECK-NEXT:    %r1 = insertvalue { i128, i64 } %r0, i64 %q.r, 1
; CHECK-NEXT:    ret { i128, i64 } %r1
; CHECK-NEXT:  }
entry:
  br i1 %c, label %then, label %join

then:
  %q.dec = add i64 %q, -1
  %s = add i128 %a, %b
  br label %join

join:
  %q.r = phi i64 [ %q.dec, %then ], [ %q, %entry ]
  %s.r = phi i128 [ %s, %then ], [ %a, %entry ]
  %r0 = insertvalue { i128, i64 } poison, i128 %s.r, 0
  %r1 = insertvalue { i128, i64 } %r0, i64 %q.r, 1
  ret { i128, i64 } %r1
}

; An extra use of the carry must prevent the prototype from discounting
; this group as one wide add. The conditional branch and PHIs should remain.
define { i64, i64, i64 } @add_carry_extra_carry_use(i64 %a.lo, i64 %a.hi, i64 %b.lo, i64 %b.hi, i64 %q, i1 %c) {
; CHECK-LABEL: define { i64, i64, i64 } @add_carry_extra_carry_use(
; CHECK:         br i1 %c, label %then, label %join
; CHECK:         %carry = extractvalue { i64, i1 } %s, 1
; CHECK:         %carry.ext = zext i1 %carry to i64
; CHECK:         %hi.extra = select i1 %carry, i64 %hi, i64 0
; CHECK:         %q.r = phi i64 [ %q.dec, %then ], [ %q, %entry ]
; CHECK-NEXT:    %lo.r = phi i64 [ %lo, %then ], [ %a.lo, %entry ]
; CHECK-NEXT:    %hi.r = phi i64 [ %hi.extra, %then ], [ %a.hi, %entry ]
; CHECK:         ret { i64, i64, i64 } %r2
; CHECK-NEXT:  }
entry:
  br i1 %c, label %then, label %join

then:
  %q.dec = add i64 %q, -1
  %s = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a.lo, i64 %b.lo)
  %lo = extractvalue { i64, i1 } %s, 0
  %carry = extractvalue { i64, i1 } %s, 1
  %carry.ext = zext i1 %carry to i64
  %hi.tmp = add i64 %a.hi, %b.hi
  %hi = add i64 %hi.tmp, %carry.ext
  %hi.extra = select i1 %carry, i64 %hi, i64 0
  br label %join

join:
  %q.r = phi i64 [ %q.dec, %then ], [ %q, %entry ]
  %lo.r = phi i64 [ %lo, %then ], [ %a.lo, %entry ]
  %hi.r = phi i64 [ %hi.extra, %then ], [ %a.hi, %entry ]
  %r0 = insertvalue { i64, i64, i64 } poison, i64 %lo.r, 0
  %r1 = insertvalue { i64, i64, i64 } %r0, i64 %hi.r, 1
  %r2 = insertvalue { i64, i64, i64 } %r1, i64 %q.r, 2
  ret { i64, i64, i64 } %r2
}

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
