; Verify that we do not crash on always-true conditions
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 -O0
;
; This test was compiled using clang -O0 from the following source code:
;
; int test(unsigned long x)
; {
;   return x >= 0 && x <= 15;
; }

define signext i32 @test(i64 %x) {
entry:
  %x.addr = alloca i64, align 8
  store i64 %x, ptr %x.addr, align 8
  %0 = load i64, ptr %x.addr, align 8
  %cmp = icmp uge i64 %0, 0
  br i1 %cmp, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %entry
  %1 = load i64, ptr %x.addr, align 8
  %cmp1 = icmp ule i64 %1, 15
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %2 = phi i1 [ false, %entry ], [ %cmp1, %land.rhs ]
  %land.ext = zext i1 %2 to i32
  ret i32 %land.ext
}

