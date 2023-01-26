; RUN: llc -O2 -ppc-reduce-cr-logicals -print-machine-bfi -o - %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-grtev4-linux-gnu"

; First block frequency info
;CHECK:      block-frequency-info: loop_test
;CHECK-NEXT: - BB0[entry]: float = 1.0, int = 12
;CHECK-NEXT: - BB1[for.check]: float = 2.6667, int = 34
;CHECK-NEXT: - BB2[test1]: float = 1.6667, int = 21
;CHECK-NEXT: - BB3[optional1]: float = 0.625, int = 8

;CHECK:      block-frequency-info: loop_test
;CHECK:      block-frequency-info: loop_test
;CHECK:      block-frequency-info: loop_test

; Last block frequency info
;CHECK:      block-frequency-info: loop_test
;CHECK-NEXT: - BB0[entry]: float = 1.0, int = 12
;CHECK-NEXT: - BB1[for.check]: float = 2.6667, int = 34
;CHECK-NEXT: - BB2[for.check]: float = 2.1667, int = 27
;CHECK-NEXT: - BB3[test1]: float = 1.6667, int = 21
;CHECK-NEXT: - BB4[optional1]: float = 0.625, int = 8


define void @loop_test(ptr %tags, i32 %count) {
entry:
  br label %for.check
for.check:
  %count.loop = phi i32 [%count, %entry], [%count.sub, %for.latch]
  %done.count = icmp ugt i32 %count.loop, 0
  %tag_ptr = getelementptr inbounds i32, ptr %tags, i32 %count
  %tag = load i32, ptr %tag_ptr
  %done.tag = icmp eq i32 %tag, 0
  %done = and i1 %done.count, %done.tag
  br i1 %done, label %test1, label %exit, !prof !1
test1:
  %tagbit1 = and i32 %tag, 1
  %tagbit1eq0 = icmp eq i32 %tagbit1, 0
  br i1 %tagbit1eq0, label %test2, label %optional1, !prof !1
optional1:
  call void @a()
  call void @a()
  call void @a()
  call void @a()
  br label %test2
test2:
  %tagbit2 = and i32 %tag, 2
  %tagbit2eq0 = icmp eq i32 %tagbit2, 0
  br i1 %tagbit2eq0, label %test3, label %optional2, !prof !1
optional2:
  call void @b()
  call void @b()
  call void @b()
  call void @b()
  br label %test3
test3:
  %tagbit3 = and i32 %tag, 4
  %tagbit3eq0 = icmp eq i32 %tagbit3, 0
  br i1 %tagbit3eq0, label %test4, label %optional3, !prof !1
optional3:
  call void @c()
  call void @c()
  call void @c()
  call void @c()
  br label %test4
test4:
  %tagbit4 = and i32 %tag, 8
  %tagbit4eq0 = icmp eq i32 %tagbit4, 0
  br i1 %tagbit4eq0, label %for.latch, label %optional4, !prof !1
optional4:
  call void @d()
  call void @d()
  call void @d()
  call void @d()
  br label %for.latch
for.latch:
  %count.sub = sub i32 %count.loop, 1
  br label %for.check
exit:
  ret void
}

declare void @a()
declare void @b()
declare void @c()
declare void @d()

!1 = !{!"branch_weights", i32 5, i32 3}
