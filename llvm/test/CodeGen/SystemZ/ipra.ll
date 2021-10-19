; Test that the set of used phys regs used by interprocedural register
; allocation is correct for a test case where the called function (@fn2)
; itself has a call (to @fn1). @fn1 defines %r0l, while @fn2 defines
; %r0d. The RegUsageInfo for @fn2 must include %r0h.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -enable-ipra -print-regusage 2>&1 < %s \
; RUN:   | FileCheck %s
;
; CHECK: fn2 Clobbered Registers: {{.*}} $r0h

@h = external dso_local global [0 x i32], align 4
@n = external dso_local global i32*, align 8

define void @fn1() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %tmp = getelementptr inbounds [0 x i32], [0 x i32]* @h, i64 0, i64 undef
  %tmp2 = load i32, i32* %tmp
  store i32 %tmp2, i32* undef
  br label %bb1
}

define void @fn2() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  br i1 undef, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  store i32* null, i32** @n
  unreachable

bb3:                                              ; preds = %bb1
  call void @fn1()
  unreachable
}

define void @main() {
bb:
  call void @fn2()
  ret void
}
