; RUN: llc < %s -mtriple=x86_64-pc-linux -mattr=+bmi,+lzcnt | FileCheck %s

; LZCNT and TZCNT will always produce the operand size when the input operand
; is zero. This test is to verify that we efficiently select LZCNT/TZCNT
; based on the fact that the 'icmp+select' sequence is always redundant
; in every function defined below.


define i16 @test1_ctlz(i16 %v) {
  %cnt = tail call i16 @llvm.ctlz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 %v, 0
  %cond = select i1 %tobool, i16 16, i16 %cnt
  ret i16 %cond
}
; CHECK-LABEL: test1_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i32 @test2_ctlz(i32 %v) {
  %cnt = tail call i32 @llvm.ctlz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 %v, 0
  %cond = select i1 %tobool, i32 32, i32 %cnt
  ret i32 %cond
}
; CHECK-LABEL: test2_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i64 @test3_ctlz(i64 %v) {
  %cnt = tail call i64 @llvm.ctlz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 %v, 0
  %cond = select i1 %tobool, i64 64, i64 %cnt
  ret i64 %cond
}
; CHECK-LABEL: test3_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i16 @test4_ctlz(i16 %v) {
  %cnt = tail call i16 @llvm.ctlz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 0, %v
  %cond = select i1 %tobool, i16 16, i16 %cnt
  ret i16 %cond
}
; CHECK-LABEL: test4_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i32 @test5_ctlz(i32 %v) {
  %cnt = tail call i32 @llvm.ctlz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 0, %v
  %cond = select i1 %tobool, i32 32, i32 %cnt
  ret i32 %cond
}
; CHECK-LABEL: test5_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i64 @test6_ctlz(i64 %v) {
  %cnt = tail call i64 @llvm.ctlz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 0, %v
  %cond = select i1 %tobool, i64 64, i64 %cnt
  ret i64 %cond
}
; CHECK-LABEL: test6_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i16 @test7_ctlz(i16 %v) {
  %cnt = tail call i16 @llvm.ctlz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 0, %v
  %cond = select i1 %tobool, i16 %cnt, i16 16
  ret i16 %cond
}
; CHECK-LABEL: test7_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i32 @test8_ctlz(i32 %v) {
  %cnt = tail call i32 @llvm.ctlz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 0, %v
  %cond = select i1 %tobool, i32 %cnt, i32 32
  ret i32 %cond
}
; CHECK-LABEL: test8_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i64 @test9_ctlz(i64 %v) {
  %cnt = tail call i64 @llvm.ctlz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 0, %v
  %cond = select i1 %tobool, i64 %cnt, i64 64
  ret i64 %cond
}
; CHECK-LABEL: test9_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i16 @test10_ctlz(i16* %ptr) {
  %v = load i16, i16* %ptr
  %cnt = tail call i16 @llvm.ctlz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 %v, 0
  %cond = select i1 %tobool, i16 16, i16 %cnt
  ret i16 %cond
}
; CHECK-LABEL: test10_ctlz
; CHECK-NOT: movw
; CHECK: lzcnt
; CHECK-NEXT: ret


define i32 @test11_ctlz(i32* %ptr) {
  %v = load i32, i32* %ptr
  %cnt = tail call i32 @llvm.ctlz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 %v, 0
  %cond = select i1 %tobool, i32 32, i32 %cnt
  ret i32 %cond
}
; CHECK-LABEL: test11_ctlz
; CHECK-NOT: movd
; CHECK: lzcnt
; CHECK-NEXT: ret


define i64 @test12_ctlz(i64* %ptr) {
  %v = load i64, i64* %ptr
  %cnt = tail call i64 @llvm.ctlz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 %v, 0
  %cond = select i1 %tobool, i64 64, i64 %cnt
  ret i64 %cond
}
; CHECK-LABEL: test12_ctlz
; CHECK-NOT: movq
; CHECK: lzcnt
; CHECK-NEXT: ret


define i16 @test13_ctlz(i16* %ptr) {
  %v = load i16, i16* %ptr
  %cnt = tail call i16 @llvm.ctlz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 0, %v
  %cond = select i1 %tobool, i16 16, i16 %cnt
  ret i16 %cond
}
; CHECK-LABEL: test13_ctlz
; CHECK-NOT: movw
; CHECK: lzcnt
; CHECK-NEXT: ret


define i32 @test14_ctlz(i32* %ptr) {
  %v = load i32, i32* %ptr
  %cnt = tail call i32 @llvm.ctlz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 0, %v
  %cond = select i1 %tobool, i32 32, i32 %cnt
  ret i32 %cond
}
; CHECK-LABEL: test14_ctlz
; CHECK-NOT: movd
; CHECK: lzcnt
; CHECK-NEXT: ret


define i64 @test15_ctlz(i64* %ptr) {
  %v = load i64, i64* %ptr
  %cnt = tail call i64 @llvm.ctlz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 0, %v
  %cond = select i1 %tobool, i64 64, i64 %cnt
  ret i64 %cond
}
; CHECK-LABEL: test15_ctlz
; CHECK-NOT: movq
; CHECK: lzcnt
; CHECK-NEXT: ret


define i16 @test16_ctlz(i16* %ptr) {
  %v = load i16, i16* %ptr
  %cnt = tail call i16 @llvm.ctlz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 0, %v
  %cond = select i1 %tobool, i16 %cnt, i16 16
  ret i16 %cond
}
; CHECK-LABEL: test16_ctlz
; CHECK-NOT: movw
; CHECK: lzcnt
; CHECK-NEXT: ret


define i32 @test17_ctlz(i32* %ptr) {
  %v = load i32, i32* %ptr
  %cnt = tail call i32 @llvm.ctlz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 0, %v
  %cond = select i1 %tobool, i32 %cnt, i32 32
  ret i32 %cond
}
; CHECK-LABEL: test17_ctlz
; CHECK-NOT: movd
; CHECK: lzcnt
; CHECK-NEXT: ret


define i64 @test18_ctlz(i64* %ptr) {
  %v = load i64, i64* %ptr
  %cnt = tail call i64 @llvm.ctlz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 0, %v
  %cond = select i1 %tobool, i64 %cnt, i64 64
  ret i64 %cond
}
; CHECK-LABEL: test18_ctlz
; CHECK-NOT: movq
; CHECK: lzcnt
; CHECK-NEXT: ret


define i16 @test1_cttz(i16 %v) {
  %cnt = tail call i16 @llvm.cttz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 %v, 0
  %cond = select i1 %tobool, i16 16, i16 %cnt
  ret i16 %cond
}
; CHECK-LABEL: test1_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i32 @test2_cttz(i32 %v) {
  %cnt = tail call i32 @llvm.cttz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 %v, 0
  %cond = select i1 %tobool, i32 32, i32 %cnt
  ret i32 %cond
}
; CHECK-LABEL: test2_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i64 @test3_cttz(i64 %v) {
  %cnt = tail call i64 @llvm.cttz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 %v, 0
  %cond = select i1 %tobool, i64 64, i64 %cnt
  ret i64 %cond
}
; CHECK-LABEL: test3_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i16 @test4_cttz(i16 %v) {
  %cnt = tail call i16 @llvm.cttz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 0, %v
  %cond = select i1 %tobool, i16 16, i16 %cnt
  ret i16 %cond
}
; CHECK-LABEL: test4_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i32 @test5_cttz(i32 %v) {
  %cnt = tail call i32 @llvm.cttz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 0, %v
  %cond = select i1 %tobool, i32 32, i32 %cnt
  ret i32 %cond
}
; CHECK-LABEL: test5_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i64 @test6_cttz(i64 %v) {
  %cnt = tail call i64 @llvm.cttz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 0, %v
  %cond = select i1 %tobool, i64 64, i64 %cnt
  ret i64 %cond
}
; CHECK-LABEL: test6_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i16 @test7_cttz(i16 %v) {
  %cnt = tail call i16 @llvm.cttz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 0, %v
  %cond = select i1 %tobool, i16 %cnt, i16 16
  ret i16 %cond
}
; CHECK-LABEL: test7_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i32 @test8_cttz(i32 %v) {
  %cnt = tail call i32 @llvm.cttz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 0, %v
  %cond = select i1 %tobool, i32 %cnt, i32 32
  ret i32 %cond
}
; CHECK-LABEL: test8_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i64 @test9_cttz(i64 %v) {
  %cnt = tail call i64 @llvm.cttz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 0, %v
  %cond = select i1 %tobool, i64 %cnt, i64 64
  ret i64 %cond
}
; CHECK-LABEL: test9_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i16 @test10_cttz(i16* %ptr) {
  %v = load i16, i16* %ptr
  %cnt = tail call i16 @llvm.cttz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 %v, 0
  %cond = select i1 %tobool, i16 16, i16 %cnt
  ret i16 %cond
}
; CHECK-LABEL: test10_cttz
; CHECK-NOT: movw
; CHECK: tzcnt
; CHECK-NEXT: ret


define i32 @test11_cttz(i32* %ptr) {
  %v = load i32, i32* %ptr
  %cnt = tail call i32 @llvm.cttz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 %v, 0
  %cond = select i1 %tobool, i32 32, i32 %cnt
  ret i32 %cond
}
; CHECK-LABEL: test11_cttz
; CHECK-NOT: movd
; CHECK: tzcnt
; CHECK-NEXT: ret


define i64 @test12_cttz(i64* %ptr) {
  %v = load i64, i64* %ptr
  %cnt = tail call i64 @llvm.cttz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 %v, 0
  %cond = select i1 %tobool, i64 64, i64 %cnt
  ret i64 %cond
}
; CHECK-LABEL: test12_cttz
; CHECK-NOT: movq
; CHECK: tzcnt
; CHECK-NEXT: ret


define i16 @test13_cttz(i16* %ptr) {
  %v = load i16, i16* %ptr
  %cnt = tail call i16 @llvm.cttz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 0, %v
  %cond = select i1 %tobool, i16 16, i16 %cnt
  ret i16 %cond
}
; CHECK-LABEL: test13_cttz
; CHECK-NOT: movw
; CHECK: tzcnt
; CHECK-NEXT: ret


define i32 @test14_cttz(i32* %ptr) {
  %v = load i32, i32* %ptr
  %cnt = tail call i32 @llvm.cttz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 0, %v
  %cond = select i1 %tobool, i32 32, i32 %cnt
  ret i32 %cond
}
; CHECK-LABEL: test14_cttz
; CHECK-NOT: movd
; CHECK: tzcnt
; CHECK-NEXT: ret


define i64 @test15_cttz(i64* %ptr) {
  %v = load i64, i64* %ptr
  %cnt = tail call i64 @llvm.cttz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 0, %v
  %cond = select i1 %tobool, i64 64, i64 %cnt
  ret i64 %cond
}
; CHECK-LABEL: test15_cttz
; CHECK-NOT: movq
; CHECK: tzcnt
; CHECK-NEXT: ret


define i16 @test16_cttz(i16* %ptr) {
  %v = load i16, i16* %ptr
  %cnt = tail call i16 @llvm.cttz.i16(i16 %v, i1 true)
  %tobool = icmp eq i16 0, %v
  %cond = select i1 %tobool, i16 %cnt, i16 16
  ret i16 %cond
}
; CHECK-LABEL: test16_cttz
; CHECK-NOT: movw
; CHECK: tzcnt
; CHECK-NEXT: ret


define i32 @test17_cttz(i32* %ptr) {
  %v = load i32, i32* %ptr
  %cnt = tail call i32 @llvm.cttz.i32(i32 %v, i1 true)
  %tobool = icmp eq i32 0, %v
  %cond = select i1 %tobool, i32 %cnt, i32 32
  ret i32 %cond
}
; CHECK-LABEL: test17_cttz
; CHECK-NOT: movd
; CHECK: tzcnt
; CHECK-NEXT: ret


define i64 @test18_cttz(i64* %ptr) {
  %v = load i64, i64* %ptr
  %cnt = tail call i64 @llvm.cttz.i64(i64 %v, i1 true)
  %tobool = icmp eq i64 0, %v
  %cond = select i1 %tobool, i64 %cnt, i64 64
  ret i64 %cond
}
; CHECK-LABEL: test18_cttz
; CHECK-NOT: movq
; CHECK: tzcnt
; CHECK-NEXT: ret

define i16 @test1b_ctlz(i16 %v) {
  %cnt = tail call i16 @llvm.ctlz.i16(i16 %v, i1 true)
  %tobool = icmp ne i16 %v, 0
  %cond = select i1 %tobool, i16 16, i16 %cnt
  ret i16 %cond
}
; CHECK-LABEL: test1b_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i32 @test2b_ctlz(i32 %v) {
  %cnt = tail call i32 @llvm.ctlz.i32(i32 %v, i1 true)
  %tobool = icmp ne i32 %v, 0
  %cond = select i1 %tobool, i32 32, i32 %cnt
  ret i32 %cond
}
; CHECK-LABEL: test2b_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i64 @test3b_ctlz(i64 %v) {
  %cnt = tail call i64 @llvm.ctlz.i64(i64 %v, i1 true)
  %tobool = icmp ne i64 %v, 0
  %cond = select i1 %tobool, i64 64, i64 %cnt
  ret i64 %cond
}
; CHECK-LABEL: test3b_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i16 @test4b_ctlz(i16 %v) {
  %cnt = tail call i16 @llvm.ctlz.i16(i16 %v, i1 true)
  %tobool = icmp ne i16 %v, 0
  %cond = select i1 %tobool, i16 %cnt, i16 16
  ret i16 %cond
}
; CHECK-LABEL: test4b_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i32 @test5b_ctlz(i32 %v) {
  %cnt = tail call i32 @llvm.ctlz.i32(i32 %v, i1 true)
  %tobool = icmp ne i32 %v, 0
  %cond = select i1 %tobool, i32 %cnt, i32 32
  ret i32 %cond
}
; CHECK-LABEL: test5b_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i64 @test6b_ctlz(i64 %v) {
  %cnt = tail call i64 @llvm.ctlz.i64(i64 %v, i1 true)
  %tobool = icmp ne i64 %v, 0
  %cond = select i1 %tobool, i64 %cnt, i64 64
  ret i64 %cond
}
; CHECK-LABEL: test6b_ctlz
; CHECK: lzcnt
; CHECK-NEXT: ret


define i16 @test1b_cttz(i16 %v) {
  %cnt = tail call i16 @llvm.cttz.i16(i16 %v, i1 true)
  %tobool = icmp ne i16 %v, 0
  %cond = select i1 %tobool, i16 16, i16 %cnt
  ret i16 %cond
}
; CHECK-LABEL: test1b_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i32 @test2b_cttz(i32 %v) {
  %cnt = tail call i32 @llvm.cttz.i32(i32 %v, i1 true)
  %tobool = icmp ne i32 %v, 0
  %cond = select i1 %tobool, i32 32, i32 %cnt
  ret i32 %cond
}
; CHECK-LABEL: test2b_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i64 @test3b_cttz(i64 %v) {
  %cnt = tail call i64 @llvm.cttz.i64(i64 %v, i1 true)
  %tobool = icmp ne i64 %v, 0
  %cond = select i1 %tobool, i64 64, i64 %cnt
  ret i64 %cond
}
; CHECK-LABEL: test3b_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i16 @test4b_cttz(i16 %v) {
  %cnt = tail call i16 @llvm.cttz.i16(i16 %v, i1 true)
  %tobool = icmp ne i16 %v, 0
  %cond = select i1 %tobool, i16 %cnt, i16 16
  ret i16 %cond
}
; CHECK-LABEL: test4b_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i32 @test5b_cttz(i32 %v) {
  %cnt = tail call i32 @llvm.cttz.i32(i32 %v, i1 true)
  %tobool = icmp ne i32 %v, 0
  %cond = select i1 %tobool, i32 %cnt, i32 32
  ret i32 %cond
}
; CHECK-LABEL: test5b_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


define i64 @test6b_cttz(i64 %v) {
  %cnt = tail call i64 @llvm.cttz.i64(i64 %v, i1 true)
  %tobool = icmp ne i64 %v, 0
  %cond = select i1 %tobool, i64 %cnt, i64 64
  ret i64 %cond
}
; CHECK-LABEL: test6b_cttz
; CHECK: tzcnt
; CHECK-NEXT: ret


declare i64 @llvm.cttz.i64(i64, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i16 @llvm.cttz.i16(i16, i1)
declare i64 @llvm.ctlz.i64(i64, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i16 @llvm.ctlz.i16(i16, i1)

