; RUN: llc -O0 -mtriple=riscv32 -mattr=+m -mattr=+xcvsimd -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

declare i32 @llvm.riscv.cv.simd.add.h(i32, i32, i32)

define i32 @test.cv.add.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.add.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.add.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.add.h(i32 %a, i32 %b, i32 0)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.add.b(i32, i32)

define i32 @test.cv.add.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.add.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.add.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.add.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.add.sc.h(i32, i32)

define i32 @test.cv.add.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.add.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.add.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.add.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.add.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.add.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.add.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.add.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.add.sc.b(i32, i32)

define i32 @test.cv.add.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.add.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.add.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.add.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.add.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.add.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.add.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.add.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sub.h(i32, i32, i32)

define i32 @test.cv.sub.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sub.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sub.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sub.h(i32 %a, i32 %b, i32 0)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sub.b(i32, i32)

define i32 @test.cv.sub.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sub.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sub.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sub.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sub.sc.h(i32, i32)

define i32 @test.cv.sub.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sub.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sub.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sub.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.sub.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.sub.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sub.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sub.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sub.sc.b(i32, i32)

define i32 @test.cv.sub.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sub.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sub.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sub.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.sub.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.sub.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sub.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sub.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.avg.h(i32, i32)

define i32 @test.cv.avg.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.avg.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avg.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avg.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.avg.b(i32, i32)

define i32 @test.cv.avg.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.avg.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avg.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avg.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.avg.sc.h(i32, i32)

define i32 @test.cv.avg.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.avg.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avg.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avg.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.avg.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.avg.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avg.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avg.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.avg.sc.b(i32, i32)

define i32 @test.cv.avg.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.avg.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avg.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avg.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.avg.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.avg.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avg.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avg.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.avgu.h(i32, i32)

define i32 @test.cv.avgu.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.avgu.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avgu.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avgu.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.avgu.b(i32, i32)

define i32 @test.cv.avgu.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.avgu.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avgu.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avgu.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.avgu.sc.h(i32, i32)

define i32 @test.cv.avgu.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.avgu.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avgu.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avgu.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.avgu.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.avgu.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avgu.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avgu.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.avgu.sc.b(i32, i32)

define i32 @test.cv.avgu.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.avgu.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avgu.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avgu.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.avgu.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.avgu.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.avgu.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.avgu.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.min.h(i32, i32)

define i32 @test.cv.min.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.min.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.min.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.min.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.min.b(i32, i32)

define i32 @test.cv.min.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.min.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.min.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.min.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.min.sc.h(i32, i32)

define i32 @test.cv.min.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.min.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.min.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.min.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.min.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.min.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.min.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.min.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.min.sc.b(i32, i32)

define i32 @test.cv.min.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.min.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.min.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.min.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.min.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.min.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.min.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.min.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.minu.h(i32, i32)

define i32 @test.cv.minu.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.minu.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.minu.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.minu.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.minu.b(i32, i32)

define i32 @test.cv.minu.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.minu.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.minu.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.minu.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.minu.sc.h(i32, i32)

define i32 @test.cv.minu.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.minu.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.minu.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.minu.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.minu.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.minu.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.minu.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.minu.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.minu.sc.b(i32, i32)

define i32 @test.cv.minu.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.minu.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.minu.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.minu.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.minu.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.minu.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.minu.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.minu.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.max.h(i32, i32)

define i32 @test.cv.max.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.max.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.max.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.max.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.max.b(i32, i32)

define i32 @test.cv.max.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.max.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.max.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.max.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.max.sc.h(i32, i32)

define i32 @test.cv.max.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.max.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.max.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.max.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.max.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.max.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.max.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.max.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.max.sc.b(i32, i32)

define i32 @test.cv.max.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.max.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.max.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.max.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.max.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.max.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.max.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.max.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.maxu.h(i32, i32)

define i32 @test.cv.maxu.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.maxu.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.maxu.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.maxu.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.maxu.b(i32, i32)

define i32 @test.cv.maxu.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.maxu.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.maxu.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.maxu.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.maxu.sc.h(i32, i32)

define i32 @test.cv.maxu.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.maxu.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.maxu.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.maxu.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.maxu.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.maxu.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.maxu.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.maxu.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.maxu.sc.b(i32, i32)

define i32 @test.cv.maxu.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.maxu.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.maxu.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.maxu.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.maxu.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.maxu.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.maxu.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.maxu.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.srl.h(i32, i32)

define i32 @test.cv.srl.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.srl.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.srl.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.srl.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.srl.b(i32, i32)

define i32 @test.cv.srl.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.srl.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.srl.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.srl.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.srl.sc.h(i32, i32)

define i32 @test.cv.srl.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.srl.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.srl.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.srl.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.srl.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.srl.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.srl.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.srl.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.srl.sc.b(i32, i32)

define i32 @test.cv.srl.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.srl.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.srl.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.srl.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.srl.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.srl.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.srl.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.srl.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sra.h(i32, i32)

define i32 @test.cv.sra.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sra.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sra.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sra.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sra.b(i32, i32)

define i32 @test.cv.sra.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sra.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sra.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sra.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sra.sc.h(i32, i32)

define i32 @test.cv.sra.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sra.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sra.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sra.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.sra.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.sra.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sra.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sra.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sra.sc.b(i32, i32)

define i32 @test.cv.sra.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sra.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sra.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sra.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.sra.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.sra.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sra.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sra.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sll.h(i32, i32)

define i32 @test.cv.sll.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sll.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sll.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sll.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sll.b(i32, i32)

define i32 @test.cv.sll.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sll.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sll.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sll.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sll.sc.h(i32, i32)

define i32 @test.cv.sll.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sll.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sll.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sll.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.sll.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.sll.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sll.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sll.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sll.sc.b(i32, i32)

define i32 @test.cv.sll.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sll.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sll.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sll.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.sll.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.sll.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sll.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sll.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.or.h(i32, i32)

define i32 @test.cv.or.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.or.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.or.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.or.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.or.b(i32, i32)

define i32 @test.cv.or.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.or.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.or.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.or.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.or.sc.h(i32, i32)

define i32 @test.cv.or.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.or.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.or.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.or.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.or.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.or.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.or.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.or.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.or.sc.b(i32, i32)

define i32 @test.cv.or.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.or.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.or.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.or.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.or.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.or.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.or.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.or.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.xor.h(i32, i32)

define i32 @test.cv.xor.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.xor.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.xor.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.xor.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.xor.b(i32, i32)

define i32 @test.cv.xor.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.xor.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.xor.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.xor.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.xor.sc.h(i32, i32)

define i32 @test.cv.xor.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.xor.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.xor.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.xor.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.xor.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.xor.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.xor.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.xor.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.xor.sc.b(i32, i32)

define i32 @test.cv.xor.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.xor.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.xor.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.xor.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.xor.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.xor.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.xor.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.xor.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.and.h(i32, i32)

define i32 @test.cv.and.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.and.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.and.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.and.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.and.b(i32, i32)

define i32 @test.cv.and.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.and.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.and.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.and.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.and.sc.h(i32, i32)

define i32 @test.cv.and.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.and.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.and.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.and.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.and.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.and.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.and.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.and.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.and.sc.b(i32, i32)

define i32 @test.cv.and.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.and.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.and.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.and.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.and.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.and.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.and.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.and.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.abs.h(i32)

define i32 @test.cv.abs.h(i32 %a) {
; CHECK-LABEL: test.cv.abs.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.abs.h a0, a0
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.abs.h(i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.abs.b(i32)

define i32 @test.cv.abs.b(i32 %a) {
; CHECK-LABEL: test.cv.abs.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.abs.b a0, a0
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.abs.b(i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotup.h(i32, i32)

define i32 @test.cv.dotup.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotup.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotup.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotup.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotup.b(i32, i32)

define i32 @test.cv.dotup.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotup.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotup.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotup.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotup.sc.h(i32, i32)

define i32 @test.cv.dotup.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotup.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotup.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotup.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.dotup.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.dotup.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotup.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotup.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotup.sc.b(i32, i32)

define i32 @test.cv.dotup.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotup.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotup.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotup.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.dotup.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.dotup.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotup.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotup.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotusp.h(i32, i32)

define i32 @test.cv.dotusp.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotusp.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotusp.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotusp.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotusp.b(i32, i32)

define i32 @test.cv.dotusp.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotusp.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotusp.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotusp.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotusp.sc.h(i32, i32)

define i32 @test.cv.dotusp.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotusp.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotusp.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotusp.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.dotusp.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.dotusp.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotusp.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotusp.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotusp.sc.b(i32, i32)

define i32 @test.cv.dotusp.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotusp.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotusp.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotusp.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.dotusp.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.dotusp.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotusp.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotusp.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotsp.h(i32, i32)

define i32 @test.cv.dotsp.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotsp.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotsp.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotsp.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotsp.b(i32, i32)

define i32 @test.cv.dotsp.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotsp.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotsp.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotsp.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotsp.sc.h(i32, i32)

define i32 @test.cv.dotsp.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotsp.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotsp.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotsp.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.dotsp.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.dotsp.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotsp.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotsp.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.dotsp.sc.b(i32, i32)

define i32 @test.cv.dotsp.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.dotsp.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotsp.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotsp.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.dotsp.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.dotsp.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.dotsp.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.dotsp.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotup.h(i32, i32, i32)

define i32 @test.cv.sdotup.h(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotup.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotup.h a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotup.h(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotup.b(i32, i32, i32)

define i32 @test.cv.sdotup.b(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotup.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotup.b a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotup.b(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotup.sc.h(i32, i32, i32)

define i32 @test.cv.sdotup.sc.h(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotup.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotup.sc.h a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotup.sc.h(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

define i32 @test.cv.sdotup.sci.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sdotup.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotup.sci.h a0, a1, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotup.sc.h(i32 %b, i32 5, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotup.sc.b(i32, i32, i32)

define i32 @test.cv.sdotup.sc.b(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotup.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotup.sc.b a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotup.sc.b(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

define i32 @test.cv.sdotup.sci.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sdotup.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotup.sci.b a0, a1, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotup.sc.b(i32 %b, i32 5, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotusp.h(i32, i32, i32)

define i32 @test.cv.sdotusp.h(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotusp.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotusp.h a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotusp.h(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotusp.b(i32, i32, i32)

define i32 @test.cv.sdotusp.b(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotusp.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotusp.b a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotusp.b(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotusp.sc.h(i32, i32, i32)

define i32 @test.cv.sdotusp.sc.h(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotusp.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotusp.sc.h a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotusp.sc.h(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

define i32 @test.cv.sdotusp.sci.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sdotusp.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotusp.sci.h a0, a1, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotusp.sc.h(i32 %b, i32 5, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotusp.sc.b(i32, i32, i32)

define i32 @test.cv.sdotusp.sc.b(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotusp.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotusp.sc.b a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotusp.sc.b(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

define i32 @test.cv.sdotusp.sci.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sdotusp.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotusp.sci.b a0, a1, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotusp.sc.b(i32 %b, i32 5, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotsp.h(i32, i32, i32)

define i32 @test.cv.sdotsp.h(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotsp.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotsp.h a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotsp.h(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotsp.b(i32, i32, i32)

define i32 @test.cv.sdotsp.b(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotsp.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotsp.b a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotsp.b(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotsp.sc.h(i32, i32, i32)

define i32 @test.cv.sdotsp.sc.h(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotsp.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotsp.sc.h a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotsp.sc.h(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

define i32 @test.cv.sdotsp.sci.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sdotsp.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotsp.sci.h a0, a1, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotsp.sc.h(i32 %b, i32 5, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.sdotsp.sc.b(i32, i32, i32)

define i32 @test.cv.sdotsp.sc.b(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.sdotsp.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotsp.sc.b a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotsp.sc.b(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

define i32 @test.cv.sdotsp.sci.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sdotsp.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sdotsp.sci.b a0, a1, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sdotsp.sc.b(i32 %b, i32 5, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.extract.h(i32, i32)

define i32 @test.cv.extract.h(i32 %a) {
; CHECK-LABEL: test.cv.extract.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.extract.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.extract.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.extract.b(i32, i32)

define i32 @test.cv.extract.b(i32 %a) {
; CHECK-LABEL: test.cv.extract.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.extract.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.extract.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.extractu.h(i32, i32)

define i32 @test.cv.extractu.h(i32 %a) {
; CHECK-LABEL: test.cv.extractu.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.extractu.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.extractu.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.extractu.b(i32, i32)

define i32 @test.cv.extractu.b(i32 %a) {
; CHECK-LABEL: test.cv.extractu.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.extractu.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.extractu.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.insert.h(i32, i32, i32)

define i32 @test.cv.insert.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.insert.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.insert.h a0, a1, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.insert.h(i32 %a, i32 %b, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.insert.b(i32, i32, i32)

define i32 @test.cv.insert.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.insert.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.insert.b a0, a1, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.insert.b(i32 %a, i32 %b, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.shuffle.h(i32, i32)

define i32 @test.cv.shuffle.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.shuffle.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.shuffle.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.shuffle.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.shuffle.b(i32, i32)

define i32 @test.cv.shuffle.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.shuffle.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.shuffle.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.shuffle.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.shuffle.sci.h(i32, i32)

define i32 @test.cv.shuffle.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.shuffle.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.shuffle.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.shuffle.sci.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.shuffle.sci.b(i32, i32)

define i32 @test.cv.shufflei0.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.shufflei0.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.shufflei0.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.shuffle.sci.b(i32 %a, i32 5)
  ret i32 %1
}

define i32 @test.cv.shufflei1.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.shufflei1.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.shufflei1.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.shuffle.sci.b(i32 %a, i32 69)
  ret i32 %1
}

define i32 @test.cv.shufflei2.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.shufflei2.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.shufflei2.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.shuffle.sci.b(i32 %a, i32 133)
  ret i32 %1
}

define i32 @test.cv.shufflei3.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.shufflei3.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.shufflei3.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.shuffle.sci.b(i32 %a, i32 197)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.shuffle2.h(i32, i32, i32)

define i32 @test.cv.shuffle2.h(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.shuffle2.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.shuffle2.h a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.shuffle2.h(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.shuffle2.b(i32, i32, i32)

define i32 @test.cv.shuffle2.b(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.shuffle2.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.shuffle2.b a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.shuffle2.b(i32 %b, i32 %c, i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.packlo.h(i32, i32)

define i32 @test.cv.packlo.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.packlo.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.pack a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.packlo.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.packhi.h(i32, i32)

define i32 @test.cv.packhi.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.packhi.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.pack.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.packhi.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.packhi.b(i32, i32, i32)

define i32 @test.cv.packhi.b(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.packhi.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.packhi.b a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.packhi.b(i32 %a, i32 %b, i32 %c)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.packlo.b(i32, i32, i32)

define i32 @test.cv.packlo.b(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.packlo.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.packlo.b a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.packlo.b(i32 %a, i32 %b, i32 %c)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpeq.h(i32, i32)

define i32 @test.cv.cmpeq.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpeq.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpeq.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpeq.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpeq.b(i32, i32)

define i32 @test.cv.cmpeq.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpeq.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpeq.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpeq.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpeq.sc.h(i32, i32)

define i32 @test.cv.cmpeq.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpeq.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpeq.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpeq.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpeq.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.cmpeq.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpeq.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpeq.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpeq.sc.b(i32, i32)

define i32 @test.cv.cmpeq.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpeq.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpeq.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpeq.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpeq.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.cmpeq.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpeq.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpeq.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpne.h(i32, i32)

define i32 @test.cv.cmpne.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpne.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpne.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpne.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpne.b(i32, i32)

define i32 @test.cv.cmpne.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpne.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpne.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpne.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpne.sc.h(i32, i32)

define i32 @test.cv.cmpne.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpne.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpne.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpne.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpne.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.cmpne.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpne.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpne.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpne.sc.b(i32, i32)

define i32 @test.cv.cmpne.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpne.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpne.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpne.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpne.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.cmpne.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpne.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpne.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgt.h(i32, i32)

define i32 @test.cv.cmpgt.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgt.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgt.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgt.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgt.b(i32, i32)

define i32 @test.cv.cmpgt.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgt.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgt.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgt.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgt.sc.h(i32, i32)

define i32 @test.cv.cmpgt.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgt.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgt.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgt.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpgt.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.cmpgt.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgt.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgt.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgt.sc.b(i32, i32)

define i32 @test.cv.cmpgt.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgt.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgt.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgt.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpgt.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.cmpgt.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgt.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgt.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpge.h(i32, i32)

define i32 @test.cv.cmpge.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpge.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpge.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpge.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpge.b(i32, i32)

define i32 @test.cv.cmpge.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpge.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpge.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpge.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpge.sc.h(i32, i32)

define i32 @test.cv.cmpge.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpge.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpge.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpge.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpge.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.cmpge.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpge.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpge.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpge.sc.b(i32, i32)

define i32 @test.cv.cmpge.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpge.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpge.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpge.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpge.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.cmpge.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpge.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpge.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmplt.h(i32, i32)

define i32 @test.cv.cmplt.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmplt.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmplt.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmplt.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmplt.b(i32, i32)

define i32 @test.cv.cmplt.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmplt.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmplt.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmplt.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmplt.sc.h(i32, i32)

define i32 @test.cv.cmplt.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmplt.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmplt.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmplt.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmplt.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.cmplt.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmplt.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmplt.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmplt.sc.b(i32, i32)

define i32 @test.cv.cmplt.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmplt.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmplt.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmplt.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmplt.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.cmplt.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmplt.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmplt.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmple.h(i32, i32)

define i32 @test.cv.cmple.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmple.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmple.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmple.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmple.b(i32, i32)

define i32 @test.cv.cmple.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmple.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmple.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmple.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmple.sc.h(i32, i32)

define i32 @test.cv.cmple.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmple.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmple.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmple.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmple.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.cmple.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmple.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmple.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmple.sc.b(i32, i32)

define i32 @test.cv.cmple.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmple.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmple.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmple.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmple.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.cmple.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmple.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmple.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgtu.h(i32, i32)

define i32 @test.cv.cmpgtu.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgtu.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgtu.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgtu.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgtu.b(i32, i32)

define i32 @test.cv.cmpgtu.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgtu.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgtu.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgtu.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgtu.sc.h(i32, i32)

define i32 @test.cv.cmpgtu.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgtu.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgtu.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgtu.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpgtu.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.cmpgtu.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgtu.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgtu.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgtu.sc.b(i32, i32)

define i32 @test.cv.cmpgtu.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgtu.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgtu.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgtu.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpgtu.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.cmpgtu.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgtu.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgtu.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgeu.h(i32, i32)

define i32 @test.cv.cmpgeu.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgeu.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgeu.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgeu.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgeu.b(i32, i32)

define i32 @test.cv.cmpgeu.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgeu.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgeu.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgeu.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgeu.sc.h(i32, i32)

define i32 @test.cv.cmpgeu.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgeu.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgeu.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgeu.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpgeu.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.cmpgeu.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgeu.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgeu.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpgeu.sc.b(i32, i32)

define i32 @test.cv.cmpgeu.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpgeu.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgeu.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgeu.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpgeu.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.cmpgeu.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpgeu.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpgeu.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpltu.h(i32, i32)

define i32 @test.cv.cmpltu.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpltu.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpltu.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpltu.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpltu.b(i32, i32)

define i32 @test.cv.cmpltu.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpltu.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpltu.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpltu.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpltu.sc.h(i32, i32)

define i32 @test.cv.cmpltu.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpltu.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpltu.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpltu.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpltu.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.cmpltu.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpltu.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpltu.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpltu.sc.b(i32, i32)

define i32 @test.cv.cmpltu.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpltu.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpltu.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpltu.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpltu.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.cmpltu.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpltu.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpltu.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpleu.h(i32, i32)

define i32 @test.cv.cmpleu.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpleu.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpleu.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpleu.h(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpleu.b(i32, i32)

define i32 @test.cv.cmpleu.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpleu.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpleu.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpleu.b(i32 %a, i32 %b)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpleu.sc.h(i32, i32)

define i32 @test.cv.cmpleu.sc.h(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpleu.sc.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpleu.sc.h a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpleu.sc.h(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpleu.sci.h(i32 %a) {
; CHECK-LABEL: test.cv.cmpleu.sci.h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpleu.sci.h a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpleu.sc.h(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cmpleu.sc.b(i32, i32)

define i32 @test.cv.cmpleu.sc.b(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.cmpleu.sc.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpleu.sc.b a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpleu.sc.b(i32 %a, i32 %b)
  ret i32 %1
}

define i32 @test.cv.cmpleu.sci.b(i32 %a) {
; CHECK-LABEL: test.cv.cmpleu.sci.b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cmpleu.sci.b a0, a0, 5
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cmpleu.sc.b(i32 %a, i32 5)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cplxmul.r(i32, i32, i32, i32)

define i32 @test.cv.cplxmul.r(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.cplxmul.r:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cplxmul.r a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cplxmul.r(i32 %a, i32 %b, i32 %c, i32 0)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cplxmul.i(i32, i32, i32, i32)

define i32 @test.cv.cplxmul.i(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.cplxmul.i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cplxmul.i a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cplxmul.i(i32 %a, i32 %b, i32 %c, i32 0)
  ret i32 %1
}

define i32 @test.cv.cplxmul.r.div2(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.cplxmul.r.div2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cplxmul.r.div2 a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cplxmul.r(i32 %a, i32 %b, i32 %c, i32 1)
  ret i32 %1
}

define i32 @test.cv.cplxmul.i.div2(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.cplxmul.i.div2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cplxmul.i.div2 a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cplxmul.i(i32 %a, i32 %b, i32 %c, i32 1)
  ret i32 %1
}

define i32 @test.cv.cplxmul.r.div4(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.cplxmul.r.div4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cplxmul.r.div4 a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cplxmul.r(i32 %a, i32 %b, i32 %c, i32 2)
  ret i32 %1
}

define i32 @test.cv.cplxmul.i.div4(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.cplxmul.i.div4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cplxmul.i.div4 a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cplxmul.i(i32 %a, i32 %b, i32 %c, i32 2)
  ret i32 %1
}

define i32 @test.cv.cplxmul.r.div8(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.cplxmul.r.div8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cplxmul.r.div8 a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cplxmul.r(i32 %a, i32 %b, i32 %c, i32 3)
  ret i32 %1
}

define i32 @test.cv.cplxmul.i.div8(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test.cv.cplxmul.i.div8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cplxmul.i.div8 a0, a1, a2
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cplxmul.i(i32 %a, i32 %b, i32 %c, i32 3)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.cplxconj(i32)

define i32 @test.cv.cplxconj(i32 %a) {
; CHECK-LABEL: test.cv.cplxconj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.cplxconj a0, a0
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.cplxconj(i32 %a)
  ret i32 %1
}

declare i32 @llvm.riscv.cv.simd.subrotmj(i32, i32, i32)

define i32 @test.cv.subrotmj(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.subrotmj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.subrotmj a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.subrotmj(i32 %a, i32 %b, i32 0)
  ret i32 %1
}

define i32 @test.cv.subrotmj.div2(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.subrotmj.div2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.subrotmj.div2 a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.subrotmj(i32 %a, i32 %b, i32 1)
  ret i32 %1
}

define i32 @test.cv.subrotmj.div4(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.subrotmj.div4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.subrotmj.div4 a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.subrotmj(i32 %a, i32 %b, i32 2)
  ret i32 %1
}

define i32 @test.cv.subrotmj.div8(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.subrotmj.div8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.subrotmj.div8 a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.subrotmj(i32 %a, i32 %b, i32 3)
  ret i32 %1
}

define i32 @test.cv.add.div2(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.add.div2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.add.div2 a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.add.h(i32 %a, i32 %b, i32 1)
  ret i32 %1
}

define i32 @test.cv.add.div4(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.add.div4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.add.div4 a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.add.h(i32 %a, i32 %b, i32 2)
  ret i32 %1
}

define i32 @test.cv.add.div8(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.add.div8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.add.div8 a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.add.h(i32 %a, i32 %b, i32 3)
  ret i32 %1
}

define i32 @test.cv.sub.div2(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sub.div2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sub.div2 a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sub.h(i32 %a, i32 %b, i32 1)
  ret i32 %1
}

define i32 @test.cv.sub.div4(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sub.div4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sub.div4 a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sub.h(i32 %a, i32 %b, i32 2)
  ret i32 %1
}

define i32 @test.cv.sub.div8(i32 %a, i32 %b) {
; CHECK-LABEL: test.cv.sub.div8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cv.sub.div8 a0, a0, a1
; CHECK-NEXT:    ret
  %1 = call i32 @llvm.riscv.cv.simd.sub.h(i32 %a, i32 %b, i32 3)
  ret i32 %1
}
