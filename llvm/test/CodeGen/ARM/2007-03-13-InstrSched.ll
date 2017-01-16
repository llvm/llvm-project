; REQUIRES: asserts
; RUN: llc -mtriple arm-apple-darwin -relocation-model pic -mattr=+v6 -filetype asm -o - %s | FileCheck -check-prefix CHECK-R9 %s
; RUN: llc -mtriple arm-apple-darwin -relocation-model pic -mattr=+v6,+reserve-r9 -ifcvt-limit=0 -stats -o /dev/null %s 2>&1 | FileCheck -check-prefix CHECK-ASM-PRINTER %s

define void @test(i32 %tmp56222, i32 %tmp36224, i32 %tmp46223, i32 %i.0196.0.ph, i32 %tmp8, i32* %tmp1011, i32** %tmp1, i32* %d2.1.out, i32* %d3.1.out, i32* %d0.1.out, i32* %d1.1.out) {
newFuncRoot:
  br label %bb74

bb78.exitStub:
  store i32 %d2.1, i32* %d2.1.out
  store i32 %d3.1, i32* %d3.1.out
  store i32 %d0.1, i32* %d0.1.out
  store i32 %d1.1, i32* %d1.1.out
  ret void

bb74:
  %fp.1.rec = phi i32 [ 0, %newFuncRoot ], [ %tmp71.rec, %bb26 ]
  %fm.1.in = phi i32* [ %tmp71, %bb26 ], [ %tmp1011, %newFuncRoot ]
  %d0.1 = phi i32 [ %tmp44, %bb26 ], [ 8192, %newFuncRoot ]
  %d1.1 = phi i32 [ %tmp54, %bb26 ], [ 8192, %newFuncRoot ]
  %d2.1 = phi i32 [ %tmp64, %bb26 ], [ 8192, %newFuncRoot ]
  %d3.1 = phi i32 [ %tmp69, %bb26 ], [ 8192, %newFuncRoot ]
  %fm.1 = load i32, i32* %fm.1.in
  icmp eq i32 %fp.1.rec, %tmp8
  br i1 %0, label %bb78.exitStub, label %bb26

bb26:
  %tmp28 = getelementptr i32*, i32** %tmp1, i32 %fp.1.rec
  %tmp30 = load i32*, i32** %tmp28
  %tmp33 = getelementptr i32, i32* %tmp30, i32 %i.0196.0.ph
  %tmp34 = load i32, i32* %tmp33
  %tmp38 = getelementptr i32, i32* %tmp30, i32 %tmp36224
  %tmp39 = load i32, i32* %tmp38
  %tmp42 = mul i32 %tmp34, %fm.1
  %tmp44 = add i32 %tmp42, %d0.1
  %tmp48 = getelementptr i32, i32* %tmp30, i32 %tmp46223
  %tmp49 = load i32, i32* %tmp48
  %tmp52 = mul i32 %tmp39, %fm.1
  %tmp54 = add i32 %tmp52, %d1.1
  %tmp58 = getelementptr i32, i32* %tmp30, i32 %tmp56222
  %tmp59 = load i32, i32* %tmp58
  %tmp62 = mul i32 %tmp49, %fm.1
  %tmp64 = add i32 %tmp62, %d2.1
  %tmp67 = mul i32 %tmp59, %fm.1
  %tmp69 = add i32 %tmp67, %d3.1
  %tmp71.rec = add i32 %fp.1.rec, 1
  %tmp71 = getelementptr i32, i32* %tmp1011, i32 %tmp71.rec
  br label %bb74
}

; CHECK-R9: r9
; CHECK-ASM-PRINTER: asm-printer

