; RUN: llc < %s -mtriple=thumbv8m.base   | FileCheck %s
; RUN: llc < %s -mtriple=thumbebv8m.base | FileCheck %s

define i32 @test_tt(i8* readnone %p) #0 {
entry:
  %0 = tail call i32 @llvm.arm.cmse.tt(i8* %p)
  ret i32 %0
}
; CHECK-LABEL: test_tt:
; CHECK: tt r{{[0-9]+}}, r{{[0-9]+}}

declare i32 @llvm.arm.cmse.tt(i8*) #1

define i32 @test_ttt(i8* readnone %p) #0 {
entry:
  %0 = tail call i32 @llvm.arm.cmse.ttt(i8* %p)
  ret i32 %0
}
; CHECK-LABEL: test_ttt:
; CHECK: ttt r{{[0-9]+}}, r{{[0-9]+}}

declare i32 @llvm.arm.cmse.ttt(i8*) #1

define i32 @test_tta(i8* readnone %p) #0 {
entry:
  %0 = tail call i32 @llvm.arm.cmse.tta(i8* %p)
  ret i32 %0
}
; CHECK-LABEL: test_tta:
; CHECK: tta r{{[0-9]+}}, r{{[0-9]+}}

declare i32 @llvm.arm.cmse.tta(i8*) #1

define i32 @test_ttat(i8* readnone %p) #0 {
entry:
  %0 = tail call i32 @llvm.arm.cmse.ttat(i8* %p)
  ret i32 %0
}
; CHECK-LABEL: test_ttat:
; CHECK: ttat r{{[0-9]+}}, r{{[0-9]+}}

declare i32 @llvm.arm.cmse.ttat(i8*) #1

attributes #0 = { nounwind readnone "target-features"="+8msecext"}
attributes #1 = { nounwind readnone }
