; RUN: llc < %s -mtriple=ve | FileCheck %s

define i8* @test1() nounwind {
; CHECK-LABEL: test1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s0, 0, %s9
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %ret = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %ret
}

define i8* @test2() nounwind {
; CHECK-LABEL: test2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld %s0, (, %s9)
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
entry:
  %ret = tail call i8* @llvm.frameaddress(i32 2)
  ret i8* %ret
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
