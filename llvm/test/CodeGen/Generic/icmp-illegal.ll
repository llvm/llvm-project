; RUN: llc < %s | FileCheck %s

; NVPTX fails to LowerFormalArguments for arg size > i128
; the arg byte size must be one of the {16, 8, 4, 2}
; XFAIL: nvptx

; CHECK-LABEL: test_ult
define i1 @test_ult(i256 %a) nounwind {
  %1 = icmp ult i256 %a, -6432394258550908438
  ret i1 %1
}

; CHECK-LABEL: test_ule
define i1 @test_ule(i256 %a) nounwind {
  %1 = icmp ule i256 %a, -6432394258550908438
  ret i1 %1
}

; CHECK-LABEL: test_ugt
define i1 @test_ugt(i256 %a) nounwind {
  %1 = icmp ugt i256 %a, -6432394258550908438
  ret i1 %1
}

; CHECK-LABEL: test_uge
define i1 @test_uge(i256 %a) nounwind {
  %1 = icmp uge i256 %a, -6432394258550908438
  ret i1 %1
}

; CHECK-LABEL: test_slt
define i1 @test_slt(i256 %a) nounwind {
  %1 = icmp slt i256 %a, -6432394258550908438
  ret i1 %1
}

; CHECK-LABEL: test_sle
define i1 @test_sle(i256 %a) nounwind {
  %1 = icmp sle i256 %a, -6432394258550908438
  ret i1 %1
}

; CHECK-LABEL: test_sgt
define i1 @test_sgt(i256 %a) nounwind {
  %1 = icmp sgt i256 %a, -6432394258550908438
  ret i1 %1
}

; CHECK-LABEL: test_sge
define i1 @test_sge(i256 %a) nounwind {
  %1 = icmp sge i256 %a, -6432394258550908438
  ret i1 %1
}
