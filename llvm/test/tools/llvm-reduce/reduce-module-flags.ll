; Test keeping one module flag
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=named-metadata --test=FileCheck --test-arg=--check-prefix=CHECK-INTERESTINGNESS0 --test-arg=%s --test-arg=--input-file %s -o %t.0
; RUN: FileCheck --check-prefix=RESULT0 %s < %t.0

; Test keeping two module flags
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=named-metadata --test=FileCheck --test-arg=--check-prefix=CHECK-INTERESTINGNESS1 --test-arg=%s --test-arg=--input-file %s -o %t.1
; RUN: FileCheck --check-prefix=RESULT1 %s < %t.1


; Test removing all module flags
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=named-metadata --test=FileCheck --test-arg=--check-prefix=CHECK-INTERESTINGNESS2 --test-arg=%s --test-arg=--input-file %s -o %t.2
; RUN: FileCheck --check-prefix=RESULT2 %s < %t.2


; CHECK-INTERESTINGNESS0: "openmp-device"

; CHECK-INTERESTINGNESS1: "wchar_size"
; CHECK-INTERESTINGNESS1: "openmp"

; CHECK-INTERESTINGNESS2: !llvm.module.flags

; RESULT0: !llvm.module.flags = !{!0}
; RESULT0: !0 = !{i32 7, !"openmp-device", i32 50}


; RESULT1: !llvm.module.flags = !{!0, !1}
; RESULT1: !0 = !{i32 1, !"wchar_size", i32 4}
; RESULT1: !1 = !{i32 7, !"openmp", i32 50}


; RESULT2: !llvm.module.flags = !{}

!llvm.module.flags = !{!0, !1, !2, !3, !4}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"openmp", i32 50}
!3 = !{i32 7, !"openmp-device", i32 50}
!4 = !{i32 8, !"PIC Level", i32 1}
