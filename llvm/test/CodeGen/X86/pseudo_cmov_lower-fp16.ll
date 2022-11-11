; RUN: llc < %s -mtriple=i386-linux-gnu -mattr=+avx512fp16 -mattr=+avx512vl -o - | FileCheck %s

; This test checks that only a single je gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; CHECK-LABEL: foo1:
; CHECK: je
; CHECK-NOT: je
define <8 x half> @foo1(i32 %v1, <8 x half> %v2, <8 x half> %v3, <8 x half> %v4) nounwind {
entry:
  %cmp = icmp eq i32 %v1, 0
  %t1 = select i1 %cmp, <8 x half> %v2, <8 x half> %v3
  %t2 = select i1 %cmp, <8 x half> %v3, <8 x half> %v4
  %sub = fsub <8 x half> %t1, %t2
  ret <8 x half> %sub
}

; This test checks that only a single ja gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR. This combines
; all the supported types together into one long string of selects based
; on the same condition.
; CHECK-LABEL: foo2:
; CHECK: ja
; CHECK-NOT: ja
define void @foo2(i32 %v1,
                  half %v32, half %v33,
                  <8 x half> %v52, <8 x half> %v53,
                  <16 x half> %v122, <16 x half> %v123,
                  <32 x half> %v132, <32 x half> %v133,
                  ptr %dst) nounwind {
entry:
  %add.ptr31 = getelementptr inbounds i8, ptr %dst, i32 2

  %add.ptr51 = getelementptr inbounds i8, ptr %dst, i32 4

  %add.ptr121 = getelementptr inbounds i8, ptr %dst, i32 20

  %add.ptr131 = getelementptr inbounds i8, ptr %dst, i32 52

  ; These operations are necessary, because select of two single use loads
  ; ends up getting optimized into a select of two leas, followed by a
  ; single load of the selected address.

  %t33 = fadd half %v33, %v32
  %t53 = fadd <8 x half> %v53, %v52
  %t123 = fadd <16 x half> %v123, %v122
  %t133 = fadd <32 x half> %v133, %v132

  %cmp = icmp ugt i32 %v1, 31
  %t31 = select i1 %cmp, half %v32, half %t33
  %t51 = select i1 %cmp, <8 x half> %v52, <8 x half> %t53
  %t121 = select i1 %cmp, <16 x half> %v122, <16 x half> %t123
  %t131 = select i1 %cmp, <32 x half> %v132, <32 x half> %t133

  store half %t31, ptr %add.ptr31, align 2
  store <8 x half> %t51, ptr %add.ptr51, align 16
  store <16 x half> %t121, ptr %add.ptr121, align 32
  store <32 x half> %t131, ptr %add.ptr131, align 64

  ret void
}

; This test checks that only a single jne gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
define dso_local <32 x half> @foo3(<32 x half> %a, <32 x half> %b, i1 zeroext %sign) local_unnamed_addr #0 {
; CHECK-LABEL: foo3:
; CHECK: jne
; CHECK-NOT: jne
entry:
  %spec.select = select i1 %sign, <32 x half> %a, <32 x half> %b
  ret <32 x half> %spec.select
}

; This test checks that only a single jne gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
define dso_local <16 x half> @foo4(<16 x half> %a, <16 x half> %b, i1 zeroext %sign) local_unnamed_addr #0 {
; CHECK-LABEL: foo4:
; CHECK: jne
; CHECK-NOT: jne
entry:
  %spec.select = select i1 %sign, <16 x half> %a, <16 x half> %b
  ret <16 x half> %spec.select
}

; This test checks that only a single jne gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
define dso_local <8 x half> @foo5(<8 x half> %a, <8 x half> %b, i1 zeroext %sign) local_unnamed_addr #0 {
; CHECK-LABEL: foo5:
; CHECK: jne
; CHECK-NOT: jne
entry:
  %spec.select = select i1 %sign, <8 x half> %a, <8 x half> %b
  ret <8 x half> %spec.select
}
