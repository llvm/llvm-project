; RUN: llc -mtriple=thumbv7m-linux-gnu -o - %s | FileCheck %s --check-prefix=T2
; RUN: llc -mtriple=thumbv6m-linux-gnu -o - %s | FileCheck %s --check-prefix=T1

declare void @foo(double)
declare i32 @llvm.arm.space(i32, i32)

define i32 @test_tbh(i1 %tst, i32 %sw, i32 %l) {
  br label %complex

; T2-LABEL: test_tbh:
; T2: [[ANCHOR:.LCPI[0-9_]+]]:
; T2: tbh [pc, r{{[0-9]+}}, lsl #1]
; T2-NEXT: @ BB#1
; T2-NEXT: LJTI
; T2-NEXT: .short	(.LBB0_[[x:[0-9]+]]-([[ANCHOR]]+4))/2
; T2-NEXT: .short	(.LBB0_{{[0-9]+}}-([[ANCHOR]]+4))/2
; T2-NEXT: .short	(.LBB0_{{[0-9]+}}-([[ANCHOR]]+4))/2
; T2-NEXT: .short	(.LBB0_[[x]]-([[ANCHOR]]+4))/2

; T1-LABEL: test_tbh:
; T1: lsls [[x:r[0-9]+]], r4, #1
; T1: add [[x]], pc
; T1: ldrh [[x]], {{\[}}[[x]], #4]
; T1: lsls [[x]], [[x]], #1
; T1: [[ANCHOR:.LCPI[0-9_]+]]:
; T1: add pc, [[x]]
; T1-NEXT: @ BB#2
; T1-NEXT: .p2align 2
; T1-NEXT: LJTI
; T1-NEXT: .short	(.LBB0_[[x:[0-9]+]]-([[ANCHOR]]+4))/2
; T1-NEXT: .short	(.LBB0_{{[0-9]+}}-([[ANCHOR]]+4))/2
; T1-NEXT: .short	(.LBB0_{{[0-9]+}}-([[ANCHOR]]+4))/2
; T1-NEXT: .short	(.LBB0_[[x]]-([[ANCHOR]]+4))/2

complex:
  call void @foo(double 12345.0)
  switch i32 %sw, label %second [ i32 0, label %other
                                  i32 1, label %third
                                  i32 2, label %end
                                  i32 3, label %other ]

second:
  ret i32 43
third:
  call i32 @llvm.arm.space(i32 970, i32 undef)
  ret i32 0

other:
  call void @bar()
  unreachable

end:
  ret i32 42
}

declare void @bar()
