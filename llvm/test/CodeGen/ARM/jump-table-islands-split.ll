; RUN: llc -mtriple=thumbv7s-apple-ios8.0 -o - %s | FileCheck %s

declare void @foo(double)
declare i32 @llvm.arm.space(i32, i32)

; The constpool entry used to call @foo should be directly between where we want
; the tbb and its table. Fortunately, the flow is simple enough that we can
; eliminate the entry calculation (ADD) and use the ADR as the base.
;
; I'm hoping this won't be fragile, but if it does break the most likely fix is
; adjusting the @llvm.arm.space call slightly. If this happens too many times
; the test should probably be removed.
define i32 @test_jumptable_not_adjacent(i1 %tst, i32 %sw, i32 %l) {
; CHECK-LABEL: test_jumptable_not_adjacent:
; CHECK:     vldr {{d[0-9]+}}, [[DBL_CONST:LCPI[0-9]+_[0-9]+]]
; [...]
; CHECK:     adr.w r[[BASE:[0-9]+]], [[JUMP_TABLE:LJTI[0-9]+_[0-9]+]]
; CHECK-NOT: r[[BASE]]

; CHECK: [[TBB_KEY:LCPI[0-9]+_[0-9]+]]:
; CHECK-NEXT:     tbb [r[[BASE]], {{r[0-9]+}}]

; CHECK: [[DBL_CONST]]:
; CHECK:     .long
; CHECK:     .long
; CHECK: [[JUMP_TABLE]]:
; CHECK:     .byte (LBB{{[0-9]+}}_{{[0-9]+}}-([[TBB_KEY]]+4)

  br label %complex

complex:
  call void @foo(double 12345.0)
  call i32 @llvm.arm.space(i32 970, i32 undef)
  switch i32 %sw, label %second [ i32 0, label %other
                                  i32 1, label %third
                                  i32 2, label %end
                                  i32 3, label %other ]

second:
  ret i32 43
third:
  ret i32 0

other:
  call void @bar()
  unreachable

end:
  ret i32 42
}

declare void @bar()
