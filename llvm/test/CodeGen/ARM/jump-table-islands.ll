; RUN: llc -mtriple=armv7-apple-ios8.0 -o - %s | FileCheck %s

%BigInt = type i8500

define %BigInt @test_moved_jumptable(i1 %tst, i32 %sw, %BigInt %l) {
; CHECK-LABEL: test_moved_jumptable:

; CHECK:   adr {{r[0-9]+}}, [[JUMP_TABLE:LJTI[0-9]+_[0-9]+]]
; CHECK:   b [[SKIP_TABLE:LBB[0-9]+_[0-9]+]]

; CHECK: [[JUMP_TABLE]]:
; CHECK:   .data_region jt32
; CHECK:   .long LBB{{[0-9]+_[0-9]+}}-[[JUMP_TABLE]]

; CHECK: [[SKIP_TABLE]]:
; CHECK:   add pc, {{r[0-9]+|lr}}, {{r[0-9]+|lr}}
  br i1 %tst, label %simple, label %complex

simple:
  br label %end

complex:
  switch i32 %sw, label %simple [ i32 0, label %other
                                  i32 1, label %third
                                  i32 5, label %end
                                  i32 6, label %other ]

third:
  ret %BigInt 0

other:
  call void @bar()
  unreachable

end:
  %val = phi %BigInt [ %l, %complex ], [ -1, %simple ]
  ret %BigInt %val
}

declare void @bar()
