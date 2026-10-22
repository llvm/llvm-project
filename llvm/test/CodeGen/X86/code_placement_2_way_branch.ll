; RUN: llc -mtriple=x86_64-linux -verify-machineinstrs -two-way-branch-opt=cold-fallthrough < %s | FileCheck %s --check-prefixes=CHECK,COLD-FT
; RUN: llc -mtriple=x86_64-linux -verify-machineinstrs -two-way-branch-opt=none < %s | FileCheck %s --check-prefixes=CHECK,COLD-FT
; RUN: llc -mtriple=x86_64-linux -verify-machineinstrs -two-way-branch-opt=hot-fallthrough < %s | FileCheck %s --check-prefixes=CHECK,HOT-FT

define void @foo() !prof !1 {
; Test that two-way branches are optimized based on `-two-way-branch-opt`.
;
; +--------+   5   +--------+
; | if.then| <---- | entry  |
; +--------+       +--------+
;   |  |             |
;   |  |             | 10
;   |  |             v
;   |  |           +--------+
;   |  |           | if.else|
;   |  |           +--------+
;   |  |             |
;   |  |             | 10
;   |  |             v
;   |  |     4     +--------+
;   |  +---------> | if.end |
;   |              +--------+
;   |                |
;   |                | 14
;   |                v
;   |     1        +--------+
;   +------------> |  end   |
;                  +--------+
;
; CHECK-LABEL: foo:
; CHECK: if.else
; CHECK: .LBB0_3:  # %if.end
; CHECK: .LBB0_4:  # %end
; CHECK: if.then
; COLD-FT: jne .LBB0_3
; HOT-FT: je .LBB0_4
; COLD-FT: jmp .LBB0_4
; HOT-FT: jmp .LBB0_3

entry:
  call void @e()
  %call1 = call zeroext i1 @a()
  br i1 %call1, label %if.then, label %if.else, !prof !2

if.then:
  call void @f()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %if.end, label %end, !prof !3

if.else:
  call void @g()
  br label %if.end

if.end:
  call void @h()
  br label %end

end:
  ret void
}

declare zeroext i1 @a()
declare void @e()
declare void @g()
declare void @f()
declare void @h()

!1 = !{!"function_entry_count", i64 15}
!2 = !{!"branch_weights", i32 5, i32 10}
!3 = !{!"branch_weights", i32 4, i32 1}
