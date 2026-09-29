; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: @simple_range = external global i32, !absolute_symbol [[META0:![0-9]+]]
@simple_range = external global i32, !absolute_symbol !0

; Unlike !range, this accepts -1, -1
; CHECK: @full_range = external global i32, !absolute_symbol [[META1:![0-9]+]]
@full_range = external global i32, !absolute_symbol !1

; CHECK: @multiple_ranges = external global i32, !absolute_symbol [[META2:![0-9]+]]
@multiple_ranges = external global i32, !absolute_symbol !2

!0 = !{i64 4096, i64 8192}
!1 = !{i64 -1, i64 -1}
!2 = !{i64 256, i64 512, i64 1024, i64 4096}
;.
; CHECK: [[META0]] = !{i64 4096, i64 8192}
; CHECK: [[META1]] = !{i64 -1, i64 -1}
; CHECK: [[META2]] = !{i64 256, i64 512, i64 1024, i64 4096}
;.
