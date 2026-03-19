; RUN: opt < %s -passes='rtsan' -S | FileCheck %s

declare void @declared_realtime_function() sanitize_realtime #0

declare void @declared_blocking_function() sanitize_realtime_blocking #0

; RealtimeSanitizer pass should ignore attributed functions that are just declarations
; CHECK: declared_realtime_function
; CHECK-EMPTY:
; CHECK: declared_blocking_function
; CHECK-EMPTY:
