; RUN: not opt < %s -passes=instrumentor -instrumentor-read-config-files=%S/test_filter_not_error_config.json -S 2>&1 | FileCheck %s

@X = dso_local global i32 0, align 4

; CHECK: error: malformed filter expression for instrumentation opportunity 'global': expected boolean value at position 1
; CHECK-NEXT: Filter: !name
