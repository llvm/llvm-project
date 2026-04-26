; This test checks if the global merge func pass should not build module summary.

; RUN: opt --passes=global-merge-func %s -o /dev/null

@0 = global { { ptr, i32, i32 } } { { ptr, i32, i32 } { ptr null, i32 19, i32 5 } }
@1 = global { { ptr, i32, i32 } } { { ptr, i32, i32 } { ptr null, i32 22, i32 5 } }
