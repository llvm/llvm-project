; RUN: lli -jit-kind=orc-lazy -orc-lazy-debug=funcs-to-stdout \
; RUN:   %s | FileCheck --check-prefix=CHECK-PER-FUNCTION %s
; RUN: lli -jit-kind=orc-lazy -per-module-lazy -orc-lazy-debug=funcs-to-stdout \
; RUN:   %s | FileCheck --check-prefix=CHECK-WHOLE-MODULE %s
;
; CHECK-PER-FUNCTION-NOT: foo
; CHECK-WHOLE-MODULE: foo
;
; Checks that the whole module is emitted when -per-module-lazy is enabled,
; even though foo is not called.
; Also checks that the foo function is not emitted when -per-module-lazy is off.

define void @foo() {
entry:
  ret void
}

define i32 @main(i32 %argc, i8** nocapture readnone %argv) {
entry:
  ret i32 0
}
