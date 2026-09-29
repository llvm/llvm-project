; RUN: lli -jit-kind=orc-lazy -orc-lazy-debug=mods-to-stdout %s | FileCheck %s
;
; CHECK: module-flags.ll.submodule
; CHECK-NOT: Module End
; CHECK: The Answer is {{.*}}42

define i32 @main() {
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"The Answer is ", i32 42}
