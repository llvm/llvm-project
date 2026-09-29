; RUN: llc < %s -emulated-tls -mtriple=aarch64-linux-ohos \
; RUN:     | FileCheck -check-prefix=EMUTLS_CHECK %s
 
%struct.__res_state = type { [5 x i8] }
 
@foo = dso_local thread_local global %struct.__res_state { [5 x i8] c"\01\02\03\04\05" }, align 1
 
@bar = hidden thread_local(initialexec) alias %struct.__res_state, ptr @foo
 
define dso_local i32 @main() {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  store i8 0, ptr @bar, align 1
  ; EMUTLS_CHECK: adrp    x0, __emutls_v.foo
  ; EMUTLS_CHECK-NEXT: add     x0, x0, :lo12:__emutls_v.foo
  ret i32 0
}
