; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes='function(memprof),memprof-module' -S | FileCheck %s --check-prefix=EMPTY
; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes='function(memprof),memprof-module' -S -memprof-runtime-default-options="verbose=1" | FileCheck %s

define i32 @main() {
entry:
  ret i32 0
}

; EMPTY-NOT: memprof_default_options_str

; CHECK: $__memprof_default_options_str = comdat any
; CHECK: @__memprof_default_options_str = constant [10 x i8] c"verbose=1\00", comdat
