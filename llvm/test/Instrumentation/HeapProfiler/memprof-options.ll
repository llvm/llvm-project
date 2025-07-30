; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes='function(memprof),memprof-module' -S | FileCheck %s --check-prefixes=CHECK,EMPTY
; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes='function(memprof),memprof-module' -S -memprof-runtime-default-options="verbose=1" | FileCheck %s --check-prefixes=CHECK,VERBOSE

define i32 @main() {
entry:
  ret i32 0
}

; CHECK: $__memprof_default_options_str = comdat any
; EMPTY: @__memprof_default_options_str = constant [1 x i8] zeroinitializer, comdat
; VERBOSE: @__memprof_default_options_str = constant [10 x i8] c"verbose=1\00", comdat
