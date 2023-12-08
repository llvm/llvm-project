; Test to ensure that the filename provided by clang in the module flags
; metadata results in the expected __memprof_profile_filename insertion.

; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes='function(memprof),memprof-module' -S | FileCheck --check-prefixes=CHECK %s

define i32 @main() {
entry:
  ret i32 0
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"MemProfProfileFilename", !"/tmp/memprof.profraw"}

; CHECK: $__memprof_profile_filename = comdat any
; CHECK: @__memprof_profile_filename = constant [21 x i8] c"/tmp/memprof.profraw\00", comdat
