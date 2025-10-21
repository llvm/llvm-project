; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=M1 %s

; Check that __cfi_check is emitted on the full LTO side with
; attributes preserved.

; M0: define void @f()
define void @f() !type !{!"f1", i32 0} {
  ret void 
}

; M1: define void @__cfi_check() #0
define void @__cfi_check() #0 {
  ret void
}

; M1: attributes #0 = { "branch-target-enforcement" }
attributes #0 = { "branch-target-enforcement" }
