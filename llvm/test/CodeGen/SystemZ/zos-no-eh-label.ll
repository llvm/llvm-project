; RUN: llc -mtriple s390x-ibm-zos < %s | FileCheck %s

define signext i32 @_Z9computeitv() personality ptr @__zos_cxx_personality_v2 {
  ret i32 0
}

declare i32 @__zos_cxx_personality_v2(...)

; The personality function is unused, therefore check that it is not referenced.
; There should also be no exception table.
; CHECK-NOT: __zos_cxx_personality_v2
; CHECK-NOT: GCC_except_table
