; RUN: mlir-translate -import-llvm %s | mlir-translate -mlir-to-llvmir | FileCheck %s

define void @synthetic() !prof !0 {
  ret void
}

define void @with_import_guid() !prof !1 {
  ret void
}

define void @synthetic_with_import_guid() !prof !2 {
  ret void
}

!0 = !{!"synthetic_function_entry_count", i64 7}
!1 = !{!"function_entry_count", i64 7, i64 1234, i64 4, i64 1234}
!2 = !{!"synthetic_function_entry_count", i64 7, i64 1234}

; CHECK: define void @synthetic()
; CHECK-SAME: !prof ![[SYNTH:[0-9]+]]

; CHECK: define void @with_import_guid()
; CHECK-SAME: !prof ![[IMPORTS:[0-9]+]]

; CHECK: define void @synthetic_with_import_guid()
; CHECK-SAME: !prof ![[SYNTH_IMPORTS:[0-9]+]]

; CHECK-DAG: ![[SYNTH]] = !{!"synthetic_function_entry_count", i64 7}
; CHECK-DAG: ![[IMPORTS]] = !{!"function_entry_count", i64 7, i64 4, i64 1234}
; CHECK-DAG: ![[SYNTH_IMPORTS]] = !{!"synthetic_function_entry_count", i64 7, i64 1234}
