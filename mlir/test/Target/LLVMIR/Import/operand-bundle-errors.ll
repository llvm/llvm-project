; RUN: mlir-translate -import-llvm -split-input-file -verify-diagnostics %s -o /dev/null
; expected-error-re @-2 {{unsupported metadata: !{{[0-9]+}} = distinct !{!"payload"}}}

declare void @callee()

define void @call_with_unsupported_metadata() {
  call void @callee() ["tag"(metadata !0)]
  ret void
}

!0 = distinct !{!"payload"}

; // -----

declare void @callee()
declare i32 @personality(...)

define void @invoke_with_unsupported_constant() personality ptr @personality {
  ; expected-error @unknown {{unhandled constant: ptr no_cfi @callee since no_cfi is unsupported}}
  invoke void @callee() ["tag"(ptr no_cfi @callee)] to label %normal unwind label %unwind

normal:
  ret void

unwind:
  %exception = landingpad {ptr, i32} cleanup
  resume {ptr, i32} %exception
}
