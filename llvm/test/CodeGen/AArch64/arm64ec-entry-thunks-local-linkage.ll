; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s

; Validates when local linkage functions get a thunk generated.

; Being called does not cause a thunk to be generated or the symbol name to be mangled.
; CHECK-NOT: "#does_not_have_addr_taken":
; CHECK-NOT:  $ientry_thunk$cdecl$v$f;
define internal void @does_not_have_addr_taken(float) nounwind {
  ret void
}
define void @calls_does_not_have_addr_taken() nounwind {
  call void @does_not_have_addr_taken(float 0.0)
  ret void
}

; Having an address taken does cause a thunk to be generated and the symbol name to be mangled.
; CHECK: "#has_addr_taken":
; CHECK: $ientry_thunk$cdecl$v$i8;
define internal void @has_addr_taken(i64) nounwind {
  ret void
}
@points_to_has_addr_taken = global ptr @has_addr_taken
