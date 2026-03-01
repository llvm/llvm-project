; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

define dso_local i32 @override_me1() "loader-replaceable" {
entry:
  ret i32 1
}

define dso_local i32 @override_me2() "loader-replaceable" {
entry:
  ret i32 2
}

define dso_local i32 @dont_override_me() {
entry:
  ret i32 3
}

; CHECK:              .section        .drectve,"yni"
; CHECK-NEXT:         .def    override_me1_$fo$;
; CHECK-NEXT:         .scl    2;
; CHECK-NEXT:         .type   0;
; CHECK-NEXT:         .endef
; CHECK-NEXT:         .def    override_me1_$fo_default$;
; CHECK-NEXT:         .scl    2;
; CHECK-NEXT:         .type   0;
; CHECK-NEXT:         .endef
; CHECK-NEXT:         .ascii  " /ALTERNATENAME:override_me1_$fo$=override_me1_$fo_default$"
; CHECK-NEXT:         .def    override_me2_$fo$;
; CHECK-NEXT:         .scl    2;
; CHECK-NEXT:         .type   0;
; CHECK-NEXT:         .endef
; CHECK-NEXT:         .def    override_me2_$fo_default$;
; CHECK-NEXT:         .scl    2;
; CHECK-NEXT:         .type   0;
; CHECK-NEXT:         .endef
; CHECK-NEXT:         .ascii  " /ALTERNATENAME:override_me2_$fo$=override_me2_$fo_default$"
; CHECK-NEXT:         .data
; CHECK-NEXT: override_me1_$fo_default$:
; CHECK-NEXT: override_me2_$fo_default$:
; CHECK-NEXT:         .zero   1
