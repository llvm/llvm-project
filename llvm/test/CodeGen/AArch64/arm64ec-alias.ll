; RUN: llc -mtriple arm64ec-windows-msvc -filetype asm -o - %s | FileCheck %s

define void @func() {
  ret void
}

define dso_local void @patchable_func() hybrid_patchable {
  ret void
}

@func_alias = alias void (), ptr @func
@func_alias2 = alias void (), ptr @func_alias
@patchable_alias = alias void (), ptr @patchable_func

; CHECK:              .weak_anti_dep  func_alias
; CHECK-NEXT: func_alias = "#func_alias"
; CHECK-NEXT:         .weak_anti_dep  func_alias2
; CHECK-NEXT: func_alias2 = "#func_alias2"
; CHECK-NEXT:         .weak_anti_dep  func
; CHECK-NEXT: func = "#func"
; CHECK:              .weak_anti_dep  patchable_alias
; CHECK-NEXT: patchable_alias = "#patchable_alias"

; CHECK:              .globl  "#func_alias"
; CHECK-NEXT:         .def    "#func_alias";
; CHECK-NEXT:         .scl    2;
; CHECK-NEXT:         .type   32;
; CHECK-NEXT:         .endef
; CHECK-NEXT: "#func_alias" = "#func"
; CHECK-NEXT:         .globl  "#func_alias2"
; CHECK-NEXT:         .def    "#func_alias2";
; CHECK-NEXT:         .scl    2;
; CHECK-NEXT:         .type   32;
; CHECK-NEXT:         .endef
; CHECK-NEXT: "#func_alias2" = "#func_alias"

; CHECK:              .globl  "#patchable_alias"
; CHECK-NEXT:         .def    "#patchable_alias";
; CHECK-NEXT:         .scl    2;
; CHECK-NEXT:         .type   32;
; CHECK-NEXT:         .endef
; CHECK-NEXT: "#patchable_alias" = "#patchable_func"
