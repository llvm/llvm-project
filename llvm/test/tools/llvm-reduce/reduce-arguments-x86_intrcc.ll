; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=arguments --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefix=REDUCED

@gv = global i32 0

; INTERESTING-LABEL: void @func(
; INTERESTING-SAME: i32 %other.keep

; REDUCED: define x86_intrcc void @func(ptr byval(i32) %k, i32 %other.keep)
define x86_intrcc void @func(ptr byval(i32) %k, i32 %other.keep, i32 %other.drop) {
  store i32 %other.keep, ptr @gv
  ret void
}

; INTERESTING-LABEL: void @extern_decl(
; INTERESTING-SAME: i32

; REDUCED: declare x86_intrcc void @extern_decl(ptr byval(i32))
declare x86_intrcc void @extern_decl(ptr byval(i32), i32, i32)

; INTERESTING-LABEL: void @callsite(
; INTERESTING: call
; REDUCED: call x86_intrcc void @func(ptr byval(i32) %k, i32 %other.keep)
define void @callsite(ptr %k, i32 %other.keep, i32 %other.drop) {
  call x86_intrcc void @func(ptr byval(i32) %k, i32 %other.keep, i32 %other.drop)
  ret void
}

; INTERESTING-LABEL: void @keep_none(
; REDUCED-LABEL: define x86_intrcc void @keep_none()
define x86_intrcc void @keep_none(ptr byval(i32) %k, i32 %other0, float %other1) {
  store i32 %other0, ptr @gv
  store float %other1, ptr @gv
  ret void
}
