; RUN: not llc -mtriple=amdgcn -mcpu=gfx90a -filetype=null %s 2>&1 | FileCheck %s

; Make sure illegal type uses are correctly diagnosed

; CHECK: error: couldn't allocate input reg for constraint 'VA'
define void @use_A_i8(i8 %x) {
  call void asm sideeffect "; use $0", "^VA"(i8 %x)
  ret void
}

; CHECK: error: couldn't allocate output register for constraint 'VA'
define i8 @def_A_i8() {
  %ret = call i8 asm sideeffect "; def $0", "=^VA"()
  ret i8 %ret
}

; CHECK: error: couldn't allocate input reg for constraint 'VA'
define void @use_A_i1(i1 %x) {
  call void asm sideeffect "; use $0", "^VA"(i1 %x)
  ret void
}

; CHECK: error: couldn't allocate output register for constraint 'VA'
define i1 @def_A_i1() {
  %ret = call i1 asm sideeffect "; def $0", "=^VA"()
  ret i1 %ret
}
