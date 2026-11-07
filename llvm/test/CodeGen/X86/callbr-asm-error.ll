; RUN: not llc -mtriple=x86_64-linux-gnu -verify-machineinstrs < %s 2>&1 | FileCheck %s

; After an error in a callbr's asm, nothing branches to its indirect targets,
; and their outputs are undefined. This uses the default -O2 pipeline, since
; BranchFolding is what rejected an indirect target with no INLINEASM_BR.

; CHECK: error: could not allocate output register for constraint '{foo}'
define i32 @bad_register() {
entry:
  %r = callbr i32 asm "", "={foo},!i"()
          to label %a [label %b]
a:
  ret i32 %r
b:
  ret i32 %r
}

; CHECK: error: invalid operand for inline asm constraint 'i'
define i32 @not_an_immediate(i32 %x) {
entry:
  %r = callbr i32 asm "", "=r,i,!i"(i32 %x)
          to label %a [label %b]
a:
  ret i32 %r
b:
  ret i32 %r
}

; A struct of outputs, where 'k' needs AVX-512.
; CHECK: error: could not allocate output register for constraint 'k'
define i32 @struct_output() {
entry:
  %r = callbr { i32, i32 } asm "", "=r,=kr,!i"()
          to label %a [label %b]
a:
  ret i32 0
b:
  %r1 = extractvalue { i32, i32 } %r, 1
  ret i32 %r1
}

; The indirect target doesn't use the output, so it has no landing pad.
; CHECK: error: could not allocate output register for constraint '{foo}'
define i32 @no_landing_pad(i1 %c) {
entry:
  br i1 %c, label %asm, label %a
asm:
  %r = callbr i32 asm "", "={foo},!i"()
          to label %a [label %b]
a:
  ret i32 0
b:
  ret i32 1
}

; A phi in an indirect target loses its value from the callbr's block.
; CHECK: error: could not allocate input reg for constraint '{foo}'
define i32 @phi_in_indirect_target(i1 %c) {
entry:
  br i1 %c, label %asm, label %b
asm:
  callbr void asm "", "{foo},!i"(i32 0)
          to label %a [label %b]
a:
  ret i32 0
b:
  %p = phi i32 [ 1, %entry ], [ 2, %asm ]
  ret i32 %p
}
