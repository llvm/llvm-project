; RUN: llc -mtriple x86_64-pc-linux < %s | FileCheck --check-prefix=CTOR %s
; RUN: llc -mtriple x86_64-pc-linux -use-init-array < %s | FileCheck --check-prefix=INIT-ARRAY %s
@llvm.global_ctors = appending global [2 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @f }, { i32, void ()* } { i32 15, void ()* @g }]

define void @f() {
entry:
  ret void
}

define void @g() {
entry:
  ret void
}

; CTOR:		.section	.ctors.65520,"aw",@progbits
; CTOR-NEXT:	.align	8
; CTOR-NEXT:	.quad	g
; CTOR-NEXT:	.section	.ctors,"aw",@progbits
; CTOR-NEXT:	.align	8
; CTOR-NEXT:	.quad	f

; INIT-ARRAY:		.section	.init_array.15,"aw",@init_array
; INIT-ARRAY-NEXT:	.align	8
; INIT-ARRAY-NEXT:	.quad	g
; INIT-ARRAY-NEXT:	.section	.init_array,"aw",@init_array
; INIT-ARRAY-NEXT:	.align	8
; INIT-ARRAY-NEXT:	.quad	f
