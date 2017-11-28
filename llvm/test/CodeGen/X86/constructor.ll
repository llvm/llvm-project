; RUN: llc -mtriple x86_64-pc-linux -use-ctors < %s | FileCheck --check-prefix=CTOR %s
; RUN: llc -mtriple x86_64-unknown-freebsd -use-ctors < %s | FileCheck --check-prefix=CTOR %s
; RUN: llc -mtriple x86_64-pc-solaris2.11 -use-ctors < %s | FileCheck --check-prefix=CTOR %s
; RUN: llc -mtriple x86_64-pc-linux < %s | FileCheck --check-prefix=INIT-ARRAY %s
; RUN: llc -mtriple x86_64-unknown-freebsd < %s | FileCheck --check-prefix=INIT-ARRAY %s
; RUN: llc -mtriple x86_64-pc-solaris2.11 < %s | FileCheck --check-prefix=INIT-ARRAY %s
; RUN: llc -mtriple x86_64-unknown-nacl < %s | FileCheck --check-prefix=NACL %s
; RUN: llc -mtriple i586-intel-elfiamcu -use-ctors < %s | FileCheck %s --check-prefix=MCU-CTORS
; RUN: llc -mtriple i586-intel-elfiamcu < %s | FileCheck %s --check-prefix=MCU-INIT-ARRAY
; RUN: llc -mtriple x86_64-win32-gnu < %s | FileCheck --check-prefix=COFF-CTOR %s
@llvm.global_ctors = appending global [3 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @f, i8* null}, { i32, void ()*, i8* } { i32 15, void ()* @g, i8* @v }, { i32, void ()*, i8* } { i32 55555, void ()* @h, i8* @v }]

@v = weak_odr global i8 0

define void @f() {
entry:
  ret void
}

define void @g() {
entry:
  ret void
}

define void @h() {
entry:
  ret void
}

; CTOR:		.section	.ctors.65520,"aGw",@progbits,v,comdat
; CTOR-NEXT:	.p2align	3
; CTOR-NEXT:	.quad	g
; CTOR-NEXT:	.section	.ctors.09980,"aGw",@progbits,v,comdat
; CTOR-NEXT:	.p2align	3
; CTOR-NEXT:	.quad	h
; CTOR-NEXT:	.section	.ctors,"aw",@progbits
; CTOR-NEXT:	.p2align	3
; CTOR-NEXT:	.quad	f

; INIT-ARRAY:		.section	.init_array.15,"aGw",@init_array,v,comdat
; INIT-ARRAY-NEXT:	.p2align	3
; INIT-ARRAY-NEXT:	.quad	g
; INIT-ARRAY-NEXT:	.section	.init_array.55555,"aGw",@init_array,v,comdat
; INIT-ARRAY-NEXT:	.p2align	3
; INIT-ARRAY-NEXT:	.quad	h
; INIT-ARRAY-NEXT:	.section	.init_array,"aw",@init_array
; INIT-ARRAY-NEXT:	.p2align	3
; INIT-ARRAY-NEXT:	.quad	f

; NACL:		.section	.init_array.15,"aGw",@init_array,v,comdat
; NACL-NEXT:	.p2align	2
; NACL-NEXT:	.long	g
; NACL-NEXT:	.section	.init_array.55555,"aGw",@init_array,v,comdat
; NACL-NEXT:	.p2align	2
; NACL-NEXT:	.long	h
; NACL-NEXT:	.section	.init_array,"aw",@init_array
; NACL-NEXT:	.p2align	2
; NACL-NEXT:	.long	f

; MCU-CTORS:         .section        .ctors,"aw",@progbits
; MCU-INIT-ARRAY:    .section        .init_array,"aw",@init_array

; COFF-CTOR:		.section	.ctors.65520,"dw",associative,v
; COFF-CTOR-NEXT:	.p2align	3
; COFF-CTOR-NEXT:	.quad	g
; COFF-CTOR-NEXT:	.section	.ctors.09980,"dw",associative,v
; COFF-CTOR-NEXT:	.p2align	3
; COFF-CTOR-NEXT:	.quad	h
; COFF-CTOR-NEXT:	.section	.ctors,"dw"
; COFF-CTOR-NEXT:	.p2align	3
; COFF-CTOR-NEXT:	.quad	f
