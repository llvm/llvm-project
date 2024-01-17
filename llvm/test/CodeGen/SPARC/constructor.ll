; RUN: llc -mtriple sparc-sun-solaris2.11 -use-ctors < %s | FileCheck --check-prefix=CTOR %s
; RUN: llc -mtriple sparc-sun-solaris2.11 < %s | FileCheck --check-prefix=INIT-ARRAY %s
@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @f, i8* null}, { i32, void ()*, i8* } { i32 15, void ()* @g, i8* @v }]

@v = weak_odr global i8 0

define void @f() {
entry:
  ret void
}

define void @g() {
entry:
  ret void
}

; CTOR:      .section      .ctors,"aw"
; CTOR-NEXT: .p2align      2
; CTOR-NEXT: .word  f
; CTOR-NEXT: .section      .ctors.65520,"awG",@progbits,v,comdat{{$}}
; CTOR-NEXT: .p2align      2
; CTOR-NEXT: .word  g

; INIT-ARRAY:    .section  .init_array.15,"awG",@init_array,v,comdat{{$}}
; INIT-ARRAY-NEXT: .p2align  2
; INIT-ARRAY-NEXT: .word g
; INIT-ARRAY-NEXT: .section  .init_array,"aw"
; INIT-ARRAY-NEXT: .p2align  2
; INIT-ARRAY-NEXT: .word f
