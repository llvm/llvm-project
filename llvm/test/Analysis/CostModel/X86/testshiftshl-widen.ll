; RUN: llc -x86-experimental-vector-widening-legalization -mtriple=x86_64-apple-darwin -mcpu=core2 < %s | FileCheck --check-prefix=SSE2-CODEGEN %s
; RUN: opt -x86-experimental-vector-widening-legalization -mtriple=x86_64-apple-darwin -mcpu=core2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE2 %s

%shifttype = type <2 x i16>
define %shifttype @shift2i16(%shifttype %a, %shifttype %b) {
entry:
  ; SSE2-LABEL: shift2i16
  ; SSE2: cost of 32 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift2i16
  ; SSE2-CODEGEN: pmullw

  %0 = shl %shifttype %a , %b
  ret %shifttype %0
}

%shifttype4i16 = type <4 x i16>
define %shifttype4i16 @shift4i16(%shifttype4i16 %a, %shifttype4i16 %b) {
entry:
  ; SSE2-LABEL: shift4i16
  ; SSE2: cost of 32 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift4i16
  ; SSE2-CODEGEN: pmullw

  %0 = shl %shifttype4i16 %a , %b
  ret %shifttype4i16 %0
}

%shifttype8i16 = type <8 x i16>
define %shifttype8i16 @shift8i16(%shifttype8i16 %a, %shifttype8i16 %b) {
entry:
  ; SSE2-LABEL: shift8i16
  ; SSE2: cost of 32 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift8i16
  ; SSE2-CODEGEN: pmullw

  %0 = shl %shifttype8i16 %a , %b
  ret %shifttype8i16 %0
}

%shifttype16i16 = type <16 x i16>
define %shifttype16i16 @shift16i16(%shifttype16i16 %a, %shifttype16i16 %b) {
entry:
  ; SSE2-LABEL: shift16i16
  ; SSE2: cost of 64 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift16i16
  ; SSE2-CODEGEN: pmullw

  %0 = shl %shifttype16i16 %a , %b
  ret %shifttype16i16 %0
}

%shifttype32i16 = type <32 x i16>
define %shifttype32i16 @shift32i16(%shifttype32i16 %a, %shifttype32i16 %b) {
entry:
  ; SSE2-LABEL: shift32i16
  ; SSE2: cost of 128 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift32i16
  ; SSE2-CODEGEN: pmullw

  %0 = shl %shifttype32i16 %a , %b
  ret %shifttype32i16 %0
}

%shifttype2i32 = type <2 x i32>
define %shifttype2i32 @shift2i32(%shifttype2i32 %a, %shifttype2i32 %b) {
entry:
  ; SSE2-LABEL: shift2i32
  ; SSE2: cost of 10 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift2i32
  ; SSE2-CODEGEN: pmuludq

  %0 = shl %shifttype2i32 %a , %b
  ret %shifttype2i32 %0
}

%shifttype4i32 = type <4 x i32>
define %shifttype4i32 @shift4i32(%shifttype4i32 %a, %shifttype4i32 %b) {
entry:
  ; SSE2-LABEL: shift4i32
  ; SSE2: cost of 10 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift4i32
  ; SSE2-CODEGEN: pmuludq

  %0 = shl %shifttype4i32 %a , %b
  ret %shifttype4i32 %0
}

%shifttype8i32 = type <8 x i32>
define %shifttype8i32 @shift8i32(%shifttype8i32 %a, %shifttype8i32 %b) {
entry:
  ; SSE2-LABEL: shift8i32
  ; SSE2: cost of 20 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift8i32
  ; SSE2-CODEGEN: pmuludq

  %0 = shl %shifttype8i32 %a , %b
  ret %shifttype8i32 %0
}

%shifttype16i32 = type <16 x i32>
define %shifttype16i32 @shift16i32(%shifttype16i32 %a, %shifttype16i32 %b) {
entry:
  ; SSE2-LABEL: shift16i32
  ; SSE2: cost of 40 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift16i32
  ; SSE2-CODEGEN: pmuludq

  %0 = shl %shifttype16i32 %a , %b
  ret %shifttype16i32 %0
}

%shifttype32i32 = type <32 x i32>
define %shifttype32i32 @shift32i32(%shifttype32i32 %a, %shifttype32i32 %b) {
entry:
  ; SSE2-LABEL: shift32i32
  ; SSE2: cost of 80 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift32i32
  ; SSE2-CODEGEN: pmuludq

  %0 = shl %shifttype32i32 %a , %b
  ret %shifttype32i32 %0
}

%shifttype2i64 = type <2 x i64>
define %shifttype2i64 @shift2i64(%shifttype2i64 %a, %shifttype2i64 %b) {
entry:
  ; SSE2-LABEL: shift2i64
  ; SSE2: cost of 4 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift2i64
  ; SSE2-CODEGEN: psllq

  %0 = shl %shifttype2i64 %a , %b
  ret %shifttype2i64 %0
}

%shifttype4i64 = type <4 x i64>
define %shifttype4i64 @shift4i64(%shifttype4i64 %a, %shifttype4i64 %b) {
entry:
  ; SSE2-LABEL: shift4i64
  ; SSE2: cost of 8 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift4i64
  ; SSE2-CODEGEN: psllq

  %0 = shl %shifttype4i64 %a , %b
  ret %shifttype4i64 %0
}

%shifttype8i64 = type <8 x i64>
define %shifttype8i64 @shift8i64(%shifttype8i64 %a, %shifttype8i64 %b) {
entry:
  ; SSE2-LABEL: shift8i64
  ; SSE2: cost of 16 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift8i64
  ; SSE2-CODEGEN: psllq

  %0 = shl %shifttype8i64 %a , %b
  ret %shifttype8i64 %0
}

%shifttype16i64 = type <16 x i64>
define %shifttype16i64 @shift16i64(%shifttype16i64 %a, %shifttype16i64 %b) {
entry:
  ; SSE2-LABEL: shift16i64
  ; SSE2: cost of 32 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift16i64
  ; SSE2-CODEGEN: psllq

  %0 = shl %shifttype16i64 %a , %b
  ret %shifttype16i64 %0
}

%shifttype32i64 = type <32 x i64>
define %shifttype32i64 @shift32i64(%shifttype32i64 %a, %shifttype32i64 %b) {
entry:
  ; SSE2-LABEL: shift32i64
  ; SSE2: cost of 64 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift32i64
  ; SSE2-CODEGEN: psllq

  %0 = shl %shifttype32i64 %a , %b
  ret %shifttype32i64 %0
}

%shifttype2i8 = type <2 x i8>
define %shifttype2i8 @shift2i8(%shifttype2i8 %a, %shifttype2i8 %b) {
entry:
  ; SSE2-LABEL: shift2i8
  ; SSE2: cost of 26 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift2i8
  ; SSE2-CODEGEN: psllw

  %0 = shl %shifttype2i8 %a , %b
  ret %shifttype2i8 %0
}

%shifttype4i8 = type <4 x i8>
define %shifttype4i8 @shift4i8(%shifttype4i8 %a, %shifttype4i8 %b) {
entry:
  ; SSE2-LABEL: shift4i8
  ; SSE2: cost of 26 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift4i8
  ; SSE2-CODEGEN: psllw

  %0 = shl %shifttype4i8 %a , %b
  ret %shifttype4i8 %0
}

%shifttype8i8 = type <8 x i8>
define %shifttype8i8 @shift8i8(%shifttype8i8 %a, %shifttype8i8 %b) {
entry:
  ; SSE2-LABEL: shift8i8
  ; SSE2: cost of 26 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift8i8
  ; SSE2-CODEGEN: psllw

  %0 = shl %shifttype8i8 %a , %b
  ret %shifttype8i8 %0
}

%shifttype16i8 = type <16 x i8>
define %shifttype16i8 @shift16i8(%shifttype16i8 %a, %shifttype16i8 %b) {
entry:
  ; SSE2-LABEL: shift16i8
  ; SSE2: cost of 26 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift16i8
  ; SSE2-CODEGEN: psllw

  %0 = shl %shifttype16i8 %a , %b
  ret %shifttype16i8 %0
}

%shifttype32i8 = type <32 x i8>
define %shifttype32i8 @shift32i8(%shifttype32i8 %a, %shifttype32i8 %b) {
entry:
  ; SSE2-LABEL: shift32i8
  ; SSE2: cost of 52 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift32i8
  ; SSE2-CODEGEN: psllw

  %0 = shl %shifttype32i8 %a , %b
  ret %shifttype32i8 %0
}

; Test shift by a constant vector.

%shifttypec = type <2 x i16>
define %shifttypec @shift2i16const(%shifttypec %a, %shifttypec %b) {
entry:
  ; SSE2-LABEL: shift2i16const
  ; SSE2: cost of 1 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift2i16const
  ; SSE2-CODEGEN: psllw $3

  %0 = shl %shifttypec %a , <i16 3, i16 3>
  ret %shifttypec %0
}

%shifttypec4i16 = type <4 x i16>
define %shifttypec4i16 @shift4i16const(%shifttypec4i16 %a, %shifttypec4i16 %b) {
entry:
  ; SSE2-LABEL: shift4i16const
  ; SSE2: cost of 1 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift4i16const
  ; SSE2-CODEGEN: psllw $3

  %0 = shl %shifttypec4i16 %a , <i16 3, i16 3, i16 3, i16 3>
  ret %shifttypec4i16 %0
}

%shifttypec8i16 = type <8 x i16>
define %shifttypec8i16 @shift8i16const(%shifttypec8i16 %a, %shifttypec8i16 %b) {
entry:
  ; SSE2-LABEL: shift8i16const
  ; SSE2: cost of 1 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift8i16const
  ; SSE2-CODEGEN: psllw $3

  %0 = shl %shifttypec8i16 %a , <i16 3, i16 3, i16 3, i16 3,
                                  i16 3, i16 3, i16 3, i16 3>
  ret %shifttypec8i16 %0
}

%shifttypec16i16 = type <16 x i16>
define %shifttypec16i16 @shift16i16const(%shifttypec16i16 %a,
                                         %shifttypec16i16 %b) {
entry:
  ; SSE2-LABEL: shift16i16const
  ; SSE2: cost of 2 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift16i16const
  ; SSE2-CODEGEN: psllw $3

  %0 = shl %shifttypec16i16 %a , <i16 3, i16 3, i16 3, i16 3,
                                   i16 3, i16 3, i16 3, i16 3,
                                   i16 3, i16 3, i16 3, i16 3,
                                   i16 3, i16 3, i16 3, i16 3>
  ret %shifttypec16i16 %0
}

%shifttypec32i16 = type <32 x i16>
define %shifttypec32i16 @shift32i16const(%shifttypec32i16 %a,
                                        %shifttypec32i16 %b) {
entry:
  ; SSE2-LABEL: shift32i16const
  ; SSE2: cost of 4 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift32i16const
  ; SSE2-CODEGEN: psllw $3

  %0 = shl %shifttypec32i16 %a , <i16 3, i16 3, i16 3, i16 3,
                                   i16 3, i16 3, i16 3, i16 3,
                                   i16 3, i16 3, i16 3, i16 3,
                                   i16 3, i16 3, i16 3, i16 3,
                                   i16 3, i16 3, i16 3, i16 3,
                                   i16 3, i16 3, i16 3, i16 3,
                                   i16 3, i16 3, i16 3, i16 3,
                                   i16 3, i16 3, i16 3, i16 3>
  ret %shifttypec32i16 %0
}

%shifttypec2i32 = type <2 x i32>
define %shifttypec2i32 @shift2i32c(%shifttypec2i32 %a, %shifttypec2i32 %b) {
entry:
  ; SSE2-LABEL: shift2i32c
  ; SSE2: cost of 1 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift2i32c
  ; SSE2-CODEGEN: pslld $3

  %0 = shl %shifttypec2i32 %a , <i32 3, i32 3>
  ret %shifttypec2i32 %0
}

%shifttypec4i32 = type <4 x i32>
define %shifttypec4i32 @shift4i32c(%shifttypec4i32 %a, %shifttypec4i32 %b) {
entry:
  ; SSE2-LABEL: shift4i32c
  ; SSE2: cost of 1 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift4i32c
  ; SSE2-CODEGEN: pslld $3

  %0 = shl %shifttypec4i32 %a , <i32 3, i32 3, i32 3, i32 3>
  ret %shifttypec4i32 %0
}

%shifttypec8i32 = type <8 x i32>
define %shifttypec8i32 @shift8i32c(%shifttypec8i32 %a, %shifttypec8i32 %b) {
entry:
  ; SSE2-LABEL: shift8i32c
  ; SSE2: cost of 2 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift8i32c
  ; SSE2-CODEGEN: pslld $3

  %0 = shl %shifttypec8i32 %a , <i32 3, i32 3, i32 3, i32 3,
                                  i32 3, i32 3, i32 3, i32 3>
  ret %shifttypec8i32 %0
}

%shifttypec16i32 = type <16 x i32>
define %shifttypec16i32 @shift16i32c(%shifttypec16i32 %a, %shifttypec16i32 %b) {
entry:
  ; SSE2-LABEL: shift16i32c
  ; SSE2: cost of 4 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift16i32c
  ; SSE2-CODEGEN: pslld $3

  %0 = shl %shifttypec16i32 %a , <i32 3, i32 3, i32 3, i32 3,
                                   i32 3, i32 3, i32 3, i32 3,
                                   i32 3, i32 3, i32 3, i32 3,
                                   i32 3, i32 3, i32 3, i32 3>
  ret %shifttypec16i32 %0
}

%shifttypec32i32 = type <32 x i32>
define %shifttypec32i32 @shift32i32c(%shifttypec32i32 %a, %shifttypec32i32 %b) {
entry:
  ; SSE2-LABEL: shift32i32c
  ; SSE2: cost of 8 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift32i32c
  ; SSE2-CODEGEN: pslld $3
  %0 = shl %shifttypec32i32 %a , <i32 3, i32 3, i32 3, i32 3,
                                   i32 3, i32 3, i32 3, i32 3,
                                   i32 3, i32 3, i32 3, i32 3,
                                   i32 3, i32 3, i32 3, i32 3,
                                   i32 3, i32 3, i32 3, i32 3,
                                   i32 3, i32 3, i32 3, i32 3,
                                   i32 3, i32 3, i32 3, i32 3,
                                   i32 3, i32 3, i32 3, i32 3>
  ret %shifttypec32i32 %0
}

%shifttypec2i64 = type <2 x i64>
define %shifttypec2i64 @shift2i64c(%shifttypec2i64 %a, %shifttypec2i64 %b) {
entry:
  ; SSE2-LABEL: shift2i64c
  ; SSE2: cost of 1 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift2i64c
  ; SSE2-CODEGEN: psllq $3

  %0 = shl %shifttypec2i64 %a , <i64 3, i64 3>
  ret %shifttypec2i64 %0
}

%shifttypec4i64 = type <4 x i64>
define %shifttypec4i64 @shift4i64c(%shifttypec4i64 %a, %shifttypec4i64 %b) {
entry:
  ; SSE2-LABEL: shift4i64c
  ; SSE2: cost of 2 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift4i64c
  ; SSE2-CODEGEN: psllq $3

  %0 = shl %shifttypec4i64 %a , <i64 3, i64 3, i64 3, i64 3>
  ret %shifttypec4i64 %0
}

%shifttypec8i64 = type <8 x i64>
define %shifttypec8i64 @shift8i64c(%shifttypec8i64 %a, %shifttypec8i64 %b) {
entry:
  ; SSE2-LABEL: shift8i64c
  ; SSE2: cost of 4 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift8i64c
  ; SSE2-CODEGEN: psllq $3

 %0 = shl %shifttypec8i64 %a , <i64 3, i64 3, i64 3, i64 3,
                                 i64 3, i64 3, i64 3, i64 3>
  ret %shifttypec8i64 %0
}

%shifttypec16i64 = type <16 x i64>
define %shifttypec16i64 @shift16i64c(%shifttypec16i64 %a, %shifttypec16i64 %b) {
entry:
  ; SSE2-LABEL: shift16i64c
  ; SSE2: cost of 8 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift16i64c
  ; SSE2-CODEGEN: psllq $3

  %0 = shl %shifttypec16i64 %a , <i64 3, i64 3, i64 3, i64 3,
                                   i64 3, i64 3, i64 3, i64 3,
                                   i64 3, i64 3, i64 3, i64 3,
                                   i64 3, i64 3, i64 3, i64 3>
  ret %shifttypec16i64 %0
}

%shifttypec32i64 = type <32 x i64>
define %shifttypec32i64 @shift32i64c(%shifttypec32i64 %a, %shifttypec32i64 %b) {
entry:
  ; SSE2-LABEL: shift32i64c
  ; SSE2: cost of 16 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift32i64c
  ; SSE2-CODEGEN: psllq $3

  %0 = shl %shifttypec32i64 %a ,<i64 3, i64 3, i64 3, i64 3,
                                  i64 3, i64 3, i64 3, i64 3,
                                  i64 3, i64 3, i64 3, i64 3,
                                  i64 3, i64 3, i64 3, i64 3,
                                  i64 3, i64 3, i64 3, i64 3,
                                  i64 3, i64 3, i64 3, i64 3,
                                  i64 3, i64 3, i64 3, i64 3,
                                  i64 3, i64 3, i64 3, i64 3>
  ret %shifttypec32i64 %0
}

%shifttypec2i8 = type <2 x i8>
define %shifttypec2i8 @shift2i8c(%shifttypec2i8 %a, %shifttypec2i8 %b) {
entry:
  ; SSE2-LABEL: shift2i8c
  ; SSE2: cost of 2 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift2i8c
  ; SSE2-CODEGEN: psllw $3

  %0 = shl %shifttypec2i8 %a , <i8 3, i8 3>
  ret %shifttypec2i8 %0
}

%shifttypec4i8 = type <4 x i8>
define %shifttypec4i8 @shift4i8c(%shifttypec4i8 %a, %shifttypec4i8 %b) {
entry:
  ; SSE2-LABEL: shift4i8c
  ; SSE2: cost of 2 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift4i8c
  ; SSE2-CODEGEN: psllw $3

  %0 = shl %shifttypec4i8 %a , <i8 3, i8 3, i8 3, i8 3>
  ret %shifttypec4i8 %0
}

%shifttypec8i8 = type <8 x i8>
define %shifttypec8i8 @shift8i8c(%shifttypec8i8 %a, %shifttypec8i8 %b) {
entry:
  ; SSE2-LABEL: shift8i8c
  ; SSE2: cost of 2 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift8i8c
  ; SSE2-CODEGEN: psllw $3

  %0 = shl %shifttypec8i8 %a , <i8 3, i8 3, i8 3, i8 3,
                                 i8 3, i8 3, i8 3, i8 3>
  ret %shifttypec8i8 %0
}

%shifttypec16i8 = type <16 x i8>
define %shifttypec16i8 @shift16i8c(%shifttypec16i8 %a, %shifttypec16i8 %b) {
entry:
  ; SSE2-LABEL: shift16i8c
  ; SSE2: cost of 2 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift16i8c
  ; SSE2-CODEGEN: psllw $3

  %0 = shl %shifttypec16i8 %a , <i8 3, i8 3, i8 3, i8 3,
                                  i8 3, i8 3, i8 3, i8 3,
                                  i8 3, i8 3, i8 3, i8 3,
                                  i8 3, i8 3, i8 3, i8 3>
  ret %shifttypec16i8 %0
}

%shifttypec32i8 = type <32 x i8>
define %shifttypec32i8 @shift32i8c(%shifttypec32i8 %a, %shifttypec32i8 %b) {
entry:
  ; SSE2-LABEL: shift32i8c
  ; SSE2: cost of 4 {{.*}} shl
  ; SSE2-CODEGEN-LABEL: shift32i8c
  ; SSE2-CODEGEN: psllw $3

  %0 = shl %shifttypec32i8 %a , <i8 3, i8 3, i8 3, i8 3,
                                  i8 3, i8 3, i8 3, i8 3,
                                  i8 3, i8 3, i8 3, i8 3,
                                  i8 3, i8 3, i8 3, i8 3,
                                  i8 3, i8 3, i8 3, i8 3,
                                  i8 3, i8 3, i8 3, i8 3,
                                  i8 3, i8 3, i8 3, i8 3,
                                  i8 3, i8 3, i8 3, i8 3>
  ret %shifttypec32i8 %0
}
