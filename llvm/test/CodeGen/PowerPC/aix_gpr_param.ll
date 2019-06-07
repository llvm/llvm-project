; RUN: llc -mtriple powerpc-ibm-aix-xcoff -stop-after=machine-cp < %s | \
; RUN: FileCheck --check-prefix=32BIT %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp < %s | \
; RUN: FileCheck --check-prefix=64BIT %s

define void @call_test_char() {
entry:
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 97
; 32BIT: BL_NOP <mcsymbol .test_char>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 97
; 64BIT: BL8_NOP <mcsymbol .test_char>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  call void @test_char(i8 signext 97)
  ret void
}

define void @call_test_chars() {
entry:
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 97
; 32BIT: $r4 = LI 97
; 32BIT: $r5 = LI 97
; 32BIT: $r6 = LI 97
; 32BIT: BL_NOP <mcsymbol .test_chars>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit killed $r4, implicit killed $r5, implicit killed $r6, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 97
; 64BIT: $x4 = LI8 97
; 64BIT: $x5 = LI8 97
; 64BIT: $x6 = LI8 97
; 64BIT: BL8_NOP <mcsymbol .test_chars>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit killed $x4, implicit killed $x5, implicit killed $x6, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  call void @test_chars(i8 signext 97, i8 signext 97, i8 signext 97, i8 signext 97)
  ret void
}

define void @call_test_chars_mix() {
entry:
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 97
; 32BIT: $r4 = LI 225
; 32BIT: $r5 = LI 97
; 32BIT: $r6 = LI -31
; 32BIT: BL_NOP <mcsymbol .test_chars_mix>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit killed $r4, implicit killed $r5, implicit killed $r6, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 97
; 64BIT: $x4 = LI8 225
; 64BIT: $x5 = LI8 97
; 64BIT: $x6 = LI8 -31
; 64BIT: BL8_NOP <mcsymbol .test_chars_mix>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit killed $x4, implicit killed $x5, implicit killed $x6, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  call void @test_chars_mix(i8 signext 97, i8 zeroext -31, i8 zeroext 97, i8 signext -31)
  ret void
}

define void @call_test_int() {
entry:
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 1
; 32BIT: BL_NOP <mcsymbol .test_int>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 1
; 64BIT: BL8_NOP <mcsymbol .test_int>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  call void @test_int(i32 1)
  ret void
}

define void @call_test_ints() {
entry:
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 1
; 32BIT: $r4 = LI 1
; 32BIT: $r5 = LI 1
; 32BIT: $r6 = LI 1
; 32BIT: $r7 = LI 1
; 32BIT: $r8 = LI 1
; 32BIT: $r9 = LI 1
; 32BIT: $r10 = LI 1
; 32BIT: BL_NOP <mcsymbol .test_ints>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit killed $r4, implicit killed $r5, implicit killed $r6, implicit killed $r7, implicit killed $r8, implicit killed $r9, implicit killed $r10, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 1
; 64BIT: $x4 = LI8 1
; 64BIT: $x5 = LI8 1
; 64BIT: $x6 = LI8 1
; 64BIT: $x7 = LI8 1
; 64BIT: $x8 = LI8 1
; 64BIT: $x9 = LI8 1
; 64BIT: $x10 = LI8 1
; 64BIT: BL8_NOP <mcsymbol .test_ints>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit killed $x4, implicit killed $x5, implicit killed $x6, implicit killed $x7, implicit killed $x8, implicit killed $x9, implicit killed $x10, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  call void @test_ints(i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1)
  ret void
}

define void @call_test_ints_64bit() {
entry:
; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: renamable $x3 = LI8 1
; 64BIT: renamable $x5 = RLDICR killed renamable $x3, 31, 32
; 64BIT: $x3 = LI8 1
; 64BIT: $x4 = LI8 1
; 64BIT: $x6 = LIS8 32768
; 64BIT: $x7 = LI8 1
; 64BIT: $x8 = LI8 1
; 64BIT: $x9 = LI8 1
; 64BIT: $x10 = LI8 1
; 64BIT: BL8_NOP <mcsymbol .test_ints_64bit>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit killed $x4, implicit $x5, implicit killed $x6, implicit killed $x7, implicit killed $x8, implicit killed $x9, implicit killed $x10, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  call void @test_ints_64bit(i32 signext 1, i32 zeroext 1, i32 zeroext 2147483648, i32 signext -2147483648, i32 signext 1, i32 signext 1, i32 signext 1, i32 signext 1)
  ret void
}

define void @call_test_i1() {
entry:
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 1
; 32BIT: BL_NOP <mcsymbol .test_i1>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 1
; 64BIT: BL8_NOP <mcsymbol .test_i1>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  call void @test_i1(i1 1)
  ret void
}

define void @call_test_i64() {
entry:
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 0
; 32BIT: $r4 = LI 1
; 32BIT: BL_NOP <mcsymbol .test_i64>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit killed $r4, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 1
; 64BIT: BL8_NOP <mcsymbol .test_i64>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  call void @test_i64(i64 1)
  ret void
}

define void @call_test_int_ptr() {
entry:
  %b = alloca i32, align 4
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: renamable $r3 = ADDI %stack.0.b, 0
; 32BIT: BL_NOP <mcsymbol .test_int_ptr>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: renamable $x3 = ADDI8 %stack.0.b, 0
; 64BIT: BL8_NOP <mcsymbol .test_int_ptr>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

  store i32 0, i32* %b, align 4
  call void @test_int_ptr(i32* %b)
  ret void
}

declare void @test_char(i8 signext)

declare void @test_chars(i8 signext, i8 signext, i8 signext, i8 signext)

declare void @test_chars_mix(i8 signext, i8 zeroext, i8 zeroext, i8 signext)

declare void @test_int(i32)

declare void @test_ints(i32, i32, i32, i32, i32, i32, i32, i32)

declare void @test_ints_64bit(i32 signext, i32 zeroext, i32 zeroext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext)

declare void @test_i1(i1)

declare void @test_i64(i64)

declare void @test_int_ptr(i32*)
