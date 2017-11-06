; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O0 -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-BE
; RUN: llc -O0 -mcpu=pwr9 -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-BE

; The following testcases take one halfword element from the second vector and
; inserts it at various locations in the first vector
define <8 x i16> @shuffle_vector_halfword_0_8(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_0_8
; CHECK: vsldoi 3, 3, 3, 8
; CHECK: vinserth 2, 3, 14
; CHECK-BE-LABEL: shuffle_vector_halfword_0_8
; CHECK-BE: vsldoi 3, 3, 3, 10
; CHECK-BE: vinserth 2, 3, 0
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_1_15(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_1_15
; CHECK: vsldoi 3, 3, 3, 10
; CHECK: vinserth 2, 3, 12
; CHECK-BE-LABEL: shuffle_vector_halfword_1_15
; CHECK-BE: vsldoi 3, 3, 3, 8
; CHECK-BE: vinserth 2, 3, 2
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 15, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_2_9(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_2_9
; CHECK: vsldoi 3, 3, 3, 6
; CHECK: vinserth 2, 3, 10
; CHECK-BE-LABEL: shuffle_vector_halfword_2_9
; CHECK-BE: vsldoi 3, 3, 3, 12
; CHECK-BE: vinserth 2, 3, 4
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 9, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_3_13(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_3_13
; CHECK: vsldoi 3, 3, 3, 14
; CHECK: vinserth 2, 3, 8
; CHECK-BE-LABEL: shuffle_vector_halfword_3_13
; CHECK-BE: vsldoi 3, 3, 3, 4
; CHECK-BE: vinserth 2, 3, 6
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 13, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_4_10(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_4_10
; CHECK: vsldoi 3, 3, 3, 4
; CHECK: vinserth 2, 3, 6
; CHECK-BE-LABEL: shuffle_vector_halfword_4_10
; CHECK-BE: vsldoi 3, 3, 3, 14
; CHECK-BE: vinserth 2, 3, 8
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 10, i32 5, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_5_14(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_5_14
; CHECK: vsldoi 3, 3, 3, 12
; CHECK: vinserth 2, 3, 4
; CHECK-BE-LABEL: shuffle_vector_halfword_5_14
; CHECK-BE: vsldoi 3, 3, 3, 6
; CHECK-BE: vinserth 2, 3, 10
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 14, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_6_11(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_6_11
; CHECK: vsldoi 3, 3, 3, 2
; CHECK: vinserth 2, 3, 2
; CHECK-BE-LABEL: shuffle_vector_halfword_6_11
; CHECK-BE: vinserth 2, 3, 12
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 11, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_7_12(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_7_12
; CHECK: vinserth 2, 3, 0
; CHECK-BE-LABEL: shuffle_vector_halfword_7_12
; CHECK-BE: vsldoi 3, 3, 3, 2
; CHECK-BE: vinserth 2, 3, 14
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 12>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_8_1(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_8_1
; CHECK: vsldoi 2, 2, 2, 6
; CHECK: vinserth 3, 2, 14
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_halfword_8_1
; CHECK-BE: vsldoi 2, 2, 2, 12
; CHECK-BE: vinserth 3, 2, 0
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i16> %vecins
}

; The following testcases take one halfword element from the first vector and
; inserts it at various locations in the second vector
define <8 x i16> @shuffle_vector_halfword_9_7(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_9_7
; CHECK: vsldoi 2, 2, 2, 10
; CHECK: vinserth 3, 2, 12
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_halfword_9_7
; CHECK-BE: vsldoi 2, 2, 2, 8
; CHECK-BE: vinserth 3, 2, 2
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 8, i32 7, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_10_4(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_10_4
; CHECK: vinserth 3, 2, 10
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_halfword_10_4
; CHECK-BE: vsldoi 2, 2, 2, 2
; CHECK-BE: vinserth 3, 2, 4
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 8, i32 9, i32 4, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_11_2(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_11_2
; CHECK: vsldoi 2, 2, 2, 4
; CHECK: vinserth 3, 2, 8
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_halfword_11_2
; CHECK-BE: vsldoi 2, 2, 2, 14
; CHECK-BE: vinserth 3, 2, 6
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 2, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_12_6(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_12_6
; CHECK: vsldoi 2, 2, 2, 12
; CHECK: vinserth 3, 2, 6
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_halfword_12_6
; CHECK-BE: vsldoi 2, 2, 2, 6
; CHECK-BE: vinserth 3, 2, 8
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 6, i32 13, i32 14, i32 15>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_13_3(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_13_3
; CHECK: vsldoi 2, 2, 2, 2
; CHECK: vinserth 3, 2, 4
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_halfword_13_3
; CHECK-BE: vinserth 3, 2, 10
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 3, i32 14, i32 15>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_14_5(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_14_5
; CHECK: vsldoi 2, 2, 2, 14
; CHECK: vinserth 3, 2, 2
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_halfword_14_5
; CHECK-BE: vsldoi 2, 2, 2, 4
; CHECK-BE: vinserth 3, 2, 12
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 5, i32 15>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_15_0(<8 x i16> %a, <8 x i16> %b) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_15_0
; CHECK: vsldoi 2, 2, 2, 8
; CHECK: vinserth 3, 2, 0
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_halfword_15_0
; CHECK-BE: vsldoi 2, 2, 2, 10
; CHECK-BE: vinserth 3, 2, 14
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 0>
  ret <8 x i16> %vecins
}

; The following testcases use the same vector in both arguments of the
; shufflevector.  If halfword element 3 in BE mode(or 4 in LE mode) is the one
; we're attempting to insert, then we can use the vector insert instruction
define <8 x i16> @shuffle_vector_halfword_0_4(<8 x i16> %a) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_0_4
; CHECK: vinserth 2, 2, 14
; CHECK-BE-LABEL: shuffle_vector_halfword_0_4
; CHECK-BE-NOT: vinserth
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 4, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_1_3(<8 x i16> %a) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_1_3
; CHECK-NOT: vinserth
; CHECK-BE-LABEL: shuffle_vector_halfword_1_3
; CHECK-BE: vinserth 2, 2, 2
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 0, i32 3, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_2_3(<8 x i16> %a) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_2_3
; CHECK-NOT: vinserth
; CHECK-BE-LABEL: shuffle_vector_halfword_2_3
; CHECK-BE: vinserth 2, 2, 4
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 0, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_3_4(<8 x i16> %a) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_3_4
; CHECK: vinserth 2, 2, 8
; CHECK-BE-LABEL: shuffle_vector_halfword_3_4
; CHECK-BE-NOT: vinserth
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 4, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_4_3(<8 x i16> %a) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_4_3
; CHECK-NOT: vinserth
; CHECK-BE-LABEL: shuffle_vector_halfword_4_3
; CHECK-BE: vinserth 2, 2, 8
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 3, i32 5, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_5_3(<8 x i16> %a) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_5_3
; CHECK-NOT: vinserth
; CHECK-BE-LABEL: shuffle_vector_halfword_5_3
; CHECK-BE: vinserth 2, 2, 10
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 3, i32 6, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_6_4(<8 x i16> %a) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_6_4
; CHECK: vinserth 2, 2, 2
; CHECK-BE-LABEL: shuffle_vector_halfword_6_4
; CHECK-BE-NOT: vinserth
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 4, i32 7>
  ret <8 x i16> %vecins
}

define <8 x i16> @shuffle_vector_halfword_7_4(<8 x i16> %a) {
entry:
; CHECK-LABEL: shuffle_vector_halfword_7_4
; CHECK: vinserth 2, 2, 0
; CHECK-BE-LABEL: shuffle_vector_halfword_7_4
; CHECK-BE-NOT: vinserth
  %vecins = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 4>
  ret <8 x i16> %vecins
}

; The following testcases take one byte element from the second vector and
; inserts it at various locations in the first vector
define <16 x i8> @shuffle_vector_byte_0_16(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_0_16
; CHECK: vsldoi 3, 3, 3, 8
; CHECK: vinsertb 2, 3, 15
; CHECK-BE-LABEL: shuffle_vector_byte_0_16
; CHECK-BE: vsldoi 3, 3, 3, 9
; CHECK-BE: vinsertb 2, 3, 0
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_1_25(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_1_25
; CHECK: vsldoi 3, 3, 3, 15
; CHECK: vinsertb 2, 3, 14
; CHECK-BE-LABEL: shuffle_vector_byte_1_25
; CHECK-BE: vsldoi 3, 3, 3, 2
; CHECK-BE: vinsertb 2, 3, 1
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 25, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_2_18(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_2_18
; CHECK: vsldoi 3, 3, 3, 6
; CHECK: vinsertb 2, 3, 13
; CHECK-BE-LABEL: shuffle_vector_byte_2_18
; CHECK-BE: vsldoi 3, 3, 3, 11
; CHECK-BE: vinsertb 2, 3, 2
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 18, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_3_27(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_3_27
; CHECK: vsldoi 3, 3, 3, 13
; CHECK: vinsertb 2, 3, 12
; CHECK-BE-LABEL: shuffle_vector_byte_3_27
; CHECK-BE: vsldoi 3, 3, 3, 4
; CHECK-BE: vinsertb 2, 3, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 27, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_4_20(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_4_20
; CHECK: vsldoi 3, 3, 3, 4
; CHECK: vinsertb 2, 3, 11
; CHECK-BE-LABEL: shuffle_vector_byte_4_20
; CHECK-BE: vsldoi 3, 3, 3, 13
; CHECK-BE: vinsertb 2, 3, 4
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_5_29(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_5_29
; CHECK: vsldoi 3, 3, 3, 11
; CHECK: vinsertb 2, 3, 10
; CHECK-BE-LABEL: shuffle_vector_byte_5_29
; CHECK-BE: vsldoi 3, 3, 3, 6
; CHECK-BE: vinsertb 2, 3, 5
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 29, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_6_22(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_6_22
; CHECK: vsldoi 3, 3, 3, 2
; CHECK: vinsertb 2, 3, 9
; CHECK-BE-LABEL: shuffle_vector_byte_6_22
; CHECK-BE: vsldoi 3, 3, 3, 15
; CHECK-BE: vinsertb 2, 3, 6
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 22, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_7_31(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_7_31
; CHECK: vsldoi 3, 3, 3, 9
; CHECK: vinsertb 2, 3, 8
; CHECK-BE-LABEL: shuffle_vector_byte_7_31
; CHECK-BE: vsldoi 3, 3, 3, 8
; CHECK-BE: vinsertb 2, 3, 7
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 31, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_8_24(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_8_24
; CHECK: vinsertb 2, 3, 7
; CHECK-BE-LABEL: shuffle_vector_byte_8_24
; CHECK-BE: vsldoi 3, 3, 3, 1
; CHECK-BE: vinsertb 2, 3, 8
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_9_17(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_9_17
; CHECK: vsldoi 3, 3, 3, 7
; CHECK: vinsertb 2, 3, 6
; CHECK-BE-LABEL: shuffle_vector_byte_9_17
; CHECK-BE: vsldoi 3, 3, 3, 10
; CHECK-BE: vinsertb 2, 3, 9
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 17, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_10_26(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_10_26
; CHECK: vsldoi 3, 3, 3, 14
; CHECK: vinsertb 2, 3, 5
; CHECK-BE-LABEL: shuffle_vector_byte_10_26
; CHECK-BE: vsldoi 3, 3, 3, 3
; CHECK-BE: vinsertb 2, 3, 10
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 26, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_11_19(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_11_19
; CHECK: vsldoi 3, 3, 3, 5
; CHECK: vinsertb 2, 3, 4
; CHECK-BE-LABEL: shuffle_vector_byte_11_19
; CHECK-BE: vsldoi 3, 3, 3, 12
; CHECK-BE: vinsertb 2, 3, 11
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 19, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_12_28(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_12_28
; CHECK: vsldoi 3, 3, 3, 12
; CHECK: vinsertb 2, 3, 3
; CHECK-BE-LABEL: shuffle_vector_byte_12_28
; CHECK-BE: vsldoi 3, 3, 3, 5
; CHECK-BE: vinsertb 2, 3, 12
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 28, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_13_21(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_13_21
; CHECK: vsldoi 3, 3, 3, 3
; CHECK: vinsertb 2, 3, 2
; CHECK-BE-LABEL: shuffle_vector_byte_13_21
; CHECK-BE: vsldoi 3, 3, 3, 14
; CHECK-BE: vinsertb 2, 3, 13
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 21, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_14_30(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_14_30
; CHECK: vsldoi 3, 3, 3, 10
; CHECK: vinsertb 2, 3, 1
; CHECK-BE-LABEL: shuffle_vector_byte_14_30
; CHECK-BE: vsldoi 3, 3, 3, 7
; CHECK-BE: vinsertb 2, 3, 14
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 30, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_15_23(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_15_23
; CHECK: vsldoi 3, 3, 3, 1
; CHECK: vinsertb 2, 3, 0
; CHECK-BE-LABEL: shuffle_vector_byte_15_23
; CHECK-BE: vinsertb 2, 3, 15
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 23>
  ret <16 x i8> %vecins
}

; The following testcases take one byte element from the first vector and
; inserts it at various locations in the second vector
define <16 x i8> @shuffle_vector_byte_16_8(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_16_8
; CHECK: vinsertb 3, 2, 15
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_16_8
; CHECK-BE: vsldoi 2, 2, 2, 1
; CHECK-BE: vinsertb 3, 2, 0
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_17_1(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_17_1
; CHECK: vsldoi 2, 2, 2, 7
; CHECK: vinsertb 3, 2, 14
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_17_1
; CHECK-BE: vsldoi 2, 2, 2, 10
; CHECK-BE: vinsertb 3, 2, 1
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_18_10(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_18_10
; CHECK: vsldoi 2, 2, 2, 14
; CHECK: vinsertb 3, 2, 13
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_18_10
; CHECK-BE: vsldoi 2, 2, 2, 3
; CHECK-BE: vinsertb 3, 2, 2
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 10, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_19_3(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_19_3
; CHECK: vsldoi 2, 2, 2, 5
; CHECK: vinsertb 3, 2, 12
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_19_3
; CHECK-BE: vsldoi 2, 2, 2, 12
; CHECK-BE: vinsertb 3, 2, 3
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_20_12(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_20_12
; CHECK: vsldoi 2, 2, 2, 12
; CHECK: vinsertb 3, 2, 11
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_20_12
; CHECK-BE: vsldoi 2, 2, 2, 5
; CHECK-BE: vinsertb 3, 2, 4
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 12, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_21_5(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_21_5
; CHECK: vsldoi 2, 2, 2, 3
; CHECK: vinsertb 3, 2, 10
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_21_5
; CHECK-BE: vsldoi 2, 2, 2, 14
; CHECK-BE: vinsertb 3, 2, 5
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 5, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_22_14(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_22_14
; CHECK: vsldoi 2, 2, 2, 10
; CHECK: vinsertb 3, 2, 9
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_22_14
; CHECK-BE: vsldoi 2, 2, 2, 7
; CHECK-BE: vinsertb 3, 2, 6
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 14, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_23_7(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_23_7
; CHECK: vsldoi 2, 2, 2, 1
; CHECK: vinsertb 3, 2, 8
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_23_7
; CHECK-BE: vinsertb 3, 2, 7
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_24_0(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_24_0
; CHECK: vsldoi 2, 2, 2, 8
; CHECK: vinsertb 3, 2, 7
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_24_0
; CHECK-BE: vsldoi 2, 2, 2, 9
; CHECK-BE: vinsertb 3, 2, 8
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_25_9(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_25_9
; CHECK: vsldoi 2, 2, 2, 15
; CHECK: vinsertb 3, 2, 6
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_25_9
; CHECK-BE: vsldoi 2, 2, 2, 2
; CHECK-BE: vinsertb 3, 2, 9
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 9, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_26_2(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_26_2
; CHECK: vsldoi 2, 2, 2, 6
; CHECK: vinsertb 3, 2, 5
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_26_2
; CHECK-BE: vsldoi 2, 2, 2, 11
; CHECK-BE: vinsertb 3, 2, 10
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 2, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_27_11(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_27_11
; CHECK: vsldoi 2, 2, 2, 13
; CHECK: vinsertb 3, 2, 4
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_27_11
; CHECK-BE: vsldoi 2, 2, 2, 4
; CHECK-BE: vinsertb 3, 2, 11
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 11, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_28_4(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_28_4
; CHECK: vsldoi 2, 2, 2, 4
; CHECK: vinsertb 3, 2, 3
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_28_4
; CHECK-BE: vsldoi 2, 2, 2, 13
; CHECK-BE: vinsertb 3, 2, 12
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 4, i32 29, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_29_13(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_29_13
; CHECK: vsldoi 2, 2, 2, 11
; CHECK: vinsertb 3, 2, 2
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_29_13
; CHECK-BE: vsldoi 2, 2, 2, 6
; CHECK-BE: vinsertb 3, 2, 13
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 13, i32 30, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_30_6(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_30_6
; CHECK: vsldoi 2, 2, 2, 2
; CHECK: vinsertb 3, 2, 1
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_30_6
; CHECK-BE: vsldoi 2, 2, 2, 15
; CHECK-BE: vinsertb 3, 2, 14
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 6, i32 31>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_31_15(<16 x i8> %a, <16 x i8> %b) {
entry:
; CHECK-LABEL: shuffle_vector_byte_31_15
; CHECK: vsldoi 2, 2, 2, 9
; CHECK: vinsertb 3, 2, 0
; CHECK: vmr 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_31_15
; CHECK-BE: vsldoi 2, 2, 2, 8
; CHECK-BE: vinsertb 3, 2, 15
; CHECK-BE: vmr 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 15>
  ret <16 x i8> %vecins
}

; The following testcases use the same vector in both arguments of the
; shufflevector.  If byte element 7 in BE mode(or 8 in LE mode) is the one
; we're attempting to insert, then we can use the vector insert instruction
define <16 x i8> @shuffle_vector_byte_0_7(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_0_7
; CHECK-NOT: vinsertb
; CHECK-BE-LABEL: shuffle_vector_byte_0_7
; CHECK-BE: vinsertb 2, 2, 0
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 7, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_1_8(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_1_8
; CHECK: vinsertb 2, 2, 14
; CHECK-BE-LABEL: shuffle_vector_byte_1_8
; CHECK-BE-NOT: vinsertb
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 8, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_2_8(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_2_8
; CHECK: vinsertb 2, 2, 13
; CHECK-BE-LABEL: shuffle_vector_byte_2_8
; CHECK-BE-NOT: vinsertb
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 8, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_3_7(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_3_7
; CHECK-NOT: vinsertb
; CHECK-BE-LABEL: shuffle_vector_byte_3_7
; CHECK-BE: vinsertb 2, 2, 3
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 7, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_4_7(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_4_7
; CHECK-NOT: vinsertb
; CHECK-BE-LABEL: shuffle_vector_byte_4_7
; CHECK-BE: vinsertb 2, 2, 4
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_5_8(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_5_8
; CHECK: vinsertb 2, 2, 10
; CHECK-BE-LABEL: shuffle_vector_byte_5_8
; CHECK-BE-NOT: vinsertb
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_6_8(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_6_8
; CHECK: vinsertb 2, 2, 9
; CHECK-BE-LABEL: shuffle_vector_byte_6_8
; CHECK-BE-NOT: vinsertb
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_7_8(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_7_8
; CHECK: vinsertb 2, 2, 8
; CHECK-BE-LABEL: shuffle_vector_byte_7_8
; CHECK-BE-NOT: vinsertb
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_8_7(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_8_7
; CHECK-NOT: vinsertb
; CHECK-BE-LABEL: shuffle_vector_byte_8_7
; CHECK-BE: vinsertb 2, 2, 8
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 7, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_9_7(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_9_7
; CHECK-NOT: vinsertb
; CHECK-BE-LABEL: shuffle_vector_byte_9_7
; CHECK-BE: vinsertb 2, 2, 9
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 7, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_10_7(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_10_7
; CHECK-NOT: vinsertb
; CHECK-BE-LABEL: shuffle_vector_byte_10_7
; CHECK-BE: vinsertb 2, 2, 10
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 7, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_11_8(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_11_8
; CHECK: vinsertb 2, 2, 4
; CHECK-BE-LABEL: shuffle_vector_byte_11_8
; CHECK-BE-NOT: vinsertb
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 8, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_12_8(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_12_8
; CHECK: vinsertb 2, 2, 3
; CHECK-BE-LABEL: shuffle_vector_byte_12_8
; CHECK-BE-NOT: vinsertb
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 8, i32 13, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_13_7(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_13_7
; CHECK-NOT: vinsertb
; CHECK-BE-LABEL: shuffle_vector_byte_13_7
; CHECK-BE: vinsertb 2, 2, 13
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 7, i32 14, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_14_7(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_14_7
; CHECK-NOT: vinsertb
; CHECK-BE-LABEL: shuffle_vector_byte_14_7
; CHECK-BE: vinsertb 2, 2, 14
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 7, i32 15>
  ret <16 x i8> %vecins
}

define <16 x i8> @shuffle_vector_byte_15_8(<16 x i8> %a) {
entry:
; CHECK-LABEL: shuffle_vector_byte_15_8
; CHECK: vinsertb 2, 2, 0
; CHECK-BE-LABEL: shuffle_vector_byte_15_8
; CHECK-BE-NOT: vinsertb
  %vecins = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 8>
  ret <16 x i8> %vecins
}
