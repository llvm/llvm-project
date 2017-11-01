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

