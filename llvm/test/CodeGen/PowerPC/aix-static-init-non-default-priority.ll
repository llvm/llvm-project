; RUN: llc -mtriple powerpc-ibm-aix-xcoff -mcpu=ppc -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mcpu=ppc -verify-machineinstrs < %s | FileCheck %s

@llvm.global_ctors = appending global [5 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @cf1, ptr null }, { i32, ptr, ptr } { i32 21, ptr @cf2, ptr null }, { i32, ptr, ptr } { i32 81, ptr @cf3, ptr null }, { i32, ptr, ptr } { i32 1125, ptr @cf4, ptr null }, { i32, ptr, ptr } { i32 64512, ptr @cf5, ptr null }]
@llvm.global_dtors = appending global [5 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 20, ptr @df1, ptr null }, { i32, ptr, ptr } { i32 80, ptr @df2, ptr null }, { i32, ptr, ptr } { i32 1124, ptr @df3, ptr null }, { i32, ptr, ptr } { i32 64511, ptr @df4, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @df5, ptr null }]

define i32 @cf1(i32 %a) {
  ret i32 %a
}

define void @cf2() {
  ret void
}

define void @cf3() {
  ret void
}

define void @cf4() {
  ret void
}

define void @cf5() {
  ret void
}

define i32 @df1(i32 %a) {
  ret i32 %a
}

define void @df2() {
  ret void
}

define void @df3() {
  ret void
}

define void @df4() {
  ret void
}

define void @df5() {
  ret void
}

; CHECK:   .globl  cf1[DS]
; CHECK:   .globl  .cf1
; CHECK:   .align  2
; CHECK:   .csect cf1[DS]
; CHECK: __sinit00000000_clang_f6a1bc9396775a64c6249effda300afe_0: # @cf1
; CHECK: .cf1:
; CHECK: .__sinit00000000_clang_f6a1bc9396775a64c6249effda300afe_0:

; CHECK:   .globl  cf2[DS]
; CHECK:   .globl  .cf2
; CHECK:   .align  2
; CHECK:   .csect cf2[DS]
; CHECK: __sinit00000024_clang_f6a1bc9396775a64c6249effda300afe_1: # @cf2
; CHECK: .cf2:
; CHECK: .__sinit00000024_clang_f6a1bc9396775a64c6249effda300afe_1:

; CHECK:   .globl  cf3[DS]
; CHECK:   .globl  .cf3
; CHECK:   .align  2
; CHECK:   .csect cf3[DS]
; CHECK: __sinit000003ec_clang_f6a1bc9396775a64c6249effda300afe_2: # @cf3
; CHECK: .cf3:
; CHECK: .__sinit000003ec_clang_f6a1bc9396775a64c6249effda300afe_2:

; CHECK:   .globl  cf4[DS]
; CHECK:   .globl  .cf4
; CHECK:   .align  2
; CHECK:   .csect cf4[DS]
; CHECK: __sinit00008c55_clang_f6a1bc9396775a64c6249effda300afe_3: # @cf4
; CHECK: .cf4:
; CHECK: .__sinit00008c55_clang_f6a1bc9396775a64c6249effda300afe_3:

; CHECK:   .globl  cf5[DS]
; CHECK:   .globl  .cf5
; CHECK:   .align  2
; CHECK:   .csect cf5[DS]
; CHECK: __sinit7ffffc01_clang_f6a1bc9396775a64c6249effda300afe_4: # @cf5
; CHECK: .cf5:
; CHECK: .__sinit7ffffc01_clang_f6a1bc9396775a64c6249effda300afe_4:

; CHECK:   .globl  df1[DS]
; CHECK:   .globl  .df1
; CHECK:   .align  2
; CHECK:   .csect df1[DS]
; CHECK: __sterm00000014_clang_f6a1bc9396775a64c6249effda300afe_0: # @df1
; CHECK: .df1:
; CHECK: .__sterm00000014_clang_f6a1bc9396775a64c6249effda300afe_0:

; CHECK:   .globl  df2[DS]
; CHECK:   .globl  .df2
; CHECK:   .align  2
; CHECK:   .csect df2[DS]
; CHECK: __sterm000003d4_clang_f6a1bc9396775a64c6249effda300afe_1: # @df2
; CHECK: .df2:
; CHECK: .__sterm000003d4_clang_f6a1bc9396775a64c6249effda300afe_1:

; CHECK:   .globl  df3[DS]
; CHECK:   .globl  .df3
; CHECK:   .align  2
; CHECK:   .csect df3[DS]
; CHECK: __sterm000007ff_clang_f6a1bc9396775a64c6249effda300afe_2: # @df3
; CHECK: .df3:
; CHECK: .__sterm000007ff_clang_f6a1bc9396775a64c6249effda300afe_2:

; CHECK:   .globl  df4[DS]
; CHECK:   .globl  .df4
; CHECK:   .align  2
; CHECK:   .csect df4[DS]
; CHECK: __sterm7fff2211_clang_f6a1bc9396775a64c6249effda300afe_3: # @df4
; CHECK: .df4:
; CHECK: .__sterm7fff2211_clang_f6a1bc9396775a64c6249effda300afe_3:

; CHECK:   .globl  df5[DS]
; CHECK:   .globl  .df5
; CHECK:   .align  2
; CHECK:   .csect df5[DS]
; CHECK: __sterm80000000_clang_f6a1bc9396775a64c6249effda300afe_4: # @df5
; CHECK: .df5:
; CHECK: .__sterm80000000_clang_f6a1bc9396775a64c6249effda300afe_4:

; CHECK:   .globl  __sinit00000000_clang_f6a1bc9396775a64c6249effda300afe_0
; CHECK:   .globl  .__sinit00000000_clang_f6a1bc9396775a64c6249effda300afe_0
; CHECK:   .globl  __sinit00000024_clang_f6a1bc9396775a64c6249effda300afe_1
; CHECK:   .globl  .__sinit00000024_clang_f6a1bc9396775a64c6249effda300afe_1
; CHECK:   .globl  __sinit000003ec_clang_f6a1bc9396775a64c6249effda300afe_2
; CHECK:   .globl  .__sinit000003ec_clang_f6a1bc9396775a64c6249effda300afe_2
; CHECK:   .globl  __sinit00008c55_clang_f6a1bc9396775a64c6249effda300afe_3
; CHECK:   .globl  .__sinit00008c55_clang_f6a1bc9396775a64c6249effda300afe_3
; CHECK:   .globl  __sinit7ffffc01_clang_f6a1bc9396775a64c6249effda300afe_4
; CHECK:   .globl  .__sinit7ffffc01_clang_f6a1bc9396775a64c6249effda300afe_4
; CHECK:   .globl  __sterm00000014_clang_f6a1bc9396775a64c6249effda300afe_0
; CHECK:   .globl  .__sterm00000014_clang_f6a1bc9396775a64c6249effda300afe_0
; CHECK:   .globl  __sterm000003d4_clang_f6a1bc9396775a64c6249effda300afe_1
; CHECK:   .globl  .__sterm000003d4_clang_f6a1bc9396775a64c6249effda300afe_1
; CHECK:   .globl  __sterm000007ff_clang_f6a1bc9396775a64c6249effda300afe_2
; CHECK:   .globl  .__sterm000007ff_clang_f6a1bc9396775a64c6249effda300afe_2
; CHECK:   .globl  __sterm7fff2211_clang_f6a1bc9396775a64c6249effda300afe_3
; CHECK:   .globl  .__sterm7fff2211_clang_f6a1bc9396775a64c6249effda300afe_3
; CHECK:   .globl  __sterm80000000_clang_f6a1bc9396775a64c6249effda300afe_4
; CHECK:   .globl  .__sterm80000000_clang_f6a1bc9396775a64c6249effda300afe_4
