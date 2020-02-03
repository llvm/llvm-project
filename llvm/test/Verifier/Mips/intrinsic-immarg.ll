; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @ld_b(<16 x i8> * %ptr, i8 * %ldptr, i32 %offset) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %offset
  ; CHECK-NEXT: %a = call <16 x i8> @llvm.mips.ld.b(i8* %ldptr, i32 %offset)
  %a = call <16 x i8> @llvm.mips.ld.b(i8* %ldptr, i32 %offset)
  store <16 x i8> %a, <16 x i8> * %ptr, align 16
  ret void
}

define void @st_b(<16 x i8> * %ptr, i8 * %ldptr, i32 %offset, i8 * %stptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %offset
  ; CHECK-NEXT: call void @llvm.mips.st.b(<16 x i8> %a, i8* %stptr, i32 %offset)
  %a = call <16 x i8> @llvm.mips.ld.b(i8* %ldptr, i32 0)
  call void @llvm.mips.st.b(<16 x i8> %a, i8* %stptr, i32 %offset)
  ret void
}

define void @ld_w(<4 x i32> * %ptr, i8 * %ldptr, i32 %offset) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %offset
  ; CHECK-NEXT: %a = call <4 x i32> @llvm.mips.ld.w(i8* %ldptr, i32 %offset)
  %a = call <4 x i32> @llvm.mips.ld.w(i8* %ldptr, i32 %offset)
  store <4 x i32> %a, <4 x i32> * %ptr, align 16
  ret void
}

define void @st_w(<8 x i16> * %ptr, i8 * %ldptr, i32 %offset, i8 * %stptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %offset
  ; CHECK-NEXT: call void @llvm.mips.st.w(<4 x i32> %a, i8* %stptr, i32 %offset)
  %a = call <4 x i32> @llvm.mips.ld.w(i8* %ldptr, i32 0)
  call void @llvm.mips.st.w(<4 x i32> %a, i8* %stptr, i32 %offset)
  ret void
}

define void @ld_h(<8 x i16> * %ptr, i8 * %ldptr, i32 %offset) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %offset
  ; CHECK-NEXT: %a = call <8 x i16> @llvm.mips.ld.h(i8* %ldptr, i32 %offset)
  %a = call <8 x i16> @llvm.mips.ld.h(i8* %ldptr, i32 %offset)
  store <8 x i16> %a, <8 x i16> * %ptr, align 16
  ret void
}

define void @st_h(<8 x i16> * %ptr, i8 * %ldptr, i32 %offset, i8 * %stptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %offset
  ; CHECK-NEXT: call void @llvm.mips.st.h(<8 x i16> %a, i8* %stptr, i32 %offset)
  %a = call <8 x i16> @llvm.mips.ld.h(i8* %ldptr, i32 0)
  call void @llvm.mips.st.h(<8 x i16> %a, i8* %stptr, i32 %offset)
  ret void
}

define void @ld_d(<2 x i64> * %ptr, i8 * %ldptr, i32 %offset) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %offset
  ; CHECK-NEXT: %a = call <2 x i64> @llvm.mips.ld.d(i8* %ldptr, i32 %offset)
  %a = call <2 x i64> @llvm.mips.ld.d(i8* %ldptr, i32 %offset)
  store <2 x i64> %a, <2 x i64> * %ptr, align 16
  ret void
}

define void @st_d(<2 x i64> * %ptr, i8 * %ldptr, i32 %offset, i8 * %stptr) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %offset
  ; CHECK-NEXT: call void @llvm.mips.st.d(<2 x i64> %a, i8* %stptr, i32 %offset)
  %a = call <2 x i64> @llvm.mips.ld.d(i8* %ldptr, i32 0)
  call void @llvm.mips.st.d(<2 x i64> %a, i8* %stptr, i32 %offset)
  ret void
}

declare <16 x i8> @llvm.mips.ld.b(i8*, i32)
declare <8 x i16> @llvm.mips.ld.h(i8*, i32)
declare <4 x i32> @llvm.mips.ld.w(i8*, i32)
declare <2 x i64> @llvm.mips.ld.d(i8*, i32)
declare void @llvm.mips.st.b(<16 x i8>, i8*, i32)
declare void @llvm.mips.st.h(<8 x i16>, i8*, i32)
declare void @llvm.mips.st.w(<4 x i32>, i8*, i32)
declare void @llvm.mips.st.d(<2 x i64>, i8*, i32)
