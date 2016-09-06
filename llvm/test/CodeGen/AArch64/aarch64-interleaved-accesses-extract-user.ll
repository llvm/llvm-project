; RUN: opt < %s -mtriple=aarch64 -interleaved-access -S | FileCheck %s

; CHECK-LABEL: @extract_user_basic(
; CHECK: %ldN = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0v4i32
; CHECK: %[[R:.+]] = extractvalue { <4 x i32>, <4 x i32> } %ldN, 0
; CHECK: extractelement <4 x i32> %[[R]], i64 1
define void @extract_user_basic(<8 x i32>* %A, i1 %C) {
entry:
  %L = load <8 x i32>, <8 x i32>* %A, align 8
  %S = shufflevector <8 x i32> %L, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  br i1 %C, label %if.then, label %if.merge

if.then:
  %E = extractelement <8 x i32> %L, i32 2
  br label %if.merge

if.merge:
  ret void
}

; CHECK-LABEL: @extract_user_multi(
; CHECK: %ldN = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0v4i32
; CHECK: %[[R:.+]] = extractvalue { <4 x i32>, <4 x i32> } %ldN, 0
; CHECK: extractelement <4 x i32> %[[R]], i64 0
; CHECK: extractelement <4 x i32> %[[R]], i64 1
define void @extract_user_multi(<8 x i32>* %A, i1 %C) {
entry:
  %L = load <8 x i32>, <8 x i32>* %A, align 8
  %S = shufflevector <8 x i32> %L, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  br i1 %C, label %if.then, label %if.merge

if.then:
  %E1 = extractelement <8 x i32> %L, i32 0
  br label %if.merge

if.merge:
  %E2 = extractelement <8 x i32> %L, i32 2
  ret void
}

; CHECK-LABEL: @extract_user_multi_no_dom(
; CHECK-NOT: %ldN = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0v4i32
define void @extract_user_multi_no_dom(<8 x i32>* %A, i1 %C) {
entry:
  %L = load <8 x i32>, <8 x i32>* %A, align 8
  %E1 = extractelement <8 x i32> %L, i32 0
  br i1 %C, label %if.then, label %if.merge

if.then:
  %S = shufflevector <8 x i32> %L, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %E2 = extractelement <8 x i32> %L, i32 2
  br label %if.merge

if.merge:
  ret void
}

; CHECK-LABEL: @extract_user_wrong_const_index(
; CHECK-NOT: %ldN = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0v4i32
define void @extract_user_wrong_const_index(<8 x i32>* %A) {
entry:
  %L = load <8 x i32>, <8 x i32>* %A, align 8
  %S = shufflevector <8 x i32> %L, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %E = extractelement <8 x i32> %L, i32 1
  ret void
}

; CHECK-LABEL: @extract_user_undef_index(
; CHECK-NOT: %ldN = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0v4i32
define void @extract_user_undef_index(<8 x i32>* %A) {
entry:
  %L = load <8 x i32>, <8 x i32>* %A, align 8
  %S = shufflevector <8 x i32> %L, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %E = extractelement <8 x i32> %L, i32 undef
  ret void
}

; CHECK-LABEL: @extract_user_var_index(
; CHECK-NOT: %ldN = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0v4i32
define void @extract_user_var_index(<8 x i32>* %A, i32 %I) {
entry:
  %L = load <8 x i32>, <8 x i32>* %A, align 8
  %S = shufflevector <8 x i32> %L, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %E = extractelement <8 x i32> %L, i32 %I
  ret void
}
