; RUN: llvm-reduce %s -o %t --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --match-full-lines --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefixes=REDUCED,INTERESTING

; REDUCED-LABEL: define void @func(i32 %k, i32* %Local, i32* %Global, float* %0) {

; Keep one reference to the original value.
; INTERESTING: %[[LOCAL:Local[0-9]*]] = alloca i32, align 4
; INTERESTING: store i32 42, i32* %[[LOCAL]], align 4
; INTERESTING: store i32 42, i32* @Global, align 4

; Everything else must use the function argument.
; REDUCED: store i32 21, i32* %Local, align 4
; REDUCED: store i32 21, i32* %Global, align 4
; REDUCED: store i32 0, i32* %Local, align 4
; REDUCED: store i32 0, i32* %Global, align 4
; REDUCED: store float 0.000000e+00, float* %0, align 4

; Do not add any arguments for %Keep and @GlobalKeep.
; INTERESTING: %[[KEEP:LocalKeep[0-9]*]] = add i32 %k, 21
; INTERESTING: store i32 %[[KEEP]], i32* @GlobalKeep, align 4

; INTERESTING-LABEL: define void @func_caller() {
; REDUCED:             call void @func(i32 21, i32* undef, i32* undef, float* undef)


@Global = global i32 42
@GlobalKeep = global i32 42

define void @func(i32 %k) {
entry:
  %Local = alloca i32, align 4

  store i32 42, i32* %Local, align 4
  store i32 42, i32* @Global, align 4

  store i32 21, i32* %Local, align 4
  store i32 21, i32* @Global, align 4

  store i32 0, i32* %Local, align 4
  store i32 0, i32* @Global, align 4

  store float 0.000000e+00,  float* bitcast (i32* @Global to float*), align 4

  %LocalKeep = add i32 %k, 21
  store i32 %LocalKeep, i32* @GlobalKeep, align 4

  ret void
}


define void @func_caller() {
entry:
  call void @func(i32 21)
  ret void
}

