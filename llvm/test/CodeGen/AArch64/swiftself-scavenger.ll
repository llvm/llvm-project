; RUN: llc -o - %s | FileCheck %s
; Check that we reserve an emergency spill slot, even if we added an extra
; CSR spill for the values used by the swiftself parameter.
; CHECK-LABEL: func:
; CHECK: str [[REG:x[0-9]+]], [sp, #8]
; CHECK: add [[REG]], sp, #248
; CHECK: str xzr, [{{\s*}}[[REG]], #32760]
; CHECK: ldr [[REG]], [sp, #8]
target triple = "arm64-apple-ios"

@ptr8 = external global i8*
@ptr64 = external global i64

define hidden swiftcc void @func(i8* swiftself %arg) #0 {
bb:
  %stack0 = alloca i8*, i32 5000, align 8
  %stack1 = alloca i8*, i32 32, align 8

  %v0  = load volatile i64, i64* @ptr64, align 8
  %v1  = load volatile i64, i64* @ptr64, align 8
  %v2  = load volatile i64, i64* @ptr64, align 8
  %v3  = load volatile i64, i64* @ptr64, align 8
  %v4  = load volatile i64, i64* @ptr64, align 8
  %v5  = load volatile i64, i64* @ptr64, align 8
  %v6  = load volatile i64, i64* @ptr64, align 8
  %v7  = load volatile i64, i64* @ptr64, align 8
  %v8  = load volatile i64, i64* @ptr64, align 8
  %v9  = load volatile i64, i64* @ptr64, align 8
  %v10 = load volatile i64, i64* @ptr64, align 8
  %v11 = load volatile i64, i64* @ptr64, align 8
  %v12 = load volatile i64, i64* @ptr64, align 8
  %v13 = load volatile i64, i64* @ptr64, align 8
  %v14 = load volatile i64, i64* @ptr64, align 8
  %v15 = load volatile i64, i64* @ptr64, align 8
  %v16 = load volatile i64, i64* @ptr64, align 8
  %v17 = load volatile i64, i64* @ptr64, align 8
  %v18 = load volatile i64, i64* @ptr64, align 8
  %v19 = load volatile i64, i64* @ptr64, align 8
  %v20 = load volatile i64, i64* @ptr64, align 8
  %v21 = load volatile i64, i64* @ptr64, align 8
  %v22 = load volatile i64, i64* @ptr64, align 8
  %v23 = load volatile i64, i64* @ptr64, align 8
  %v24 = load volatile i64, i64* @ptr64, align 8
  %v25 = load volatile i64, i64* @ptr64, align 8

  ; this should exceed stack-relative addressing limits and need an emergency
  ; spill slot.
  %s = getelementptr inbounds i8*, i8** %stack0, i64 4092
  store volatile i8* null, i8** %s
  store volatile i8* null, i8** %stack1

  store volatile i64 %v0,  i64* @ptr64, align 8
  store volatile i64 %v1,  i64* @ptr64, align 8
  store volatile i64 %v2,  i64* @ptr64, align 8
  store volatile i64 %v3,  i64* @ptr64, align 8
  store volatile i64 %v4,  i64* @ptr64, align 8
  store volatile i64 %v5,  i64* @ptr64, align 8
  store volatile i64 %v6,  i64* @ptr64, align 8
  store volatile i64 %v7,  i64* @ptr64, align 8
  store volatile i64 %v8,  i64* @ptr64, align 8
  store volatile i64 %v9,  i64* @ptr64, align 8
  store volatile i64 %v10, i64* @ptr64, align 8
  store volatile i64 %v11, i64* @ptr64, align 8
  store volatile i64 %v12, i64* @ptr64, align 8
  store volatile i64 %v13, i64* @ptr64, align 8
  store volatile i64 %v14, i64* @ptr64, align 8
  store volatile i64 %v15, i64* @ptr64, align 8
  store volatile i64 %v16, i64* @ptr64, align 8
  store volatile i64 %v17, i64* @ptr64, align 8
  store volatile i64 %v18, i64* @ptr64, align 8
  store volatile i64 %v19, i64* @ptr64, align 8
  store volatile i64 %v20, i64* @ptr64, align 8
  store volatile i64 %v21, i64* @ptr64, align 8
  store volatile i64 %v22, i64* @ptr64, align 8
  store volatile i64 %v23, i64* @ptr64, align 8
  store volatile i64 %v24, i64* @ptr64, align 8
  store volatile i64 %v25, i64* @ptr64, align 8
  
  ; use swiftself parameter late so it stays alive throughout the function.
  store volatile i8* %arg, i8** @ptr8
  ret void
}
