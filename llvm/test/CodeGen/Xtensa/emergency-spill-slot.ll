; RUN: llc -mtriple=xtensa -mcpu=esp8266 < %s -o /dev/null
; RUN: llc -mtriple=xtensa -mcpu=esp32s3 < %s -o /dev/null

; Check that this compiles. Similar to the reproducer in:
; https://github.com/espressif/llvm-project/issues/143
define i32 @store_far(ptr %in) {
  %far = alloca i32, align 4
  %buf = alloca [1500 x i8], align 1
  %v0 = load volatile i32, ptr %in, align 4
  %v1 = load volatile i32, ptr %in, align 4
  %v2 = load volatile i32, ptr %in, align 4
  %v3 = load volatile i32, ptr %in, align 4
  %v4 = load volatile i32, ptr %in, align 4
  %v5 = load volatile i32, ptr %in, align 4
  %v6 = load volatile i32, ptr %in, align 4
  %v7 = load volatile i32, ptr %in, align 4
  %v8 = load volatile i32, ptr %in, align 4
  %v9 = load volatile i32, ptr %in, align 4
  %v10 = load volatile i32, ptr %in, align 4
  %v11 = load volatile i32, ptr %in, align 4
  %v12 = load volatile i32, ptr %in, align 4
  %v13 = load volatile i32, ptr %in, align 4
  store i32 %v0, ptr %far, align 4
  %x1 = xor i32 %v0, %v1
  %x2 = xor i32 %x1, %v2
  %x3 = xor i32 %x2, %v3
  %x4 = xor i32 %x3, %v4
  %x5 = xor i32 %x4, %v5
  %x6 = xor i32 %x5, %v6
  %x7 = xor i32 %x6, %v7
  %x8 = xor i32 %x7, %v8
  %x9 = xor i32 %x8, %v9
  %x10 = xor i32 %x9, %v10
  %x11 = xor i32 %x10, %v11
  %x12 = xor i32 %x11, %v12
  %x13 = xor i32 %x12, %v13
  ret i32 %x13
}
