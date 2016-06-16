; RUN: llc -mtriple=arm64-apple-darwin -enable-misched=0 -mcpu=cyclone < %s | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin -enable-misched=0 -mcpu=cyclone -fast-isel < %s | FileCheck %s --check-prefix=FAST
; RUN: llc -mtriple=arm64-apple-darwin -enable-misched=0 -mcpu=cyclone -filetype=obj -o %t %s
; RUN: llvm-objdump -triple arm64-apple-darwin -d %t | FileCheck %s --check-prefix CHECK-ENCODING

; CHECK-ENCODING-NOT: <unknown>
; CHECK-ENCODING: mov x16, #281470681743360
; CHECK-ENCODING: movk x16, #57005, lsl #16
; CHECK-ENCODING: movk x16, #48879

; One argument will be passed in register, the other will be pushed on the stack.
; Return value in x0.
define void @jscall_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: jscall_patchpoint_codegen:
; CHECK:       Ltmp
; CHECK:       str x{{.+}}, [sp]
; CHECK-NEXT:  mov  x0, x{{.+}}
; CHECK:       Ltmp
; CHECK-NEXT:  mov  x16, #281470681743360
; CHECK:  movk  x16, #57005, lsl #16
; CHECK:  movk  x16, #48879
; CHECK-NEXT:  blr x16
; FAST-LABEL:  jscall_patchpoint_codegen:
; FAST:        Ltmp
; FAST:        str x{{.+}}, [sp]
; FAST:        Ltmp
; FAST-NEXT:   mov   x16, #281470681743360
; FAST-NEXT:   movk  x16, #57005, lsl #16
; FAST-NEXT:   movk  x16, #48879
; FAST-NEXT:   blr x16
  %resolveCall2 = inttoptr i64 281474417671919 to i8*
  %result = tail call webkit_jscc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 20, i8* %resolveCall2, i32 2, i64 %p4, i64 %p2)
  %resolveCall3 = inttoptr i64 244837814038255 to i8*
  tail call webkit_jscc void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 6, i32 20, i8* %resolveCall3, i32 2, i64 %p4, i64 %result)
  ret void
}

; Test if the arguments are properly aligned and that we don't store undef arguments.
define i64 @jscall_patchpoint_codegen2(i64 %callee) {
entry:
; CHECK-LABEL: jscall_patchpoint_codegen2:
; CHECK:       Ltmp
; CHECK:       orr w[[REG:[0-9]+]], wzr, #0x6
; CHECK-NEXT:  str x[[REG]], [sp, #24]
; CHECK-NEXT:  orr w[[REG:[0-9]+]], wzr, #0x4
; CHECK-NEXT:  str w[[REG]], [sp, #16]
; CHECK-NEXT:  orr w[[REG:[0-9]+]], wzr, #0x2
; CHECK-NEXT:  str x[[REG]], [sp]
; CHECK:       Ltmp
; CHECK-NEXT:  mov  x16, #281470681743360
; CHECK-NEXT:  movk  x16, #57005, lsl #16
; CHECK-NEXT:  movk  x16, #48879
; CHECK-NEXT:  blr x16
; FAST-LABEL:  jscall_patchpoint_codegen2:
; FAST:        Ltmp
; FAST:        orr [[REG1:x[0-9]+]], xzr, #0x2
; FAST-NEXT:   orr [[REG2:w[0-9]+]], wzr, #0x4
; FAST-NEXT:   orr [[REG3:x[0-9]+]], xzr, #0x6
; FAST-NEXT:   str [[REG1]], [sp]
; FAST-NEXT:   str [[REG2]], [sp, #16]
; FAST-NEXT:   str [[REG3]], [sp, #24]
; FAST:        Ltmp
; FAST-NEXT:   mov  x16, #281470681743360
; FAST-NEXT:   movk  x16, #57005, lsl #16
; FAST-NEXT:   movk  x16, #48879
; FAST-NEXT:   blr x16
  %call = inttoptr i64 281474417671919 to i8*
  %result = call webkit_jscc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 7, i32 20, i8* %call, i32 6, i64 %callee, i64 2, i64 undef, i32 4, i32 undef, i64 6)
  ret i64 %result
}

; Test if the arguments are properly aligned and that we don't store undef arguments.
define i64 @jscall_patchpoint_codegen3(i64 %callee) {
entry:
; CHECK-LABEL: jscall_patchpoint_codegen3:
; CHECK:       Ltmp
; CHECK:       mov  w[[REG:[0-9]+]], #10
; CHECK-NEXT:  str x[[REG]], [sp, #48]
; CHECK-NEXT:  orr w[[REG:[0-9]+]], wzr, #0x8
; CHECK-NEXT:  str w[[REG]], [sp, #36]
; CHECK-NEXT:  orr w[[REG:[0-9]+]], wzr, #0x6
; CHECK-NEXT:  str x[[REG]], [sp, #24]
; CHECK-NEXT:  orr w[[REG:[0-9]+]], wzr, #0x4
; CHECK-NEXT:  str w[[REG]], [sp, #16]
; CHECK-NEXT:  orr w[[REG:[0-9]+]], wzr, #0x2
; CHECK-NEXT:  str x[[REG]], [sp]
; CHECK:       Ltmp
; CHECK-NEXT:  mov   x16, #281470681743360
; CHECK-NEXT:  movk  x16, #57005, lsl #16
; CHECK-NEXT:  movk  x16, #48879
; CHECK-NEXT:  blr x16
; FAST-LABEL:  jscall_patchpoint_codegen3:
; FAST:        Ltmp
; FAST:        orr [[REG1:x[0-9]+]], xzr, #0x2
; FAST-NEXT:   orr [[REG2:w[0-9]+]], wzr, #0x4
; FAST-NEXT:   orr [[REG3:x[0-9]+]], xzr, #0x6
; FAST-NEXT:   orr [[REG4:w[0-9]+]], wzr, #0x8
; FAST-NEXT:   mov [[REG5:x[0-9]+]], #10
; FAST-NEXT:   str [[REG1]], [sp]
; FAST-NEXT:   str [[REG2]], [sp, #16]
; FAST-NEXT:   str [[REG3]], [sp, #24]
; FAST-NEXT:   str [[REG4]], [sp, #36]
; FAST-NEXT:   str [[REG5]], [sp, #48]
; FAST:        Ltmp
; FAST-NEXT:   mov   x16, #281470681743360
; FAST-NEXT:   movk  x16, #57005, lsl #16
; FAST-NEXT:   movk  x16, #48879
; FAST-NEXT:   blr x16
  %call = inttoptr i64 281474417671919 to i8*
  %result = call webkit_jscc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 7, i32 20, i8* %call, i32 10, i64 %callee, i64 2, i64 undef, i32 4, i32 undef, i64 6, i32 undef, i32 8, i32 undef, i64 10)
  ret i64 %result
}

; CHECK-LABEL: test_i16:
; CHECK: ldrh [[BREG:w[0-9]+]], [sp]
; CHECK: add {{w[0-9]+}}, w0, [[BREG]]
define webkit_jscc zeroext i16 @test_i16(i16 zeroext %a, i16 zeroext %b) {
  %sum = add i16 %a, %b
  ret i16 %sum
}

declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)
