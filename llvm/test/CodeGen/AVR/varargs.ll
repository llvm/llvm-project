; RUN: llc -mattr=sram,movw,addsubiw < %s -mtriple=avr | FileCheck %s

declare void @llvm.va_start(ptr)
declare i16 @vsprintf(ptr nocapture, ptr nocapture, ptr)
declare void @llvm.va_end(ptr)

define i16 @varargs1(ptr nocapture %x, ...) {
; CHECK-LABEL: varargs1:
; CHECK: movw r20, r28
; CHECK: subi r20, 215
; CHECK: sbci r21, 255
; CHECK: movw r24, r28
; CHECK: adiw r24, 3
; CHECK: ldd r22, Y+39
; CHECK: ldd r23, Y+40
; CHECK: call
  %buffer = alloca [32 x i8]
  %ap = alloca ptr
  %ap1 = bitcast ptr %ap to ptr
  call void @llvm.va_start(ptr %ap1)
  %arraydecay = getelementptr inbounds [32 x i8], ptr %buffer, i16 0, i16 0
  %1 = load ptr, ptr %ap
  %call = call i16 @vsprintf(ptr %arraydecay, ptr %x, ptr %1)
  call void @llvm.va_end(ptr %ap1)
  ret i16 0
}

define i16 @varargs2(ptr nocapture %x, ...) {
; CHECK-LABEL: varargs2:
; CHECK: ldd r24, [[REG:X|Y|Z]]+{{[0-9]+}}
; CHECK: ldd r25, [[REG]]+{{[0-9]+}}
  %ap = alloca ptr
  %ap1 = bitcast ptr %ap to ptr
  call void @llvm.va_start(ptr %ap1)
  %1 = va_arg ptr %ap, i16
  call void @llvm.va_end(ptr %ap1)
  ret i16 %1
}

declare void @var1223(i16, ...)
define void @varargcall() {
; CHECK-LABEL: varargcall:
; CHECK: ldi [[REG1:r[0-9]+]], 191
; CHECK: ldi [[REG2:r[0-9]+]], 223
; CHECK: std Z+6, [[REG2]]
; CHECK: std Z+5, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 189
; CHECK: ldi [[REG2:r[0-9]+]], 205
; CHECK: std Z+4, [[REG2]]
; CHECK: std Z+3, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 205
; CHECK: ldi [[REG2:r[0-9]+]], 171
; CHECK: std Z+2, [[REG2]]
; CHECK: std Z+1, [[REG1]]
; CHECK: call
; CHECK: adiw r30, 6
  tail call void (i16, ...) @var1223(i16 -21555, i16 -12867, i16 -8257)
  ret void
}
