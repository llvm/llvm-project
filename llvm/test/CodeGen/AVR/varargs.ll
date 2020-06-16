; RUN: llc -mattr=sram,movw,addsubiw < %s -march=avr | FileCheck %s

declare void @llvm.va_start(i8*)
declare i16 @vsprintf(i8* nocapture, i8* nocapture, i8*)
declare void @llvm.va_end(i8*)

define i16 @varargs1(i8* nocapture %x, ...) {
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
  %ap = alloca i8*
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %arraydecay = getelementptr inbounds [32 x i8], [32 x i8]* %buffer, i16 0, i16 0
  %1 = load i8*, i8** %ap
  %call = call i16 @vsprintf(i8* %arraydecay, i8* %x, i8* %1)
  call void @llvm.va_end(i8* %ap1)
  ret i16 0
}

define i16 @varargs2(i8* nocapture %x, ...) {
; CHECK-LABEL: varargs2:
; CHECK: ldd r24, [[REG:X|Y|Z]]+{{[0-9]+}}
; CHECK: ldd r25, [[REG]]+{{[0-9]+}}
  %ap = alloca i8*
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %1 = va_arg i8** %ap, i16
  call void @llvm.va_end(i8* %ap1)
  ret i16 %1
}

declare void @var1223(i16, ...)
define void @varargcall() {
; CHECK-LABEL: varargcall:
; CHECK: ldi [[REG1:r[0-9]+]], 189
; CHECK: ldi [[REG2:r[0-9]+]], 205
; CHECK: push [[REG2]]
; CHECK: push [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 191
; CHECK: ldi [[REG2:r[0-9]+]], 223
; CHECK: push [[REG2]]
; CHECK: push [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 205
; CHECK: ldi [[REG2:r[0-9]+]], 171
; CHECK: push [[REG2]]
; CHECK: push [[REG1]]
; CHECK: call
; CHECK: adiw r30, 6
  tail call void (i16, ...) @var1223(i16 -21555, i16 -12867, i16 -8257)
  ret void
}
