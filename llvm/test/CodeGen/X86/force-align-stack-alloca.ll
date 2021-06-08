; This test is attempting to detect when we request forced re-alignment of the
; stack to an alignment greater than would be available due to the ABI. We
; arbitrarily force alignment up to 32-bytes for i386 hoping that this will
; exceed any ABI provisions.
;
; RUN: llc < %s -mcpu=generic -stackrealign -stack-alignment=32 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

define i32 @f(i8* %p) nounwind {
entry:
  %0 = load i8, i8* %p
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

define i64 @g(i32 %i) nounwind {
; CHECK-LABEL: g:
; CHECK:      pushl  %ebp
; CHECK-NEXT: movl   %esp, %ebp
; CHECK-NEXT: pushl
; CHECK-NEXT: pushl
; CHECK-NEXT: andl   $-32, %esp
; CHECK-NEXT: subl   $32, %esp
;
; Now setup the base pointer (%esi).
; CHECK-NEXT: movl   %esp, %esi
; CHECK-NOT:         {{[^ ,]*}}, %esp
;
; The next adjustment of the stack is due to the alloca.
; CHECK:      movl   %{{...}}, %esp
; CHECK-NOT:         {{[^ ,]*}}, %esp
;
; Next we set up the memset call.
; CHECK:      subl   $20, %esp
; CHECK-NOT:         {{[^ ,]*}}, %esp
; CHECK:      pushl
; CHECK:      pushl
; CHECK:      pushl
; CHECK:      calll  memset
;
; Deallocating 32 bytes of outgoing call frame for memset and
; allocating 28 bytes for calling f yields a 4-byte adjustment:
; CHECK-NEXT: addl   $4, %esp
; CHECK-NOT:         {{[^ ,]*}}, %esp
;
; And move on to call 'f', and then restore the stack.
; CHECK:      pushl
; CHECK-NOT:         {{[^ ,]*}}, %esp
; CHECK:      calll  f
; CHECK-NEXT: addl   $32, %esp
; CHECK-NOT:         {{[^ ,]*}}, %esp
;
; Restore %esp from %ebp (frame pointer) and subtract the size of
; zone with callee-saved registers to pop them.
; This is the state prior to stack realignment and the allocation of VLAs.
; CHECK-NOT:  popl
; CHECK:      leal   -8(%ebp), %esp
; CHECK-NEXT: popl
; CHECK-NEXT: popl
; CHECK-NEXT: popl   %ebp
; CHECK-NEXT: ret

entry:
  br label %if.then

if.then:
  %0 = alloca i8, i32 %i
  call void @llvm.memset.p0i8.i32(i8* %0, i8 0, i32 %i, i1 false)
  %call = call i32 @f(i8* %0)
  %conv = sext i32 %call to i64
  ret i64 %conv
}

declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1) nounwind
