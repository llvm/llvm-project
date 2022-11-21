; RUN: llc < %s -mtriple=armv7-apple-darwin -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -arm-atomic-cfg-tidy=0 | FileCheck %s -check-prefix=THUMB
; rdar://7998649

%struct.foo = type { i64, i64 }

define zeroext i8 @t(%struct.foo* %this, i1 %tst) noreturn optsize {
entry:
; ARM-LABEL:       t:
; ARM-DAG:       mov r[[ADDR:[0-9]+]], #8
; ARM-DAG:       mov [[VAL:r[0-9]+]], #0
; ARM:       str [[VAL]], [r[[ADDR]]], r0

; THUMB-LABEL:     t:
; THUMB-DAG:       movs r[[ADDR:[0-9]+]], #8
; THUMB-DAG:       movs [[VAL:r[0-9]+]], #0
; THUMB-NOT: str {{[a-z0-9]+}}, [{{[a-z0-9]+}}], {{[a-z0-9]+}}
; THUMB:     str [[VAL]], [r[[ADDR]]]
  %0 = getelementptr inbounds %struct.foo, %struct.foo* %this, i32 0, i32 1 ; <i64*> [#uses=1]
  store i32 0, i32* inttoptr (i32 8 to i32*), align 8
  br i1 %tst, label %bb.nph96, label %bb3

bb3:                                              ; preds = %entry
  %1 = load i64, i64* %0, align 4                      ; <i64> [#uses=0]
  ret i8 42

bb.nph96:                                         ; preds = %entry
  ret i8 3
}
