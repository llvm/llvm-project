; RUN: llc -relocation-model pic < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-freebsd"

@x1 = external thread_local global i32, align 4

define void @x3() #0 {
entry:
  %0 = load i32, i32* @x1, align 4
  %cond = icmp eq i32 %0, 92
  br i1 %cond, label %sw.bb, label %sw.epilog

sw.bb:                                            ; preds = %entry
  call void @x2(i8* null)
  unreachable

sw.epilog:                                        ; preds = %entry
  ret void
}

declare void @x2(i8*)

attributes #0 = { optsize }

; CHECK-LABEL: x3:
; CHECK:         addl    $_GLOBAL_OFFSET_TABLE_+(.Ltmp0-.L0$pb), %[[REG:.*]]
; CHECK-NEXT:    leal    x1@TLSGD(,%[[REG]]), %eax
; CHECK-NEXT:    calll   ___tls_get_addr@PLT
; CHECK-NEXT:    cmpl    $92, (%eax)
