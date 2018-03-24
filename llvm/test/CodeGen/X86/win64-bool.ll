; RUN: llc < %s -mtriple=x86_64-windows-msvc | FileCheck %s --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-windows-gnu | FileCheck %s --check-prefix=CHECK

define i32 @pass_bool(i1 zeroext %b) {
entry:
  %cond = select i1 %b, i32 66, i32 0
  ret i32 %cond
}

; CHECK-LABEL: pass_bool:
; CHECK-DAG: testb %cl, %cl
; CHECK-DAG: movl    $66,
; CHECK:     cmovel {{.*}}, %eax
; CHECK:     retq

define zeroext i1 @ret_true() {
entry:
  ret i1 true
}

; CHECK-LABEL: ret_true:
; CHECK:     movb $1, %al
; CHECK:     retq
