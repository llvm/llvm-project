; RUN: llc < %s -mtriple=i686-windows-msvc | FileCheck %s
; RUN: llc < %s -mtriple=i686-windows-gnu | FileCheck %s

define x86_fastcallcc i32 @pass_fast_bool(i1 inreg zeroext %b) {
entry:
  %cond = select i1 %b, i32 66, i32 0
  ret i32 %cond
}

; CHECK-LABEL: @pass_fast_bool@4:
; CHECK-DAG: testb %cl, %cl
; CHECK-DAG: movl    $66,
; CHECK:     retl

define x86_vectorcallcc i32 @pass_vector_bool(i1 inreg zeroext %b) {
entry:
  %cond = select i1 %b, i32 66, i32 0
  ret i32 %cond
}

; CHECK-LABEL: pass_vector_bool@@4:
; CHECK-DAG: testb %cl, %cl
; CHECK-DAG: movl    $66,
; CHECK:     retl

define zeroext i1 @ret_true() {
entry:
  ret i1 true
}

; CHECK-LABEL: ret_true:
; CHECK:     movb $1, %al
; CHECK:     retl
