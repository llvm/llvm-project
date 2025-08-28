; RUN: llc < %s -mtriple=nvptx64 -O2 -mcpu=sm_52 -mattr=+ptx62 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_52 -mattr=+ptx62 | %ptxas-verify %}

declare i32 @llvm.nvvm.activemask()

; CHECK-LABEL: activemask(
;
;      CHECK: activemask.b32  %[[REG:.+]];
; CHECK-NEXT: st.param.b32    [func_retval0], %[[REG]];
; CHECK-NEXT: ret;
define dso_local i32 @activemask() {
entry:
  %mask = call i32 @llvm.nvvm.activemask()
  ret i32 %mask
}

; CHECK-LABEL: convergent(
;
;      CHECK: activemask.b32  %[[REG:.+]];
;      CHECK: activemask.b32  %[[REG]];
;      CHECK: .param.b32    [func_retval0], %[[REG]];
; CHECK-NEXT: ret;
define dso_local i32 @convergent(i1 %cond) {
entry:
  br i1 %cond, label %if.else, label %if.then

if.then:
  %0 = call i32 @llvm.nvvm.activemask()
  br label %if.end

if.else:
  %1 = call i32 @llvm.nvvm.activemask()
  br label %if.end

if.end:
  %mask = phi i32 [ %0, %if.then ], [ %1, %if.else ]
  ret i32 %mask
}
