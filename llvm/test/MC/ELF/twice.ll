; Check that there is no persistent state in the ELF emitter that crashes us
; when we try to reuse the pass manager
; RUN: llc -compile-twice -filetype=obj %s -o -

; RUN: llc -compile-twice -filetype=obj -save-temp-labels %s -o - | llvm-objdump -d - | FileCheck %s

; CHECK-LABEL: <foo>:
; CHECK:         je {{.*}} <.LBB0_2>
; CHECK-LABEL: <.LBB0_2>:

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

@var = external local_unnamed_addr global i32

define dso_local void @foo(i32 %a) {
entry:
  %tobool.not = icmp eq i32 %a, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  store i32 1, ptr @var
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}
