; RUN: opt %loadPolly -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true -disable-output < %s
;
; CHECK: Execution Context: [p_0_loaded_from_currpc] -> {  :  }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@currpc = external global i32, align 4
@inbuff = external global i8*, align 8

; Function Attrs: uwtable
define void @_Z13dotableswitchP9Classfile() {
entry:
  br i1 undef, label %for.end, label %while.body

while.body:                                       ; preds = %while.body, %entry
  store i8* undef, i8** @inbuff, align 8
  %0 = load i32, i32* @currpc, align 4
  %rem = and i32 %0, 3
  %tobool = icmp eq i32 %rem, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %while.end
  br label %for.body

for.end:                                          ; preds = %while.end, %entry
  ret void
}
