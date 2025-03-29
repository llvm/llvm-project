; RUN: opt %loadNPMPolly -S -passes=polly-codegen < %s | FileCheck %s

; CHECK: polly.start

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"

define void @fixup_gotos(ptr %A, ptr %data) nounwind {
entry:
  br label %if

if:
  %cond = icmp eq ptr %A, null
  br i1 %cond, label %last, label %then

then:
  store i32 1, ptr %data, align 4
  br label %last

last:
  ret void
}
