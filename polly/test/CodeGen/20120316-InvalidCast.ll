; RUN: opt %loadPolly %defaultOpts -polly-codegen %s

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"
target triple = "hexagon-unknown-linux-gnu"

define void @fixup_gotos(i32* %A, i32* %data) nounwind {
entry:
  br label %if

if:
  %cond = icmp eq i32* %A, null
  br i1 %cond, label %last, label %then

then:
  store i32 1, i32* %data, align 4
  br label %last

last:
  ret void
}
