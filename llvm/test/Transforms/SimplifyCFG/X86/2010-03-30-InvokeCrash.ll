; RUN: opt -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output < %s
; END.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @bar(i32)

define void @foo() personality i32 (...)* @__gxx_personality_v0 {
entry:
 invoke void @bar(i32 undef)
         to label %r unwind label %u

r:                                                ; preds = %entry
 ret void

u:                                                ; preds = %entry
 %val = landingpad { i8*, i32 }
          cleanup
 resume { i8*, i32 } %val
}

declare i32 @__gxx_personality_v0(...)
