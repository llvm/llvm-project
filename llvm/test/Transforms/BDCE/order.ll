; RUN: opt -bdce -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @__gxx_personality_v0(...)

define fastcc void @_ZN11__sanitizerL12TestRegistryEPNS_14ThreadRegistryEb() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br i1 undef, label %if.else, label %entry.if.end_crit_edge

if.else:
  ret void

invoke.cont70:
  store i32 %call71, i32* undef, align 4
  br label %if.else

; CHECK-LABEL: @_ZN11__sanitizerL12TestRegistryEPNS_14ThreadRegistryEb
; CHECK: store i32 %call71

lpad65.loopexit.split-lp.loopexit.split-lp:
  br label %if.else

lpad65.loopexit.split-lp.loopexit.split-lp.loopexit:
  %lpad.loopexit1121 = landingpad { i8*, i32 }
          cleanup
  br label %lpad65.loopexit.split-lp.loopexit.split-lp

entry.if.end_crit_edge:
  %call71 = invoke i32 @_ZN11__sanitizer14ThreadRegistry12CreateThreadEmbjPv()
          to label %invoke.cont70 unwind label %lpad65.loopexit.split-lp.loopexit.split-lp.loopexit
}

declare i32 @_ZN11__sanitizer14ThreadRegistry12CreateThreadEmbjPv()

attributes #0 = { uwtable }

