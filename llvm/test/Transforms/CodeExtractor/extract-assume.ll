; RUN: opt -passes="function(slp-vectorizer),module(hotcoldsplit),function(slp-vectorizer,print<assumptions>)" -hotcoldsplit-threshold=-1 -disable-output %s 2>&1 | FileCheck %s
;
; Make sure this compiles. Check that function assumption cache is refreshed
; after extracting blocks with assume calls from the function.

; CHECK: Cached assumptions for function: fun
; CHECK-NEXT: Cached assumptions for function: fun.cold
; CHECK-NOT: icmp uge

declare void @fun2(i32) #0

define void @fun(i32 %x) {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  ret void

if.else:
  %cmp = icmp uge i32 %x, 64
  call void @llvm.assume(i1 %cmp)
  call void @fun2(i32 %x)
  unreachable
}

declare void @llvm.assume(i1) #1

attributes #0 = { alwaysinline }
attributes #1 = { nounwind }
