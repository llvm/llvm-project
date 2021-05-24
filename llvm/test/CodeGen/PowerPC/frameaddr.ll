; RUN: llc -verify-machineinstrs < %s -mcpu=pwr7 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare void @llvm.eh.sjlj.longjmp(i8*) #1

define i8* @main() #0 {
entry:
  %0 = call i8* @llvm.frameaddress(i32 0)
  ret i8* %0

; CHECK: @main
; CHECK: mr 3, 1
}

define i8* @foo() #3 { ; naked
entry:
  %0 = call i8* @llvm.frameaddress(i32 0)
  ret i8* %0

; CHECK: @foo
; CHECK: mr 3, 1
}

define i8* @bar() #0 {
entry:
  %x = alloca [100000 x i8]                       ; <[100000 x i8]*> [#uses=1]
  %x1 = bitcast [100000 x i8]* %x to i8*          ; <i8*> [#uses=1]
  call void @use(i8* %x1) nounwind
  %0 = call i8* @llvm.frameaddress(i32 0)
  ret i8* %0

; Note that if we start eliminating non-leaf frame pointers by default, this
; will need to be updated.
; CHECK: @bar
; CHECK: mr 3, 31
}

declare void @use(i8*)

declare i8* @llvm.frameaddress(i32) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind naked "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

