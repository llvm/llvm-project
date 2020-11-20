; RUN: opt -S -inline < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s

; By inlining foo, an alloca is created in main to hold the byval argument, so
; a lifetime marker should be generated as well by default.

%struct.foo = type { i32, [16 x i32] }

@gFoo = global %struct.foo zeroinitializer, align 8

define i32 @foo(%struct.foo* byval(%struct.foo) align 8 %f, i32 %a) {
entry:
  %a1 = getelementptr inbounds %struct.foo, %struct.foo* %f, i32 0, i32 1
  %arrayidx = getelementptr inbounds [16 x i32], [16 x i32]* %a1, i32 0, i32 %a
  %tmp2 = load i32, i32* %arrayidx, align 1
  ret i32 %tmp2
}

define i32 @main(i32 %argc, i8** %argv) {
; CHECK-LABEL: @main
; CHECK: llvm.lifetime.start
; CHECK: memcpy
entry:
  %call = call i32 @foo(%struct.foo* byval(%struct.foo) align 8 @gFoo, i32 %argc)
  ret i32 %call
}
