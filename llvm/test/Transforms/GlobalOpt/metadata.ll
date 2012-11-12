; RUN: opt -S -globalopt < %s | FileCheck %s

; PR6112 - When globalopt does RAUW(@G, %G), the metadata reference should drop
; to null.  Function local metadata that references @G from a different function
; to that containing %G should likewise drop to null.
@G = internal global i8** null

define i32 @main(i32 %argc, i8** %argv) {
; CHECK: @main
; CHECK: %G = alloca
  store i8** %argv, i8*** @G
  ret i32 0
}

define void @foo(i32 %x) {
  call void @llvm.foo(metadata !{i8*** @G, i32 %x})
; CHECK: call void @llvm.foo(metadata !{null, i32 %x})
  ret void
}

declare void @llvm.foo(metadata) nounwind readnone

!named = !{!0}

!0 = metadata !{i8*** @G}
; CHECK: !0 = metadata !{null}
