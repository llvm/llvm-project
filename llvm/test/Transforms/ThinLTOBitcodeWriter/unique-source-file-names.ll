; RUN: opt -thinlto-bc -thin-link-bitcode-file=%t2 -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 1 -o %t1 %t
; RUN: llvm-dis -o - %t1 | FileCheck %s

source_filename = "unique-source-file-names.c"

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @f, ptr null }]

; CHECK: @g.45934e8a5251fb7adbecfff71a4e70ed =
@g = internal global i8 42, !type !0

declare void @sink(ptr)

define internal void @f() {
  call void @sink(ptr @g)
  ret void
}

!0 = !{i32 0, !"typeid"}

!llvm.module.flags = !{!1}
!1 = !{i32 5, !"Unique Source File Identifier", !2}
!2 = !{!"unique-source-file-names.c"}
