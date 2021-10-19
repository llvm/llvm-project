target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

define i32 @foo(i32 %goo) {
entry:
  %goo.addr = alloca i32, align 4
  store i32 %goo, i32* %goo.addr, align 4
  %0 = load i32, i32* %goo.addr, align 4
  %1 = load i32, i32* %goo.addr, align 4
  %mul = mul nsw i32 %0, %1
  ret i32 %mul
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.md = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, enums: !4)
!1 = !DIFile(filename: "foo.cpp", directory: "tmp")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!5 = !{!4}
