; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

; CHECK:      function declaration may not have a !dbg attachment
declare !dbg !4 void @f1()

define void @f2() !dbg !4 {
  unreachable
}

; CHECK:      function must have a single !dbg attachment
define void @f3() !dbg !4 !dbg !4 {
  unreachable
}

; CHECK-NOT:  !dbg
; CHECK:      function !dbg attachment must be a subprogram
; CHECK-NEXT: void ()* @bar
; CHECK-NEXT: !{{[0-9]+}} = !{}
define void @bar() !dbg !6 {
  unreachable
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!1}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2)
!2 = !DIFile(filename: "t.c", directory: "/path/to/dir")
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !2, unit: !1)
!6 = !{}
