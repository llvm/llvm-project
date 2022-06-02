; RUN: llvm-as %s -disable-output 2>&1 | FileCheck %s

; CHECK:      function declaration may only have a unique !dbg attachment
declare !dbg !4 void @f1()

; CHECK-NOT:      function declaration may only have a unique !dbg attachment
declare !dbg !6 void @f5()

; CHECK:      function must have a single !dbg attachment
define void @f2() !dbg !4 !dbg !4 {
  unreachable
}

; CHECK:      DISubprogram attached to more than one function
define void @f3() !dbg !4 {
  unreachable
}

; CHECK:      DISubprogram attached to more than one function
define void @f4() !dbg !4 {
  unreachable
}

; CHECK-NOT:  !dbg
; CHECK:      function !dbg attachment must be a subprogram
; CHECK-NEXT: ptr @bar
; CHECK-NEXT: !{{[0-9]+}} = !{}
define void @bar() !dbg !3 {
  unreachable
}

; CHECK: warning: ignoring invalid debug info
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!1}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, retainedTypes: !5)
!2 = !DIFile(filename: "t.c", directory: "/path/to/dir")
!3 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !2, unit: !1)
!5 = !{!6}
!6 = !DISubprogram(name: "f5", scope: !1, file: !2, unit: !1, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
