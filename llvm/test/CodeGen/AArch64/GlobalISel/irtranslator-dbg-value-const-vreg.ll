; RUN: llc -O0 -mtriple=aarch64-apple-macosx -global-isel \
; RUN:   -stop-after=ir-translator < %s | FileCheck %s

;; A debug value of a constant that no DBG_VALUE operand can hold, such as the
;; address of an external global, falls back to the vreg the function
;; materializes the constant into anyway.

@ext = external global i64

;; The debug value comes first in the block order, so this only works once the
;; whole function is translated.
; CHECK-LABEL: name: used_later
; CHECK: [[EXT:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @ext
; CHECK: DBG_VALUE [[EXT]](p0), $noreg, ![[#]], !DIExpression()
define void @used_later(ptr %p, i1 %c) !dbg !6 {
entry:
    #dbg_value(ptr @ext, !9, !DIExpression(), !10)
  br i1 %c, label %use, label %exit, !dbg !10

use:
  store ptr @ext, ptr %p, align 8, !dbg !10
  br label %exit, !dbg !10

exit:
  ret void, !dbg !10
}

;; Nothing else needs the address, and materializing it only for the debug value
;; would make codegen depend on debug info.
; CHECK-LABEL: name: unused
; CHECK-NOT: G_GLOBAL_VALUE
; CHECK: DBG_VALUE $noreg, 0, ![[#]], !DIExpression()
; CHECK-NOT: G_GLOBAL_VALUE
define void @unused() !dbg !11 {
entry:
    #dbg_value(ptr @ext, !12, !DIExpression(), !13)
  ret void, !dbg !13
}

;; A poison debug value terminates the location, even if the function also has
;; a vreg for poison.
; CHECK-LABEL: name: kill
; CHECK: G_IMPLICIT_DEF
; CHECK: DBG_VALUE $noreg, 0, ![[#]], !DIExpression()
define void @kill(ptr %p) !dbg !14 {
entry:
  store ptr poison, ptr %p, align 8, !dbg !16
    #dbg_value(ptr poison, !15, !DIExpression(), !16)
  ret void, !dbg !16
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: false, emissionKind: FullDebug)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !DISubroutineType(types: !{null})
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "used_later", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!9 = !DILocalVariable(name: "x", scope: !6, file: !3, line: 2, type: !7)
!10 = !DILocation(line: 2, column: 1, scope: !6)
!11 = distinct !DISubprogram(name: "unused", scope: !3, file: !3, line: 5, type: !4, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !2)
!12 = !DILocalVariable(name: "y", scope: !11, file: !3, line: 6, type: !7)
!13 = !DILocation(line: 6, column: 1, scope: !11)
!14 = distinct !DISubprogram(name: "kill", scope: !3, file: !3, line: 9, type: !4, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !2)
!15 = !DILocalVariable(name: "z", scope: !14, file: !3, line: 10, type: !7)
!16 = !DILocation(line: 10, column: 1, scope: !14)
