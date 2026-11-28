; RUN: opt -passes=inline -S < %s | FileCheck %s

;; A call result used only by debug records should be replaced by the inlined
;; return value instead of becoming poison.

define internal i32 @forward(i32 %x, ptr %p) {
  store volatile i32 %x, ptr %p, align 4
  ret i32 %x
}

define internal i32 @forward_cond(i32 %x, ptr %p, i1 %c) {
entry:
  br i1 %c, label %then, label %exit

then:
  store volatile i32 %x, ptr %p, align 4
  br label %exit

exit:
  ret i32 %x
}

; CHECK-LABEL: define void @single_block(
; CHECK-SAME: i32 [[X:%.*]], ptr
; CHECK: #dbg_value(i32 [[X]], ![[#]], !DIExpression(),
define void @single_block(i32 %x, ptr %p) !dbg !4 {
  %r = call i32 @forward(i32 %x, ptr %p), !dbg !10
    #dbg_value(i32 %r, !8, !DIExpression(), !10)
  ret void, !dbg !10
}

; CHECK-LABEL: define void @multi_block(
; CHECK-SAME: i32 [[X:%.*]], ptr
; CHECK: forward_cond.exit:
; CHECK-NEXT: #dbg_value(i32 [[X]], ![[#]], !DIExpression(),
define void @multi_block(i32 %x, ptr %p, i1 %c) !dbg !11 {
  %r = call i32 @forward_cond(i32 %x, ptr %p, i1 %c), !dbg !13
    #dbg_value(i32 %r, !12, !DIExpression(), !13)
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !{null})
!4 = distinct !DISubprogram(name: "single_block", scope: !1, file: !1, line: 1, type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILocalVariable(name: "a", scope: !4, file: !1, line: 2, type: !5)
!10 = !DILocation(line: 2, scope: !4)
!11 = distinct !DISubprogram(name: "multi_block", scope: !1, file: !1, line: 5, type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DILocalVariable(name: "c", scope: !11, file: !1, line: 6, type: !5)
!13 = !DILocation(line: 6, scope: !11)
