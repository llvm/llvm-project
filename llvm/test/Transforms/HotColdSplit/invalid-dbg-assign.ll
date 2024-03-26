; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=-1 -S %s | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -passes=hotcoldsplit -hotcoldsplit-threshold=-1 -S %s | FileCheck %s
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

; CHECK: define void @foo
; CHECK-NOT: dbg.assign
; CHECK: call void @foo.cold
; CHECK-NOT: dbg.assign
; CHECK: define internal void @foo.cold
; CHECK-NOT: dbg.assign
define void @foo() !dbg !10 {
  %buf.i = alloca i32, align 4, !DIAssignID !8
  br i1 false, label %if.else, label %if.then

if.then:                                          ; preds = %0
  call void @llvm.dbg.assign(metadata i1 undef, metadata !9, metadata !DIExpression(), metadata !8, metadata ptr %buf.i, metadata !DIExpression()), !dbg !14
  unreachable

if.else:                                          ; preds = %0
  ret void
}


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "blah", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2)
!1 = !DIFile(filename: "blah", directory: "blah")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DIAssignID()
!9 = !DILocalVariable(name: "buf", scope: !10, file: !1, line: 1774, type: !13)
!10 = distinct !DISubprogram(name: "blah", scope: !1, file: !1, line: 1771, type: !11, scopeLine: 1773, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(cc: DW_CC_nocall, types: !7)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!14 = !DILocation(line: 0, scope: !10)
