; RUN: opt -hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; The block "if.end" in @foo is extracted into a new function, @foo.cold.1.
; Check the following:

; CHECK-LABEL: define {{.*}}@foo.cold.1

; - The llvm.dbg.value intrinsic pointing to an argument in @foo (%arg1) is
;   dropped
; CHECK-NOT: llvm.dbg.value

; - Instructions without locations in the original function have no
;   location in the new function
; CHECK:      [[ADD1:%.*]] = add i32 %{{.*}}, 1{{$}}

; - Ditto (see above), calls are not special
; CHECK-NEXT: call void @sink(i32 [[ADD1]])

; - Line locations are preserved
; CHECK-NEXT: call void @sink(i32 [[ADD1]]), !dbg [[LINE1:![0-9]+]]

; - llvm.dbg.value intrinsics for values local to @foo.cold.1 are preserved
; CHECK-NEXT: llvm.dbg.value(metadata i32 [[ADD1]], metadata [[VAR1:![0-9]+]], metadata !DIExpression()), !dbg [[LINE1]]

; - Expressions inside of dbg.value intrinsics are preserved
; CHECK-NEXT: llvm.dbg.value(metadata i32 [[ADD1]], metadata [[VAR1]], metadata !DIExpression(DW_OP_constu, 1, DW_OP_plus, DW_OP_stack_value)

; - The DISubprogram for @foo.cold.1 has an empty DISubroutineType
; CHECK: [[FILE:![0-9]+]] = !DIFile(filename: "<stdin>"
; CHECK: [[EMPTY_MD:![0-9]+]] = !{}
; CHECK: [[EMPTY_TYPE:![0-9]+]] = !DISubroutineType(types: [[EMPTY_MD]])
; CHECK: [[NEWSCOPE:![0-9]+]] = distinct !DISubprogram(name: "foo.cold.1", linkageName: "foo.cold.1", scope: null, file: [[FILE]], type: [[EMPTY_TYPE]], spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized

; - Line locations in @foo.cold.1 point to the new scope for @foo.cold.1
; CHECK: [[LINE1]] = !DILocation(line: 1, column: 1, scope: [[NEWSCOPE]])

define void @foo(i32 %arg1) !dbg !6 {
entry:
  %var = add i32 0, 0, !dbg !11
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  ret void

if.end:                                           ; preds = %entry
  call void @llvm.dbg.value(metadata i32 %arg1, metadata !9, metadata !DIExpression()), !dbg !11
  %add1 = add i32 %arg1, 1
  call void @sink(i32 %add1)
  call void @sink(i32 %add1), !dbg !11
  call void @llvm.dbg.value(metadata i32 %add1, metadata !9, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.value(metadata i32 %add1, metadata !9, metadata !DIExpression(DW_OP_constu, 1, DW_OP_plus, DW_OP_stack_value)), !dbg !11
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

declare void @sink(i32) cold

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 7}
!4 = !{i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !6)
