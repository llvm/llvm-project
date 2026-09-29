; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

; Hoisted inlinable calls need to have accurate scope information, but we're
; allowed to erase the line information.

source_filename = "t.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

; Function Attrs: noinline nounwind readnone uwtable
define float @fabsf(float %f) #0 !dbg !7 {
entry:
  %conv = fpext float %f to double, !dbg !9
  %call = call double @fabs(double %conv) #1, !dbg !10
  %conv1 = fptrunc double %call to float, !dbg !11
  ret float %conv1, !dbg !12
}

; Function Attrs: nounwind readnone
declare double @fabs(double) #1

; Function Attrs: noinline nounwind uwtable
define void @hoistit(i32 %cond, float %f) #2 !dbg !13 {
entry:
  %tobool = icmp ne i32 %cond, 0, !dbg !14
  br i1 %tobool, label %if.then, label %if.else, !dbg !14

if.then:                                          ; preds = %entry
  %call = call float @fabsf(float %f) #1, !dbg !15
  call void @useit1(float %call), !dbg !16
  br label %if.end, !dbg !18

if.else:                                          ; preds = %entry
  %call1 = call float @fabsf(float %f) #1, !dbg !19
  call void @useit2(float %call1), !dbg !20
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !21
}

; CHECK-LABEL: define void @hoistit
; CHECK-SAME: 		!dbg ![[sp_hoistit:[0-9]+]]
; CHECK: call float @fabsf(float %f) {{.*}} !dbg ![[dbgloc:[0-9]+]]
; CHECK: br i1 %tobool, label %if.then, label %if.else

; CHECK: ![[sp_hoistit]] = distinct !DISubprogram(name: "hoistit", {{.*}})
; CHECK: ![[dbgloc]] = !DILocation({{.*}}, scope: ![[sp_hoistit]])

declare void @useit1(float)

declare void @useit2(float)

attributes #0 = { noinline nounwind readnone uwtable }
attributes #1 = { nounwind readnone willreturn }
attributes #2 = { noinline nounwind uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 5.0.0 "}
!7 = distinct !DISubprogram(name: "fabsf", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 5, column: 22, scope: !7)
!10 = !DILocation(line: 5, column: 17, scope: !7)
!11 = !DILocation(line: 5, column: 10, scope: !7)
!12 = !DILocation(line: 5, column: 3, scope: !7)
!13 = distinct !DISubprogram(name: "hoistit", scope: !1, file: !1, line: 7, type: !8, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 8, column: 7, scope: !13)
!15 = !DILocation(line: 9, column: 12, scope: !13)
!16 = !DILocation(line: 9, column: 5, scope: !17)
!17 = !DILexicalBlockFile(scope: !13, file: !1, discriminator: 1)
!18 = !DILocation(line: 10, column: 3, scope: !13)
!19 = !DILocation(line: 11, column: 12, scope: !13)
!20 = !DILocation(line: 11, column: 5, scope: !17)
!21 = !DILocation(line: 13, column: 1, scope: !13)
