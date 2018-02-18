; RUN: llc -O0 -mtriple=aarch64-- -stop-after=livedebugvalues -fast-isel=true < %s | FileCheck %s

; CHECK: ![[LOCAL:[0-9]+]] = !DILocalVariable(name: "__vla_expr",
; CHECK: DBG_VALUE {{.*}} ![[LOCAL]]

; Function Attrs: noinline nounwind optnone uwtable
define void @foo(i32 %n) local_unnamed_addr #0 !dbg !7 {
entry:
  %0 = zext i32 %n to i64, !dbg !11
  %1 = call i8* @llvm.stacksave(), !dbg !12
  call void @llvm.dbg.value(metadata i64 %0, metadata !13, metadata !DIExpression()), !dbg !12
  %vla.i = alloca i32, i64 %0, align 16, !dbg !12
  call void @llvm.stackrestore(i8* %1), !dbg !12
  ret void, !dbg !12
}

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #1

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.c", directory: "/path/to/build")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 39, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 2, column: 5, scope: !7)
!12 = !DILocation(line: 4, column: 5, scope: !7)
!13 = !DILocalVariable(name: "__vla_expr", scope: !14, type: !15, flags: DIFlagArtificial)
!14 = distinct !DILexicalBlock(scope: !7, file: !1, line: 32, column: 31)
!15 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
