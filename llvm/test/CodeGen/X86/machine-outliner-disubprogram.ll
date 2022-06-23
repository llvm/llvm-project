; Ensure that we can correctly emit a compile unit for outlined functions and
; that we correctly emit DISubprograms for those functions.
; Also make sure that the DISubprograms reference the generated unit.
; make sure that if there are two outlined functions in the program, 
; RUN: llc %s -verify-machineinstrs -enable-machine-outliner -mtriple=x86_64-apple-darwin -o /dev/null -print-after=machine-outliner
define void @f6() #0 !dbg !8 {
entry:
  %dog = alloca i32, align 4
  %cat = alloca i32, align 4
  %pangolin = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %dog, metadata !11, metadata !DIExpression()), !dbg !13
  store i32 16, ptr %dog, align 4, !dbg !13
  call void @llvm.dbg.declare(metadata ptr %cat, metadata !14, metadata !DIExpression()), !dbg !15
  store i32 32, ptr %cat, align 4, !dbg !15
  call void @llvm.dbg.declare(metadata ptr %pangolin, metadata !16, metadata !DIExpression()), !dbg !17
  store i32 48, ptr %pangolin, align 4, !dbg !17
  ret void, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define void @f5() #0 !dbg !19 {
entry:
  %dog = alloca i32, align 4
  %cat = alloca i32, align 4
  %pangolin = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %dog, metadata !20, metadata !DIExpression()), !dbg !21
  store i32 16, ptr %dog, align 4, !dbg !21
  call void @llvm.dbg.declare(metadata ptr %cat, metadata !22, metadata !DIExpression()), !dbg !23
  store i32 32, ptr %cat, align 4, !dbg !23
  call void @llvm.dbg.declare(metadata ptr %pangolin, metadata !24, metadata !DIExpression()), !dbg !25
  store i32 48, ptr %pangolin, align 4, !dbg !25
  ret void, !dbg !26
}

define void @f4() #0 !dbg !27 {
entry:
  %dog = alloca i32, align 4
  %cat = alloca i32, align 4
  %pangolin = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %dog, metadata !28, metadata !DIExpression()), !dbg !29
  store i32 16, ptr %dog, align 4, !dbg !29
  call void @llvm.dbg.declare(metadata ptr %cat, metadata !30, metadata !DIExpression()), !dbg !31
  store i32 32, ptr %cat, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %pangolin, metadata !32, metadata !DIExpression()), !dbg !33
  store i32 48, ptr %pangolin, align 4, !dbg !33
  ret void, !dbg !34
}

define i32 @f1() #0 !dbg !35 {
entry:
  %dog = alloca i32, align 4
  %cat = alloca i32, align 4
  %pangolin = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %dog, metadata !38, metadata !DIExpression()), !dbg !39
  store i32 1, ptr %dog, align 4, !dbg !39
  call void @llvm.dbg.declare(metadata ptr %cat, metadata !40, metadata !DIExpression()), !dbg !41
  store i32 2, ptr %cat, align 4, !dbg !41
  call void @llvm.dbg.declare(metadata ptr %pangolin, metadata !42, metadata !DIExpression()), !dbg !43
  store i32 3, ptr %pangolin, align 4, !dbg !43
  store i32 16, ptr %dog, align 4, !dbg !44
  %0 = load i32, ptr %dog, align 4, !dbg !45
  ret i32 %0, !dbg !46
}

define i32 @f2() #0 !dbg !47 {
entry:
  %dog = alloca i32, align 4
  %cat = alloca i32, align 4
  %pangolin = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %dog, metadata !48, metadata !DIExpression()), !dbg !49
  store i32 1, ptr %dog, align 4, !dbg !49
  call void @llvm.dbg.declare(metadata ptr %cat, metadata !50, metadata !DIExpression()), !dbg !51
  store i32 2, ptr %cat, align 4, !dbg !51
  call void @llvm.dbg.declare(metadata ptr %pangolin, metadata !52, metadata !DIExpression()), !dbg !53
  store i32 3, ptr %pangolin, align 4, !dbg !53
  store i32 16, ptr %dog, align 4, !dbg !54
  %0 = load i32, ptr %dog, align 4, !dbg !55
  ret i32 %0, !dbg !56
}

define i32 @f3() #0 !dbg !57 {
entry:
  %dog = alloca i32, align 4
  %cat = alloca i32, align 4
  %pangolin = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %dog, metadata !58, metadata !DIExpression()), !dbg !59
  store i32 1, ptr %dog, align 4, !dbg !59
  call void @llvm.dbg.declare(metadata ptr %cat, metadata !60, metadata !DIExpression()), !dbg !61
  store i32 2, ptr %cat, align 4, !dbg !61
  call void @llvm.dbg.declare(metadata ptr %pangolin, metadata !62, metadata !DIExpression()), !dbg !63
  store i32 3, ptr %pangolin, align 4, !dbg !63
  store i32 16, ptr %dog, align 4, !dbg !64
  %0 = load i32, ptr %dog, align 4, !dbg !65
  ret i32 %0, !dbg !66
}

define i32 @main() #0 !dbg !67 {
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  call void @llvm.dbg.declare(metadata ptr %a, metadata !68, metadata !DIExpression()), !dbg !69
  store i32 4, ptr %a, align 4, !dbg !69
  %call = call i32 @f1() #2, !dbg !70
  %call1 = call i32 @f2() #2, !dbg !71
  %call2 = call i32 @f3() #2, !dbg !72
  ret i32 0, !dbg !73
}

; CHECK: distinct !DISubprogram(name: "OUTLINED_FUNCTION_1",
; CHECK-SAME: scope: !1,
; CHECK-SAME: file: !1,
; CHECK-SAME: type: [[TYPE:![0-9]+]],
; CHECK-SAME: isLocal: false, 
; CHECK-SAME: isDefinition: true,
; CHECK-SAME: flags: DIFlagArtificial,
; CHECK-SAME: isOptimized: true,
; CHECK-SAME: unit: !0,
; CHECK-SAME: retainedNodes: [[VARS:![0-9]+]]

; CHECK: distinct !DISubprogram(name: "OUTLINED_FUNCTION_0",
; CHECK-SAME: scope: !1,
; CHECK-SAME: file: !1,
; CHECK-SAME: type: [[TYPE]],
; CHECK-SAME: isLocal: false, 
; CHECK-SAME: isDefinition: true,
; CHECK-SAME: flags: DIFlagArtificial,
; CHECK-SAME: isOptimized: true,
; CHECK-SAME: unit: !0,
; CHECK-SAME: retainedNodes: [[VARS]]

attributes #0 = { noinline noredzone nounwind optnone ssp uwtable "frame-pointer"="all"  }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noredzone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang"}
!8 = distinct !DISubprogram(name: "f6", scope: !1, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocalVariable(name: "dog", scope: !8, file: !1, line: 4, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 4, column: 16, scope: !8)
!14 = !DILocalVariable(name: "cat", scope: !8, file: !1, line: 5, type: !12)
!15 = !DILocation(line: 5, column: 16, scope: !8)
!16 = !DILocalVariable(name: "pangolin", scope: !8, file: !1, line: 6, type: !12)
!17 = !DILocation(line: 6, column: 16, scope: !8)
!18 = !DILocation(line: 7, column: 1, scope: !8)
!19 = distinct !DISubprogram(name: "f5", scope: !1, file: !1, line: 9, type: !9, isLocal: false, isDefinition: true, scopeLine: 9, isOptimized: false, unit: !0, retainedNodes: !2)
!20 = !DILocalVariable(name: "dog", scope: !19, file: !1, line: 10, type: !12)
!21 = !DILocation(line: 10, column: 16, scope: !19)
!22 = !DILocalVariable(name: "cat", scope: !19, file: !1, line: 11, type: !12)
!23 = !DILocation(line: 11, column: 16, scope: !19)
!24 = !DILocalVariable(name: "pangolin", scope: !19, file: !1, line: 12, type: !12)
!25 = !DILocation(line: 12, column: 16, scope: !19)
!26 = !DILocation(line: 13, column: 1, scope: !19)
!27 = distinct !DISubprogram(name: "f4", scope: !1, file: !1, line: 15, type: !9, isLocal: false, isDefinition: true, scopeLine: 15, isOptimized: false, unit: !0, retainedNodes: !2)
!28 = !DILocalVariable(name: "dog", scope: !27, file: !1, line: 16, type: !12)
!29 = !DILocation(line: 16, column: 16, scope: !27)
!30 = !DILocalVariable(name: "cat", scope: !27, file: !1, line: 17, type: !12)
!31 = !DILocation(line: 17, column: 16, scope: !27)
!32 = !DILocalVariable(name: "pangolin", scope: !27, file: !1, line: 18, type: !12)
!33 = !DILocation(line: 18, column: 16, scope: !27)
!34 = !DILocation(line: 19, column: 1, scope: !27)
!35 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 21, type: !36, isLocal: false, isDefinition: true, scopeLine: 21, isOptimized: false, unit: !0, retainedNodes: !2)
!36 = !DISubroutineType(types: !37)
!37 = !{!12}
!38 = !DILocalVariable(name: "dog", scope: !35, file: !1, line: 22, type: !12)
!39 = !DILocation(line: 22, column: 16, scope: !35)
!40 = !DILocalVariable(name: "cat", scope: !35, file: !1, line: 23, type: !12)
!41 = !DILocation(line: 23, column: 16, scope: !35)
!42 = !DILocalVariable(name: "pangolin", scope: !35, file: !1, line: 24, type: !12)
!43 = !DILocation(line: 24, column: 16, scope: !35)
!44 = !DILocation(line: 25, column: 7, scope: !35)
!45 = !DILocation(line: 26, column: 10, scope: !35)
!46 = !DILocation(line: 26, column: 3, scope: !35)
!47 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 29, type: !36, isLocal: false, isDefinition: true, scopeLine: 29, isOptimized: false, unit: !0, retainedNodes: !2)
!48 = !DILocalVariable(name: "dog", scope: !47, file: !1, line: 30, type: !12)
!49 = !DILocation(line: 30, column: 16, scope: !47)
!50 = !DILocalVariable(name: "cat", scope: !47, file: !1, line: 31, type: !12)
!51 = !DILocation(line: 31, column: 16, scope: !47)
!52 = !DILocalVariable(name: "pangolin", scope: !47, file: !1, line: 32, type: !12)
!53 = !DILocation(line: 32, column: 16, scope: !47)
!54 = !DILocation(line: 33, column: 7, scope: !47)
!55 = !DILocation(line: 34, column: 10, scope: !47)
!56 = !DILocation(line: 34, column: 3, scope: !47)
!57 = distinct !DISubprogram(name: "f3", scope: !1, file: !1, line: 37, type: !36, isLocal: false, isDefinition: true, scopeLine: 37, isOptimized: false, unit: !0, retainedNodes: !2)
!58 = !DILocalVariable(name: "dog", scope: !57, file: !1, line: 38, type: !12)
!59 = !DILocation(line: 38, column: 16, scope: !57)
!60 = !DILocalVariable(name: "cat", scope: !57, file: !1, line: 39, type: !12)
!61 = !DILocation(line: 39, column: 16, scope: !57)
!62 = !DILocalVariable(name: "pangolin", scope: !57, file: !1, line: 40, type: !12)
!63 = !DILocation(line: 40, column: 16, scope: !57)
!64 = !DILocation(line: 41, column: 7, scope: !57)
!65 = !DILocation(line: 42, column: 10, scope: !57)
!66 = !DILocation(line: 42, column: 3, scope: !57)
!67 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 45, type: !36, isLocal: false, isDefinition: true, scopeLine: 45, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!68 = !DILocalVariable(name: "a", scope: !67, file: !1, line: 46, type: !12)
!69 = !DILocation(line: 46, column: 16, scope: !67)
!70 = !DILocation(line: 47, column: 3, scope: !67)
!71 = !DILocation(line: 48, column: 3, scope: !67)
!72 = !DILocation(line: 49, column: 3, scope: !67)
!73 = !DILocation(line: 51, column: 3, scope: !67)
