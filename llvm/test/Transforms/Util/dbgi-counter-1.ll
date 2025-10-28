; RUN: opt -S -passes=dbg-counter < %s 2>&1 | FileCheck %s  

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: Function: foo
; CHECK-NEXT: #dbg_value: 0
; CHECK-NEXT: #dbg_declare: 3
; CHECK-NEXT: #dbg_assign: 0

define dso_local i32 @foo() #0 !dbg !10 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
    #dbg_declare(ptr %1, !16, !DIExpression(), !17)
  store i32 21, ptr %1, align 4, !dbg !17
    #dbg_declare(ptr %2, !18, !DIExpression(), !19)
  store i32 22, ptr %2, align 4, !dbg !19
    #dbg_declare(ptr %3, !20, !DIExpression(), !22)
  store i32 23, ptr %3, align 4, !dbg !22
  %4 = load i32, ptr %1, align 4, !dbg !23
  store i32 %4, ptr %3, align 4, !dbg !24
  %5 = load i32, ptr %2, align 4, !dbg !25
  store i32 %5, ptr %1, align 4, !dbg !26
  %6 = load i32, ptr %1, align 4, !dbg !27
  %7 = load i32, ptr %2, align 4, !dbg !28
  %8 = add nsw i32 %6, %7, !dbg !29
  ret i32 %8, !dbg !30
}

; Function Attrs: noinline nounwind optnone uwtable

; CHECK: Function: main
; CHECK-NEXT: #dbg_value: 0
; CHECK-NEXT: #dbg_declare: 1
; CHECK-NEXT: #dbg_assign: 0

define dso_local i32 @main() #0 !dbg !31 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, ptr %1, align 4
    #dbg_declare(ptr %2, !32, !DIExpression(), !33)
  %3 = call i32 @foo(), !dbg !34
  store i32 %3, ptr %2, align 4, !dbg !33
  %4 = load i32, ptr %2, align 4, !dbg !35
  ret i32 %4, !dbg !36
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 22.0.0git (https://github.com/anamaoh/llvm-project.git 782a91e1fc94d9c82495f60afc5ed5edd72de776)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/home/ana-marija/Documents/foo.c", directory: "/home/ana-marija/Desktop/LLVM/llvm-project", checksumkind: CSK_MD5, checksum: "22bc4641d6e8e63df371a758d86cafff")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 22.0.0git (https://github.com/anamaoh/llvm-project.git 782a91e1fc94d9c82495f60afc5ed5edd72de776)"}
!10 = distinct !DISubprogram(name: "foo", scope: !11, file: !11, line: 1, type: !12, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!11 = !DIFile(filename: "Documents/foo.c", directory: "/home/ana-marija", checksumkind: CSK_MD5, checksum: "22bc4641d6e8e63df371a758d86cafff")
!12 = !DISubroutineType(types: !13)
!13 = !{!14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{}
!16 = !DILocalVariable(name: "X", scope: !10, file: !11, line: 2, type: !14)
!17 = !DILocation(line: 2, column: 6, scope: !10)
!18 = !DILocalVariable(name: "Y", scope: !10, file: !11, line: 3, type: !14)
!19 = !DILocation(line: 3, column: 6, scope: !10)
!20 = !DILocalVariable(name: "Z", scope: !21, file: !11, line: 5, type: !14)
!21 = distinct !DILexicalBlock(scope: !10, file: !11, line: 4, column: 2)
!22 = !DILocation(line: 5, column: 8, scope: !21)
!23 = !DILocation(line: 6, column: 8, scope: !21)
!24 = !DILocation(line: 6, column: 6, scope: !21)
!25 = !DILocation(line: 8, column: 6, scope: !10)
!26 = !DILocation(line: 8, column: 4, scope: !10)
!27 = !DILocation(line: 9, column: 9, scope: !10)
!28 = !DILocation(line: 9, column: 13, scope: !10)
!29 = !DILocation(line: 9, column: 11, scope: !10)
!30 = !DILocation(line: 9, column: 2, scope: !10)
!31 = distinct !DISubprogram(name: "main", scope: !11, file: !11, line: 12, type: !12, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!32 = !DILocalVariable(name: "A", scope: !31, file: !11, line: 13, type: !14)
!33 = !DILocation(line: 13, column: 6, scope: !31)
!34 = !DILocation(line: 13, column: 10, scope: !31)
!35 = !DILocation(line: 14, column: 9, scope: !31)
!36 = !DILocation(line: 14, column: 2, scope: !31)