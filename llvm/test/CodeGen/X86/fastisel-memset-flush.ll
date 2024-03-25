; RUN: llc -mtriple=x86_64 -O0 < %s | FileCheck %s

define dso_local void @foo() !dbg !7 {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4, !dbg !9
  %0 = bitcast ptr %a to ptr, !dbg !10
  call void @llvm.memset.p0.i64(ptr align 4 %0, i8 -86, i64 4, i1 false), !dbg !10
  %1 = bitcast ptr %a to ptr, !dbg !11
  call void @other(ptr %1), !dbg !12
  ret void, !dbg !13
}
; CHECK:      callq memset
; CHECK-NEXT: .loc 1 9 9
; CHECK-NEXT: leaq
; CHECK-NEXT: .loc 1 9 3
; CHECK-NEXT: callq other

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

declare dso_local void @other(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0 (https://github.com/llvm/llvm-project eaae6fdf67e1f61599331d69a41a7dafe6199667)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "memset-test.c", directory: "/home/probinson/projects/scratch")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project eaae6fdf67e1f61599331d69a41a7dafe6199667)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, type: !8, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 6, column: 7, scope: !7)
!10 = !DILocation(line: 8, column: 3, scope: !7)
!11 = !DILocation(line: 9, column: 9, scope: !7)
!12 = !DILocation(line: 9, column: 3, scope: !7)
!13 = !DILocation(line: 10, column: 1, scope: !7)
