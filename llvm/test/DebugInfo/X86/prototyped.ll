; RUN: rm -rf %t
; RUN: mkdir %t

; RUN: llc -mtriple x86_64-unknown-linux-gnu -O0 -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s

; RUN: sed -e "s/LANG_C/LANG_C89/" %s > %t/test.ll
; RUN: llc -mtriple=x86_64-pc-linux-gnu -filetype=obj -o %t/test.o < %t/test.ll
; RUN: llvm-dwarfdump -debug-info %t/test.o | FileCheck %s

; RUN: sed -e "s/LANG_C/LANG_C99/" %s > %t/test.ll
; RUN: llc -mtriple=x86_64-pc-linux-gnu -filetype=obj -o %t/test.o < %t/test.ll
; RUN: llvm-dwarfdump -debug-info %t/test.o | FileCheck %s

; RUN: sed -e "s/LANG_C/LANG_C11/" %s > %t/test.ll
; RUN: llc -mtriple=x86_64-pc-linux-gnu -filetype=obj -o %t/test.o < %t/test.ll
; RUN: llvm-dwarfdump -debug-info %t/test.o | FileCheck %s

; RUN: sed -e "s/LANG_C/LANG_ObjC/" %s > %t/test.ll
; RUN: llc -mtriple=x86_64-pc-linux-gnu -filetype=obj -o %t/test.o < %t/test.ll
; RUN: llvm-dwarfdump -debug-info %t/test.o | FileCheck %s

; Generated from this simple example, compiled as C code:
; void (*x)(void);
; void y(void) { }

; CHECK: DW_TAG_subroutine_type
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_AT_prototyped (true)

; CHECK: DW_TAG_subprogram
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_AT_prototyped (true)

@x = dso_local global ptr null, align 8, !dbg !0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @y() #0 !dbg !16 {
entry:
  ret void, !dbg !18
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "clang version 16.0.0 (git@github.com:llvm/llvm-project.git 4ffde47ab8c2790c8ee2867eccb9f41c329c9e99)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "02049edb47c3f3ebb7dfd6ed0658ec6b")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 8, !"PIC Level", i32 2}
!12 = !{i32 7, !"PIE Level", i32 2}
!13 = !{i32 7, !"uwtable", i32 2}
!14 = !{i32 7, !"frame-pointer", i32 2}
!15 = !{!"clang version 16.0.0 (git@github.com:llvm/llvm-project.git 4ffde47ab8c2790c8ee2867eccb9f41c329c9e99)"}
!16 = distinct !DISubprogram(name: "y", scope: !3, file: !3, line: 2, type: !6, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
!17 = !{}
!18 = !DILocation(line: 2, column: 16, scope: !16)
