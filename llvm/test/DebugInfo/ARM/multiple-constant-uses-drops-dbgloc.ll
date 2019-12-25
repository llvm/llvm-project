; RUN: llc -filetype=asm -asm-verbose=0 < %s | FileCheck %s

; char ch;
; int b;
;
; void proc (void)
; {
;     ch = 'A';
;     b = 0; // <== this should have correct location
; }

; CHECK: .loc 1 7 7
; CHECK: mov  r{{[0-9]}}, #0

source_filename = "test/DebugInfo/ARM/multiple-constant-uses-drops-dbgloc.ll"
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabihf"

@ch = common global i8 0, align 1, !dbg !0
@b = common global i32 0, align 4, !dbg !5

; Function Attrs: nounwind
define void @proc() #0 !dbg !13 {
entry:
  store i8 65, i8* @ch, align 1, !dbg !17
  store i32 0, i32* @b, align 4, !dbg !18
  ret void, !dbg !19
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a8" "target-features"="+neon,+vfp3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "ch", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!3 = !DIFile(filename: "test.c", directory: "/home/user/clang/build")
!4 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 1, !"min_enum_size", i32 4}
!12 = !{!"clang version 3.7.0"}
!13 = distinct !DISubprogram(name: "proc", scope: !3, file: !3, line: 4, type: !14, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{null}
!16 = !{}
!17 = !DILocation(line: 6, column: 8, scope: !13)
!18 = !DILocation(line: 7, column: 7, scope: !13)
!19 = !DILocation(line: 8, column: 1, scope: !13)

