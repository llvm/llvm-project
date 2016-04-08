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

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabihf"

@ch = common global i8 0, align 1
@b = common global i32 0, align 4

; Function Attrs: nounwind
define void @proc() #0 !dbg !4 {
entry:
  store i8 65, i8* @ch, align 1, !dbg !17
  store i32 0, i32* @b, align 4, !dbg !18
  ret void, !dbg !19
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a8" "target-features"="+neon,+vfp3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, subprograms: !3, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/home/user/clang/build")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "proc", scope: !1, file: !1, line: 4, type: !5, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8, !10}
!8 = !DIGlobalVariable(name: "ch", scope: !0, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, variable: i8* @ch)
!9 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!10 = !DIGlobalVariable(name: "b", scope: !0, file: !1, line: 2, type: !11, isLocal: false, isDefinition: true, variable: i32* @b)
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 1, !"min_enum_size", i32 4}
!16 = !{!"clang version 3.7.0 (http://llvm.org/git/clang.git 9b0abb9df531ef7928c8182120e1869affca17d5) (http://llvm.org/git/llvm.git b1e759524dd94f7ce1e24935daed8383927e96c1)"}
!17 = !DILocation(line: 6, column: 8, scope: !4)
!18 = !DILocation(line: 7, column: 7, scope: !4)
!19 = !DILocation(line: 8, column: 1, scope: !4)
