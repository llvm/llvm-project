; RUN: opt < %s -instcombine -o /dev/null  -pass-remarks-output=%t -S \
; RUN:     -pass-remarks=instcombine 2>&1 | FileCheck %s
; RUN: cat %t | FileCheck -check-prefix=YAML %s
; RUN: opt < %s -passes='require<opt-remark-emit>,instcombine' -o /dev/null \
; RUN:     -pass-remarks-output=%t -S -pass-remarks=instcombine 2>&1 | FileCheck %s
; RUN: cat %t | FileCheck -check-prefix=YAML %s

; CHECK:      remark: libcalls-opt-remarks.c:10:10: folded strlen(select) to select of constants{{$}}
; CHECK-NOT:  remark:

; YAML:      --- !Passed
; YAML-NEXT: Pass:            instcombine
; YAML-NEXT: Name:            simplify-libcalls
; YAML-NEXT: DebugLoc:        { File: libcalls-opt-remarks.c, Line: 10, Column: 10 }
; YAML-NEXT: Function:        f1
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'folded strlen(select) to select of constants'
; YAML-NEXT: ...

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i32 @strlen(i8*)

@hello = constant [6 x i8] c"hello\00"
@longer = constant [7 x i8] c"longer\00"

define i32 @f1(i1) !dbg !7 {
  %hello = getelementptr [6 x i8], [6 x i8]* @hello, i32 0, i32 0, !dbg !10
  %longer = getelementptr [7 x i8], [7 x i8]* @longer, i32 0, i32 0, !dbg !12
  %2 = select i1 %0, i8* %hello, i8* %longer, !dbg !9
  %3 = call i32 @strlen(i8* %2), !dbg !14
  ret i32 %3, !dbg !16
}



!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Apple LLVM version 8.1.0 (clang-802.0.42)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "libcalls-opt-remarks.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"Apple LLVM version 8.1.0 (clang-802.0.42)"}
!7 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 9, type: !8, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 10, column: 17, scope: !7)
!10 = !DILocation(line: 10, column: 24, scope: !11)
!11 = !DILexicalBlockFile(scope: !7, file: !1, discriminator: 1)
!12 = !DILocation(line: 10, column: 32, scope: !13)
!13 = !DILexicalBlockFile(scope: !7, file: !1, discriminator: 2)
!14 = !DILocation(line: 10, column: 10, scope: !15)
!15 = !DILexicalBlockFile(scope: !7, file: !1, discriminator: 3)
!16 = !DILocation(line: 10, column: 3, scope: !15)
