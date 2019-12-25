; RUN: llc -mcpu=core2 -mtriple=i686-pc-win32 < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mcpu=core2 -mtriple=i686-pc-win32 < %s -filetype=obj | llvm-readobj --codeview | FileCheck %s --check-prefix=OBJ

; This C++ source should run and you should be able to step through the volatile
; modifications to x in windbg.
; # 1 "t.cpp" 2
; volatile int x;
;
; # 1 "./t.h" 1
; inline void f() {
;   x += 3;
; }
; # 3 "t.cpp" 2
; inline __forceinline void g() {
;   x += 2;
;   f();
;   x += 2;
; }
; int main() {
;   x += 1;
;   g();
;   x += 1;
;   return x;
; }

; ASM: _main:                                  # @main
; ASM: Lfunc_begin0:
; ASM:         .cv_func_id 0
; ASM: # %bb.0:                                 # %entry
; ASM:         .cv_file        1 "D:\\src\\llvm\\build\\t.cpp"
; ASM:         .cv_loc 0 1 9 5       # t.cpp:9:5
; ASM:         incl    "?x@@3HC"
; ASM:         .cv_inline_site_id 1 within 0 inlined_at 1 10 3
; ASM:         .cv_loc 1 1 4 5                 # t.cpp:4:5
; ASM:         addl    $2, "?x@@3HC"
; ASM:         .cv_file        2 "D:\\src\\llvm\\build\\t.h"
; ASM:         .cv_inline_site_id 2 within 1 inlined_at 1 5 3
; ASM:         .cv_loc 2 2 2 5                 # ./t.h:2:5
; ASM:         addl    $3, "?x@@3HC"
; ASM:         .cv_loc 1 1 6 5                 # t.cpp:6:5
; ASM:         addl    $2, "?x@@3HC"
; ASM:         .cv_loc 0 1 11 5                # t.cpp:11:5
; ASM:         incl    "?x@@3HC"
; ASM:         .cv_loc 0 1 12 10               # t.cpp:12:10
; ASM:         movl    "?x@@3HC", %eax
; ASM:         .cv_loc 0 1 12 3                # t.cpp:12:3
; ASM:         retl

; OBJ: Subsection [
; OBJ:   SubSectionType: InlineeLines (0xF6)
; OBJ:   SubSectionSize: 0x1C
; OBJ:   InlineeSourceLine {
; OBJ:     Inlinee: g (0x1002)
; OBJ:     FileID: D:\src\llvm\build\t.cpp (0x0)
; OBJ:     SourceLineNum: 3
; OBJ:   }
; OBJ:   InlineeSourceLine {
; OBJ:     Inlinee: f (0x1003)
; OBJ:     FileID: D:\src\llvm\build\t.h (0x8)
; OBJ:     SourceLineNum: 1
; OBJ:   }
; OBJ: ]

; OBJ: Subsection [
; OBJ:   SubSectionType: Symbols (0xF1)
; OBJ:   {{.*}}Proc{{.*}}Sym {
; OBJ:     Kind: S_GPROC32_ID (0x1147)
; OBJ:     FunctionType: main (0x1005)
; OBJ:     CodeOffset: _main+0x0
; OBJ:     Segment: 0x0
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     DisplayName: main
; OBJ:     LinkageName: _main
; OBJ:   }

; Previously, g's InlineSiteSym referenced t.h, which was wasteful.
; OBJ:        InlineSiteSym {
; OBJ:          Inlinee: g (0x1002)
; OBJ:          BinaryAnnotations [
; OBJ-NEXT:       ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x6, LineOffset: 1}
; OBJ-NEXT:       ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x7, LineOffset: 1}
; OBJ-NEXT:       ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x7, LineOffset: 1}
; OBJ-NEXT:       ChangeCodeLength: 0x7
; OBJ-NEXT:     ]
; OBJ:        }

; OBJ:   InlineSiteSym {
; OBJ:     Inlinee: f (0x1003)
; OBJ:     BinaryAnnotations [
; OBJ-NEXT:  ChangeCodeOffsetAndLineOffset: {CodeOffset: 0xD, LineOffset: 1}
; OBJ-NEXT:  ChangeCodeLength: 0x7
; OBJ-NEXT:]
; OBJ:   }
; OBJ:   InlineSiteEnd {
; OBJ:   }
; OBJ:   InlineSiteEnd {
; OBJ:   }
; OBJ:   ProcEnd {
; OBJ:   }
; OBJ: ]

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

@"\01?x@@3HC" = global i32 0, align 4, !dbg !0

; Function Attrs: norecurse nounwind uwtable
define i32 @main() local_unnamed_addr #0 !dbg !12 {
entry:
  %0 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !15, !tbaa !16
  %add = add nsw i32 %0, 1, !dbg !15
  store volatile i32 %add, i32* @"\01?x@@3HC", align 4, !dbg !15, !tbaa !16
  %1 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !20, !tbaa !16
  %add.i = add nsw i32 %1, 2, !dbg !20
  store volatile i32 %add.i, i32* @"\01?x@@3HC", align 4, !dbg !20, !tbaa !16
  %2 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !25, !tbaa !16
  %add.i.i = add nsw i32 %2, 3, !dbg !25
  store volatile i32 %add.i.i, i32* @"\01?x@@3HC", align 4, !dbg !25, !tbaa !16
  %3 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !29, !tbaa !16
  %add1.i = add nsw i32 %3, 2, !dbg !29
  store volatile i32 %add1.i, i32* @"\01?x@@3HC", align 4, !dbg !29, !tbaa !16
  %4 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !30, !tbaa !16
  %add1 = add nsw i32 %4, 1, !dbg !30
  store volatile i32 %add1, i32* @"\01?x@@3HC", align 4, !dbg !30, !tbaa !16
  %5 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !31, !tbaa !16
  ret i32 %5, !dbg !32
}

attributes #0 = { norecurse nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", linkageName: "\01?x@@3HC", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.9.0 (trunk 275430) (llvm/trunk 275433)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"CodeView", i32 1}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"PIC Level", i32 2}
!11 = !{!"clang version 3.9.0 (trunk 275430) (llvm/trunk 275433)"}
!12 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 8, type: !13, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!13 = !DISubroutineType(types: !14)
!14 = !{!7}
!15 = !DILocation(line: 9, column: 5, scope: !12)
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C++ TBAA"}
!20 = !DILocation(line: 4, column: 5, scope: !21, inlinedAt: !24)
!21 = distinct !DISubprogram(name: "g", linkageName: "\01?g@@YAXXZ", scope: !3, file: !3, line: 3, type: !22, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!22 = !DISubroutineType(types: !23)
!23 = !{null}
!24 = distinct !DILocation(line: 10, column: 3, scope: !12)
!25 = !DILocation(line: 2, column: 5, scope: !26, inlinedAt: !28)
!26 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !27, file: !27, line: 1, type: !22, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!27 = !DIFile(filename: "./t.h", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!28 = distinct !DILocation(line: 5, column: 3, scope: !21, inlinedAt: !24)
!29 = !DILocation(line: 6, column: 5, scope: !21, inlinedAt: !24)
!30 = !DILocation(line: 11, column: 5, scope: !12)
!31 = !DILocation(line: 12, column: 10, scope: !12)
!32 = !DILocation(line: 12, column: 3, scope: !12)

