; RUN: llc -mcpu=core2 -mtriple=i686-pc-win32 < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mcpu=core2 -mtriple=i686-pc-win32 < %s -filetype=obj | llvm-readobj -codeview | FileCheck %s --check-prefix=OBJ

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
; ASM: # BB#0:                                 # %entry
; ASM:         .cv_file        1 "D:\\src\\llvm\\build\\t.cpp"
; ASM:         .cv_loc 0 1 9 5 is_stmt 0       # t.cpp:9:5
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
; OBJ:   ProcStart {
; OBJ:     Kind: S_GPROC32_ID (0x1147)
; OBJ:     FunctionType: main (0x1005)
; OBJ:     CodeOffset: _main+0x0
; OBJ:     Segment: 0x0
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     DisplayName: main
; OBJ:     LinkageName: _main
; OBJ:   }

; Previously, g's InlineSite referenced t.h, which was wasteful.
; OBJ:        InlineSite {
; OBJ:          Inlinee: g (0x1002)
; OBJ:          BinaryAnnotations [
; OBJ-NEXT:       ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x6, LineOffset: 1}
; OBJ-NEXT:       ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x7, LineOffset: 1}
; OBJ-NEXT:       ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x7, LineOffset: 1}
; OBJ-NEXT:       ChangeCodeLength: 0x7
; OBJ-NEXT:     ]
; OBJ:        }

; OBJ:   InlineSite {
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

@"\01?x@@3HC" = global i32 0, align 4

; Function Attrs: norecurse nounwind uwtable
define i32 @main() local_unnamed_addr #0 !dbg !11 {
entry:
  %0 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !14, !tbaa !15
  %add = add nsw i32 %0, 1, !dbg !14
  store volatile i32 %add, i32* @"\01?x@@3HC", align 4, !dbg !14, !tbaa !15
  %1 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !19, !tbaa !15
  %add.i = add nsw i32 %1, 2, !dbg !19
  store volatile i32 %add.i, i32* @"\01?x@@3HC", align 4, !dbg !19, !tbaa !15
  %2 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !24, !tbaa !15
  %add.i.i = add nsw i32 %2, 3, !dbg !24
  store volatile i32 %add.i.i, i32* @"\01?x@@3HC", align 4, !dbg !24, !tbaa !15
  %3 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !28, !tbaa !15
  %add1.i = add nsw i32 %3, 2, !dbg !28
  store volatile i32 %add1.i, i32* @"\01?x@@3HC", align 4, !dbg !28, !tbaa !15
  %4 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !29, !tbaa !15
  %add1 = add nsw i32 %4, 1, !dbg !29
  store volatile i32 %add1, i32* @"\01?x@@3HC", align 4, !dbg !29, !tbaa !15
  %5 = load volatile i32, i32* @"\01?x@@3HC", align 4, !dbg !30, !tbaa !15
  ret i32 %5, !dbg !31
}

attributes #0 = { norecurse nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 275430) (llvm/trunk 275433)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "x", linkageName: "\01?x@@3HC", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, variable: i32* @"\01?x@@3HC")
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"CodeView", i32 1}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"clang version 3.9.0 (trunk 275430) (llvm/trunk 275433)"}
!11 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 8, type: !12, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{!6}
!14 = !DILocation(line: 9, column: 5, scope: !11)
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C++ TBAA"}
!19 = !DILocation(line: 4, column: 5, scope: !20, inlinedAt: !23)
!20 = distinct !DISubprogram(name: "g", linkageName: "\01?g@@YAXXZ", scope: !1, file: !1, line: 3, type: !21, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!21 = !DISubroutineType(types: !22)
!22 = !{null}
!23 = distinct !DILocation(line: 10, column: 3, scope: !11)
!24 = !DILocation(line: 2, column: 5, scope: !25, inlinedAt: !27)
!25 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !26, file: !26, line: 1, type: !21, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!26 = !DIFile(filename: "./t.h", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!27 = distinct !DILocation(line: 5, column: 3, scope: !20, inlinedAt: !23)
!28 = !DILocation(line: 6, column: 5, scope: !20, inlinedAt: !23)
!29 = !DILocation(line: 11, column: 5, scope: !11)
!30 = !DILocation(line: 12, column: 10, scope: !11)
!31 = !DILocation(line: 12, column: 3, scope: !11)
