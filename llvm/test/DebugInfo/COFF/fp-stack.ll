; RUN: llc -mtriple=i686-windows-msvc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=i686-windows-msvc < %s -filetype=obj | llvm-readobj -codeview - | FileCheck %s --check-prefix=OBJ
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

define double @f(double %p1) !dbg !4 {
entry:
  %sub = fsub double -0.000000e+00, %p1, !dbg !16
  tail call void @llvm.dbg.value(metadata double %sub, i64 0, metadata !10, metadata !14), !dbg !15
  ret double %sub
}

; ASM:         .cv_def_range    Lfunc_begin0 Lfunc_end0, "A\021\200\000\000\000"
; OBJ:    DefRangeRegister {
; OBJ:      Register: 128
; OBJ:      MayHaveNoName: 0
; OBJ:      LocalVariableAddrRange {
; OBJ:        OffsetStart: .text+0x0
; OBJ:        ISectStart: 0x0
; OBJ:        Range: 0x7
; OBJ:      }
; OBJ:    }

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 261537) (llvm/trunk 261463)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !5, file: !5, line: 2, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, variables: !9)
!5 = !DIFile(filename: "t.ii", directory: "/")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!9 = !{!10}
!10 = !DILocalVariable(name: "p1", arg: 1, scope: !4, file: !5, line: 2, type: !8)
!11 = !{i32 2, !"CodeView", i32 1}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.9.0 (trunk 261537) (llvm/trunk 261463)"}
!14 = !DIExpression()
!15 = !DILocation(line: 2, scope: !4)
!16 = !DILocation(line: 3, scope: !4)
