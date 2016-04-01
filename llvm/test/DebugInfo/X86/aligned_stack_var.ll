; RUN: llc %s -mtriple=x86_64-pc-linux-gnu -O0 -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; If stack is realigned, we shouldn't describe locations of local
; variables by giving offset from the frame pointer (%rbp):
; push %rpb
; mov  %rsp,%rbp
; and  ALIGNMENT,%rsp ; (%rsp and %rbp are different now)
; It's better to use offset from %rsp instead.

; DW_AT_location of variable "x" shouldn't be equal to
; (DW_OP_fbreg: .*): DW_OP_fbreg has code 0x91

; CHECK: {{0x.* DW_TAG_variable}}
; CHECK-NOT: {{DW_AT_location.*DW_FORM_block1.*0x.*91}}
; CHECK: NULL

define void @_Z3runv() nounwind uwtable !dbg !5 {
entry:
  %x = alloca i32, align 32
  call void @llvm.dbg.declare(metadata i32* %x, metadata !9, metadata !DIExpression()), !dbg !12
  ret void, !dbg !13
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.2 (trunk 155696:155697) (llvm/trunk 155696)", isOptimized: false, emissionKind: FullDebug, file: !14, enums: !1, retainedTypes: !1, subprograms: !3, globals: !1, imports:  !1)
!1 = !{}
!3 = !{!5}
!5 = distinct !DISubprogram(name: "run", linkageName: "_Z3runv", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !14, scope: !6, type: !7, variables: !1)
!6 = !DIFile(filename: "test.cc", directory: "/home/samsonov/debuginfo")
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocalVariable(name: "x", line: 2, scope: !10, file: !6, type: !11)
!10 = distinct !DILexicalBlock(line: 1, column: 12, file: !14, scope: !5)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 2, column: 7, scope: !10)
!13 = !DILocation(line: 3, column: 1, scope: !10)
!14 = !DIFile(filename: "test.cc", directory: "/home/samsonov/debuginfo")
!15 = !{i32 1, !"Debug Info Version", i32 3}
