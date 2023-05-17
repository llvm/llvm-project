; RUN: llc -O0 -asm-verbose -mtriple=x86_64-macosx -debugger-tune=gdb < %s | FileCheck %s
; CHECK-NOT: .asciz "X" ## External Name
; CHECK: .asciz "Y" ## External Name
; Test to check type with no definition is listed in pubtypes section.
%struct.X = type opaque
%struct.Y = type { i32 }

define i32 @foo(ptr %x, ptr %y) nounwind ssp !dbg !1 {
entry:
  %x_addr = alloca ptr                     ; <ptr> [#uses=1]
  %y_addr = alloca ptr                     ; <ptr> [#uses=1]
  %retval = alloca i32                            ; <ptr> [#uses=2]
  %0 = alloca i32                                 ; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata ptr %x_addr, metadata !0, metadata !DIExpression()), !dbg !13
  store ptr %x, ptr %x_addr
  call void @llvm.dbg.declare(metadata ptr %y_addr, metadata !14, metadata !DIExpression()), !dbg !13
  store ptr %y, ptr %y_addr
  store i32 0, ptr %0, align 4, !dbg !13
  %1 = load i32, ptr %0, align 4, !dbg !13            ; <i32> [#uses=1]
  store i32 %1, ptr %retval, align 4, !dbg !13
  br label %return, !dbg !13

return:                                           ; preds = %entry
  %retval1 = load i32, ptr %retval, !dbg !13          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!20}

!0 = !DILocalVariable(name: "x", line: 7, arg: 1, scope: !1, file: !2, type: !7)
!1 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !3, scopeLine: 7, file: !18, scope: !2, type: !4)
!2 = !DIFile(filename: "a.c", directory: "/tmp/")
!3 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: false, emissionKind: FullDebug, file: !18, enums: !19, retainedTypes: !19, imports:  null)
!4 = !DISubroutineType(types: !5)
!5 = !{!6, !7, !9}
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !18, scope: !2, baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "X", line: 3, flags: DIFlagFwdDecl, file: !18, scope: !2)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !18, scope: !2, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "Y", line: 4, size: 32, align: 32, file: !18, scope: !2, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 5, size: 32, align: 32, file: !18, scope: !10, baseType: !6)
!13 = !DILocation(line: 7, scope: !1)
!14 = !DILocalVariable(name: "y", line: 7, arg: 2, scope: !1, file: !2, type: !9)
!15 = !DILocation(line: 7, scope: !16)
!16 = distinct !DILexicalBlock(line: 7, column: 0, file: !18, scope: !1)
!18 = !DIFile(filename: "a.c", directory: "/tmp/")
!19 = !{}
!20 = !{i32 1, !"Debug Info Version", i32 3}
