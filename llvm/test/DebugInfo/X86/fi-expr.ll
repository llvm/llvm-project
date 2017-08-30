; RUN: llc -mtriple=x86_64-apple-darwin -o - %s -filetype=obj \
; RUN:   | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; A hand-crafted FrameIndex location with a DW_OP_deref.
; CHECK: DW_TAG_formal_parameter
;                                          fbreg -8, deref
; CHECK-NEXT: DW_AT_location {{.*}} (DW_OP_fbreg -8, DW_OP_deref)
; CHECK-NEXT: DW_AT_name {{.*}} "foo"

define void @f(i8* %bar) !dbg !6 {
entry:
  %foo.addr = alloca i8*
  store i8* %bar, i8** %foo.addr
  call void @llvm.dbg.declare(metadata i8** %foo.addr, metadata !12, metadata !13), !dbg !14
  ret void, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !DILocalVariable(name: "foo", arg: 1, scope: !6, file: !1, line: 1, type: !10)
!13 = !DIExpression(DW_OP_deref)
!14 = !DILocation(line: 1, scope: !6)
!15 = !DILocation(line: 1, scope: !6)
