; RUN: llc -o - %s -filetype=obj -frame-pointer=all | llvm-dwarfdump - | FileCheck %s

target triple="aarch64--"

define void @f() sanitize_hwaddress !dbg !6 {
entry:
  %x = call i8* @g(i32 0)
  %a = alloca [128 x i8]
  %b = alloca [128 x i8]
  ; CHECK: DW_AT_location (DW_OP_fbreg
  call void @llvm.dbg.declare(metadata [128 x i8]* %a, metadata !12, metadata !DIExpression()), !dbg !14
  ; CHECK: DW_AT_location (DW_OP_fbreg
  call void @llvm.dbg.declare(metadata [128 x i8]* %b, metadata !13, metadata !DIExpression()), !dbg !14
  ret void, !dbg !15
}

declare i8* @g(i32)

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "x.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang"}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags:
DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !DILocalVariable(name: "a", scope: !6, file: !1, line: 1, type: !9)
!13 = !DILocalVariable(name: "b", scope: !6, file: !1, line: 1, type: !9)
!14 = !DILocation(line: 1, column: 29, scope: !6)
!15 = !DILocation(line: 1, column: 37, scope: !6)
