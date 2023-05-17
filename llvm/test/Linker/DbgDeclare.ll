; RUN: llvm-link %s %p/DbgDeclare2.ll -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s
; Test if metadata in dbg.declare is mapped properly or not.

; rdar://13089880
; CHECK: define i32 @main(i32 %argc, ptr %argv)
; CHECK: call void @llvm.dbg.declare(metadata ptr %argc.addr, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: call void @llvm.dbg.declare(metadata ptr %argv.addr, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: define void @test(i32 %argc, ptr %argv)
; CHECK: call void @llvm.dbg.declare(metadata ptr %argc.addr, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: call void @llvm.dbg.declare(metadata ptr %argv.addr, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: call void @llvm.dbg.declare(metadata ptr %i, metadata !{{[0-9]+}}, metadata {{.*}})

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

define i32 @main(i32 %argc, ptr %argv) uwtable ssp !dbg !5 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  store i32 0, ptr %retval
  store i32 %argc, ptr %argc.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %argc.addr, metadata !14, metadata !DIExpression()), !dbg !15
  store ptr %argv, ptr %argv.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %argv.addr, metadata !16, metadata !DIExpression()), !dbg !15
  %0 = load i32, ptr %argc.addr, align 4, !dbg !17
  %1 = load ptr, ptr %argv.addr, align 8, !dbg !17
  call void @test(i32 %0, ptr %1), !dbg !17
  ret i32 0, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @test(i32, ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 173515)", isOptimized: true, emissionKind: FullDebug, file: !20, enums: !2, retainedTypes: !2, globals: !2)
!2 = !{}
!5 = distinct !DISubprogram(name: "main", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !20, scope: null, type: !7, retainedNodes: !2)
!6 = !DIFile(filename: "main.cpp", directory: "/private/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9, !10}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !12)
!12 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!14 = !DILocalVariable(name: "argc", line: 3, arg: 1, scope: !5, file: !6, type: !9)
!15 = !DILocation(line: 3, scope: !5)
!16 = !DILocalVariable(name: "argv", line: 3, arg: 2, scope: !5, file: !6, type: !10)
!17 = !DILocation(line: 5, scope: !18)
!18 = distinct !DILexicalBlock(line: 4, column: 0, file: !20, scope: !5)
!19 = !DILocation(line: 6, scope: !18)
!20 = !DIFile(filename: "main.cpp", directory: "/private/tmp")
!21 = !{i32 1, !"Debug Info Version", i32 3}
