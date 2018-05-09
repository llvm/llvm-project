; RUN: llc < %s -filetype=obj | llvm-dwarfdump -v -debug-info - | FileCheck %s 
;
; CHECK:      DW_TAG_array_type
; CHECK-NEXT: DW_AT_GNU_vector [DW_FORM_flag_present] (true)
; CHECK-NEXT: DW_AT_byte_size [DW_FORM_data1] (0x20)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @test() !dbg !9 {
  %1 = alloca <6 x float>, align 32
  call void @llvm.dbg.declare(metadata <6 x float>* %1, metadata !13, metadata !DIExpression()), !dbg !19
  %2 = bitcast <6 x float>* %1 to i8*, !dbg !20
  call void @foo(i8* %2), !dbg !21
  ret void, !dbg !22
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare dso_local void @foo(i8*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "small.c", directory: "/dbg/info")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang"}
!9 = distinct !DISubprogram(name: "test", scope: !10, file: !10, line: 3, type: !11, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!10 = !DIFile(filename: "./small.c", directory: "/dbg/info")
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocalVariable(name: "v01", scope: !9, file: !10, line: 4, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "vec6f", file: !10, line: 1, baseType: !15)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 256, flags: DIFlagVector, elements: !17)
!16 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!17 = !{!18}
!18 = !DISubrange(count: 6)
!19 = !DILocation(line: 4, column: 9, scope: !9)
!20 = !DILocation(line: 5, column: 7, scope: !9)
!21 = !DILocation(line: 5, column: 3, scope: !9)
!22 = !DILocation(line: 6, column: 1, scope: !9)
