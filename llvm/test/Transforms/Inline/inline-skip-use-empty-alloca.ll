; RUN: opt -opaque-pointers=0 < %s -S -passes=inline | FileCheck %s
; RUN: opt -opaque-pointers=0 < %s -S -strip-debug -passes=inline | FileCheck %s

; https://bugs.llvm.org/show_bug.cgi?id=43291
; The purpose of this test is to check if there is use_empty in the inner loop when scanning
; blocks of allocas, if the alloca is use_empty, skip it when allocas inlining.

define void @foo(i16 %k) !dbg !6 {
  call void @llvm.dbg.value(metadata i16 %k, metadata !14, metadata !DIExpression()), !dbg !15
  %alloca_a = alloca i32
  call void @llvm.dbg.value(metadata i16 %k, metadata !14, metadata !DIExpression()), !dbg !15
  %alloca_b = alloca [3 x i32]
  call void @llvm.dbg.value(metadata i16 %k, metadata !14, metadata !DIExpression()), !dbg !15
  %alloca_c = alloca i32
  call void @llvm.dbg.value(metadata i16 %k, metadata !14, metadata !DIExpression()), !dbg !15
  %alloca_a..1 = bitcast i32* %alloca_a to i8**, !dbg !16
  %alloca_c..1 = bitcast i32* %alloca_c to i8**, !dbg !16
  %_tmp23 = icmp ne i16 %k, 0
  br i1 %_tmp23, label %bb1, label %bb2

bb1:                                              ; preds = %0
  %_tmp28 = getelementptr [3 x i32], [3 x i32]* %alloca_b, i16 0, i64 0
  store i32 0, i32* %_tmp28
  br label %bb2

bb2:                                              ; preds = %bb1, %0
  ret void
}

define i16 @bar() !dbg !17 {
  ; CHECK-LABEL: bar
  ; CHECK: %alloca_a.i = alloca i32
  ; CHECK-NOT: %alloca_b.i = alloca [3 x i32]
  ; CHECK: %alloca_c.i = alloca i32
  call void @foo(i16 0), !dbg !20
  br label %bb1

bb1:                                              ; preds = %0
  ret i16 0
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "Foo", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2)
!1 = !DIFile(filename: "foo2.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"Foo"}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 17, type: !7, scopeLine: 17, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !13}
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "int64_t", file: !10, line: 76, baseType: !11)
!10 = !DIFile(filename: "stdint.h", directory: "/tmp")
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "__i64_t", file: !1, baseType: !12)
!12 = !DIBasicType(name: "signed long long", size: 64, encoding: DW_ATE_signed)
!13 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!14 = !DILocalVariable(name: "k", arg: 1, scope: !6, file: !1, line: 17, type: !13)
!15 = !DILocation(line: 0, scope: !6)
!16 = !DILocation(line: 19, column: 3, scope: !6)
!17 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 8, type: !18, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{!13}
!20 = !DILocation(line: 8, column: 7, scope: !17)
