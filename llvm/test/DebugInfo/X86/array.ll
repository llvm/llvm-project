; ModuleID = 'array.c'
;
; From (clang -g -c -O1):
;
; void f(int* p) {
;   p[0] = 42;
; }
;
; int main(int argc, char** argv) {
;   int array[4] = { 0, 1, 2, 3 };
;   f(array);
;   return array[0];
; }
;
; RUN: llc -filetype=asm %s -o - | FileCheck %s
; Test that we only emit register-indirect locations for the array array.
; rdar://problem/14874886
;
; CHECK:     ##DEBUG_VALUE: main:array <- [%R{{.*}}+0]
; CHECK-NOT: ##DEBUG_VALUE: main:array <- %R{{.*}}
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

@main.array = private unnamed_addr constant [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16

; Function Attrs: nounwind ssp uwtable
define void @f(i32* nocapture %p) #0 !dbg !4 {
  tail call void @llvm.dbg.value(metadata i32* %p, i64 0, metadata !11, metadata !DIExpression()), !dbg !28
  store i32 42, i32* %p, align 4, !dbg !29, !tbaa !30
  ret void, !dbg !34
}

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32 %argc, i8** nocapture readnone %argv) #0 !dbg !12 {
  %array = alloca [4 x i32], align 16
  tail call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata !19, metadata !DIExpression()), !dbg !35
  tail call void @llvm.dbg.value(metadata i8** %argv, i64 0, metadata !20, metadata !DIExpression()), !dbg !35
  tail call void @llvm.dbg.value(metadata [4 x i32]* %array, i64 0, metadata !21, metadata !DIExpression()), !dbg !36
  %1 = bitcast [4 x i32]* %array to i8*, !dbg !36
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast ([4 x i32]* @main.array to i8*), i64 16, i1 false), !dbg !36
  tail call void @llvm.dbg.value(metadata [4 x i32]* %array, i64 0, metadata !21, metadata !DIExpression()), !dbg !36
  %2 = getelementptr inbounds [4 x i32], [4 x i32]* %array, i64 0, i64 0, !dbg !37
  call void @f(i32* %2), !dbg !37
  tail call void @llvm.dbg.value(metadata [4 x i32]* %array, i64 0, metadata !21, metadata !DIExpression()), !dbg !36
  %3 = load i32, i32* %2, align 16, !dbg !38, !tbaa !30
  ret i32 %3, !dbg !38
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !26}
!llvm.ident = !{!27}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "array.c", directory: "")
!2 = !{}
!3 = !{!4, !12}
!4 = distinct !DISubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !10)
!5 = !DIFile(filename: "array.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DILocalVariable(name: "p", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!12 = distinct !DISubprogram(name: "main", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 5, file: !1, scope: !5, type: !13, variables: !18)
!13 = !DISubroutineType(types: !14)
!14 = !{!9, !9, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !16)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !17)
!17 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!18 = !{!19, !20, !21}
!19 = !DILocalVariable(name: "argc", line: 5, arg: 1, scope: !12, file: !5, type: !9)
!20 = !DILocalVariable(name: "argv", line: 5, arg: 2, scope: !12, file: !5, type: !15)
!21 = !DILocalVariable(name: "array", line: 6, scope: !12, file: !5, type: !22)
!22 = !DICompositeType(tag: DW_TAG_array_type, size: 128, align: 32, baseType: !9, elements: !23)
!23 = !{!24}
!24 = !DISubrange(count: 4)
!25 = !{i32 2, !"Dwarf Version", i32 2}
!26 = !{i32 1, !"Debug Info Version", i32 3}
!27 = !{!"clang version 3.5.0 "}
!28 = !DILocation(line: 1, scope: !4)
!29 = !DILocation(line: 2, scope: !4)
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !32, i64 0}
!32 = !{!"omnipotent char", !33, i64 0}
!33 = !{!"Simple C/C++ TBAA"}
!34 = !DILocation(line: 3, scope: !4)
!35 = !DILocation(line: 5, scope: !12)
!36 = !DILocation(line: 6, scope: !12)
!37 = !DILocation(line: 7, scope: !12)
!38 = !DILocation(line: 8, scope: !12)
