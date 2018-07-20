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
; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump -v - ---debug-info | FileCheck %s --check-prefix=DWARF

; CHECK-LABEL: _main:
; CHECK: movaps {{.*}}, (%rsp)
; CHECK: movq    %rsp, %rdi
; CHECK: callq _f

; DWARF: DW_TAG_variable
; DWARF-NEXT:              DW_AT_location [DW_FORM_exprloc]      (DW_OP_fbreg +0)
; DWARF-NEXT:              DW_AT_name [DW_FORM_strp]     ( {{.*}} = "array")

; ModuleID = '/tmp/array.c'
source_filename = "/tmp/array.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

@main.array = private unnamed_addr constant [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16

; Function Attrs: nounwind ssp uwtable
define void @f(i32* nocapture %p) local_unnamed_addr #0 !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata i32* %p, metadata !14, metadata !15), !dbg !16
  store i32 42, i32* %p, align 4, !dbg !17, !tbaa !18
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 !dbg !23 {
entry:
  %array = alloca [4 x i32], align 16
  tail call void @llvm.dbg.value(metadata i32 %argc, metadata !30, metadata !15), !dbg !36
  tail call void @llvm.dbg.value(metadata i8** %argv, metadata !31, metadata !15), !dbg !37
  %0 = bitcast [4 x i32]* %array to i8*, !dbg !38
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0) #3, !dbg !38
  tail call void @llvm.dbg.declare(metadata [4 x i32]* %array, metadata !32, metadata !15), !dbg !39
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 nonnull %0, i8* align 16 bitcast ([4 x i32]* @main.array to i8*), i64 16, i1 false), !dbg !39
  %arraydecay = getelementptr inbounds [4 x i32], [4 x i32]* %array, i64 0, i64 0, !dbg !40
  call void @f(i32* nonnull %arraydecay), !dbg !41
  %1 = load i32, i32* %arraydecay, align 16, !dbg !42, !tbaa !18
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0) #3, !dbg !43
  ret i32 %1, !dbg !44
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 (trunk 303873) (llvm/trunk 303875)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/array.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 5.0.0 (trunk 303873) (llvm/trunk 303875)"}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !13)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14}
!14 = !DILocalVariable(name: "p", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!15 = !DIExpression()
!16 = !DILocation(line: 1, column: 13, scope: !8)
!17 = !DILocation(line: 2, column: 8, scope: !8)
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 3, column: 1, scope: !8)
!23 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !24, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !29)
!24 = !DISubroutineType(types: !25)
!25 = !{!12, !12, !26}
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 64)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64)
!28 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!29 = !{!30, !31, !32}
!30 = !DILocalVariable(name: "argc", arg: 1, scope: !23, file: !1, line: 5, type: !12)
!31 = !DILocalVariable(name: "argv", arg: 2, scope: !23, file: !1, line: 5, type: !26)
!32 = !DILocalVariable(name: "array", scope: !23, file: !1, line: 6, type: !33)
!33 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 128, elements: !34)
!34 = !{!35}
!35 = !DISubrange(count: 4)
!36 = !DILocation(line: 5, column: 14, scope: !23)
!37 = !DILocation(line: 5, column: 27, scope: !23)
!38 = !DILocation(line: 6, column: 3, scope: !23)
!39 = !DILocation(line: 6, column: 7, scope: !23)
!40 = !DILocation(line: 7, column: 5, scope: !23)
!41 = !DILocation(line: 7, column: 3, scope: !23)
!42 = !DILocation(line: 8, column: 10, scope: !23)
!43 = !DILocation(line: 9, column: 1, scope: !23)
!44 = !DILocation(line: 8, column: 3, scope: !23)
