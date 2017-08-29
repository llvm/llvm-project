; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t.o -filetype=obj -O0
; RUN: llvm-dwarfdump %t.o | FileCheck %s
;
; Test that on x86_64, the 32-bit subregister esi is emitted as
; subregister of the 64-bit rsi.
;
; rdar://problem/16015314
;
; CHECK:  .debug_info contents:
; CHECK:  DW_TAG_variable
; CHECK-NEXT:  DW_AT_location [DW_FORM_data4]	(
; CHECK-NEXT:    {{.*}} - {{.*}}: DW_OP_reg4 RSI)
; CHECK-NEXT:  DW_AT_name [DW_FORM_strp]{{.*}} "a"
;
; struct bar {
;   int a;
;   int b;
; };
;
; void doSomething() __attribute__ ((noinline));
;
; void doSomething(struct bar *b)
; {
;   int a = b->a;
;   printf("%d\n", a); // set breakpoint here
; }
;
; int main()
; {
;   struct bar myBar = { 3, 4 };
;   doSomething(&myBar);
;   return 0;
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.bar = type { i32, i32 }

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@main.myBar = private unnamed_addr constant %struct.bar { i32 3, i32 4 }, align 4

; Function Attrs: noinline nounwind ssp uwtable
define void @doSomething(%struct.bar* nocapture readonly %b) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.bar* %b, metadata !15, metadata !DIExpression()), !dbg !25
  %a1 = getelementptr inbounds %struct.bar, %struct.bar* %b, i64 0, i32 0, !dbg !26
  %0 = load i32, i32* %a1, align 4, !dbg !26, !tbaa !27
  tail call void @llvm.dbg.value(metadata i32 %0, metadata !16, metadata !DIExpression()), !dbg !26
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32 %0) #4, !dbg !32
  ret void, !dbg !33
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #2

; Function Attrs: nounwind ssp uwtable
define i32 @main() #3 !dbg !17 {
entry:
  %myBar = alloca i64, align 8, !dbg !34
  %tmpcast = bitcast i64* %myBar to %struct.bar*, !dbg !34
  tail call void @llvm.dbg.declare(metadata %struct.bar* %tmpcast, metadata !21, metadata !DIExpression()), !dbg !34
  store i64 17179869187, i64* %myBar, align 8, !dbg !34
  call void @doSomething(%struct.bar* %tmpcast), !dbg !35
  ret i32 0, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { nounwind ssp uwtable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 ", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "subregisters.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "doSomething", line: 10, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 11, file: !1, scope: !5, type: !6, variables: !14)
!5 = !DIFile(filename: "subregisters.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "bar", line: 3, size: 64, align: 32, file: !1, elements: !10)
!10 = !{!11, !13}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 4, size: 32, align: 32, file: !1, scope: !9, baseType: !12)
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 5, size: 32, align: 32, offset: 32, file: !1, scope: !9, baseType: !12)
!14 = !{!15, !16}
!15 = !DILocalVariable(name: "b", line: 10, arg: 1, scope: !4, file: !5, type: !8)
!16 = !DILocalVariable(name: "a", line: 12, scope: !4, file: !5, type: !12)
!17 = distinct !DISubprogram(name: "main", line: 16, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 17, file: !1, scope: !5, type: !18, variables: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{!12}
!20 = !{!21}
!21 = !DILocalVariable(name: "myBar", line: 18, scope: !17, file: !5, type: !9)
!22 = !{i32 2, !"Dwarf Version", i32 2}
!23 = !{i32 1, !"Debug Info Version", i32 3}
!24 = !{!"clang version 3.5 "}
!25 = !DILocation(line: 10, scope: !4)
!26 = !DILocation(line: 12, scope: !4)
!27 = !{!28, !29, i64 0}
!28 = !{!"bar", !29, i64 0, !29, i64 4}
!29 = !{!"int", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C/C++ TBAA"}
!32 = !DILocation(line: 13, scope: !4)
!33 = !DILocation(line: 14, scope: !4)
!34 = !DILocation(line: 18, scope: !17)
!35 = !DILocation(line: 19, scope: !17)
!36 = !DILocation(line: 20, scope: !17)
