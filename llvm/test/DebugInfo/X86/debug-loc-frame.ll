; REQUIRES: object-emission

; Check that when variables are allocated on the stack we generate debug locations
; for the stack location directly instead of generating a register+offset indirection.

; RUN: llc -O2 -filetype=obj -disable-post-ra -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN: | llvm-dwarfdump -v - | FileCheck %s
;
; int data = 17;
; int sum  = 0;
; int zero = 0;
; int *ptr;
;
; extern void foo(int i, int *p);
;
; int main()
; {
;   int val;
;   val = data;
;   foo(1, &val);
;   foo(2, &data);
;   return zero;
; }
;
; CHECK:      .debug_info contents
; CHECK:      DW_TAG_subprogram
; CHECK-NOT:  NULL
; CHECK:      DW_TAG_variable
; CHECK:      DW_AT_location [DW_FORM_sec_offset] ({{.*}}
; CHECK-NEXT:   [{{0x.*}}, {{0x.*}}): DW_OP_reg0 RAX
; CHECK-NEXT:   [{{0x.*}}, {{0x.*}}): DW_OP_breg7 RSP+4, DW_OP_deref)
; CHECK-NEXT: DW_AT_name {{.*}}"val"

; ModuleID = 'frame.c'
source_filename = "frame.c"

@data = global i32 17, align 4, !dbg !0
@sum = local_unnamed_addr global i32 0, align 4, !dbg !6
@zero = local_unnamed_addr global i32 0, align 4, !dbg !9
@ptr = common local_unnamed_addr global i32* null, align 8, !dbg !11

define i32 @main() local_unnamed_addr !dbg !17 {
entry:
  %val = alloca i32, align 4
  %0 = bitcast i32* %val to i8*, !dbg !22
  call void @llvm.lifetime.start(i64 4, i8* %0), !dbg !22
  %1 = load i32, i32* @data, align 4, !dbg !23, !tbaa !24
  tail call void @llvm.dbg.value(metadata i32 %1, metadata !21, metadata !28), !dbg !29
  store i32 %1, i32* %val, align 4, !dbg !30, !tbaa !24
  tail call void @llvm.dbg.value(metadata i32* %val, metadata !21, metadata !31), !dbg !29
  call void @foo(i32 1, i32* nonnull %val), !dbg !32
  call void @foo(i32 2, i32* nonnull @data), !dbg !33
  %2 = load i32, i32* @zero, align 4, !dbg !34, !tbaa !24
  call void @llvm.lifetime.end(i64 4, i8* %0), !dbg !35
  ret i32 %2, !dbg !36
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #0

declare void @foo(i32, i32*) local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15}
!llvm.ident = !{!16}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "data", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.9.0 (trunk 273961)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "frame.c", directory: "/home/user/test")
!4 = !{}
!5 = !{!0, !6, !9, !11}
!6 = distinct !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "sum", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = !DIGlobalVariable(name: "zero", scope: !2, file: !3, line: 3, type: !8, isLocal: false, isDefinition: true)
!11 = distinct !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = !DIGlobalVariable(name: "ptr", scope: !2, file: !3, line: 4, type: !13, isLocal: false, isDefinition: true)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64)
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{!"clang version 3.9.0 (trunk 273961)"}
!17 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 8, type: !18, isLocal: false, isDefinition: true, scopeLine: 9, isOptimized: true, unit: !2, variables: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{!8}
!20 = !{!21}
!21 = !DILocalVariable(name: "val", scope: !17, file: !3, line: 10, type: !8)
!22 = !DILocation(line: 10, column: 3, scope: !17)
!23 = !DILocation(line: 11, column: 9, scope: !17)
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C/C++ TBAA"}
!28 = !DIExpression()
!29 = !DILocation(line: 10, column: 7, scope: !17)
!30 = !DILocation(line: 11, column: 7, scope: !17)
!31 = !DIExpression(DW_OP_deref)
!32 = !DILocation(line: 12, column: 3, scope: !17)
!33 = !DILocation(line: 13, column: 3, scope: !17)
!34 = !DILocation(line: 14, column: 10, scope: !17)
!35 = !DILocation(line: 15, column: 1, scope: !17)
!36 = !DILocation(line: 14, column: 3, scope: !17)

