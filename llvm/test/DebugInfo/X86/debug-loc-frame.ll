; REQUIRES: object-emission

; Check that when variables are allocated on the stack we generate debug locations
; for the stack location directly instead of generating a register+offset indirection.

; RUN: llc -O2 -filetype=obj -disable-post-ra -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN: | llvm-dwarfdump - | FileCheck %s
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
; CHECK:      DW_AT_location [DW_FORM_sec_offset] ([[DEBUGLOCOFFSET:0x[0-9a-f]+]]){{[[:space:]].*}}"val"

; See that 'val' has at least one location entry with a DW_op_breg? operand.
; The DWARF DW_op_breg* ops are encoded from 0x70 to 0x8f, but checking for an
; op in the range from 0x70 to 0x7f should suffice because that range covers
; all integer GPRs.
;
; CHECK: .debug_loc contents:
; CHECK-NOT: .debug{{.*}} contents
; CHECK: [[DEBUGLOCOFFSET]]: Beginning
; CHECK-NOT: {{0x[0-9a-f]+}}: Beginning
; CHECK: Location description: 7{{[0-9a-f] .*}}

; ModuleID = 'frame.c'
source_filename = "frame.c"

@data = global i32 17, align 4, !dbg !4
@sum = local_unnamed_addr global i32 0, align 4, !dbg !6
@zero = local_unnamed_addr global i32 0, align 4, !dbg !7
@ptr = common local_unnamed_addr global i32* null, align 8, !dbg !8

; Function Attrs: nounwind uwtable
define i32 @main() local_unnamed_addr !dbg !13 {
entry:
  %val = alloca i32, align 4
  %0 = bitcast i32* %val to i8*, !dbg !18
  call void @llvm.lifetime.start(i64 4, i8* %0), !dbg !18
  %1 = load i32, i32* @data, align 4, !dbg !19, !tbaa !20
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !17, metadata !24), !dbg !25
  store i32 %1, i32* %val, align 4, !dbg !26, !tbaa !20
  tail call void @llvm.dbg.value(metadata i32* %val, i64 0, metadata !17, metadata !27), !dbg !25
  call void @foo(i32 1, i32* nonnull %val), !dbg !28
  call void @foo(i32 2, i32* nonnull @data), !dbg !29
  %2 = load i32, i32* @zero, align 4, !dbg !30, !tbaa !20
  call void @llvm.lifetime.end(i64 4, i8* %0), !dbg !31
  ret i32 %2, !dbg !32
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture)

declare void @foo(i32, i32*) local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 273961)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "frame.c", directory: "/home/user/test")
!2 = !{}
!3 = !{!4, !6, !7, !8}
!4 = distinct !DIGlobalVariable(name: "data", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true)
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = distinct !DIGlobalVariable(name: "sum", scope: !0, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true)
!7 = distinct !DIGlobalVariable(name: "zero", scope: !0, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true)
!8 = distinct !DIGlobalVariable(name: "ptr", scope: !0, file: !1, line: 4, type: !9, isLocal: false, isDefinition: true)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.9.0 (trunk 273961)"}
!13 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 8, type: !14, isLocal: false, isDefinition: true, scopeLine: 9, isOptimized: true, unit: !0, variables: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!5}
!16 = !{!17}
!17 = !DILocalVariable(name: "val", scope: !13, file: !1, line: 10, type: !5)
!18 = !DILocation(line: 10, column: 3, scope: !13)
!19 = !DILocation(line: 11, column: 9, scope: !13)
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DIExpression()
!25 = !DILocation(line: 10, column: 7, scope: !13)
!26 = !DILocation(line: 11, column: 7, scope: !13)
!27 = !DIExpression(DW_OP_deref)
!28 = !DILocation(line: 12, column: 3, scope: !13)
!29 = !DILocation(line: 13, column: 3, scope: !13)
!30 = !DILocation(line: 14, column: 10, scope: !13)
!31 = !DILocation(line: 15, column: 1, scope: !13)
!32 = !DILocation(line: 14, column: 3, scope: !13)
