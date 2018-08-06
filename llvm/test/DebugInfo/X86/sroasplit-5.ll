; RUN: opt %s -sroa -verify -S -o - | FileCheck %s
; From:
; struct prog_src_register {
;   unsigned : 4;       
;   int Index : 12 + 1; 
;   unsigned : 12;      
;   unsigned : 4;       
;   int : 12 + 1        
; } src_reg_for_float() {
;   struct prog_src_register a;
;   memset(&a, 0, sizeof(a));
;   int local = a.Index;
;   return a;
; }
; ModuleID = 'pr22495.c'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; When SROA is creating new smaller allocas, it may add padding.
;
; There should be no debug info for the padding.
; CHECK-NOT: DW_OP_LLVM_fragment, 56
; CHECK: DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NOT: DW_OP_LLVM_fragment, 56
; CHECK: DIExpression(DW_OP_LLVM_fragment, 32, 24)
; CHECK-NOT: DW_OP_LLVM_fragment, 56
%struct.prog_src_register = type { i32, i24 }

; Function Attrs: nounwind
define i64 @src_reg_for_float() #0 !dbg !4 {
entry:
  %retval = alloca %struct.prog_src_register, align 4
  %a = alloca %struct.prog_src_register, align 4
  %local = alloca i32, align 4
  call void @llvm.dbg.declare(metadata %struct.prog_src_register* %a, metadata !16, metadata !17), !dbg !18
  %0 = bitcast %struct.prog_src_register* %a to i8*, !dbg !19
  call void @llvm.memset.p0i8.i64(i8* align 4 %0, i8 0, i64 8, i1 false), !dbg !19
  call void @llvm.dbg.declare(metadata i32* %local, metadata !20, metadata !17), !dbg !21
  %1 = bitcast %struct.prog_src_register* %a to i32*, !dbg !21
  %bf.load = load i32, i32* %1, align 4, !dbg !21
  %bf.shl = shl i32 %bf.load, 15, !dbg !21
  %bf.ashr = ashr i32 %bf.shl, 19, !dbg !21
  store i32 %bf.ashr, i32* %local, align 4, !dbg !21
  %2 = bitcast %struct.prog_src_register* %retval to i8*, !dbg !22
  %3 = bitcast %struct.prog_src_register* %a to i8*, !dbg !22
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 8, i1 false), !dbg !22
  %4 = bitcast %struct.prog_src_register* %retval to i64*, !dbg !22
  %5 = load i64, i64* %4, align 1, !dbg !22
  ret i64 %5, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) #2

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.7.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "src_reg_for_float", line: 7, isLocal: false, isDefinition: true, isOptimized: false, unit: !0, scopeLine: 7, file: !5, scope: !6, type: !7, retainedNodes: !2)
!5 = !DIFile(filename: "pr22495.c", directory: "")
!6 = !DIFile(filename: "pr22495.c", directory: "")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "prog_src_register", line: 1, size: 64, align: 32, file: !5, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "Index", line: 3, size: 13, align: 32, offset: 4, file: !5, scope: !9, baseType: !12)
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.7.0 "}
!16 = !DILocalVariable(name: "a", line: 8, scope: !4, file: !6, type: !9)
!17 = !DIExpression()
!18 = !DILocation(line: 8, scope: !4)
!19 = !DILocation(line: 9, scope: !4)
!20 = !DILocalVariable(name: "local", line: 10, scope: !4, file: !6, type: !12)
!21 = !DILocation(line: 10, scope: !4)
!22 = !DILocation(line: 11, scope: !4)
