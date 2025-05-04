; RUN: llc -O0 -mtriple=x86_64-apple-darwin -filetype=asm %s -o - | FileCheck %s
; Ensure that we generate an indirect location for the variable length array a.
; CHECK: ##DEBUG_VALUE: vla:a <- [DW_OP_deref] [{{\$r[a-z]+}}+0]
; CHECK: DW_OP_breg{{[0-9]}}
; rdar://problem/13658587
;
; generated from:
;
; int vla(int n) {
;   int a[n];
;   a[0] = 42;
;   return a[n-1];
; }
;
; int main(int argc, char** argv) {
;    return vla(argc);
; }

; ModuleID = 'vla.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Function Attrs: nounwind ssp uwtable
define i32 @vla(i32 %n) nounwind ssp uwtable !dbg !4 {
entry:
  %n.addr = alloca i32, align 4
  %saved_stack = alloca ptr
  %cleanup.dest.slot = alloca i32
  store i32 %n, ptr %n.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %n.addr, metadata !15, metadata !DIExpression()), !dbg !16
  %0 = load i32, ptr %n.addr, align 4, !dbg !17
  %1 = zext i32 %0 to i64, !dbg !17
  %2 = call ptr @llvm.stacksave(), !dbg !17
  store ptr %2, ptr %saved_stack, !dbg !17
  %vla = alloca i32, i64 %1, align 16, !dbg !17
  call void @llvm.dbg.declare(metadata ptr %vla, metadata !18, metadata !DIExpression(DW_OP_deref)), !dbg !17
  store i32 42, ptr %vla, align 4, !dbg !22
  %3 = load i32, ptr %n.addr, align 4, !dbg !23
  %sub = sub nsw i32 %3, 1, !dbg !23
  %idxprom = sext i32 %sub to i64, !dbg !23
  %arrayidx1 = getelementptr inbounds i32, ptr %vla, i64 %idxprom, !dbg !23
  %4 = load i32, ptr %arrayidx1, align 4, !dbg !23
  store i32 1, ptr %cleanup.dest.slot
  %5 = load ptr, ptr %saved_stack, !dbg !24
  call void @llvm.stackrestore(ptr %5), !dbg !24
  ret i32 %4, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

; Function Attrs: nounwind
declare ptr @llvm.stacksave() nounwind

; Function Attrs: nounwind
declare void @llvm.stackrestore(ptr) nounwind

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32 %argc, ptr %argv) nounwind ssp uwtable !dbg !9 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  store i32 0, ptr %retval
  store i32 %argc, ptr %argc.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %argc.addr, metadata !25, metadata !DIExpression()), !dbg !26
  store ptr %argv, ptr %argv.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %argv.addr, metadata !27, metadata !DIExpression()), !dbg !26
  %0 = load i32, ptr %argc.addr, align 4, !dbg !28
  %call = call i32 @vla(i32 %0), !dbg !28
  ret i32 %call, !dbg !28
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.3 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "vla.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "vla", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "vla.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "main", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 7, file: !1, scope: !5, type: !10, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!8, !8, !12}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !13)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !14)
!14 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!15 = !DILocalVariable(name: "n", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!16 = !DILocation(line: 1, scope: !4)
!17 = !DILocation(line: 2, scope: !4)
!18 = !DILocalVariable(name: "a", line: 2, scope: !4, file: !5, type: !19)
!19 = !DICompositeType(tag: DW_TAG_array_type, align: 32, baseType: !8, elements: !20)
!20 = !{!21}
!21 = !DISubrange(count: -1)
!22 = !DILocation(line: 3, scope: !4)
!23 = !DILocation(line: 4, scope: !4)
!24 = !DILocation(line: 5, scope: !4)
!25 = !DILocalVariable(name: "argc", line: 7, arg: 1, scope: !9, file: !5, type: !8)
!26 = !DILocation(line: 7, scope: !9)
!27 = !DILocalVariable(name: "argv", line: 7, arg: 2, scope: !9, file: !5, type: !12)
!28 = !DILocation(line: 8, scope: !9)
!29 = !{i32 1, !"Debug Info Version", i32 3}
