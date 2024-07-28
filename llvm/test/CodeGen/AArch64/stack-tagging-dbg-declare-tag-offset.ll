; RUN: opt -aarch64-stack-tagging -stack-tagging-record-stack-history=instr -S -o - %s | FileCheck %s

;; Also test with RemoveDIs to verify that debug intrinsics immediately
;; preceding an alloca (or other instruction of interest to stack tagging) will
;; be correctly processed.
; RUN: opt --try-experimental-debuginfo-iterators -aarch64-stack-tagging -stack-tagging-record-stack-history=instr -S -o - %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android10000"

declare void @g(ptr, ptr, ptr, ptr, ptr, ptr)

define void @f() sanitize_memtag !dbg !6 {
entry:
  %nodebug0 = alloca ptr
  %nodebug1 = alloca ptr
  %nodebug2 = alloca ptr
  %nodebug3 = alloca ptr
  %a = alloca ptr
  ; CHECK: #dbg_declare{{.*}} !DIExpression(DW_OP_LLVM_tag_offset, 4)
  call void @llvm.dbg.declare(metadata ptr %a, metadata !12, metadata !DIExpression()), !dbg !14
  ; CHECK: #dbg_declare{{.*}} !DIExpression(DW_OP_LLVM_tag_offset, 4)
  call void @llvm.dbg.declare(metadata ptr %a, metadata !12, metadata !DIExpression()), !dbg !14
  %b = alloca ptr
  ; CHECK: #dbg_declare{{.*}} !DIExpression(DW_OP_LLVM_tag_offset, 5)
  call void @llvm.dbg.declare(metadata ptr %b, metadata !13, metadata !DIExpression()), !dbg !14
  ; CHECK: #dbg_declare{{.*}} !DIExpression(DW_OP_LLVM_tag_offset, 5)
  call void @llvm.dbg.declare(metadata ptr %b, metadata !13, metadata !DIExpression()), !dbg !14
  call void @g(ptr %nodebug0, ptr %nodebug1, ptr %nodebug2, ptr %nodebug3, ptr %a, ptr %b)
  ret void, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "x.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang"}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags:
DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !DILocalVariable(name: "a", scope: !6, file: !1, line: 1, type: !9)
!13 = !DILocalVariable(name: "b", scope: !6, file: !1, line: 1, type: !9)
!14 = !DILocation(line: 1, column: 29, scope: !6)
!15 = !DILocation(line: 1, column: 37, scope: !6)
