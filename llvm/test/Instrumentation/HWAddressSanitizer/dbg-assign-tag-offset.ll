; RUN: opt -passes=hwasan -S -o - %s | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -passes=hwasan -S -o - %s | FileCheck %s

source_filename = "test.ll"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @g(ptr, ptr, ptr, ptr, ptr, ptr)

; Function Attrs: sanitize_hwaddress
define void @f() #0 !dbg !7 {
entry:
  %nodebug0 = alloca ptr, align 8
  %nodebug1 = alloca ptr, align 8
  %nodebug2 = alloca ptr, align 8
  %nodebug3 = alloca ptr, align 8
  ; CHECK: %a = alloca{{.*}} !DIAssignID ![[ID1:[0-9]+]]
  %a = alloca ptr, align 8, !DIAssignID !13
  ; CHECK: #dbg_assign{{.*}} ![[ID1]]{{.*}} !DIExpression(DW_OP_LLVM_tag_offset, 32)
  call void @llvm.dbg.assign(metadata i1 undef, metadata !14, metadata !DIExpression(), metadata !13, metadata ptr %a, metadata !DIExpression()), !dbg !15
  ; CHECK: %b = alloca{{.*}} !DIAssignID ![[ID2:[0-9]+]]
  %b = alloca ptr, align 8, !DIAssignID !16
  ; CHECK: #dbg_assign{{.*}} ![[ID2]]{{.*}} !DIExpression(DW_OP_LLVM_tag_offset, 96)
  call void @llvm.dbg.assign(metadata i1 undef, metadata !17, metadata !DIExpression(), metadata !16, metadata ptr %b, metadata !DIExpression()), !dbg !15
  call void @g(ptr %nodebug0, ptr %nodebug1, ptr %nodebug2, ptr %nodebug3, ptr %a, ptr %b)
  ret void, !dbg !18
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #1

attributes #0 = { sanitize_hwaddress }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "x.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !12)
!12 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!13 = distinct !DIAssignID()
!14 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocation(line: 0, scope: !7)
!16 = distinct !DIAssignID()
!17 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 1, type: !10)
!18 = !DILocation(line: 1, column: 37, scope: !7)
