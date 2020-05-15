;; This test checks whether DWARF operator DW_OP_push_object_address
;; is accepted and processed.

; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

;; Test whether DW_OP_push_object_address is accepted.

; CHECK-LABEL:       DW_TAG_variable
; CHECK:        DW_AT_location
; CHECK-SAME:        DW_OP_push_object_address

;; Below is the original test case this IR is generated from
;;---------------------------
;;int main() {
;;int var;
;;return var;
;;}
;;---------------------------
;; step 1: generate IR using -g -O0 -S -emit-llvm
;; step 2: insert DW_OP_push_object_address in dbg.declare instruction
;; This is meaningless test case focused to test DW_OP_push_object_address.

; ModuleID = 'dwarfdump-pushobjectaddress.c'
source_filename = "dwarfdump-pushobjectaddress.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() !dbg !7 {
entry:
  %retval = alloca i32, align 4
  %var = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata i32* %var, metadata !11, metadata !DIExpression(DW_OP_push_object_address)), !dbg !12
  %0 = load i32, i32* %var, align 4, !dbg !13
  ret i32 %0, !dbg !14
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "dwarfdump-pushobjectaddress.c", directory: "/dir")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "var", scope: !7, file: !1, line: 2, type: !10)
!12 = !DILocation(line: 2, column: 5, scope: !7)
!13 = !DILocation(line: 3, column: 8, scope: !7)
!14 = !DILocation(line: 3, column: 1, scope: !7)
