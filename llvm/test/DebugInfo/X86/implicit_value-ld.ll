;; This test checks for emission of DW_OP_implicit_value operation
;; for long double type.

; RUN: llc -debugger-tune=gdb -filetype=obj %s -o - | llvm-dwarfdump - | FileCheck %s

; CHECK: .debug_info contents:
; CHECK: DW_TAG_variable
; CHECK-NEXT:  DW_AT_location        ({{.*}}
; CHECK-NEXT:                     [{{.*}}): DW_OP_implicit_value 0x10 0x00 0xf8 0x28 0x5c 0x8f 0xc2 0xf5 0xc8 0x00 0x40 0x00 0x00 0x00 0x00 0x00 0x00)
; CHECK-NEXT:  DW_AT_name    ("ld")

;; Generated from: clang -ggdb -O1
;;int main() {
;;        long double ld = 3.14;
;;        printf("dummy\n");
;;        ld *= ld;
;;        return 0;
;;}

; ModuleID = 'implicit_value-ld.c'
source_filename = "implicit_value-ld.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@str = private unnamed_addr constant [6 x i8] c"dummy\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.declare(metadata [6 x i8]* undef, metadata !12, metadata !DIExpression(DW_OP_LLVM_fragment, 80, 48)), !dbg !14
  call void @llvm.dbg.value(metadata x86_fp80 0xK4000C8F5C28F5C28F800, metadata !12, metadata !DIExpression()), !dbg !15
  %puts = call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @str, i64 0, i64 0)), !dbg !16
  call void @llvm.dbg.value(metadata x86_fp80 undef, metadata !12, metadata !DIExpression()), !dbg !15
  ret i32 0, !dbg !17
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: nofree nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nofree nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "implicit_value-ld.c", directory: "/home/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "ld", scope: !7, file: !1, line: 2, type: !13)
!13 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!14 = !DILocation(line: 2, column: 14, scope: !7)
!15 = !DILocation(line: 0, scope: !7)
!16 = !DILocation(line: 3, column: 2, scope: !7)
!17 = !DILocation(line: 5, column: 2, scope: !7)
