; RUN: opt -passes=mem2reg,instcombine %s -o - -S | FileCheck %s
;
; Test that a dbg.declare describing am alloca with volatile
; load/stores is not lowered into a dbg.value, since the alloca won't
; be elided anyway.
;
; Generated from:
;
; unsigned long long g();
; void h(unsigned long long);
; void f() {
;   volatile unsigned long long v = g();
;   if (v == 0)
;     g();
;   h(v);
; }

; CHECK: alloca i64
; CHECK-NOT: call void @llvm.dbg.value
; CHECK: call void @llvm.dbg.declare
; CHECK-NOT: call void @llvm.dbg.value

source_filename = "volatile.c"

; Function Attrs: nounwind optsize ssp uwtable
define void @f() local_unnamed_addr #0 !dbg !8 {
  %1 = alloca i64, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr %1), !dbg !15
  call void @llvm.dbg.declare(metadata ptr %1, metadata !12, metadata !DIExpression()), !dbg !15
  %2 = call i64 (...) @g() #4, !dbg !15
  store volatile i64 %2, ptr %1, align 8, !dbg !15
  %3 = load volatile i64, ptr %1, align 8, !dbg !15
  %4 = icmp eq i64 %3, 0, !dbg !15
  br i1 %4, label %5, label %7, !dbg !15

; <label>:6:                                      ; preds = %0
  %6 = call i64 (...) @g() #4, !dbg !15
  br label %7, !dbg !15

; <label>:8:                                      ; preds = %5, %0
  %8 = load volatile i64, ptr %1, align 8, !dbg !15
  call void @h(i64 %8) #4, !dbg !15
  call void @llvm.lifetime.end.p0(i64 8, ptr %1), !dbg !15
  ret void, !dbg !15
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: optsize
declare i64 @g(...) local_unnamed_addr #3

; Function Attrs: optsize
declare void @h(i64) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

attributes #0 = { nounwind optsize ssp uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { optsize }
attributes #4 = { optsize }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "volatile.c", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0, retainedNodes: !11)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !{!12}
!12 = !DILocalVariable(name: "v", scope: !8, file: !1, line: 4, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !14)
!14 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!15 = !DILocation(line: 4, column: 3, scope: !8)
