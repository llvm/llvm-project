; This test verifies that the debug info for an external Fortran module
; is correctly generated.
;
; To generate the test source, compile the following two files in order in
; the same directory (as the second compilation uses the .mod from the first):
; external_module.f90 (to compile: <fortran compiler> -g -c external_module.f90)
;   module external_module
;     real :: dummy
;   end module external_module
;
; em.f90 (to compile: <fortran compierl> -g -llvm-emit -c -S em.f90)
;   program use_external_module
;   use external_module
;   implicit none
;
;     real :: x
;     x = 2.0 + dummy
;
;   end program use_external_module
;
; The test would be in em.ll.

; RUN: llc -filetype=obj  %s -o - | llvm-dwarfdump - | FileCheck %s

; CHECK: [[DIE_ID:0x[0-9a-f]+]]: DW_TAG_module
; CHECK-NEXT:                      DW_AT_name    ("external_module")
; CHECK-NEXT:                      DW_AT_declaration     (true)

; CHECK:      DW_TAG_imported_module
; CHECK-NEXT:   DW_AT_decl_file
; CHECK-NEXT:   DW_AT_decl_line
; CHECK-NEXT:   DW_AT_import  ([[DIE_ID]])

; When the debugger sees the module being imported is a declaration,
; it should go to the global scope to find the module's definition.

; ModuleID = 'em.f90'
source_filename = "em.f90"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@external_module_mp_dummy_ = available_externally global float 0.000000e+00, align 8, !dbg !0
@0 = internal unnamed_addr constant i32 2

; Function Attrs: noinline nounwind uwtable
define void @MAIN__() #0 !dbg !2 {
alloca_0:
  %"var$1" = alloca [8 x i64], align 8
  %"use_external_module_$X" = alloca float, align 8
  call void @llvm.dbg.declare(metadata ptr %"use_external_module_$X", metadata !13, metadata !DIExpression()), !dbg !17
  %func_result = call i32 @for_set_reentrancy(ptr @0), !dbg !18
  %external_module_mp_dummy__fetch = load float, ptr @external_module_mp_dummy_, align 1, !dbg !19
  %add = fadd reassoc ninf nsz arcp contract afn float 2.000000e+00, %external_module_mp_dummy__fetch, !dbg !20
  store float %add, ptr %"use_external_module_$X", align 1, !dbg !19
  ret void, !dbg !21
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @for_set_reentrancy(ptr)

attributes #0 = { noinline nounwind uwtable "intel-lang"="fortran" "min-legal-vector-width"="0" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!15, !16}
!llvm.dbg.cu = !{!6}
!omp_offload.info = !{}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "dummy", linkageName: "external_module_mp_dummy_", scope: !2, file: !3, line: 1, type: !14, isLocal: false, isDefinition: true)
!2 = distinct !DISubprogram(name: "use_external_module", linkageName: "MAIN__", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !12)
!3 = !DIFile(filename: "em.f90", directory: "tests")
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "Intel(R) Fortran 21.0-2165", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, globals: !8, splitDebugInlining: false, nameTableKind: None)
!7 = !{}
!8 = !{!0}
!9 = !{}
!10 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !2, entity: !11, file: !3, line: 2)
!11 = !DIModule(scope: !2, name: "external_module", isDecl: true)
!12 = !{!10, !13}
!13 = !DILocalVariable(name: "x", scope: !2, file: !3, line: 5, type: !14)
!14 = !DIBasicType(name: "REAL*4", size: 32, encoding: DW_ATE_float)
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !DILocation(line: 5, column: 12, scope: !2)
!18 = !DILocation(line: 1, column: 9, scope: !2)
!19 = !DILocation(line: 6, column: 4, scope: !2)
!20 = !DILocation(line: 6, column: 12, scope: !2)
!21 = !DILocation(line: 8, column: 1, scope: !2)
