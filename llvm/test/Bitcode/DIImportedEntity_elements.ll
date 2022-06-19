;; This test checks processing of DIImportedEntity with elements field.
; REQUIRES: x86_64-linux

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: !DIImportedEntity(tag: DW_TAG_imported_module
; CHECK-SAME: elements: [[ELEMENTS:![0-9]+]]
; CHECK: [[ELEMENTS]] = !{[[ELEMENT:![0-9]+]]}
; CHECK: [[ELEMENT]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "var4"

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;This test case is generated from
;;module mymod
;;  integer :: var1 = 11
;;  integer :: var2 = 12
;;  integer :: var3 = 13
;;end module mymod
;;
;;Program main
;;  call use_renamed()
;;  contains
;;    subroutine use_renamed()
;;      use mymod, var4 => var1
;;      print *, var4
;;    end subroutine use_renamed
;;end program main
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; ModuleID = 'DIImportedEntity_elements.ll'
source_filename = "/tmp/usemodulealias.ll"

%struct_mymod_8_ = type <{ [12 x i8] }>
%struct.struct_ul_MAIN__348 = type { i8* }

@_mymod_8_ = global %struct_mymod_8_ <{ [12 x i8] c"\0B\00\00\00\0C\00\00\00\0D\00\00\00" }>, align 64, !dbg !0, !dbg !7, !dbg !10
@.C330_MAIN_ = internal constant i32 0
@.C364_main_use_renamed = internal constant i32 25
@.C330_main_use_renamed = internal constant i32 0
@.C331_main_use_renamed = internal constant i64 0
@.C359_main_use_renamed = internal constant i32 6
@.C357_main_use_renamed = internal constant [18 x i8] c"usemodulealias.f90"
@.C352_main_use_renamed = internal constant i32 12

define i32 @mymod_() {
.L.entry:
  ret i32 undef
}

define void @MAIN_() !dbg !15 {
L.entry:
  %.S0000_353 = alloca %struct.struct_ul_MAIN__348, align 8
  %0 = bitcast i32* @.C330_MAIN_ to i8*
  %1 = bitcast void (...)* @fort_init to void (i8*, ...)*
  call void (i8*, ...) %1(i8* %0)
  br label %L.LB2_357

L.LB2_357:                                        ; preds = %L.entry
  %2 = bitcast %struct.struct_ul_MAIN__348* %.S0000_353 to i64*, !dbg !23
  call void @main_use_renamed(i64* %2), !dbg !23
  ret void, !dbg !24
}

define internal void @main_use_renamed(i64* noalias %.S0000) !dbg !14 {
L.entry:
  ret void, !dbg !25
}

declare void @fort_init(...)

!llvm.module.flags = !{!21, !22}
!llvm.dbg.cu = !{!4}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var1", scope: !2, file: !3, line: 2, type: !9, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: !4, name: "mymod", file: !3, line: 1)
!3 = !DIFile(filename: "DIImportedEntity_elements.f90", directory: "/tmp")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, flags: "'+flang usemodulealias.f90 -g -S -emit-llvm'", runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !12, nameTableKind: None)
!5 = !{}
!6 = !{!0, !7, !10}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 4))
!8 = distinct !DIGlobalVariable(name: "var2", scope: !2, file: !3, line: 3, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression(DW_OP_plus_uconst, 8))
!11 = distinct !DIGlobalVariable(name: "var3", scope: !2, file: !3, line: 4, type: !9, isLocal: false, isDefinition: true)
!12 = !{!13}
!13 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !14, entity: !2, file: !3, line: 10, elements: !19)
!14 = distinct !DISubprogram(name: "use_renamed", scope: !15, file: !3, line: 10, type: !18, scopeLine: 10, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4)
!15 = distinct !DISubprogram(name: "main", scope: !4, file: !3, line: 7, type: !16, scopeLine: 7, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !4)
!16 = !DISubroutineType(cc: DW_CC_program, types: !17)
!17 = !{null}
!18 = !DISubroutineType(types: !17)
!19 = !{!20}
!20 = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "var4", scope: !14, entity: !1, file: !3, line: 10)
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !DILocation(line: 8, column: 1, scope: !15)
!24 = !DILocation(line: 9, column: 1, scope: !15)
!25 = !DILocation(line: 13, column: 1, scope: !14)
