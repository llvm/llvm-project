; RUN: llc < %s -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; struct A {
;   void thiscallcc();
; };
; void A::thiscallcc() {}
; void cdeclcc() {}
; void __fastcall fastcallcc() {}
; void __stdcall stdcallcc() {}
; void __vectorcall vectorcallcc() {}
; $ clang -g t.cpp -emit-llvm -S -o t.ll -O1

; CHECK: .debug_abbrev contents:

; CHECK: [[subroutine_abbrev:\[[0-9]+\]]] DW_TAG_subroutine_type      DW_CHILDREN_yes
; CHECK-NEXT:         DW_AT_type      DW_FORM_ref4
; CHECK-NEXT:         DW_AT_calling_convention        DW_FORM_data1

; CHECK: .debug_info contents:

; CHECK:  DW_TAG_subroutine_type [[subroutine_abbrev]] *
; CHECK-NEXT:         DW_AT_type [DW_FORM_ref4]       {{.*}}
; CHECK-NEXT:         DW_AT_calling_convention [DW_FORM_data1]        (DW_CC_BORLAND_msfastcall)

; CHECK: DW_TAG_subprogram [{{.*}}] *
; CHECK:                 DW_AT_low_pc
; CHECK:                 DW_AT_high_pc
; CHECK:                 DW_AT_frame_base
; CHECK:                 DW_AT_linkage_name
; CHECK:                 DW_AT_name
; CHECK:                 DW_AT_decl_file
; CHECK:                 DW_AT_decl_line
; CHECK:                 DW_AT_calling_convention [DW_FORM_data1]        (DW_CC_BORLAND_msfastcall)
; CHECK:                 DW_AT_type
; CHECK:                 DW_AT_external

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.0.23918"

@"\01?fptr@@3P6IHHH@ZA" = global i32 (i32, i32)* @"\01?f@@YIHHH@Z", align 4

; Function Attrs: nounwind readnone
define x86_fastcallcc i32 @"\01?f@@YIHHH@Z"(i32 inreg %a, i32 inreg %b) #0 !dbg !12 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %b, i64 0, metadata !14, metadata !16), !dbg !17
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !15, metadata !16), !dbg !18
  %add = add nsw i32 %b, %a, !dbg !19
  ret i32 %add, !dbg !20
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 272067)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "fptr", linkageName: "\01?fptr@@3P6IHHH@ZA", scope: !0, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, variable: i32 (i32, i32)** @"\01?fptr@@3P6IHHH@ZA")
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 32, align: 32)
!6 = !DISubroutineType(cc: DW_CC_BORLAND_msfastcall, types: !7)
!7 = !{!8, !8, !8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.9.0 (trunk 272067)"}
!12 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YIHHH@Z", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !13)
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "b", arg: 2, scope: !12, file: !1, line: 1, type: !8)
!15 = !DILocalVariable(name: "a", arg: 1, scope: !12, file: !1, line: 1, type: !8)
!16 = !DIExpression()
!17 = !DILocation(line: 1, column: 29, scope: !12)
!18 = !DILocation(line: 1, column: 22, scope: !12)
!19 = !DILocation(line: 1, column: 43, scope: !12)
!20 = !DILocation(line: 1, column: 34, scope: !12)
