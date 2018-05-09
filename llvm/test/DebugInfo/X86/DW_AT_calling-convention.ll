; RUN: llc < %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s

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

@"\01?fptr@@3P6IHHH@ZA" = global i32 (i32, i32)* @"\01?f@@YIHHH@Z", align 4, !dbg !0

; Function Attrs: nounwind readnone
define x86_fastcallcc i32 @"\01?f@@YIHHH@Z"(i32 inreg %a, i32 inreg %b) #0 !dbg !13 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %b, metadata !15, metadata !17), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %a, metadata !16, metadata !17), !dbg !19
  %add = add nsw i32 %b, %a, !dbg !20
  ret i32 %add, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "fptr", linkageName: "\01?fptr@@3P6IHHH@ZA", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.9.0 (trunk 272067)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32, align: 32)
!7 = !DISubroutineType(cc: DW_CC_BORLAND_msfastcall, types: !8)
!8 = !{!9, !9, !9}
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.9.0 (trunk 272067)"}
!13 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YIHHH@Z", scope: !3, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !14)
!14 = !{!15, !16}
!15 = !DILocalVariable(name: "b", arg: 2, scope: !13, file: !3, line: 1, type: !9)
!16 = !DILocalVariable(name: "a", arg: 1, scope: !13, file: !3, line: 1, type: !9)
!17 = !DIExpression()
!18 = !DILocation(line: 1, column: 29, scope: !13)
!19 = !DILocation(line: 1, column: 22, scope: !13)
!20 = !DILocation(line: 1, column: 43, scope: !13)
!21 = !DILocation(line: 1, column: 34, scope: !13)

