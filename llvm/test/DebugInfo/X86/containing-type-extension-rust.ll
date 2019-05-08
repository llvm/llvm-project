; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

; Check that any type can have a vtable holder.
; CHECK: [[SP:.*]]: DW_TAG_structure_type
; CHECK-NOT: TAG
; CHECK: DW_AT_containing_type [DW_FORM_ref4]
; CHECK: DW_AT_name [DW_FORM_strp] {{.*}}= "vtable")

; This was compiled using
; rustc -g --emit=llvm-ir t2.rs
; ... and then edited by hand, because rustc is using a somewhat older llvm.
;
; t2.rs is:
;
; // trait object test case
;
; pub trait T {
; }
;
; impl T for f64 {
; }
;
; pub fn main() {
;     let tu = &23.0f64 as &T;
; }
; t2.rs ends here ^^^

; ModuleID = 't2.cgu-0.rs'
source_filename = "t2.cgu-0.rs"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@ref.0 = internal unnamed_addr constant double 2.300000e+01, align 8
@vtable.1 = internal unnamed_addr constant { void (double*)*, i64, i64 } { void (double*)* @_ZN4core3ptr13drop_in_place17h2818a933abde117eE, i64 8, i64 8 }, align 8, !dbg !0
@__rustc_debug_gdb_scripts_section__ = linkonce_odr unnamed_addr constant [34 x i8] c"\01gdb_load_rust_pretty_printers.py\00", section ".debug_gdb_scripts", align 1

; core::ptr::drop_in_place
; Function Attrs: uwtable
define internal void @_ZN4core3ptr13drop_in_place17h2818a933abde117eE(double*) unnamed_addr #0 !dbg !11 {
start:
  %arg0 = alloca double*
  store double* %0, double** %arg0
  call void @llvm.dbg.declare(metadata double** %arg0, metadata !20, metadata !22), !dbg !23
  ret void, !dbg !24
}

; t2::main
; Function Attrs: uwtable
define internal void @_ZN2t24main17h6319e6ac7de3a097E() unnamed_addr #0 !dbg !25 {
start:
  %tu = alloca { i8*, void (i8*)** }
  call void @llvm.dbg.declare(metadata { i8*, void (i8*)** }* %tu, metadata !29, metadata !22), !dbg !37
  %0 = getelementptr inbounds { i8*, void (i8*)** }, { i8*, void (i8*)** }* %tu, i32 0, i32 0, !dbg !37
  store i8* bitcast (double* @ref.0 to i8*), i8** %0, !dbg !37
  %1 = getelementptr inbounds { i8*, void (i8*)** }, { i8*, void (i8*)** }* %tu, i32 0, i32 1, !dbg !37
  store void (i8*)** bitcast ({ void (double*)*, i64, i64 }* @vtable.1 to void (i8*)**), void (i8*)*** %1, !dbg !37
  ret void, !dbg !38
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define i32 @main(i32, i8**) unnamed_addr #2 {
top:
  %2 = load volatile i8, i8* getelementptr inbounds ([34 x i8], [34 x i8]* @__rustc_debug_gdb_scripts_section__, i32 0, i32 0), align 1
  %3 = sext i32 %0 to i64
; call std::rt::lang_start
  %4 = call i64 @_ZN3std2rt10lang_start17h2626caf1112a00beE(void ()* @_ZN2t24main17h6319e6ac7de3a097E, i64 %3, i8** %1)
  %5 = trunc i64 %4 to i32
  ret i32 %5
}

; std::rt::lang_start
declare i64 @_ZN3std2rt10lang_start17h2626caf1112a00beE(void ()*, i64, i8**) unnamed_addr #3

attributes #0 = { uwtable "no-frame-pointer-elim"="true" "probe-stack"="__rust_probestack" }
attributes #1 = { nounwind readnone }
attributes #2 = { "no-frame-pointer-elim"="true" }
attributes #3 = { "no-frame-pointer-elim"="true" "probe-stack"="__rust_probestack" }

!llvm.module.flags = !{!6, !7}
!llvm.dbg.cu = !{!8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "vtable", scope: null, file: !2, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "<unknown>", directory: "")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "vtable", file: !2, size: 64, align: 64, flags: DIFlagArtificial, elements: !4, vtableHolder: !5, identifier: "vtable")
!4 = !{}
!5 = !DIBasicType(name: "f64", size: 64, encoding: DW_ATE_float)
!6 = !{i32 1, !"PIE Level", i32 2}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !9, producer: "clang LLVM (rustc version 1.22.0-dev)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !10)
!9 = !DIFile(filename: "t2.rs", directory: "/home/tromey/Rust")
!10 = !{!0}
!11 = distinct !DISubprogram(name: "drop_in_place<f64>", linkageName: "_ZN4core3ptr18drop_in_place<f64>E", scope: !13, file: !12, line: 59, type: !15, isLocal: false, isDefinition: true, scopeLine: 59, flags: DIFlagPrototyped, isOptimized: false, unit: !8, templateParams: !18, retainedNodes: !4)
!12 = !DIFile(filename: "/home/tromey/Rust/rust/src/libcore/ptr.rs", directory: "")
!13 = !DINamespace(name: "ptr", scope: !14)
!14 = !DINamespace(name: "core", scope: null)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*mut f64", baseType: !5, size: 64, align: 64)
!18 = !{!19}
!19 = !DITemplateTypeParameter(name: "T", type: !5)
!20 = !DILocalVariable(arg: 1, scope: !11, file: !21, line: 1, type: !17)
!21 = !DIFile(filename: "t2.rs", directory: "")
!22 = !DIExpression()
!23 = !DILocation(line: 1, scope: !11)
!24 = !DILocation(line: 59, scope: !11)
!25 = distinct !DISubprogram(name: "main", linkageName: "_ZN2t24mainE", scope: !26, file: !9, line: 9, type: !27, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagMainSubprogram, isOptimized: false, unit: !8, templateParams: !4, retainedNodes: !4)
!26 = !DINamespace(name: "t2", scope: null)
!27 = !DISubroutineType(types: !28)
!28 = !{null}
!29 = !DILocalVariable(name: "tu", scope: !30, file: !9, line: 10, type: !31, align: 8)
!30 = distinct !DILexicalBlock(scope: !25, file: !9, line: 10, column: 4)
!31 = !DICompositeType(tag: DW_TAG_structure_type, name: "&T", scope: !26, file: !2, size: 128, align: 64, elements: !32, identifier: "b9f642b757d8ad3984c1e721e3ce6016d14d9322")
!32 = !{!33, !36}
!33 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !31, file: !2, baseType: !34, size: 64, align: 64, flags: DIFlagArtificial)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const u8", baseType: !35, size: 64, align: 64)
!35 = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
!36 = !DIDerivedType(tag: DW_TAG_member, name: "vtable", scope: !31, file: !2, baseType: !34, size: 64, align: 64, offset: 64, flags: DIFlagArtificial)
!37 = !DILocation(line: 10, scope: !30)
!38 = !DILocation(line: 11, scope: !25)
