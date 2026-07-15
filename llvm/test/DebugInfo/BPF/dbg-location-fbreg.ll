; RUN: llc -O2 -mtriple=bpfel -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; RUN: llc -O2 -mtriple=bpfeb -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
;
; Source:
;   #![no_std]
;   use core::hint::black_box;
;
;   #[inline(never)]
;   pub extern "C" fn entrypoint() -> u64 {
;       let mut var_a: u64 = 0x1111_2222_3333_4444;
;       let mut var_b: u64 = 0xAAAA_BBBB_CCCC_DDDD;
;       black_box(&mut var_a);
;       black_box(&mut var_b);
;       var_a ^ var_b
;   }
; Compilation flag:
;   rustc lib.rs --emit=llvm-ir -g -O --crate-type=lib

; ModuleID = 'lib.1f96fa33bef8cf6b-cgu.0'
source_filename = "lib.1f96fa33bef8cf6b-cgu.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; lib::entrypoint
; Function Attrs: noinline nounwind uwtable
define noundef i64 @_ZN3lib10entrypoint17hbcd54b7b97a1f371E() unnamed_addr #0 !dbg !6 {
start:
  %0 = alloca [8 x i8], align 8
  %1 = alloca [8 x i8], align 8
  %var_b = alloca [8 x i8], align 8
  %var_a = alloca [8 x i8], align 8
    #dbg_declare(ptr %var_a, !13, !DIExpression(), !18)
    #dbg_declare(ptr %var_b, !15, !DIExpression(), !19)
  call void @llvm.lifetime.start.p0(ptr nonnull %var_a), !dbg !20
  store i64 1229801703532086340, ptr %var_a, align 8, !dbg !21
  call void @llvm.lifetime.start.p0(ptr nonnull %var_b), !dbg !22
  store i64 -6148895925951734307, ptr %var_b, align 8, !dbg !23
    #dbg_value(ptr %var_a, !24, !DIExpression(), !35)
  call void @llvm.lifetime.start.p0(ptr nonnull %1), !dbg !37
  store ptr %var_a, ptr %1, align 8, !dbg !37
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %1) #2, !dbg !37, !srcloc !38
  call void @llvm.lifetime.end.p0(ptr nonnull %1), !dbg !37
    #dbg_value(ptr %var_b, !24, !DIExpression(), !39)
  call void @llvm.lifetime.start.p0(ptr nonnull %0), !dbg !41
  store ptr %var_b, ptr %0, align 8, !dbg !41
  call void asm sideeffect "", "r,~{memory}"(ptr nonnull %0) #2, !dbg !41, !srcloc !38
  call void @llvm.lifetime.end.p0(ptr nonnull %0), !dbg !41
  %_7 = load i64, ptr %var_a, align 8, !dbg !42, !noundef !17
  %_8 = load i64, ptr %var_b, align 8, !dbg !43, !noundef !17
  %_0 = xor i64 %_8, %_7, !dbg !42
  call void @llvm.lifetime.end.p0(ptr nonnull %var_b), !dbg !44
  call void @llvm.lifetime.end.p0(ptr nonnull %var_a), !dbg !45
  ret i64 %_0, !dbg !46
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "probe-stack"="inline-asm" "target-cpu"="penryn" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}
!llvm.dbg.cu = !{!4}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{!"rustc version 1.96.0 (ac68faa20 2026-05-25)"}
!4 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !5, producer: "clang LLVM (rustc version 1.96.0 (ac68faa20 2026-05-25))", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!5 = !DIFile(filename: "lib.rs/@/lib.1f96fa33bef8cf6b-cgu.0", directory: "/tmp/llvm-test/src")
!6 = distinct !DISubprogram(name: "entrypoint", linkageName: "_ZN3lib10entrypoint17hbcd54b7b97a1f371E", scope: !8, file: !7, line: 6, type: !9, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, templateParams: !17, retainedNodes: !12)
!7 = !DIFile(filename: "lib.rs", directory: "/tmp/llvm-test/src", checksumkind: CSK_MD5, checksum: "47ba490b7ac3b9eeccd8e8f3180f2b08")
!8 = !DINamespace(name: "lib", scope: null)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!12 = !{!13, !15}
!13 = !DILocalVariable(name: "var_a", scope: !14, file: !7, line: 7, type: !11, align: 64)
!14 = distinct !DILexicalBlock(scope: !6, file: !7, line: 7, column: 5)
!15 = !DILocalVariable(name: "var_b", scope: !16, file: !7, line: 8, type: !11, align: 64)
!16 = distinct !DILexicalBlock(scope: !14, file: !7, line: 8, column: 5)
!17 = !{}
!18 = !DILocation(line: 7, column: 9, scope: !14)
!19 = !DILocation(line: 8, column: 9, scope: !16)
!20 = !DILocation(line: 7, column: 9, scope: !6)
!21 = !DILocation(line: 7, column: 26, scope: !6)
!22 = !DILocation(line: 8, column: 9, scope: !14)
!23 = !DILocation(line: 8, column: 26, scope: !14)
!24 = !DILocalVariable(name: "dummy", arg: 1, scope: !25, file: !26, line: 490, type: !31)
!25 = distinct !DISubprogram(name: "black_box<&mut u64>", linkageName: "_ZN4core4hint9black_box17h799a64f36faa5e88E", scope: !27, file: !26, line: 490, type: !29, scopeLine: 490, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !4, templateParams: !33, retainedNodes: !32)
!26 = !DIFile(filename: "/Users/b/.rustup/toolchains/aarch64-apple-darwin/lib/rustlib/src/rust/library/core/src/hint.rs", directory: "", checksumkind: CSK_MD5, checksum: "3bdbac5c7616d584a36b114744411911")
!27 = !DINamespace(name: "hint", scope: !28)
!28 = !DINamespace(name: "core", scope: null)
!29 = !DISubroutineType(types: !30)
!30 = !{!31, !31}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&mut u64", baseType: !11, size: 64, align: 64, dwarfAddressSpace: 0)
!32 = !{!24}
!33 = !{!34}
!34 = !DITemplateTypeParameter(name: "T", type: !31)
!35 = !DILocation(line: 0, scope: !25, inlinedAt: !36)
!36 = !DILocation(line: 10, column: 5, scope: !16)
!37 = !DILocation(line: 491, column: 5, scope: !25, inlinedAt: !36)
!38 = !{i64 872325037889853}
!39 = !DILocation(line: 0, scope: !25, inlinedAt: !40)
!40 = !DILocation(line: 11, column: 5, scope: !16)
!41 = !DILocation(line: 491, column: 5, scope: !25, inlinedAt: !40)
!42 = !DILocation(line: 13, column: 5, scope: !16)
!43 = !DILocation(line: 13, column: 13, scope: !16)
!44 = !DILocation(line: 14, column: 1, scope: !14)
!45 = !DILocation(line: 14, column: 1, scope: !6)
!46 = !DILocation(line: 14, column: 2, scope: !6)

; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name ("entrypoint")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location{{.*}}(DW_OP_fbreg -24)
; CHECK: DW_AT_name ("var_a")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location{{.*}}(DW_OP_fbreg -16)
; CHECK: DW_AT_name ("var_b")
