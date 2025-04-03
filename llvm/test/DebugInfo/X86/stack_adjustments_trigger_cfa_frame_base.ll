; RUN: llc < %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s
;
; use core::hint::black_box;
;
; #[inline(never)]
; fn callee(
;     s1: &(),
;     s2: &(),
;     s3: &(),
;     s4: &(),
;     s5: &(),
;     s6: &(),
;     s7: &(),
;     s8: &(),
;     s9: &mut (),
; ) {
;     black_box(s1);
;     black_box(s2);
;     black_box(s3);
;     black_box(s4);
;     black_box(s5);
;     black_box(s6);
;     black_box(s7);
;     black_box(s8);
;     black_box(s9);
; }
;
; pub fn caller() {
;     let s = ();
;     let mut t = ();
;     callee(&s, &s, &s, &s, &s, &s, &s, &s, &mut t);
; }
;
; Test that if a call requires fiddling with the stack pointer we switch to
; using a CFA-based DW_AT_frame_base

; CHECK: DW_AT_frame_base [DW_FORM_exprloc] (DW_OP_call_frame_cfa, DW_OP_consts -{{[0-9]+}}, DW_OP_plus)
; CHECK-NOT: DW_TAG
; CHECK: _ZN10playground6caller

; ModuleID = 'playground.71f4e8b5-cgu.0'
source_filename = "playground.71f4e8b5-cgu.0"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; core::hint::black_box
define align 1 ptr @_ZN4core4hint9black_box17h9f9a3aab786d67e0E(ptr align 1 %dummy) unnamed_addr !dbg !6 {
start:
  %0 = alloca ptr, align 8
  %dummy.dbg.spill = alloca ptr, align 8
  store ptr %dummy, ptr %dummy.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %dummy.dbg.spill, metadata !15, metadata !DIExpression()), !dbg !18
  store ptr %dummy, ptr %0, align 8, !dbg !19
  call void asm sideeffect "", "r,~{memory}"(ptr %0), !dbg !19, !srcloc !20
  %1 = load ptr, ptr %0, align 8, !dbg !19, !nonnull !21, !align !22, !noundef !21
  ret ptr %1, !dbg !23
}

; core::hint::black_box
define align 1 ptr @_ZN4core4hint9black_box17hff24a8f6cdc261d0E(ptr align 1 %dummy) unnamed_addr !dbg !24 {
start:
  %0 = alloca ptr, align 8
  %dummy.dbg.spill = alloca ptr, align 8
  store ptr %dummy, ptr %dummy.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %dummy.dbg.spill, metadata !29, metadata !DIExpression()), !dbg !32
  store ptr %dummy, ptr %0, align 8, !dbg !33
  call void asm sideeffect "", "r,~{memory}"(ptr %0), !dbg !33, !srcloc !20
  %1 = load ptr, ptr %0, align 8, !dbg !33, !nonnull !21, !align !22, !noundef !21
  ret ptr %1, !dbg !34
}

; playground::callee
define internal void @_ZN10playground6callee17hf55947d3dfc887f4E(ptr align 1 %s1, ptr align 1 %s2, ptr align 1 %s3, ptr align 1 %s4, ptr align 1 %s5, ptr align 1 %s6, ptr align 1 %s7, ptr align 1 %s8, ptr align 1 %s9) unnamed_addr !dbg !35 {
start:
  %s9.dbg.spill = alloca ptr, align 8
  %s8.dbg.spill = alloca ptr, align 8
  %s7.dbg.spill = alloca ptr, align 8
  %s6.dbg.spill = alloca ptr, align 8
  %s5.dbg.spill = alloca ptr, align 8
  %s4.dbg.spill = alloca ptr, align 8
  %s3.dbg.spill = alloca ptr, align 8
  %s2.dbg.spill = alloca ptr, align 8
  %s1.dbg.spill = alloca ptr, align 8
  store ptr %s1, ptr %s1.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %s1.dbg.spill, metadata !41, metadata !DIExpression()), !dbg !50
  store ptr %s2, ptr %s2.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %s2.dbg.spill, metadata !42, metadata !DIExpression()), !dbg !51
  store ptr %s3, ptr %s3.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %s3.dbg.spill, metadata !43, metadata !DIExpression()), !dbg !52
  store ptr %s4, ptr %s4.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %s4.dbg.spill, metadata !44, metadata !DIExpression()), !dbg !53
  store ptr %s5, ptr %s5.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %s5.dbg.spill, metadata !45, metadata !DIExpression()), !dbg !54
  store ptr %s6, ptr %s6.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %s6.dbg.spill, metadata !46, metadata !DIExpression()), !dbg !55
  store ptr %s7, ptr %s7.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %s7.dbg.spill, metadata !47, metadata !DIExpression()), !dbg !56
  store ptr %s8, ptr %s8.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %s8.dbg.spill, metadata !48, metadata !DIExpression()), !dbg !57
  store ptr %s9, ptr %s9.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata ptr %s9.dbg.spill, metadata !49, metadata !DIExpression()), !dbg !58
; call core::hint::black_box
  %_10 = call align 1 ptr @_ZN4core4hint9black_box17h9f9a3aab786d67e0E(ptr align 1 %s1), !dbg !59
; call core::hint::black_box
  %_12 = call align 1 ptr @_ZN4core4hint9black_box17h9f9a3aab786d67e0E(ptr align 1 %s2), !dbg !60
; call core::hint::black_box
  %_14 = call align 1 ptr @_ZN4core4hint9black_box17h9f9a3aab786d67e0E(ptr align 1 %s3), !dbg !61
; call core::hint::black_box
  %_16 = call align 1 ptr @_ZN4core4hint9black_box17h9f9a3aab786d67e0E(ptr align 1 %s4), !dbg !62
; call core::hint::black_box
  %_18 = call align 1 ptr @_ZN4core4hint9black_box17h9f9a3aab786d67e0E(ptr align 1 %s5), !dbg !63
; call core::hint::black_box
  %_20 = call align 1 ptr @_ZN4core4hint9black_box17h9f9a3aab786d67e0E(ptr align 1 %s6), !dbg !64
; call core::hint::black_box
  %_22 = call align 1 ptr @_ZN4core4hint9black_box17h9f9a3aab786d67e0E(ptr align 1 %s7), !dbg !65
; call core::hint::black_box
  %_24 = call align 1 ptr @_ZN4core4hint9black_box17h9f9a3aab786d67e0E(ptr align 1 %s8), !dbg !66
; call core::hint::black_box
  %_26 = call align 1 ptr @_ZN4core4hint9black_box17hff24a8f6cdc261d0E(ptr align 1 %s9), !dbg !67
  ret void, !dbg !68
}

; playground::caller
define void @_ZN10playground6caller17h0397b5030166733dE() unnamed_addr !dbg !69 {
start:
  %t = alloca {}, align 1
  %s = alloca {}, align 1
  call void @llvm.dbg.declare(metadata ptr %s, metadata !73, metadata !DIExpression()), !dbg !77
  call void @llvm.dbg.declare(metadata ptr %t, metadata !75, metadata !DIExpression()), !dbg !78
; call playground::callee
  call void @_ZN10playground6callee17hf55947d3dfc887f4E(ptr align 1 %s, ptr align 1 %s, ptr align 1 %s, ptr align 1 %s, ptr align 1 %s, ptr align 1 %s, ptr align 1 %s, ptr align 1 %s, ptr align 1 %t), !dbg !79
  ret void, !dbg !80
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.dbg.cu = !{!4}

!0 = !{i32 7, !"PIC Level", i32 2}
!1 = !{i32 2, !"RtLibUseGOT", i32 1}
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !5, producer: "clang LLVM (rustc version 1.69.0-nightly (e1eaa2d5d 2023-02-06))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!5 = !DIFile(filename: "src/lib.rs/@/playground.71f4e8b5-cgu.0", directory: "/playground")
!6 = distinct !DISubprogram(name: "black_box<&()>", linkageName: "_ZN4core4hint9black_box17h9f9a3aab786d67e0E", scope: !8, file: !7, line: 294, type: !10, scopeLine: 294, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4, templateParams: !16, retainedNodes: !14)
!7 = !DIFile(filename: "/rustc/e1eaa2d5d4d1f5b7b89561a940718058d414e89c/library/core/src/hint.rs", directory: "", checksumkind: CSK_MD5, checksum: "2eba1ee5b9c26bf5eea6ed3dac7a7b79")
!8 = !DINamespace(name: "hint", scope: !9)
!9 = !DINamespace(name: "core", scope: null)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&()", baseType: !13, size: 64, align: 64, addressSpace: 0)
!13 = !DIBasicType(name: "()", encoding: DW_ATE_unsigned)
!14 = !{!15}
!15 = !DILocalVariable(name: "dummy", arg: 1, scope: !6, file: !7, line: 294, type: !12)
!16 = !{!17}
!17 = !DITemplateTypeParameter(name: "T", type: !12)
!18 = !DILocation(line: 294, column: 27, scope: !6)
!19 = !DILocation(line: 295, column: 5, scope: !6)
!20 = !{i32 382361}
!21 = !{}
!22 = !{i64 1}
!23 = !DILocation(line: 296, column: 2, scope: !6)
!24 = distinct !DISubprogram(name: "black_box<&mut ()>", linkageName: "_ZN4core4hint9black_box17hff24a8f6cdc261d0E", scope: !8, file: !7, line: 294, type: !25, scopeLine: 294, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4, templateParams: !30, retainedNodes: !28)
!25 = !DISubroutineType(types: !26)
!26 = !{!27, !27}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&mut ()", baseType: !13, size: 64, align: 64, addressSpace: 0)
!28 = !{!29}
!29 = !DILocalVariable(name: "dummy", arg: 1, scope: !24, file: !7, line: 294, type: !27)
!30 = !{!31}
!31 = !DITemplateTypeParameter(name: "T", type: !27)
!32 = !DILocation(line: 294, column: 27, scope: !24)
!33 = !DILocation(line: 295, column: 5, scope: !24)
!34 = !DILocation(line: 296, column: 2, scope: !24)
!35 = distinct !DISubprogram(name: "callee", linkageName: "_ZN10playground6callee17hf55947d3dfc887f4E", scope: !37, file: !36, line: 4, type: !38, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !4, templateParams: !21, retainedNodes: !40)
!36 = !DIFile(filename: "src/lib.rs", directory: "/playground", checksumkind: CSK_MD5, checksum: "bb1df4ba7c42e8987c349ab2cbe5f6b6")
!37 = !DINamespace(name: "playground", scope: null)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !12, !12, !12, !12, !12, !12, !12, !12, !27}
!40 = !{!41, !42, !43, !44, !45, !46, !47, !48, !49}
!41 = !DILocalVariable(name: "s1", arg: 1, scope: !35, file: !36, line: 5, type: !12)
!42 = !DILocalVariable(name: "s2", arg: 2, scope: !35, file: !36, line: 6, type: !12)
!43 = !DILocalVariable(name: "s3", arg: 3, scope: !35, file: !36, line: 7, type: !12)
!44 = !DILocalVariable(name: "s4", arg: 4, scope: !35, file: !36, line: 8, type: !12)
!45 = !DILocalVariable(name: "s5", arg: 5, scope: !35, file: !36, line: 9, type: !12)
!46 = !DILocalVariable(name: "s6", arg: 6, scope: !35, file: !36, line: 10, type: !12)
!47 = !DILocalVariable(name: "s7", arg: 7, scope: !35, file: !36, line: 11, type: !12)
!48 = !DILocalVariable(name: "s8", arg: 8, scope: !35, file: !36, line: 12, type: !12)
!49 = !DILocalVariable(name: "s9", arg: 9, scope: !35, file: !36, line: 13, type: !27)
!50 = !DILocation(line: 5, column: 5, scope: !35)
!51 = !DILocation(line: 6, column: 5, scope: !35)
!52 = !DILocation(line: 7, column: 5, scope: !35)
!53 = !DILocation(line: 8, column: 5, scope: !35)
!54 = !DILocation(line: 9, column: 5, scope: !35)
!55 = !DILocation(line: 10, column: 5, scope: !35)
!56 = !DILocation(line: 11, column: 5, scope: !35)
!57 = !DILocation(line: 12, column: 5, scope: !35)
!58 = !DILocation(line: 13, column: 5, scope: !35)
!59 = !DILocation(line: 15, column: 5, scope: !35)
!60 = !DILocation(line: 16, column: 5, scope: !35)
!61 = !DILocation(line: 17, column: 5, scope: !35)
!62 = !DILocation(line: 18, column: 5, scope: !35)
!63 = !DILocation(line: 19, column: 5, scope: !35)
!64 = !DILocation(line: 20, column: 5, scope: !35)
!65 = !DILocation(line: 21, column: 5, scope: !35)
!66 = !DILocation(line: 22, column: 5, scope: !35)
!67 = !DILocation(line: 23, column: 5, scope: !35)
!68 = !DILocation(line: 24, column: 2, scope: !35)
!69 = distinct !DISubprogram(name: "caller", linkageName: "_ZN10playground6caller17h0397b5030166733dE", scope: !37, file: !36, line: 26, type: !70, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, templateParams: !21, retainedNodes: !72)
!70 = !DISubroutineType(types: !71)
!71 = !{null}
!72 = !{!73, !75}
!73 = !DILocalVariable(name: "s", scope: !74, file: !36, line: 27, type: !13, align: 1)
!74 = distinct !DILexicalBlock(scope: !69, file: !36, line: 27, column: 5)
!75 = !DILocalVariable(name: "t", scope: !76, file: !36, line: 28, type: !13, align: 1)
!76 = distinct !DILexicalBlock(scope: !74, file: !36, line: 28, column: 5)
!77 = !DILocation(line: 27, column: 9, scope: !74)
!78 = !DILocation(line: 28, column: 9, scope: !76)
!79 = !DILocation(line: 29, column: 5, scope: !76)
!80 = !DILocation(line: 30, column: 2, scope: !81)
!81 = !DILexicalBlockFile(scope: !69, file: !36, discriminator: 0)

