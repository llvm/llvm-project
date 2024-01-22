; RUN: llc --mtriple="aarch64-" -O0 -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=AARCH
; RUN: llc --mtriple="aarch64-" -O0 -fast-isel=false -global-isel=false -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=AARCH
; RUN: llc --mtriple="aarch64-" -O0 -fast-isel -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=AARCH

; RUN: llc --mtriple="aarch64-" -O0 -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - --try-experimental-debuginfo-iterators | FileCheck %s --check-prefix=AARCH
; RUN: llc --mtriple="aarch64-" -O0 -fast-isel=false -global-isel=false -stop-after=finalize-isel %s -o - --try-experimental-debuginfo-iterators | FileCheck %s --check-prefix=AARCH
; RUN: llc --mtriple="aarch64-" -O0 -fast-isel -stop-after=finalize-isel %s -o - --try-experimental-debuginfo-iterators | FileCheck %s --check-prefix=AARCH

; AARCH-NOT:  DBG_VALUE
; AARCH:      DBG_VALUE $x22, $noreg, !{{.*}}, !DIExpression(DW_OP_LLVM_entry_value, 1)
; AARCH-NEXT: DBG_VALUE $x22, $noreg, !{{.*}}, !DIExpression(DW_OP_LLVM_entry_value, 1)
; AARCH-NOT:  DBG_VALUE

define void @foo(ptr %unused_arg, ptr swiftasync %async_arg) !dbg !6 {
  call void @llvm.dbg.value(metadata ptr %async_arg, metadata !12, metadata !DIExpression(DW_OP_LLVM_entry_value, 1)), !dbg !14
  call void @llvm.dbg.value(metadata ptr %async_arg, metadata !12, metadata !DIExpression(DW_OP_LLVM_entry_value, 1)), !dbg !14
  call void @consume(ptr %async_arg)
  ret void, !dbg !15
}

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @consume(ptr %ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "x.c", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9, !9, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !DILocalVariable(name: "a", scope: !6, file: !1, line: 1, type: !9)
!14 = !DILocation(line: 1, column: 29, scope: !6)
!15 = !DILocation(line: 1, column: 37, scope: !6)
