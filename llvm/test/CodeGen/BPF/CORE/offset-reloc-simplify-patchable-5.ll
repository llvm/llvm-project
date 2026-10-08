; RUN: llc -mtriple=bpf -filetype=obj -mcpu=v3 -disable-bpf-core-optimization %s -o %t
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t
; RUN: %python %p/../BTF/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
; RUN: llvm-objdump --no-print-imm-hex -dr --no-show-raw-insn %t | FileCheck --check-prefix=CHECK-DUMP %s

; Source:
;   struct t {
;     char c[40];
;     int ub;
;   } __attribute__((preserve_access_index));
;   void foo(volatile struct t *t) {
;     t->ub = 1;
;   }
;   int bar(volatile struct t *t) {
;     return t->ub;
;   }
; Using the following command:
;   clang -g -O2 -mcpu=v3 -S -emit-llvm --target=bpf t.c -o t.ll

@"llvm.t:0:40$0:1" = external global i64, !llvm.preserve.access.index !0 #0

; Function Attrs: nofree nounwind memory(readwrite, target_mem0: none, target_mem1: none)
define dso_local void @foo(ptr noundef %0) local_unnamed_addr #1 !dbg !20 {
    #dbg_value(ptr %0, !26, !DIExpression(), !27)
  %2 = load i64, ptr @"llvm.t:0:40$0:1", align 8
  %3 = getelementptr i8, ptr %0, i64 %2
  %4 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %3)
  store volatile i32 1, ptr %4, align 4, !dbg !28, !tbaa !29
  ret void, !dbg !31
}

; Function Attrs: nofree nounwind memory(readwrite, target_mem0: none, target_mem1: none)
define dso_local i32 @bar(ptr noundef %0) local_unnamed_addr #1 !dbg !32 {
    #dbg_value(ptr %0, !36, !DIExpression(), !37)
  %2 = load i64, ptr @"llvm.t:0:40$0:1", align 8
  %3 = getelementptr i8, ptr %0, i64 %2
  %4 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %3)
  %5 = load volatile i32, ptr %4, align 4, !dbg !38, !tbaa !29
  ret i32 %5, !dbg !39
}

; CHECK-BTF:      [1] PTR '(anon)' type_id=2
; CHECK-BTF-NEXT: [2] VOLATILE '(anon)' type_id=3
; CHECK-BTF-NEXT: [3] STRUCT 't' size=44 vlen=2
; CHECK-BTF-NEXT:         'c' type_id=5 bits_offset=0
; CHECK-BTF-NEXT:         'ub' type_id=7 bits_offset=320
; CHECK-BTF-NEXT: [4] INT 'char' size=1 bits_offset=0 nr_bits=8 encoding=SIGNED
; CHECK-BTF-NEXT: [5] ARRAY '(anon)' type_id=4 index_type_id=6 nr_elems=40
; CHECK-BTF-NEXT: [6] INT '__ARRAY_SIZE_TYPE__' size=4 bits_offset=0 nr_bits=32 encoding=(none)
; CHECK-BTF-NEXT: [7] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF-NEXT: [8] FUNC_PROTO '(anon)' ret_type_id=0 vlen=1
; CHECK-BTF-NEXT:         't' type_id=1
; CHECK-BTF-NEXT: [9] FUNC 'foo' type_id=8 linkage=global
; CHECK-BTF-NEXT: [10] FUNC_PROTO '(anon)' ret_type_id=7 vlen=1
; CHECK-BTF-NEXT:          't' type_id=1
; CHECK-BTF-NEXT: [11] FUNC 'bar' type_id=10 linkage=global

; CHECK-DUMP:      <foo>:
; CHECK-DUMP-NEXT: 0:       r2 = 40
; CHECK-DUMP-NEXT:          0000000000000000:  CO-RE <byte_off> [3] struct t::ub (0:1)
; CHECK-DUMP-NEXT: 1:       r1 += r2
; CHECK-DUMP-NEXT: 2:       w2 = 1
; CHECK-DUMP-NEXT: 3:       *(u32 *)(r1 + 0) = w2
; CHECK-DUMP-NEXT: 4:       exit
; CHECK-DUMP:      <bar>:
; CHECK-DUMP-NEXT: 5:       r2 = 40
; CHECK-DUMP-NEXT:          0000000000000028:  CO-RE <byte_off> [3] struct t::ub (0:1)
; CHECK-DUMP-NEXT: 6:       r1 += r2
; CHECK-DUMP-NEXT: 7:       w0 = *(u32 *)(r1 + 0)
; CHECK-DUMP-NEXT: 8:       exit

; Function Attrs: nofree nosync nounwind memory(none)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr) #2

attributes #0 = { "btf_ama" }
attributes #1 = { nofree nounwind memory(readwrite, target_mem0: none, target_mem1: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="v3" }
attributes #2 = { nofree nosync nounwind memory(none) }

!llvm.dbg.cu = !{!10}
!llvm.module.flags = !{!11, !12, !13, !14}
!llvm.ident = !{!15}
!llvm.errno.tbaa = !{!16}

!0 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 352, elements: !2)
!1 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/tmp", checksumkind: CSK_MD5, checksum: "c0b1e83c4a0097ba80a1a5b0da78cb75")
!2 = !{!3, !8}
!3 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !0, file: !1, line: 2, baseType: !4, size: 320)
!4 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 320, elements: !6)
!5 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!6 = !{!7}
!7 = !DISubrange(count: 40)
!8 = !DIDerivedType(tag: DW_TAG_member, name: "ub", scope: !0, file: !1, line: 3, baseType: !9, size: 32, offset: 320)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 23.0.0git (https://github.com/llvm/llvm-project.git 11727c11f833873af97d3969483f488e5251f35d)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 7, !"frame-pointer", i32 2}
!14 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!15 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git 11727c11f833873af97d3969483f488e5251f35d)"}
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C/C++ TBAA"}
!20 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !21, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !10, retainedNodes: !25, keyInstructions: true)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
!24 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !0)
!25 = !{!26}
!26 = !DILocalVariable(name: "t", arg: 1, scope: !20, file: !1, line: 5, type: !23)
!27 = !DILocation(line: 0, scope: !20)
!28 = !DILocation(line: 6, column: 9, scope: !20, atomGroup: 1, atomRank: 1)
!29 = !{!30, !17, i64 40}
!30 = !{!"t", !18, i64 0, !17, i64 40}
!31 = !DILocation(line: 7, column: 1, scope: !20, atomGroup: 2, atomRank: 1)
!32 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 8, type: !33, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !10, retainedNodes: !35, keyInstructions: true)
!33 = !DISubroutineType(types: !34)
!34 = !{!9, !23}
!35 = !{!36}
!36 = !DILocalVariable(name: "t", arg: 1, scope: !32, file: !1, line: 8, type: !23)
!37 = !DILocation(line: 0, scope: !32)
!38 = !DILocation(line: 9, column: 13, scope: !32, atomGroup: 1, atomRank: 2)
!39 = !DILocation(line: 9, column: 3, scope: !32, atomGroup: 1, atomRank: 1)
