; RUN: opt -O2 -mtriple=bpf-pc-linux -S %s -o %t1
; RUN: llc -mcpu=v3 -filetype=obj %t1 -o %t2
; RUN: llvm-objcopy --dump-section='.BTF'=%t3 %t2
; RUN: %python %p/print_btf.py %t3 | FileCheck -check-prefixes=CHECK-BTF %s
; RUN: llvm-objdump --no-print-imm-hex -dr --no-show-raw-insn %t2 | FileCheck --check-prefix=CHECK-DUMP %s

; Source code:
;   int prog1(void) {
;     return __builtin_btf_type_id(*((char *) 0), 1);
;   }
;   int prog2(void) {
;     return __builtin_btf_type_id(*((typeof(const char *) *) 0), 1);
;   }
;   int prog3(void) {
;     return __builtin_btf_type_id(*((typeof(void *) *) 0), 1);
;   }
;   extern int do_smth(int);
;   int prog4(void) {
;     return __builtin_btf_type_id(*(typeof(do_smth) *)do_smth, 1);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

define dso_local i32 @prog1() #0 !dbg !13 {
  %1 = call i64 @llvm.bpf.btf.type.id(i32 0, i64 1), !dbg !17, !llvm.preserve.access.index !3
  %2 = trunc i64 %1 to i32, !dbg !18
  ret i32 %2, !dbg !19
}

declare i64 @llvm.bpf.btf.type.id(i32, i64) #1

define dso_local i32 @prog2() #0 !dbg !20 {
  %1 = call i64 @llvm.bpf.btf.type.id(i32 1, i64 1), !dbg !21, !llvm.preserve.access.index !22
  %2 = trunc i64 %1 to i32, !dbg !24
  ret i32 %2, !dbg !25
}

define dso_local i32 @prog3() #0 !dbg !26 {
  %1 = call i64 @llvm.bpf.btf.type.id(i32 2, i64 1), !dbg !27, !llvm.preserve.access.index !28
  %2 = trunc i64 %1 to i32, !dbg !29
  ret i32 %2, !dbg !30
}

define dso_local i32 @prog4() #0 !dbg !31 {
  %1 = call i64 @llvm.bpf.btf.type.id(i32 3, i64 1), !dbg !32, !llvm.preserve.access.index !33
  %2 = trunc i64 %1 to i32, !dbg !35
  ret i32 %2, !dbg !36
}

; CHECK-BTF:      [1] FUNC_PROTO '(anon)' ret_type_id=2 vlen=0
; CHECK_BTF-NEXT: [2] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK_BTF-NEXT: [3] FUNC 'prog1' type_id=1 linkage=global
; CHECK_BTF-NEXT: [4] INT 'char' size=1 bits_offset=0 nr_bits=8 encoding=SIGNED
; CHECK_BTF-NEXT: [5] FUNC_PROTO '(anon)' ret_type_id=2 vlen=0
; CHECK_BTF-NEXT: [6] FUNC 'prog2' type_id=5 linkage=global
; CHECK_BTF-NEXT: [7] PTR '(anon)' type_id=8
; CHECK_BTF-NEXT: [8] CONST '(anon)' type_id=4
; CHECK_BTF-NEXT: [9] FUNC_PROTO '(anon)' ret_type_id=2 vlen=0
; CHECK_BTF-NEXT: [10] FUNC 'prog3' type_id=9 linkage=global
; CHECK_BTF-NEXT: [11] PTR '(anon)' type_id=0
; CHECK_BTF-NEXT: [12] FUNC_PROTO '(anon)' ret_type_id=2 vlen=0
; CHECK_BTF-NEXT: [13] FUNC 'prog4' type_id=12 linkage=global
; CHECK_BTF-NEXT: [14] FUNC_PROTO '(anon)' ret_type_id=2 vlen=1
; CHECK_BTF-NEXT:     '(anon)' type_id=2

; CHECK-DUMP:      prog1
; CHECK-DUMP-NEXT: 0:       r0 = 4 ll
; CHECK-DUMP-NEXT:          0000000000000000:  CO-RE <target_type_id> [4] char
; CHECK-DUMP:      prog2
; CHECK-DUMP-NEXT: 3:       r0 = 7 ll
; CHECK-DUMP-NEXT:          0000000000000018:  CO-RE <target_type_id> [7] <anon 7>
; CHECK-DUMP:      prog3
; CHECK-DUMP-NEXT: 6:       r0 = 11 ll
; CHECK-DUMP-NEXT:          0000000000000030:  CO-RE <target_type_id> [11] <anon 11>
; CHECK-DUMP:      prog4
; CHECK-DUMP-NEXT: 9:       r0 = 14 ll
; CHECK-DUMP-NEXT:          0000000000000048:  CO-RE <target_type_id> [14] <anon 14>

declare !dbg !37 dso_local i32 @do_smth(i32 noundef) #2

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind memory(none) }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4, !5, !6, !7}
!llvm.ident = !{!8}
!llvm.errno.tbaa = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 23.0.0git (git@github.com:yonghong-song/llvm-project.git 4be42b923d6f5074112d22da4f73f2eb9cef5adf)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/tmp/tmp3", checksumkind: CSK_MD5, checksum: "5eb5e0736b75af551ca1934d9ff741db")
!2 = !{!3}
!3 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!"clang version 23.0.0git (git@github.com:yonghong-song/llvm-project.git 4be42b923d6f5074112d22da4f73f2eb9cef5adf)"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}
!13 = distinct !DISubprogram(name: "prog1", scope: !1, file: !1, line: 1, type: !14, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!14 = !DISubroutineType(types: !15)
!15 = !{!16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DILocation(line: 2, column: 10, scope: !13, atomGroup: 1, atomRank: 3)
!18 = !DILocation(line: 2, column: 10, scope: !13, atomGroup: 1, atomRank: 2)
!19 = !DILocation(line: 2, column: 3, scope: !13, atomGroup: 1, atomRank: 1)
!20 = distinct !DISubprogram(name: "prog2", scope: !1, file: !1, line: 4, type: !14, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!21 = !DILocation(line: 5, column: 10, scope: !20, atomGroup: 1, atomRank: 3)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !3)
!24 = !DILocation(line: 5, column: 10, scope: !20, atomGroup: 1, atomRank: 2)
!25 = !DILocation(line: 5, column: 3, scope: !20, atomGroup: 1, atomRank: 1)
!26 = distinct !DISubprogram(name: "prog3", scope: !1, file: !1, line: 7, type: !14, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!27 = !DILocation(line: 8, column: 10, scope: !26, atomGroup: 1, atomRank: 3)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!29 = !DILocation(line: 8, column: 10, scope: !26, atomGroup: 1, atomRank: 2)
!30 = !DILocation(line: 8, column: 3, scope: !26, atomGroup: 1, atomRank: 1)
!31 = distinct !DISubprogram(name: "prog4", scope: !1, file: !1, line: 11, type: !14, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!32 = !DILocation(line: 12, column: 10, scope: !31, atomGroup: 1, atomRank: 3)
!33 = !DISubroutineType(types: !34)
!34 = !{!16, !16}
!35 = !DILocation(line: 12, column: 10, scope: !31, atomGroup: 1, atomRank: 2)
!36 = !DILocation(line: 12, column: 3, scope: !31, atomGroup: 1, atomRank: 1)
!37 = !DISubprogram(name: "do_smth", scope: !1, file: !1, line: 10, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !38)
!38 = !{!39}
!39 = !DILocalVariable(arg: 1, scope: !37, file: !1, line: 10, type: !16)
