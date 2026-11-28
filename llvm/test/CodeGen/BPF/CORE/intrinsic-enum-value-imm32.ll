; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; RUN: llc -mcpu=v1 -filetype=asm -o - %t1 | FileCheck --check-prefix=CHECK-V1 %s
;
; Source:
;   enum E { N = 10 } __attribute__((preserve_access_index));
;   enum U { M = 0x80000000 } __attribute__((preserve_access_index));
;   unsigned f1(unsigned hash) { return hash % N; }
;   long f2(void) { return N; }
;   unsigned long f3(void) { return M; }
;   long f4(void) { return __builtin_preserve_enum_value(*(enum E *)N, 1); }
; Compiler flag to generate IR:
;   clang -target bpf -S -O2 -g -emit-llvm -Xclang -disable-llvm-passes t.c
;
; N has type int and M has type unsigned int, so clang gives the calls for
; them a range. Their values are loaded with a 64-bit and a 32-bit mov, which
; sign-extend and zero-extend the immediate, and the conversions to long and
; unsigned long go away. The explicit builtin call has no range, so its value
; is loaded with a ld_imm64.

target triple = "bpf"

@0 = private unnamed_addr constant [5 x i8] c"N:10\00", align 1
@1 = private unnamed_addr constant [5 x i8] c"N:10\00", align 1
@2 = private unnamed_addr constant [13 x i8] c"M:2147483648\00", align 1
@3 = private unnamed_addr constant [5 x i8] c"N:10\00", align 1

; Function Attrs: nounwind
define dso_local i32 @f1(i32 noundef %0) #0 !dbg !22 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !27
    #dbg_declare(ptr %2, !26, !DIExpression(), !28)
  %3 = load i32, ptr %2, align 4, !dbg !29, !tbaa !27
  %4 = call range(i64 -2147483648, 2147483648) i64 @llvm.bpf.preserve.enum.value(i32 0, ptr @0, i64 1), !dbg !30, !llvm.preserve.access.index !3
  %5 = trunc i64 %4 to i32, !dbg !30
  %6 = urem i32 %3, %5, !dbg !31
  ret i32 %6, !dbg !32
}

; Function Attrs: nounwind memory(none)
declare i64 @llvm.bpf.preserve.enum.value(i32, ptr, i64) #1

; Function Attrs: nounwind
define dso_local i64 @f2() #0 !dbg !33 {
  %1 = call range(i64 -2147483648, 2147483648) i64 @llvm.bpf.preserve.enum.value(i32 1, ptr @1, i64 1), !dbg !37, !llvm.preserve.access.index !3
  %2 = trunc i64 %1 to i32, !dbg !38
  %3 = sext i32 %2 to i64, !dbg !39
  ret i64 %3, !dbg !40
}

; Function Attrs: nounwind
define dso_local i64 @f3() #0 !dbg !41 {
  %1 = call range(i64 0, 4294967296) i64 @llvm.bpf.preserve.enum.value(i32 2, ptr @2, i64 1), !dbg !45, !llvm.preserve.access.index !7
  %2 = trunc i64 %1 to i32, !dbg !46
  %3 = zext i32 %2 to i64, !dbg !47
  ret i64 %3, !dbg !48
}

; Function Attrs: nounwind
define dso_local i64 @f4() #0 !dbg !49 {
  %1 = call i64 @llvm.bpf.preserve.enum.value(i32 3, ptr @3, i64 1), !dbg !50, !llvm.preserve.access.index !3
  ret i64 %1, !dbg !51
}

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14, !15}
!llvm.errno.tbaa = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !10, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/tmp/llt", checksumkind: CSK_MD5, checksum: "4ab60c69614cdefa88395f1d70852785")
!2 = !{!3, !7}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !1, line: 1, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "N", value: 10)
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "U", file: !1, line: 2, baseType: !4, size: 32, elements: !8)
!8 = !{!9}
!9 = !DIEnumerator(name: "M", value: 2147483648, isUnsigned: true)
!10 = !{!11, !4, !3, !7}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 2, !"debugger-tuning", !"gdb"}
!15 = !{i32 7, !"frame-pointer", i32 2}
!17 = !{!18, !19, i64 0}
!18 = !{!"__libc_errno", !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 3, type: !23, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !25, keyInstructions: true)
!23 = !DISubroutineType(types: !24)
!24 = !{!4, !4}
!25 = !{!26}
!26 = !DILocalVariable(name: "hash", arg: 1, scope: !22, file: !1, line: 3, type: !4)
!27 = !{!19, !19, i64 0}
!28 = !DILocation(line: 3, column: 22, scope: !22)
!29 = !DILocation(line: 3, column: 37, scope: !22)
!30 = !DILocation(line: 3, column: 44, scope: !22)
!31 = !DILocation(line: 3, column: 42, scope: !22, atomGroup: 1, atomRank: 2)
!32 = !DILocation(line: 3, column: 30, scope: !22, atomGroup: 1, atomRank: 1)
!33 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 4, type: !34, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!34 = !DISubroutineType(types: !35)
!35 = !{!36}
!36 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!37 = !DILocation(line: 4, column: 24, scope: !33, atomGroup: 1, atomRank: 4)
!38 = !DILocation(line: 4, column: 24, scope: !33, atomGroup: 1, atomRank: 3)
!39 = !DILocation(line: 4, column: 24, scope: !33, atomGroup: 1, atomRank: 2)
!40 = !DILocation(line: 4, column: 17, scope: !33, atomGroup: 1, atomRank: 1)
!41 = distinct !DISubprogram(name: "f3", scope: !1, file: !1, line: 5, type: !42, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!42 = !DISubroutineType(types: !43)
!43 = !{!44}
!44 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!45 = !DILocation(line: 5, column: 33, scope: !41, atomGroup: 1, atomRank: 4)
!46 = !DILocation(line: 5, column: 33, scope: !41, atomGroup: 1, atomRank: 3)
!47 = !DILocation(line: 5, column: 33, scope: !41, atomGroup: 1, atomRank: 2)
!48 = !DILocation(line: 5, column: 26, scope: !41, atomGroup: 1, atomRank: 1)
!49 = distinct !DISubprogram(name: "f4", scope: !1, file: !1, line: 6, type: !34, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!50 = !DILocation(line: 6, column: 24, scope: !49, atomGroup: 1, atomRank: 2)
!51 = !DILocation(line: 6, column: 17, scope: !49, atomGroup: 1, atomRank: 1)

; CHECK-LABEL: f1:
; CHECK:         w0 = w1
; CHECK:         r1 = 10
; CHECK-NOT:     ll
; CHECK:         w0 %= w1
; CHECK:         exit
; CHECK-LABEL: f2:
; CHECK:         r0 = 10
; CHECK-NOT:     r0 <<=
; CHECK:         exit
; CHECK-LABEL: f3:
; CHECK:         w0 = -2147483648
; CHECK-NOT:     r0 <<=
; CHECK:         exit
; CHECK-LABEL: f4:
; CHECK:         r0 = 10 ll
; CHECK:         exit

; CHECK-V1-LABEL: f2:
; CHECK-V1:         r0 = 10
; CHECK-V1-NOT:     r0 <<=
; CHECK-V1:         exit
; CHECK-V1-LABEL: f3:
; CHECK-V1:         w0 = -2147483648
; CHECK-V1-NOT:     r0 <<=
; CHECK-V1:         exit
; CHECK-V1-LABEL: f4:
; CHECK-V1:         r0 = 10 ll
; CHECK-V1:         exit

; All four are ENUM_VALUE relocations.
; CHECK:             .long   16                              # FieldReloc
; CHECK-NEXT:        .long   {{[0-9]+}}                      # Field reloc section string offset={{[0-9]+}}
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   {{[0-9]+}}
; CHECK-NEXT:        .long   11
