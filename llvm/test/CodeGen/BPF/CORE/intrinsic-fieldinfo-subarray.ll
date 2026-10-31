; RUN: opt -O2 %s | llc -filetype=obj -o - | \
; RUN:   llvm-objdump --no-addresses --no-show-raw-insn --no-print-imm-hex -dr - | \
; RUN:   FileCheck %s --implicit-check-not=CO-RE

; Source:
;   struct S { long x; int y; } __attribute__((preserve_access_index));
;   unsigned fi_sub(struct S (*p)[2][3]) {
;     return __builtin_preserve_field_info((*p)[1], 0);
;   }
;   unsigned fi_ptr_arr(struct S (*p)[3]) {
;     return __builtin_preserve_field_info(p[1], 0);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

source_filename = "test.c"
target triple = "bpf"

%struct.S = type { i64, i32 }

; Function Attrs: nounwind
define dso_local i32 @fi_sub(ptr noundef %0) #0 !dbg !22 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !32
    #dbg_declare(ptr %2, !31, !DIExpression(), !35)
  %3 = load ptr, ptr %2, align 8, !dbg !36, !tbaa !32
  %4 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x %struct.S]]) %3, i32 1, i32 1), !dbg !37, !llvm.preserve.access.index !27
  %5 = call i32 @llvm.bpf.preserve.field.info.p0(ptr %4, i64 0), !dbg !38
  ret i32 %5, !dbg !39
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #1

; Function Attrs: nounwind memory(none)
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64 immarg) #2

; Function Attrs: nounwind
define dso_local i32 @fi_ptr_arr(ptr noundef %0) #0 !dbg !40 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8, !tbaa !32
    #dbg_declare(ptr %2, !44, !DIExpression(), !45)
  %3 = load ptr, ptr %2, align 8, !dbg !46, !tbaa !32
  %4 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %struct.S]) %3, i32 0, i32 1), !dbg !46, !llvm.preserve.access.index !3
  %5 = call i32 @llvm.bpf.preserve.field.info.p0(ptr %4, i64 0), !dbg !47
  ret i32 %5, !dbg !48
}

; CHECK-LABEL: <fi_sub>:
; CHECK:       r0 = 48
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] struct S::[3] (3)
; CHECK-NEXT:  exit
; CHECK-LABEL: <fi_ptr_arr>:
; CHECK:       r0 = 48
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] struct S::[3] (3)
; CHECK-NEXT:  exit

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}
!llvm.errno.tbaa = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!4 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 384, elements: !11)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 1, size: 128, elements: !6)
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !5, file: !1, line: 1, baseType: !8, size: 64)
!8 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !5, file: !1, line: 1, baseType: !10, size: 32, offset: 64)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 3)
!13 = !{i32 7, !"Dwarf Version", i32 5}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !{!"clang version 24.0.0git"}
!17 = !{!18, !19, i64 0}
!18 = !{!"__libc_errno", !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = distinct !DISubprogram(name: "fi_sub", scope: !1, file: !1, line: 2, type: !23, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !30, keyInstructions: true)
!23 = !DISubroutineType(types: !24)
!24 = !{!25, !26}
!25 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 64)
!27 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 768, elements: !28)
!28 = !{!29, !12}
!29 = !DISubrange(count: 2)
!30 = !{!31}
!31 = !DILocalVariable(name: "p", arg: 1, scope: !22, file: !1, line: 2, type: !26)
!32 = !{!33, !33, i64 0}
!33 = !{!"p1 _ZTS1S", !34, i64 0}
!34 = !{!"any pointer", !20, i64 0}
!35 = !DILocation(line: 2, column: 28, scope: !22)
!36 = !DILocation(line: 3, column: 42, scope: !22)
!37 = !DILocation(line: 3, column: 40, scope: !22)
!38 = !DILocation(line: 3, column: 10, scope: !22, atomGroup: 1, atomRank: 2)
!39 = !DILocation(line: 3, column: 3, scope: !22, atomGroup: 1, atomRank: 1)
!40 = distinct !DISubprogram(name: "fi_ptr_arr", scope: !1, file: !1, line: 5, type: !41, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !43, keyInstructions: true)
!41 = !DISubroutineType(types: !42)
!42 = !{!25, !3}
!43 = !{!44}
!44 = !DILocalVariable(name: "p", arg: 1, scope: !40, file: !1, line: 5, type: !3)
!45 = !DILocation(line: 5, column: 32, scope: !40)
!46 = !DILocation(line: 6, column: 40, scope: !40)
!47 = !DILocation(line: 6, column: 10, scope: !40, atomGroup: 1, atomRank: 2)
!48 = !DILocation(line: 6, column: 3, scope: !40, atomGroup: 1, atomRank: 1)
