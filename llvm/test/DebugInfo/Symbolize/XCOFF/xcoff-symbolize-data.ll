;; This file shows that llvm-symbolizer can symbolize data symbols
;; from the DWARF info on AIX. Note that AIX is big endian.

;; FIXME: use assembly rather than LLVM IR once integrated assembler supports
;; AIX assembly syntax.

; REQUIRES: powerpc-registered-target
; RUN: llc -filetype=obj -o %t -mtriple=powerpc-aix-ibm-xcoff < %s
; RUN: llvm-symbolizer --obj=%t 'DATA 0x60' 'DATA 0x61' 'DATA 0x64' 'DATA 0X68' \
; RUN:   'DATA 0x90' 'DATA 0x94' 'DATA 0X98' | \
; RUN:   FileCheck %s

;; Test an uninitialized global variable from offset 0.
; CHECK: bss_global
; CHECK-NEXT: 96 4
; CHECK-NEXT: /t.cpp:1
; CHECK-EMPTY:

;; Test an uninitialized global variable from offset 1.
; CHECK: bss_global
; CHECK-NEXT: 96 4
; CHECK-NEXT: /t.cpp:1
; CHECK-EMPTY:

;; Test an initialized global variable.
; CHECK: data_global
; CHECK-NEXT: 100 4
; CHECK-NEXT: /t.cpp:2
; CHECK-EMPTY:

;; Test a pointer type global variable.
; CHECK: str
; CHECK-NEXT: 104 4
; CHECK-NEXT: /t.cpp:4
; CHECK-EMPTY:

;; Test a function scope static variable.
;; FIXME: fix the wrong size 152
; CHECK: f()::function_global
; CHECK-NEXT: 144 152
; CHECK-NEXT: /t.cpp:8
; CHECK-EMPTY:

;; Test a global scope static variable that is used in current compilation unit.
;; FIXME: fix the wrong size 152
; CHECK: beta
; CHECK-NEXT: 148 152
; CHECK-NEXT: /t.cpp:13
; CHECK-EMPTY:

;; Test another global scope static variable that is used in current compilation unit.
;; FIXME: fix the wrong size 152
; CHECK: alpha
; CHECK-NEXT: 152 152
; CHECK-NEXT: /t.cpp:12
; CHECK-EMPTY:

;; The case is from `test/tools/llvm-symbolizer/data-location.yaml`, compiled with:
;; clang++ -g -gdwarf-3 -O3 t.cpp -nostdlib -target powerpc-aix-ibm-xcoff -S -emit-llvm

;;     cat t.cpp
;;     1	int bss_global;
;;     2	int data_global = 2;
;;     3
;;     4	const char* str =
;;     5	  "12345678";
;;     6
;;     7	int* f() {
;;     8	  static int function_global;
;;     9	  return &function_global;
;;    10	}
;;    11
;;    12	static int alpha;
;;    13	static int beta;
;;    14	int *f(bool b) { return beta ? &alpha : &beta; }
;;    15

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32"
target triple = "powerpc-ibm-aix-xcoff"

@bss_global = local_unnamed_addr global i32 0, align 4, !dbg !0
@data_global = local_unnamed_addr global i32 2, align 4, !dbg !5
@.str = private unnamed_addr constant [9 x i8] c"12345678\00", align 1, !dbg !8
@str = local_unnamed_addr global ptr @.str, align 4, !dbg !15
@_ZZ1fvE15function_global = internal global i32 0, align 4, !dbg !18
@_ZL4beta = internal global i32 0, align 4, !dbg !24
@_ZL5alpha = internal global i32 0, align 4, !dbg !26

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define noundef nonnull ptr @_Z1fv() local_unnamed_addr #0 !dbg !20 {
entry:
  ret ptr @_ZZ1fvE15function_global, !dbg !34
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none)
define noundef nonnull ptr @_Z1fb(i1 noundef zeroext %b) local_unnamed_addr #1 !dbg !35 {
entry:
  call void @llvm.dbg.value(metadata i1 %b, metadata !40, metadata !DIExpression(DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_stack_value)), !dbg !41
  %0 = load i32, ptr @_ZL4beta, align 4, !dbg !42, !tbaa !43
  %tobool.not = icmp eq i32 %0, 0, !dbg !42
  %cond = select i1 %tobool.not, ptr @_ZL4beta, ptr @_ZL5alpha, !dbg !42
  ret ptr %cond, !dbg !42
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr7" "target-features"="+altivec,+bpermd,+extdiv,+isa-v206-instructions,+vsx,-aix-small-local-exec-tls,-crbits,-crypto,-direct-move,-htm,-isa-v207-instructions,-isa-v30-instructions,-power8-vector,-power9-vector,-privileged,-quadword-atomics,-rop-protect,-spe" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr7" "target-features"="+altivec,+bpermd,+extdiv,+isa-v206-instructions,+vsx,-aix-small-local-exec-tls,-crbits,-crypto,-direct-move,-htm,-isa-v207-instructions,-isa-v30-instructions,-power8-vector,-power9-vector,-privileged,-quadword-atomics,-rop-protect,-spe" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!28, !29, !30, !31, !32}
!llvm.ident = !{!33}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "bss_global", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 18.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.cpp", directory: "/")
!4 = !{!0, !5, !8, !15, !18, !24, !26}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "data_global", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(scope: null, file: !3, line: 5, type: !10, isLocal: true, isDefinition: true)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 72, elements: !13)
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !12)
!12 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!13 = !{!14}
!14 = !DISubrange(count: 9)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "str", scope: !2, file: !3, line: 4, type: !17, isLocal: false, isDefinition: true)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 32)
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = distinct !DIGlobalVariable(name: "function_global", scope: !20, file: !3, line: 8, type: !7, isLocal: true, isDefinition: true)
!20 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !3, file: !3, line: 7, type: !21, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!21 = !DISubroutineType(types: !22)
!22 = !{!23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32)
!24 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression())
!25 = distinct !DIGlobalVariable(name: "beta", linkageName: "_ZL4beta", scope: !2, file: !3, line: 13, type: !7, isLocal: true, isDefinition: true)
!26 = !DIGlobalVariableExpression(var: !27, expr: !DIExpression())
!27 = distinct !DIGlobalVariable(name: "alpha", linkageName: "_ZL5alpha", scope: !2, file: !3, line: 12, type: !7, isLocal: true, isDefinition: true)
!28 = !{i32 7, !"Dwarf Version", i32 3}
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{i32 1, !"wchar_size", i32 2}
!31 = !{i32 8, !"PIC Level", i32 2}
!32 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!33 = !{!"clang version 18.0.0"}
!34 = !DILocation(line: 9, scope: !20)
!35 = distinct !DISubprogram(name: "f", linkageName: "_Z1fb", scope: !3, file: !3, line: 14, type: !36, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !39)
!36 = !DISubroutineType(types: !37)
!37 = !{!23, !38}
!38 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!39 = !{!40}
!40 = !DILocalVariable(name: "b", arg: 1, scope: !35, file: !3, line: 14, type: !38)
!41 = !DILocation(line: 0, scope: !35)
!42 = !DILocation(line: 14, scope: !35)
!43 = !{!44, !44, i64 0}
!44 = !{!"int", !45, i64 0}
!45 = !{!"omnipotent char", !46, i64 0}
!46 = !{!"Simple C++ TBAA"}
