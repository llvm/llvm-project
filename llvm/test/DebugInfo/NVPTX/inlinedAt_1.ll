; RUN: llc < %s -mattr=+ptx72 | FileCheck %s
;
;; Test mutual recursion with deep inlining - verifies that inlined_at information
;; is correctly emitted for multiple levels of inlining when foo() and bar() call each other.
;
; #include <stdio.h>
;
; __device__ int gg;
;
; __device__ void foo();
; __device__ void bar();
; extern __device__ void calculate();
; __device__ void foo() {
;   if (gg > 7)
;     bar();
;   calculate();
; }
;
; __device__ void bar() {
;   if (gg > 17)
;     foo();
;   calculate();
; }
;
; __global__ void kernel() {
;   foo();
; }
;
; CHECK: .loc [[FILENUM:[1-9]]] 21
; CHECK: .loc [[FILENUM]] 9 {{[0-9]*}}, function_name [[FOONAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 21
; CHECK: .loc [[FILENUM]] 16 {{[0-9]*}}, function_name [[BARNAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 10
; CHECK: .loc [[FILENUM]] 10 {{[0-9]*}}, function_name [[FOONAME]], inlined_at [[FILENUM]] 16
; CHECK: .section .debug_str
; CHECK: {
; CHECK: [[FOONAME]]:
; CHECK-NEXT: // {{.*}} _Z3foov
; CHECK: [[BARNAME]]:
; CHECK-NEXT: // {{.*}} _Z3barv
; CHECK: }
source_filename = "<unnamed>"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@gg = internal addrspace(1) global i32 0, align 4
@llvm.used = appending global [2 x ptr] [ptr @_Z6kernelv, ptr addrspacecast (ptr addrspace(1) @gg to ptr)], section "llvm.metadata"

define internal fastcc void @_Z3foov() unnamed_addr  !dbg !4 {
entry:
  %tmp = load i32, ptr addrspace(1) @gg, align 4, !dbg !6
  %cmp = icmp sgt i32 %tmp, 7, !dbg !6
  br i1 %cmp, label %if.then, label %if.end, !dbg !6

if.then:                                          ; preds = %entry
  tail call fastcc void @_Z3barv(), !dbg !8
  br label %if.end, !dbg !8

if.end:                                           ; preds = %if.then, %entry
  tail call void @_Z9calculatev(), !dbg !10
  ret void, !dbg !11
}

define internal fastcc void @_Z3barv() unnamed_addr  !dbg !12 {
entry:
  %tmp = load i32, ptr addrspace(1) @gg, align 4, !dbg !13
  %cmp = icmp sgt i32 %tmp, 17, !dbg !13
  br i1 %cmp, label %if.then, label %if.end, !dbg !13

if.then:                                          ; preds = %entry
  tail call fastcc void @_Z3foov(), !dbg !15
  br label %if.end, !dbg !15

if.end:                                           ; preds = %if.then, %entry
  tail call void @_Z9calculatev(), !dbg !17
  ret void, !dbg !18
}

declare void @_Z9calculatev() local_unnamed_addr

define void @_Z6kernelv() !dbg !19 {
entry:
  %tmp.i = load i32, ptr addrspace(1) @gg, align 4, !dbg !20
  %cmp.i = icmp sgt i32 %tmp.i, 7, !dbg !20
  br i1 %cmp.i, label %if.then.i, label %_Z3foov.exit, !dbg !20

if.then.i:                                        ; preds = %entry
  %cmp.i2 = icmp sgt i32 %tmp.i, 17, !dbg !23
  br i1 %cmp.i2, label %if.then.i10, label %_Z3barv.exit, !dbg !23

if.then.i10:                                      ; preds = %if.then.i
  tail call fastcc void @_Z3foov(), !dbg !25
  tail call void @_Z9calculatev(), !dbg !28
  tail call void @_Z9calculatev(), !dbg !29
  br label %_Z3barv.exit, !dbg !30

_Z3barv.exit:                                     ; preds = %if.then.i, %if.then.i10
  tail call void @_Z9calculatev(), !dbg !31
  br label %_Z3foov.exit, !dbg !32

_Z3foov.exit:                                     ; preds = %entry, %_Z3barv.exit
  tail call void @_Z9calculatev(), !dbg !33
  ret void, !dbg !34
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
!1 = !DIFile(filename: "t1.cu", directory: "")
!2 = !{}
!3 = !{i32 1, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 8, type: !5, scopeLine: 8, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !2)
!6 = !DILocation(line: 9, column: 3, scope: !7)
!7 = distinct !DILexicalBlock(scope: !4, file: !1, line: 8, column: 29)
!8 = !DILocation(line: 10, column: 5, scope: !9)
!9 = distinct !DILexicalBlock(scope: !7, file: !1, line: 9, column: 3)
!10 = !DILocation(line: 11, column: 3, scope: !7)
!11 = !DILocation(line: 12, column: 1, scope: !7)
!12 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 14, type: !5, scopeLine: 14, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 15, column: 3, scope: !14)
!14 = distinct !DILexicalBlock(scope: !12, file: !1, line: 14, column: 29)
!15 = !DILocation(line: 16, column: 5, scope: !16)
!16 = distinct !DILexicalBlock(scope: !14, file: !1, line: 15, column: 3)
!17 = !DILocation(line: 17, column: 3, scope: !14)
!18 = !DILocation(line: 18, column: 1, scope: !14)
!19 = distinct !DISubprogram(name: "kernel", linkageName: "_Z6kernelv", scope: !1, file: !1, line: 20, type: !5, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!20 = !DILocation(line: 9, column: 3, scope: !7, inlinedAt: !21)
!21 = distinct !DILocation(line: 21, column: 3, scope: !22)
!22 = distinct !DILexicalBlock(scope: !19, file: !1, line: 20, column: 29)
!23 = !DILocation(line: 15, column: 3, scope: !14, inlinedAt: !24)
!24 = distinct !DILocation(line: 10, column: 5, scope: !9, inlinedAt: !21)
!25 = !DILocation(line: 16, column: 5, scope: !16, inlinedAt: !26)
!26 = distinct !DILocation(line: 10, column: 5, scope: !9, inlinedAt: !27)
!27 = distinct !DILocation(line: 16, column: 5, scope: !16, inlinedAt: !24)
!28 = !DILocation(line: 17, column: 3, scope: !14, inlinedAt: !26)
!29 = !DILocation(line: 11, column: 3, scope: !7, inlinedAt: !27)
!30 = !DILocation(line: 16, column: 5, scope: !16, inlinedAt: !24)
!31 = !DILocation(line: 17, column: 3, scope: !14, inlinedAt: !24)
!32 = !DILocation(line: 10, column: 5, scope: !9, inlinedAt: !21)
!33 = !DILocation(line: 11, column: 3, scope: !7, inlinedAt: !21)
!34 = !DILocation(line: 22, column: 1, scope: !22)
