; RUN: llc < %s -mattr=+ptx72 | FileCheck %s --check-prefix=DEFAULT
; RUN: llc < %s -mattr=+ptx70 | FileCheck %s --check-prefix=PTXVERSION
; RUN: llc < %s -mattr=+ptx72 --line-info-inlined-at=false | FileCheck %s --check-prefix=NOFLAG
;
;; Test command-line flags to control inlined_at emission - verifies that PTX 7.2+ emits
;; function_name and inlined_at by default, while PTX 7.0 or --line-info-inlined-at=false disables it.
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
;     calculate();
; }
;
; __device__ void bar() {
;   if (gg > 17)
;     foo();
;     calculate();
; }
;
; __global__ void kernel() {
;   foo();
; }
;
; DEFAULT: .loc [[FILENUM:[1-9]]] 10
; DEFAULT: .loc [[FILENUM]] 15 {{[0-9]*}}, function_name [[BARNAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 10
; DEFAULT: .loc [[FILENUM]] 16 {{[0-9]*}}, function_name [[BARNAME]], inlined_at [[FILENUM]] 10
; DEFAULT: .loc [[FILENUM]] 17 {{[0-9]*}}, function_name [[BARNAME]], inlined_at [[FILENUM]] 10
; DEFAULT: .section .debug_str
; DEFAULT: {
; DEFAULT: [[BARNAME]]:
; DEFAULT-NEXT: // {{.*}} _Z3barv
; DEFAULT: }

; NOFLAG-NOT: function_name
; NOFLAG-NOT: inlined_at {{[1-9]}}
; NOFLAG-NOT: .section .debug_str

; PTXVERSION-NOT: function_name
; PTXVERSION-NOT: inlined_at {{[1-9]}}
; PTXVERSION-NOT: .section .debug_str

source_filename = "<unnamed>"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8-p6:32:32"
target triple = "nvptx64-nvidia-cuda"

@gg = internal addrspace(1) global i32 0, align 4
@llvm.used = appending global [2 x ptr] [ptr @_Z6kernelv, ptr addrspacecast (ptr addrspace(1) @gg to ptr)], section "llvm.metadata"

define internal fastcc void @_Z3foov() unnamed_addr !dbg !4 {
entry:
  %tmp = load i32, ptr addrspace(1) @gg, align 4, !dbg !6
  %cmp = icmp sgt i32 %tmp, 7, !dbg !6
  br i1 %cmp, label %if.then, label %if.end, !dbg !6

if.then:                                          ; preds = %entry
  %cmp.i = icmp samesign ugt i32 %tmp, 17, !dbg !8
  br i1 %cmp.i, label %if.then.i, label %_Z3barv.exit, !dbg !8

if.then.i:                                        ; preds = %if.then
  tail call fastcc void @_Z3foov(), !dbg !13
  br label %_Z3barv.exit, !dbg !13

_Z3barv.exit:                                     ; preds = %if.then, %if.then.i
  tail call void @_Z9calculatev(), !dbg !15
  br label %if.end, !dbg !16

if.end:                                           ; preds = %_Z3barv.exit, %entry
  tail call void @_Z9calculatev(), !dbg !17
  ret void, !dbg !18
}

declare void @_Z9calculatev() local_unnamed_addr

define void @_Z6kernelv() !dbg !19 {
entry:
  tail call fastcc void @_Z3foov(), !dbg !20
  ret void, !dbg !22
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
!1 = !DIFile(filename: "t7.cu", directory: "")
!2 = !{}
!3 = !{i32 1, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 8, type: !5, scopeLine: 8, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !2)
!6 = !DILocation(line: 9, column: 3, scope: !7)
!7 = distinct !DILexicalBlock(scope: !4, file: !1, line: 8, column: 29)
!8 = !DILocation(line: 15, column: 3, scope: !9, inlinedAt: !11)
!9 = distinct !DILexicalBlock(scope: !10, file: !1, line: 14, column: 29)
!10 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 14, type: !5, scopeLine: 14, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = distinct !DILocation(line: 10, column: 5, scope: !12)
!12 = distinct !DILexicalBlock(scope: !7, file: !1, line: 9, column: 3)
!13 = !DILocation(line: 16, column: 5, scope: !14, inlinedAt: !11)
!14 = distinct !DILexicalBlock(scope: !9, file: !1, line: 15, column: 3)
!15 = !DILocation(line: 17, column: 3, scope: !9, inlinedAt: !11)
!16 = !DILocation(line: 10, column: 5, scope: !12)
!17 = !DILocation(line: 11, column: 3, scope: !7)
!18 = !DILocation(line: 12, column: 1, scope: !7)
!19 = distinct !DISubprogram(name: "kernel", linkageName: "_Z6kernelv", scope: !1, file: !1, line: 20, type: !5, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!20 = !DILocation(line: 21, column: 3, scope: !21)
!21 = distinct !DILexicalBlock(scope: !19, file: !1, line: 20, column: 29)
!22 = !DILocation(line: 22, column: 1, scope: !21)
