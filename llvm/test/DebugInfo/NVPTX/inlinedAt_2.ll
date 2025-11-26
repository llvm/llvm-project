; RUN: llc < %s -mattr=+ptx72 | FileCheck %s
;
;; Test simple two-level inlining - verifies that inlined_at information is correctly
;; emitted when foo() calls bar() and kernel() calls foo().
;
; #include <stdio.h>
;
; __device__ int gg;
;
; __device__ void foo();
; __device__ void bar();
;
; __device__ void foo() {
;   if (gg > 7)
;     bar();
; }
;
; __device__ void bar() {
;   ++gg;
; }
;
; __global__ void kernel() {
;   foo();
; }
;
; CHECK: .loc [[FILENUM:[1-9]]] 18
; CHECK: .loc [[FILENUM]] 9 {{[0-9]*}}, function_name [[FOONAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 18
; CHECK: .loc [[FILENUM]] 10 {{[0-9]*}}, function_name [[FOONAME]], inlined_at [[FILENUM]] 18
; CHECK: .loc [[FILENUM]] 14 {{[0-9]*}}, function_name [[BARNAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 10
; CHECK: .section .debug_str
; CHECK: {
; CHECK: [[FOONAME]]:
; CHECK-NEXT: // {{.*}} _Z3foov
; CHECK: [[BARNAME]]:
; CHECK-NEXT: // {{.*}} _Z3barv
; CHECK: }

source_filename = "<unnamed>"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8-p6:32:32"
target triple = "nvptx64-nvidia-cuda"

@gg = internal addrspace(1) global i32 0, align 4
@llvm.used = appending global [2 x ptr] [ptr @_Z6kernelv, ptr addrspacecast (ptr addrspace(1) @gg to ptr)], section "llvm.metadata"

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define void @_Z6kernelv() #0 !dbg !5 {
entry:
  %tmp.i = load i32, ptr addrspace(1) @gg, align 4, !dbg !7
  %cmp.i = icmp sgt i32 %tmp.i, 7, !dbg !7
  br i1 %cmp.i, label %if.then.i, label %_Z3foov.exit, !dbg !7

if.then.i:                                        ; preds = %entry
  %inc.i.i = add nuw nsw i32 %tmp.i, 1, !dbg !12
  store i32 %inc.i.i, ptr addrspace(1) @gg, align 4, !dbg !12
  br label %_Z3foov.exit, !dbg !17

_Z3foov.exit:                                     ; preds = %entry, %if.then.i
  ret void, !dbg !18
}

attributes #0 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) "target-cpu"="sm_75" }

!llvm.dbg.cu = !{!0}
!nvvm.annotations = !{!3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
!1 = !DIFile(filename: "t2.cu", directory: "")
!2 = !{}
!3 = !{ptr @_Z6kernelv, !"kernel", i32 1}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "kernel", linkageName: "_Z6kernelv", scope: !1, file: !1, line: 17, type: !6, scopeLine: 17, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 9, column: 3, scope: !8, inlinedAt: !10)
!8 = distinct !DILexicalBlock(scope: !9, file: !1, line: 8, column: 29)
!9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 8, type: !6, scopeLine: 8, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = distinct !DILocation(line: 18, column: 3, scope: !11)
!11 = distinct !DILexicalBlock(scope: !5, file: !1, line: 17, column: 29)
!12 = !DILocation(line: 14, column: 3, scope: !13, inlinedAt: !15)
!13 = distinct !DILexicalBlock(scope: !14, file: !1, line: 13, column: 29)
!14 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 13, type: !6, scopeLine: 13, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = distinct !DILocation(line: 10, column: 5, scope: !16, inlinedAt: !10)
!16 = distinct !DILexicalBlock(scope: !8, file: !1, line: 9, column: 3)
!17 = !DILocation(line: 10, column: 5, scope: !16, inlinedAt: !10)
!18 = !DILocation(line: 19, column: 1, scope: !11)
