; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 | FileCheck %s
; RUN: %if ptxas-sm_90 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 | %ptxas-verify -arch=sm_90 %}

;; Test that .loc directives inside inline asm preserve inlined_at context.
;; When inline asm appears inside an inlined function, the .loc directives
;; injected for inline asm instructions must include function_name and
;; inlined_at attributes to avoid stripping the inlining context.
;;
;; Simplified from:
;;   __device__ int local_func(int p) {
;;     int dummy;
;;     asm volatile("ld.local.u32 %0, [0];" : "+r"(dummy));
;;     return 2 * p;
;;   }
;;   __global__ void kernel() {
;;     int r = local_func(threadIdx.x);
;;   }

target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8-p6:32:32"
target triple = "nvptx64-nvidia-cuda"

; The inline asm instruction has a DebugLoc at line 4 (in local_func),
; inlined at line 10 (in kernel). The .loc for the ld.local.u32 inside
; the inline asm must carry function_name and inlined_at.

; CHECK:       .loc [[FILEID:[0-9]+]] 4 5, function_name [[FUNCNAME:\$L__info_string[0-9]+]], inlined_at [[FILEID]] 10 3
; CHECK-NEXT:  // begin inline asm
; CHECK-NEXT:  .loc [[FILEID]] 4 5, function_name [[FUNCNAME]], inlined_at [[FILEID]] 10 3
; CHECK-NEXT:  ld.local.u32

define void @kernel() #0 !dbg !8 {
entry:
  %0 = call i32 asm sideeffect "ld.local.u32 $0, [0];", "=r"() #1, !dbg !17
  ret void, !dbg !19
}

attributes #0 = { convergent mustprogress willreturn "nvvm.kernel" "target-cpu"="sm_90" }
attributes #1 = { convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "lgenfe", isOptimized: false, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!6 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "kernel", linkageName: "_Z6kernelv", scope: !1, file: !1, line: 9, type: !9, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "void")
!12 = distinct !DISubprogram(name: "local_func", linkageName: "_Z10local_funci", scope: !1, file: !1, line: 2, type: !13, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!13 = !DISubroutineType(types: !10)
!14 = !{}
!15 = distinct !DILexicalBlock(scope: !12, file: !1, line: 2, column: 30)
!16 = !DILocation(line: 10, column: 3, scope: !8)
!17 = !DILocation(line: 4, column: 5, scope: !15, inlinedAt: !16)
!19 = !DILocation(line: 11, column: 1, scope: !8)
