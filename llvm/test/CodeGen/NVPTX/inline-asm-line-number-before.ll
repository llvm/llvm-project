; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_75 | FileCheck %s
; RUN: %if ptxas-sm_75 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_75 | %ptxas-verify -arch=sm_75 %}

;; Test that debug location info is emitted before inline asm.


;; __global__ void vectorAdd(float *v) {
;;     float a, b;
;; 
;;     // Inline PTX to perform two additions using raw strings and scoped with brackets
;;     asm volatile(R"({       
;;         add.f32 %0, %2, %3;
;;         add.f32 %1, %4, %5;
;;     })"
;;         : "=f"(a)
;;         , "=f"(b)
;;         : "f"(v[0])
;;         , "f"(v[1])
;;         , "f"(v[2])
;;         , "f"(v[3])
;;     );
;; 
;;     *v = a + b;
;; }

target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8-p6:32:32"
target triple = "nvptx64-nvidia-cuda"

; CHECK: .loc [[FILEID:[0-9]+]] 5 5
; CHECK: // begin inline asm

define void @_Z9vectorAddPf(ptr noundef %v) #0 !dbg !8 {
entry:
  %arrayidx = getelementptr inbounds float, ptr %v, i32 0, !dbg !17
  %tmp1 = load float, ptr %arrayidx, align 4, !dbg !17
  %arrayidx3 = getelementptr inbounds float, ptr %v, i32 1, !dbg !19
  %tmp4 = load float, ptr %arrayidx3, align 4, !dbg !19
  %arrayidx6 = getelementptr inbounds float, ptr %v, i32 2, !dbg !20
  %tmp7 = load float, ptr %arrayidx6, align 4, !dbg !20
  %arrayidx9 = getelementptr inbounds float, ptr %v, i32 3, !dbg !21
  %tmp10 = load float, ptr %arrayidx9, align 4, !dbg !21
  %0 = call { float, float } asm sideeffect "{\0A        add.f32 $0, $2, $3;\0A        add.f32 $1, $4, $5;\0A    }", "=f,=f,f,f,f,f"(float %tmp1, float %tmp4, float %tmp7, float %tmp10) #1, !dbg !22
  %asmresult = extractvalue { float, float } %0, 0, !dbg !22
  %asmresult11 = extractvalue { float, float } %0, 1, !dbg !22
  %add = fadd float %asmresult, %asmresult11, !dbg !26
  store float %add, ptr %v, align 4, !dbg !26
  ret void, !dbg !27
}

attributes #0 = { alwaysinline convergent mustprogress willreturn "nvvm.kernel" "target-cpu"="sm_75" }
attributes #1 = { convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "lgenfe", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "vectorAdd.cu", directory: "/test")
!6 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "vectorAdd", linkageName: "_Z9vectorAddPf", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !12}
!11 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "void")
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64, align: 64)
!13 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!14 = !{}
!17 = !DILocation(line: 14, column: 15, scope: !18)
!18 = distinct !DILexicalBlock(scope: !8, file: !1, line: 1, column: 30)
!19 = !DILocation(line: 15, column: 15, scope: !18)
!20 = !DILocation(line: 16, column: 15, scope: !18)
!21 = !DILocation(line: 17, column: 15, scope: !18)
!22 = !DILocation(line: 5, column: 5, scope: !18)
!26 = !DILocation(line: 20, column: 5, scope: !18)
!27 = !DILocation(line: 21, column: 1, scope: !18)
