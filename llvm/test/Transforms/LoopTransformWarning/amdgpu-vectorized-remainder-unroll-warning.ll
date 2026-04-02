; RUN: opt < %s -passes='loop-vectorize,loop-unroll,transform-warning' -force-vector-width=4 -force-vector-interleave=1 -disable-output -pass-remarks-missed=transform-warning 2>&1 | FileCheck %s
;
; AMDGPU variant: verifies that the improved unroll warnings work on an
; amdgpu_kernel targeting gfx90a.
;
; With noalias pointers and a unit stride, the vectorizer sets
; DisableRuntimeUnroll = true, so the remainder loop gets
; llvm.loop.unroll.runtime.disable.  The unroller can still handle
; the vectorized loop, but the remainder survives to transform-warning
; which should emit the specific "scalar remainder" message.

; CHECK-NOT: warning: {{.*}} loop not unrolled
; CHECK: warning: {{.*}} scalar remainder loop after vectorization not unrolled: the optimizer was unable to perform the requested transformation

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @pragma_unroll_kernel(ptr addrspace(1) noalias %A, ptr addrspace(1) noalias %B, i32 %n) #0 !dbg !4 {
entry:
  %cmp = icmp sgt i32 %n, 0, !dbg !8
  br i1 %cmp, label %for.body.preheader, label %for.end, !dbg !8

for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %for.body.preheader ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %A, i64 %iv, !dbg !9
  %val = load i32, ptr addrspace(1) %arrayidx, align 4, !dbg !9
  %add = add nsw i32 %val, 1, !dbg !9
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %B, i64 %iv, !dbg !9
  store i32 %add, ptr addrspace(1) %arrayidx2, align 4, !dbg !9
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %wide.trip.count, !dbg !8
  br i1 %exitcond, label %for.end, label %for.body, !dbg !8, !llvm.loop !10

for.end:
  ret void, !dbg !11
}

attributes #0 = { nounwind "target-cpu"="gfx90a" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, file: !1)
!1 = !DIFile(filename: "test.hip", directory: ".")
!2 = !{i32 2, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "pragma_unroll_kernel", line: 5, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 5, file: !1, scope: !1, type: !5, retainedNodes: !6)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = distinct !DILexicalBlock(line: 7, column: 3, file: !1, scope: !4)
!8 = !DILocation(line: 7, column: 3, scope: !7)
!9 = !DILocation(line: 8, column: 5, scope: !7)
!10 = distinct !{!10, !12}
!11 = !DILocation(line: 10, column: 1, scope: !4)
!12 = !{!"llvm.loop.unroll.enable"}
