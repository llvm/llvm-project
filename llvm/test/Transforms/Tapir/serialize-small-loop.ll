; RUN: opt < %s -serialize-small-tasks -S -o - | FileCheck %s
; RUN: opt < %s -passes='serialize-small-tasks' -S -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: argmemonly nounwind uwtable
define dso_local void @small_daxpy(double* nocapture %y, double* nocapture readonly %x, double %a) local_unnamed_addr #0 !dbg !6 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %pfor.detach, !dbg !8

pfor.cond.cleanup:                                ; preds = %pfor.inc
  sync within %syncreg, label %sync.continue, !dbg !8

pfor.detach:                                      ; preds = %pfor.inc, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc, !dbg !8

; CHECK: pfor.detach:
; CHECK-NEXT: %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %pfor.inc ]
; CHECK-NOT: detach within
; CHECK-NEXT: br label %pfor.body

pfor.body:                                        ; preds = %pfor.detach
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv, !dbg !9
  %0 = load double, double* %arrayidx, align 8, !dbg !9, !tbaa !10
  %mul1 = fmul double %0, %a, !dbg !14
  %arrayidx3 = getelementptr inbounds double, double* %y, i64 %indvars.iv, !dbg !15
  %1 = load double, double* %arrayidx3, align 8, !dbg !16, !tbaa !10
  %add4 = fadd double %1, %mul1, !dbg !16
  store double %add4, double* %arrayidx3, align 8, !dbg !16, !tbaa !10
  reattach within %syncreg, label %pfor.inc, !dbg !15

; CHECK: pfor.body:
; CHECK-NOT: reattach within
; CHECL: br label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !8
  %exitcond = icmp eq i64 %indvars.iv.next, 10, !dbg !8
  br i1 %exitcond, label %pfor.cond.cleanup, label %pfor.detach, !dbg !8, !llvm.loop !17

sync.continue:                                    ; preds = %pfor.cond.cleanup
  ret void, !dbg !20
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

attributes #0 = { argmemonly nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (git@github.com:wsmoses/Tapir-Clang.git dc9a8d2e98c088903c7ad63b576d7e73de94f4a1) (git@github.com:wsmoses/Tapir-LLVM.git 3c50c14938a10d8d50b534caa34022c4dbbc4569)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "smallloops.c", directory: "/data/compilers/tests/adhoc/tapirloops")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 7.0.0 (git@github.com:wsmoses/Tapir-Clang.git dc9a8d2e98c088903c7ad63b576d7e73de94f4a1) (git@github.com:wsmoses/Tapir-LLVM.git 3c50c14938a10d8d50b534caa34022c4dbbc4569)"}
!6 = distinct !DISubprogram(name: "small_daxpy", scope: !1, file: !1, line: 8, type: !7, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 9, column: 3, scope: !6)
!9 = !DILocation(line: 10, column: 17, scope: !6)
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
!14 = !DILocation(line: 10, column: 15, scope: !6)
!15 = !DILocation(line: 10, column: 5, scope: !6)
!16 = !DILocation(line: 10, column: 10, scope: !6)
!17 = distinct !{!17, !8, !18, !19}
!18 = !DILocation(line: 10, column: 20, scope: !6)
!19 = !{!"tapir.loop.spawn.strategy", i32 1}
!20 = !DILocation(line: 11, column: 1, scope: !6)
