; RUN: opt -loop-vectorize %s -S | FileCheck %s
; Tests that the debug intrinsic does not cause additional instructions to be
; created by SCEVExpander.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.s = type { double* }

; CHECK-LABEL: entry:
; CHECK: %[[LV:.+]] = load double*, double** %a
; CHECK-NEXT: call void @llvm.dbg.value(metadata double* %[[LV]]
; CHECK-NEXT: %[[PTI:.+]] = ptrtoint double* %[[LV]] to i64
; CHECK-NEXT: %[[MPTR:.+]] = and i64 %[[PTI]], 31
; CHECK-NEXT: %[[MCOND:.+]] = icmp eq i64 %[[MPTR]], 0
; CHECK-NEXT: br i1 false, label %scalar.ph, label %vector.scevcheck


define void @test(%struct.s* %x) !dbg !6 {
entry:
  %a = getelementptr inbounds %struct.s, %struct.s* %x, i64 0, i32 0
  %0 = load double*, double** %a, align 8
  call void @llvm.dbg.value(metadata double* %0, metadata !9, metadata !DIExpression()), !dbg !11
  %ptrint = ptrtoint double* %0 to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.1, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %0, i64 %indvars.iv
  %1 = load double, double* %arrayidx, align 16
  %add = fadd double %1, 1.000000e+00
  store double %add, double* %arrayidx, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx.1 = getelementptr inbounds double, double* %0, i64 %indvars.iv.next
  %2 = load double, double* %arrayidx.1, align 8
  %add.1 = fadd double %2, 1.000000e+00
  store double %add.1, double* %arrayidx.1, align 8
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv.next, 1
  %exitcond.1 = icmp eq i64 %indvars.iv.next, 1599
  br i1 %exitcond.1, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "file.ll", directory: "/")
!2 = !{}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test", linkageName: "test", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0)
!7 = !DISubroutineType(types: !2)
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !6)
