; RUN: opt -S -passes=licm %s | FileCheck %s
;
; LICM should null out debug locations when it hoists intrinsics that won't lower to function calls out of a loop.
; CHECK: define float @foo
; CHECK-NEXT: entry:
; CHECK-NEXT: call float @llvm.fma.f32(float %coef_0, float %coef_1, float 0.000000e+00){{$}}
; CHECK-NEXT: br label %loop.header
;
define float @foo(float* %A, float %coef_0, float %coef_1, i32 %n) !dbg !2 {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %loop.backedge ]
  %a = phi float [ 0.000000e+00, %entry ], [ %a.inc, %loop.backedge ]
  %cond = icmp ult i32 %i, %n
  br i1 %cond, label %loop.backedge, label %exit

loop.backedge:
  %i.cast = zext i32 %i to i64
  %A.ptr = getelementptr inbounds float, float* %A, i64 %i.cast
  %A.load = load float, float* %A.ptr
  %fma = call float @llvm.fma.f32(float %coef_0, float %coef_1, float 0.000000e+00), !dbg !3
  %mul = fmul float %fma, %A.load
  %a.inc = fadd float %mul, %a
  %i.inc = add i32 %i, 1
  br label %loop.header

exit:
  ret float %a
}

declare float @llvm.fma.f32(float, float, float) #1

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "source.c", directory: "/")
!2 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !5)
!3 = !DILocation(line: 4, column: 17, scope: !2)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 2, !"Debug Info Version", i32 3}
