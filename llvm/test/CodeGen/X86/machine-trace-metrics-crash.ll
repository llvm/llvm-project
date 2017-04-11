; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=sse -enable-unsafe-fp-math < %s | FileCheck %s

; The debug info in this test case was causing a crash because machine trace metrics
; did not correctly ignore debug instructions. The check lines ensure that the
; machine-combiner pass has run, reassociated the add operands, and therefore
; used machine trace metrics.

define void @PR24199() {
; CHECK-LABEL:	PR24199:
; CHECK:	addss	%xmm1, %xmm0
; CHECK:	addss	%xmm2, %xmm0

entry:
  %i = alloca %struct.A, align 8
  %tobool = icmp ne i32 undef, 0
  br i1 undef, label %if.end, label %if.then

if.then:
  br label %if.end

if.end:
  %h = phi float [ 0.0, %if.then ], [ 4.0, %entry ]
  call void @foo(%struct.A* nonnull undef)
  tail call void @llvm.dbg.value(metadata %struct.A* undef, i64 0, metadata !5, metadata !4), !dbg !6
  tail call void @llvm.dbg.value(metadata float %h, i64 0, metadata !5, metadata !4), !dbg !6
  %n0 = load float, float* undef, align 4
  %mul = fmul fast float %n0, %h
  %add = fadd fast float %mul, 1.0
  tail call void @llvm.dbg.value(metadata %struct.A* undef, i64 0, metadata !5, metadata !4), !dbg !6
  tail call void @llvm.dbg.value(metadata float %add, i64 0, metadata !5, metadata !4), !dbg !6
  %add.i = fadd fast float %add, %n0
  store float %add.i, float* undef, align 4
  %n1 = bitcast %struct.A* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %n1)
  %n2 = load <2 x float>, <2 x float>* undef, align 8
  %conv = uitofp i1 %tobool to float
  %bitcast = extractelement <2 x float> %n2, i32 0
  %factor = fmul fast float %bitcast, 2.0
  %add3 = fadd fast float %factor, %conv
  call void @bar(float %add3)
  ret void
}

%struct.A = type { float, float }

declare void @bar(float)
declare void @foo(%struct.A*)
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "24199.cpp", directory: "/bin")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(linkageName: "foo", file: !1, line: 18, isLocal: false, isDefinition: true, scopeLine: 18, unit: !0)
!4 = !DIExpression()
!5 = !DILocalVariable(name: "this", arg: 1, scope: !3, flags: DIFlagArtificial | DIFlagObjectPointer)
!6 = !DILocation(line: 0, scope: !3)


