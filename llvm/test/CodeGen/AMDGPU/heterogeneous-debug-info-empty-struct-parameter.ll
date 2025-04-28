; REQUIRES: asserts
; RUN: llc -O0 -stop-after=finalize-isel -debug-only=instr-emitter < %s 2>&1 | FileCheck --check-prefixes=AFTER-ISEL %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

%struct.empty = type { i8 }

; AFTER-ISEL: Dropping debug info for
; AFTER-ISEL-NOT: DBG_DEF

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z6callee5empty() #0 !dbg !9 {
entry:
  %tmp = alloca %struct.empty, align 1, addrspace(5)
  %0 = addrspacecast %struct.empty addrspace(5)* %tmp to %struct.empty*
  call void @llvm.dbg.def(metadata !14, metadata %struct.empty* %0), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.def(metadata, metadata) #1

; Function Attrs: convergent mustprogress noinline nounwind optnone
define dso_local void @_Z6callerv() #0 !dbg !17 {
entry:
  %agg.tmp = alloca %struct.empty, align 1, addrspace(5)
  %agg.tmp.ascast = addrspacecast %struct.empty addrspace(5)* %agg.tmp to %struct.empty*
  call void @_Z6callee5empty() #2, !dbg !20
  ret void, !dbg !21
}

attributes #0 = { convergent mustprogress noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx900" "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0 (ssh://slinder1@gerrit-git.amd.com:29418/lightning/ec/llvm-project 5f5fbe9cbea2abf8c628dfc75d472ece801cf355)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "clang/test/CodeGenHIP/<stdin>", directory: "/home/slinder1/llvm-project/amd-stg-open")
!2 = !{!3}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "empty", file: !4, line: 7, size: 8, flags: DIFlagTypePassByValue, elements: !5, identifier: "_ZTS5empty")
!4 = !DIFile(filename: "clang/test/CodeGenHIP/debug-info-empty-struct-parameter.hip", directory: "/home/slinder1/llvm-project/amd-stg-open")
!5 = !{}
!6 = !{i32 2, !"Debug Info Version", i32 4}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 14.0.0 (ssh://slinder1@gerrit-git.amd.com:29418/lightning/ec/llvm-project 5f5fbe9cbea2abf8c628dfc75d472ece801cf355)"}
!9 = distinct !DISubprogram(name: "callee", linkageName: "_Z6callee5empty", scope: !4, file: !4, line: 9, type: !10, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !3}
!12 = !{!13}
!13 = !DILocalVariable(arg: 1, scope: !9, file: !4, line: 9, type: !3)
!14 = distinct !DILifetime(object: !13, location: !DIExpr(DIOpReferrer(%struct.empty*), DIOpDeref(%struct.empty)))
!15 = !DILocation(line: 9, column: 29, scope: !9)
!16 = !DILocation(line: 9, column: 33, scope: !9)
!17 = distinct !DISubprogram(name: "caller", linkageName: "_Z6callerv", scope: !4, file: !4, line: 10, type: !18, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !5)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !DILocation(line: 10, column: 37, scope: !17)
!21 = !DILocation(line: 10, column: 54, scope: !17)
