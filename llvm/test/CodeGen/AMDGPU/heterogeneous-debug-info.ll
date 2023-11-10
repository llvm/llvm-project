 ; RUN: llc -O0 -stop-after=finalize-isel < %s | FileCheck --check-prefixes=COMMON,AFTER-ISEL %s

; RUN: llc -O0 -stop-after=regallocfast < %s | FileCheck --check-prefixes=COMMON,AFTER-RA %s
; RUN: llc -O0 -stop-after=prologepilog < %s | FileCheck --check-prefixes=COMMON,AFTER-PEI %s
; RUN: llc -O0 -stop-after=livedebugvalues < %s | FileCheck --check-prefixes=COMMON,AFTER-LDV %s
; COMMON-DAG: ![[VAR_I:[0-9]+]] = !DILocalVariable(name: "I",
; COMMON-DAG: ![[VAR_R:[0-9]+]] = !DILocalVariable(name: "R",
; AFTER-ISEL-DAG: ![[ENTRY_LIFETIME_VAR_I:[0-9]+]] = distinct !DILifetime(object: ![[VAR_I]], location: !DIExpr(DIOpReferrer(i32)))
; AFTER-ISEL-DAG: ![[STACK_LIFETIME_VAR_I:[0-9]+]] = distinct !DILifetime(object: ![[VAR_I]], location: !DIExpr(DIOpReferrer(i32)))
; AFTER-ISEL-DAG: ![[STACK_LIFETIME_VAR_R:[0-9]+]] = distinct !DILifetime(object: ![[VAR_R]], location: !DIExpr(DIOpReferrer(i32)))
; AFTER-RA-DAG: ![[ENTRY_LIFETIME_VAR_I:[0-9]+]] = distinct !DILifetime(object: ![[VAR_I]], location: !DIExpr(DIOpReferrer(i32)))
; AFTER-RA-DAG: ![[STACK_LIFETIME_VAR_I:[0-9]+]] = distinct !DILifetime(object: ![[VAR_I]], location: !DIExpr(DIOpReferrer(i32)))
; AFTER-RA-DAG: ![[STACK_LIFETIME_VAR_R:[0-9]+]] = distinct !DILifetime(object: ![[VAR_R]], location: !DIExpr(DIOpReferrer(i32)))
; AFTER-PEI-DAG: ![[ENTRY_LIFETIME_VAR_I:[0-9]+]] = distinct !DILifetime(object: ![[VAR_I]], location: !DIExpr(DIOpReferrer(i32)))
; AFTER-PEI-DAG: ![[STACK_LIFETIME_VAR_I:[0-9]+]] = distinct !DILifetime(object: ![[VAR_I]], location: !DIExpr(DIOpReferrer(i32), DIOpConstant(i32 6), DIOpShr(), DIOpReinterpret(ptr addrspace(5)), DIOpDeref(i32), DIOpConstant(i32 4), DIOpByteOffset(i32)))
; AFTER-PEI-DAG: ![[STACK_LIFETIME_VAR_R:[0-9]+]] = distinct !DILifetime(object: ![[VAR_R]], location: !DIExpr(DIOpReferrer(i32), DIOpConstant(i32 6), DIOpShr(), DIOpReinterpret(ptr addrspace(5)), DIOpDeref(i32), DIOpConstant(i32 8), DIOpByteOffset(i32)))
; AFTER-LDV-DAG: ![[ENTRY_LIFETIME_VAR_I:[0-9]+]] = distinct !DILifetime(object: ![[VAR_I]], location: !DIExpr(DIOpReferrer(i32)))
; AFTER-LDV-DAG: ![[STACK_LIFETIME_VAR_I:[0-9]+]] = distinct !DILifetime(object: ![[VAR_I]], location: !DIExpr(DIOpReferrer(i32), DIOpConstant(i32 6), DIOpShr(), DIOpReinterpret(ptr addrspace(5)), DIOpDeref(i32), DIOpConstant(i32 4), DIOpByteOffset(i32)))
; AFTER-LDV-DAG: ![[STACK_LIFETIME_VAR_R:[0-9]+]] = distinct !DILifetime(object: ![[VAR_R]], location: !DIExpr(DIOpReferrer(i32), DIOpConstant(i32 6), DIOpShr(), DIOpReinterpret(ptr addrspace(5)), DIOpDeref(i32), DIOpConstant(i32 8), DIOpByteOffset(i32)))
; COMMON-LABEL: bb.{{[0-9]}}.entry:
; COMMON: {{^$}}
; AFTER-ISEL-NOT: DBG_
; AFTER-ISEL: %[[#ARG_0_COPY_VREG:]]:vgpr_32 = COPY $vgpr0
; AFTER-ISEL-NOT: DBG_
; AFTER-ISEL: DBG_DEF ![[ENTRY_LIFETIME_VAR_I]], %[[#ARG_0_COPY_VREG]]
; AFTER-ISEL-NOT: DBG_
; AFTER-ISEL: DBG_KILL ![[ENTRY_LIFETIME_VAR_I]]
; AFTER-ISEL-NOT: DBG_
; AFTER-ISEL: DBG_DEF ![[STACK_LIFETIME_VAR_I]], %stack.1.I.addr
; AFTER-ISEL-NOT: DBG_
; AFTER-ISEL: DBG_DEF ![[STACK_LIFETIME_VAR_R]], %stack.2.R
; AFTER-ISEL-NOT: DBG_

; AFTER-RA-NOT: DBG_
; AFTER-RA: %[[#ARG_0_COPY_VREG:]]:vgpr_32 = COPY killed $vgpr0
; AFTER-RA-NOT: DBG_
; AFTER-RA: DBG_DEF ![[STACK_LIFETIME_VAR_I]], %stack.1.I.addr
; AFTER-RA-NOT: DBG_
; AFTER-RA: DBG_DEF ![[STACK_LIFETIME_VAR_R]], %stack.2.R
; AFTER-RA-NOT: DBG_

; AFTER-PEI-NOT: DBG_
; AFTER-PEI: DBG_DEF ![[STACK_LIFETIME_VAR_I]], $sgpr33
; AFTER-PEI-NOT: DBG_
; AFTER-PEI: DBG_DEF ![[STACK_LIFETIME_VAR_R]], $sgpr33
; AFTER-PEI-NOT: DBG_

; AFTER-LDV-NOT: DBG_
; AFTER-LDV: DBG_DEF ![[STACK_LIFETIME_VAR_I]], $sgpr33
; AFTER-LDV-NOT: DBG_
; AFTER-LDV: DBG_DEF ![[STACK_LIFETIME_VAR_R]], $sgpr33
; AFTER-LDV-NOT: DBG_

; COMMON-LABEL: bb.{{[0-9]}}.Flow:

; AFTER-ISEL-NOT: DBG_

; AFTER-PEI-NOT: DBG_

; AFTER-LDV: {{^[[:space:]]+$}}
; AFTER-LDV-DAG: DBG_DEF ![[STACK_LIFETIME_VAR_I]], $sgpr33
; AFTER-LDV-DAG: DBG_DEF ![[STACK_LIFETIME_VAR_R]], $sgpr33
; AFTER-LDV-NOT: DBG_

; COMMON-LABEL: bb.{{[0-9]}}.if.then:

; AFTER-ISEL-NOT: DBG_

; AFTER-PEI-NOT: DBG_

; AFTER-LDV: {{^[[:space:]]+$}}
; AFTER-LDV-DAG: DBG_DEF ![[STACK_LIFETIME_VAR_I]], $sgpr33
; AFTER-LDV-DAG: DBG_DEF ![[STACK_LIFETIME_VAR_R]], $sgpr33
; AFTER-LDV-NOT: DBG_

; COMMON-LABEL: bb.{{[0-9]}}.if.else:

; AFTER-ISEL-NOT: DBG_

; AFTER-PEI-NOT: DBG_

; AFTER-LDV: {{^[[:space:]]+$}}
; AFTER-LDV-DAG: DBG_DEF ![[STACK_LIFETIME_VAR_I]], $sgpr33
; AFTER-LDV-DAG: DBG_DEF ![[STACK_LIFETIME_VAR_R]], $sgpr33
; AFTER-LDV-NOT: DBG_

; COMMON-LABEL: bb.{{[0-9]}}.if.end:

; AFTER-ISEL-NOT: DBG_

; AFTER-PEI-NOT: DBG_

; AFTER-LDV: {{^[[:space:]]+$}}
; AFTER-LDV-DAG: DBG_DEF ![[STACK_LIFETIME_VAR_I]], $sgpr33
; AFTER-LDV-DAG: DBG_DEF ![[STACK_LIFETIME_VAR_R]], $sgpr33
; AFTER-LDV-NOT: DBG_
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"
; Function Attrs: convergent mustprogress noinline nounwind optnone
define hidden i32 @_Z13absoluteValuei(i32 %I) #2 {
entry:
  call void @llvm.dbg.def(metadata !14, metadata i32 %I)
  %retval = alloca i32, align 4, addrspace(5)
  %retval.ascast = addrspacecast ptr addrspace(5) %retval to ptr
  %I.addr = alloca i32, align 4, addrspace(5)
  call void @llvm.dbg.kill(metadata !14)
  call void @llvm.dbg.def(metadata !15, metadata ptr addrspace(5) %I.addr)
  %I.addr.ascast = addrspacecast ptr addrspace(5) %I.addr to ptr
  %R = alloca i32, align 4, addrspace(5)
  call void @llvm.dbg.def(metadata !17, metadata ptr addrspace(5) %R)
  %R.ascast = addrspacecast ptr addrspace(5) %R to ptr
  store i32 %I, ptr %I.addr.ascast, align 4
  %0 = load i32, ptr %I.addr.ascast, align 4
  %cmp = icmp slt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else
if.then:                                          ; preds = %entry
  %1 = load i32, ptr %I.addr.ascast, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %R.ascast, align 4
  br label %if.end
if.else:                                          ; preds = %entry
  %2 = load i32, ptr %I.addr.ascast, align 4
  store i32 %2, ptr %R.ascast, align 4
  br label %if.end
if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %R.ascast, align 4
  ret i32 %3
}
; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.def(metadata, metadata) #3
; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.kill(metadata) #3
attributes #3 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #0 = { convergent mustprogress noinline noreturn nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx900" "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst" }
attributes #1 = { cold noreturn nounwind }
attributes #2 = { convergent mustprogress noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx900" "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst" }
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7}
!llvm.ident = !{!8}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_11, file: !1, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 87656a3134c7c03565efca85352a58541ce68789)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "absoluteValue.hip", directory: "/rocm-gdb-symbols")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 4}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 1}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 87656a3134c7c03565efca85352a58541ce68789)"}
!9 = distinct !DISubprogram(name: "absoluteValue", scope: !0, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "I", arg: 1, scope: !9, file: !1, line: 1, type: !12)
!14 = distinct !DILifetime(object: !13, location: !DIExpr(DIOpReferrer(i32)))
!15 = distinct !DILifetime(object: !13, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!16 = !DILocalVariable(name: "R", scope: !9, file: !1, line: 2, type: !12)
!17 = distinct !DILifetime(object: !16, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
