; RUN: opt -always-inline -S < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

@global = global i32 6, !dbg.def !2

; CHECK-LABEL: define hidden i32 @_Z6calleei{{.*}}
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CALLEE_LIFETIME_I:]], metadata i32 addrspace(5)* %I{{.*}})
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CALLEE_LIFETIME_J:]], metadata i32 addrspace(5)* %J{{.*}})

; Function Attrs: alwaysinline convergent mustprogress nounwind
define hidden i32 @_Z6calleei(i32 %I) #0 !dbg !12 {
entry:
  %retval = alloca i32, align 4, addrspace(5)
  %retval.ascast = addrspacecast i32 addrspace(5)* %retval to i32*
  %I.addr = alloca i32, align 4, addrspace(5)
  %I.addr.ascast = addrspacecast i32 addrspace(5)* %I.addr to i32*
  %J = alloca i32, align 4, addrspace(5)
  %J.ascast = addrspacecast i32 addrspace(5)* %J to i32*
  store i32 %I, i32* %I.addr.ascast, align 4
  call void @llvm.dbg.def(metadata !19, metadata i32 addrspace(5)* %I.addr)
  call void @llvm.dbg.def(metadata !21, metadata i32 addrspace(5)* %J)
  %0 = load i32, i32* %I.addr.ascast, align 4
  store i32 %0, i32* %J.ascast, align 4
  %1 = load i32, i32* %J.ascast, align 4
  ret i32 %1
}

; CHECK-LABEL: define hidden i32 @_Z7caller0v{{.*}}
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CALLER0_LIFETIME_I:]], metadata i32 addrspace(5)* %I{{.*}})
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CALLER0_LIFETIME_J:]], metadata i32 addrspace(5)* %J{{.*}})

; Function Attrs: convergent mustprogress noinline nounwind optnone
define hidden i32 @_Z7caller0v() #1 !dbg !23 {
entry:
  %retval = alloca i32, align 4, addrspace(5)
  %retval.ascast = addrspacecast i32 addrspace(5)* %retval to i32*
  %call = call i32 @_Z6calleei(i32 0) #0, !dbg !26
  ret i32 %call
}

; CHECK-LABEL: define hidden i32 @_Z7caller1v{{.*}}
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CALLER1_LIFETIME_I:]], metadata i32 addrspace(5)* %I{{.*}})
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CALLER1_LIFETIME_J:]], metadata i32 addrspace(5)* %J{{.*}})

; Function Attrs: convergent mustprogress noinline nounwind optnone
define hidden i32 @_Z7caller1v() #1 !dbg !27 {
entry:
  %retval = alloca i32, align 4, addrspace(5)
  %retval.ascast = addrspacecast i32 addrspace(5)* %retval to i32*
  %call = call i32 @_Z6calleei(i32 1) #0, !dbg !28
  ret i32 %call
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.def(metadata, metadata) #2

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.kill(metadata) #2

attributes #0 = { alwaysinline convergent mustprogress nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx900" "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst" }
attributes #1 = { convergent mustprogress noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx900" "target-features"="+16-bit-insts,+ci-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst" }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }

; CHECK-LABEL: !llvm.dbg.retainedNodes =
; CHECK-SAME: !{![[#GLOBAL_COMPUTED_LIFETIME:]]}

; CHECK-DAG: ![[#EMPTY_TUPLE:]] = !{}

; CHECK-DAG: ![[#GLOBAL_COMPUTED_LIFETIME]] = distinct !DILifetime(object: ![[#GLOBAL_VARIABLE:]], location: !DIExpr(DIOpArg(0, i32)), argObjects: {![[#GLOBAL_FRAGMENT:]]})
; CHECK-DAG: ![[#GLOBAL_VARIABLE]] = !DIGlobalVariable(name: "global", {{.*}}
; CHECK-DAG: ![[#GLOBAL_FRAGMENT]] = distinct !DIFragment()

; CHECK-DAG: ![[#CALLEE_LIFETIME_I]] = distinct !DILifetime(object: ![[#CALLEE_LOCAL_VARIABLE_I:]], location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref(i32)))
; CHECK-DAG: ![[#CALLEE_LIFETIME_J]] = distinct !DILifetime(object: ![[#CALLEE_LOCAL_VARIABLE_J:]], location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref(i32)), argObjects: {![[#GLOBAL_FRAGMENT]], ![[#GLOBAL_VARIABLE]], ![[#CALLEE_LOCAL_VARIABLE_I]], ![[#CALLEE_LOCAL_FRAGMENT:]]})
; CHECK-DAG: ![[#CALLEE_LOCAL_VARIABLE_I]] = !DILocalVariable(name: "I", arg: 1, scope: ![[#CALLEE_SUBPROGRAM:]], {{.*}}
; CHECK-DAG: ![[#CALLEE_LOCAL_VARIABLE_J]] = !DILocalVariable(name: "J", scope: ![[#CALLEE_SUBPROGRAM]], {{.*}}
; CHECK-DAG: ![[#CALLEE_SUBPROGRAM]] = distinct !DISubprogram(name: "callee", {{.*}}retainedNodes: ![[#CALLEE_RETAINED_NODES:]])
; CHECK-DAG: ![[#CALLEE_RETAINED_NODES]] = !{![[#CALLEE_COMPUTED_LIFETIME:]], ![[#CALLEE_LOCAL_VARIABLE_I]], ![[#CALLEE_LOCAL_VARIABLE_J]]}
; CHECK-DAG: ![[#CALLEE_LOCAL_FRAGMENT]] = distinct !DIFragment()
; CHECK-DAG: ![[#CALLEE_COMPUTED_LIFETIME]] = distinct !DILifetime(object: ![[#CALLEE_LOCAL_FRAGMENT]], location: !DIExpr(DIOpConstant(i8 0)))

; CHECK-DAG: ![[#CALLER0_LIFETIME_I]] = distinct !DILifetime(object: ![[#CALLEE_LOCAL_VARIABLE_I:]], location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref(i32)))
; CHECK-DAG: ![[#CALLER0_LIFETIME_J]] = distinct !DILifetime(object: ![[#CALLEE_LOCAL_VARIABLE_J:]], location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref(i32)), argObjects: {![[#GLOBAL_FRAGMENT]], ![[#GLOBAL_VARIABLE]], ![[#CALLEE_LOCAL_VARIABLE_I]], ![[#CALLER0_LOCAL_FRAGMENT:]]})
; CHECK-DAG: ![[#CALLER0_SUBPROGRAM:]] = distinct !DISubprogram(name: "caller0", {{.*}}retainedNodes: ![[#EMPTY_TUPLE:]]{{.*}})
; CHECK-DAG: ![[#CALLER0_LOCAL_FRAGMENT]] = distinct !DIFragment()

; CHECK-DAG: ![[#CALLER1_LIFETIME_I]] = distinct !DILifetime(object: ![[#CALLEE_LOCAL_VARIABLE_I:]], location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref(i32)))
; CHECK-DAG: ![[#CALLER1_LIFETIME_J]] = distinct !DILifetime(object: ![[#CALLEE_LOCAL_VARIABLE_J:]], location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref(i32)), argObjects: {![[#GLOBAL_FRAGMENT]], ![[#GLOBAL_VARIABLE]], ![[#CALLEE_LOCAL_VARIABLE_I]], ![[#CALLER1_LOCAL_FRAGMENT:]]})
; CHECK-DAG: ![[#CALLER1_SUBPROGRAM:]] = distinct !DISubprogram(name: "caller1", {{.*}}retainedNodes: ![[#EMPTY_TUPLE:]]{{.*}})
; CHECK-DAG: ![[#CALLER1_LOCAL_FRAGMENT]] = distinct !DIFragment()

!llvm.dbg.retainedNodes = !{!0}
!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpArg(0, i32)), argObjects: {!2})
!1 = !DIGlobalVariable(name: "global", scope: !3, type: !15, isLocal: false, isDefinition: true)
!2 = distinct !DIFragment()
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_11, file: !4, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 87656a3134c7c03565efca85352a58541ce68789)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, imports: !5, splitDebugInlining: false, nameTableKind: None)
!4 = !DIFile(filename: "alwaysInline.hip", directory: "/rocm-gdb-symbols")
!5 = !{}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 4}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"PIC Level", i32 1}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 87656a3134c7c03565efca85352a58541ce68789)"}
!12 = distinct !DISubprogram(name: "callee", scope: !3, file: !4, line: 1, type: !13, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !15}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!17, !20, !22}
!17 = distinct !DILifetime(object: !18, location: !DIExpr(DIOpConstant(i8 0)))
!18 = distinct !DIFragment()
!19 = distinct !DILifetime(object: !20, location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref(i32)))
!20 = !DILocalVariable(name: "I", arg: 1, scope: !12, file: !4, line: 1, type: !15)
!21 = distinct !DILifetime(object: !22, location: !DIExpr(DIOpReferrer(i32 addrspace(5)*), DIOpDeref(i32)), argObjects: {!2, !1, !20, !18})
!22 = !DILocalVariable(name: "J", scope: !12, file: !4, line: 1, type: !15)
!23 = distinct !DISubprogram(name: "caller0", scope: !3, file: !4, line: 2, type: !24, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !5)
!24 = !DISubroutineType(types: !25)
!25 = !{!15}
!26 = !DILocation(line: 2, column: 2, scope: !23)
!27 = distinct !DISubprogram(name: "caller1", scope: !3, file: !4, line: 3, type: !24, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !5)
!28 = !DILocation(line: 3, column: 3, scope: !27)
