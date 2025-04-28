; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -amdgpu-lower-module-lds -S < %s | FileCheck %s
; ModuleID = 'test.cl'
source_filename = "test.cl"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; CHECK: @llvm.amdgcn.kernel.hello.lds = internal addrspace(3) global %llvm.amdgcn.kernel.hello.lds.t poison, align 8, !absolute_symbol !{{[0-9]+}}, !dbg.def ![[HELLO_FRAGMENT:[0-9]+]]
; CHECK: @llvm.amdgcn.kernel.bye.lds = internal addrspace(3) global %llvm.amdgcn.kernel.bye.lds.t poison, align 8, !absolute_symbol !{{[0-9]+}}, !dbg.def ![[BYE_FRAGMENT:[0-9]+]]

; CHECK: ![[HELLO_FRAGMENT]] = distinct !DIFragment()
; CHECK: ![[BYE_FRAGMENT]] = distinct !DIFragment()

; CHECK: distinct !DILifetime(object: ![[ONE32:[0-9]+]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i32), DIOpConstant(i32 8), DIOpByteOffset(i32)), argObjects: {![[HELLO_FRAGMENT]]})
; CHECK: ![[ONE32]] = distinct !DIGlobalVariable(name: "One32"
; CHECK: distinct !DILifetime(object: ![[TWO64:[0-9]+]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i64), DIOpConstant(i32 0), DIOpByteOffset(i64)), argObjects: {![[HELLO_FRAGMENT]]})
; CHECK: ![[TWO64]] = distinct !DIGlobalVariable(name: "Two64"
; CHECK: distinct !DILifetime(object: ![[THREE32:[0-9]+]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i32), DIOpConstant(i32 12), DIOpByteOffset(i32)), argObjects: {![[HELLO_FRAGMENT]]})
; CHECK: ![[THREE32]] = distinct !DIGlobalVariable(name: "Three32"

; CHECK: distinct !DILifetime(object: ![[FOUR64:[0-9]+]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i64), DIOpConstant(i32 0), DIOpByteOffset(i64)), argObjects: {![[BYE_FRAGMENT]]})
; CHECK: ![[FOUR64]] = distinct !DIGlobalVariable(name: "Four64"
; CHECK: distinct !DILifetime(object: ![[FIVE32:[0-9]+]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i32), DIOpConstant(i32 8), DIOpByteOffset(i32)), argObjects: {![[BYE_FRAGMENT]]})
; CHECK: ![[FIVE32]] = distinct !DIGlobalVariable(name: "Five32"
; CHECK: distinct !DILifetime(object: ![[SIX32:[0-9]+]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i32), DIOpConstant(i32 12), DIOpByteOffset(i32)), argObjects: {![[BYE_FRAGMENT]]})
; CHECK: ![[SIX32]] = distinct !DIGlobalVariable(name: "Six32"

@hello.One32 = internal addrspace(3) global i32 undef, align 4, !dbg.def !0
@hello.Two64 = internal addrspace(3) global i64 undef, align 8, !dbg.def !1
@hello.Three32 = internal addrspace(3) global i32 undef, align 4, !dbg.def !2
@bye.Four64 = internal addrspace(3) global i64 undef, align 8, !dbg.def !3
@bye.Five32 = internal addrspace(3) global i32 undef, align 4, !dbg.def !4
@bye.Six32 = internal addrspace(3) global i32 undef, align 4, !dbg.def !5

; Function Attrs: convergent noinline norecurse nounwind optnone
define protected amdgpu_kernel void @hello(ptr addrspace(1) noundef align 4 %A) #0 !dbg !12 !kernel_arg_addr_space !42 !kernel_arg_access_qual !43 !kernel_arg_type !44 !kernel_arg_base_type !44 !kernel_arg_type_qual !45 {
entry:
  %A.addr = alloca ptr addrspace(1), align 8, addrspace(5)
  store ptr addrspace(1) %A, ptr addrspace(5) %A.addr, align 8
  call void @llvm.dbg.def(metadata !46, metadata ptr addrspace(5) %A.addr), !dbg !47
  %0 = load ptr addrspace(1), ptr addrspace(5) %A.addr, align 8, !dbg !48
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %0, i64 0, !dbg !48
  %1 = load i32, ptr addrspace(1) %arrayidx, align 4, !dbg !48
  store i32 %1, ptr addrspace(3) @hello.One32, align 4, !dbg !49
  %2 = load ptr addrspace(1), ptr addrspace(5) %A.addr, align 8, !dbg !50
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 1, !dbg !50
  %3 = load i32, ptr addrspace(1) %arrayidx1, align 4, !dbg !50
  %conv = sext i32 %3 to i64, !dbg !51
  store i64 %conv, ptr addrspace(3) @hello.Two64, align 8, !dbg !52
  %4 = load ptr addrspace(1), ptr addrspace(5) %A.addr, align 8, !dbg !53
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 2, !dbg !53
  %5 = load i32, ptr addrspace(1) %arrayidx2, align 4, !dbg !53
  store i32 %5, ptr addrspace(3) @hello.Three32, align 4, !dbg !54
  ret void, !dbg !55
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.def(metadata, metadata) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define protected amdgpu_kernel void @bye(ptr addrspace(1) noundef align 4 %B) #0 !dbg !25 !kernel_arg_addr_space !42 !kernel_arg_access_qual !43 !kernel_arg_type !44 !kernel_arg_base_type !44 !kernel_arg_type_qual !45 {
entry:
  %B.addr = alloca ptr addrspace(1), align 8, addrspace(5)
  store ptr addrspace(1) %B, ptr addrspace(5) %B.addr, align 8
  call void @llvm.dbg.def(metadata !56, metadata ptr addrspace(5) %B.addr), !dbg !57
  %0 = load ptr addrspace(1), ptr addrspace(5) %B.addr, align 8, !dbg !58
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %0, i64 0, !dbg !58
  %1 = load i32, ptr addrspace(1) %arrayidx, align 4, !dbg !58
  %conv = sext i32 %1 to i64, !dbg !59
  store i64 %conv, ptr addrspace(3) @bye.Four64, align 8, !dbg !60
  %2 = load ptr addrspace(1), ptr addrspace(5) %B.addr, align 8, !dbg !61
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 1, !dbg !61
  %3 = load i32, ptr addrspace(1) %arrayidx1, align 4, !dbg !61
  store i32 %3, ptr addrspace(3) @bye.Five32, align 4, !dbg !62
  %4 = load ptr addrspace(1), ptr addrspace(5) %B.addr, align 8, !dbg !63
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 2, !dbg !63
  %5 = load i32, ptr addrspace(1) %arrayidx2, align 4, !dbg !63
  store i32 %5, ptr addrspace(3) @bye.Six32, align 4, !dbg !64
  ret void, !dbg !65
}

attributes #0 = { convergent noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,256" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1030" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!6}
!llvm.dbg.retainedNodes = !{!10, !19, !21, !23, !28, !30}
!llvm.module.flags = !{!32, !33, !34, !35, !36, !37}
!opencl.ocl.version = !{!38, !39}
!llvm.ident = !{!40, !41}

!0 = distinct !DIFragment()
!1 = distinct !DIFragment()
!2 = distinct !DIFragment()
!3 = distinct !DIFragment()
!4 = distinct !DIFragment()
!5 = distinct !DIFragment()
!6 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !7, producer: "clang version 17.0.0 (ssh://kzhuravl@gerrit-git.amd.com:29418/lightning/ec/llvm-project 76d7dcab493592ef3d677380bd7bb72eba837b6d)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !8, splitDebugInlining: false, nameTableKind: None)
!7 = !DIFile(filename: "test.cl", directory: "/home/kzhuravl/Temp", checksumkind: CSK_MD5, checksum: "2eb184c4ab001e8b6aa9a7053bca621d")
!8 = !{!9}
!9 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!10 = distinct !DILifetime(object: !11, location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i32)), argObjects: {!0})
!11 = distinct !DIGlobalVariable(name: "One32", scope: !12, file: !7, line: 2, type: !16, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
!12 = distinct !DISubprogram(name: "hello", scope: !7, file: !7, line: 1, type: !13, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !17)
!13 = !DISubroutineType(cc: DW_CC_LLVM_OpenCLKernel, types: !14)
!14 = !{null, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, memorySpace: DW_MSPACE_LLVM_global)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DILocalVariable(name: "A", arg: 1, scope: !12, file: !7, line: 1, type: !15, memorySpace: DW_MSPACE_LLVM_private)
!19 = distinct !DILifetime(object: !20, location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i64)), argObjects: {!1})
!20 = distinct !DIGlobalVariable(name: "Two64", scope: !12, file: !7, line: 3, type: !9, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
!21 = distinct !DILifetime(object: !22, location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i32)), argObjects: {!2})
!22 = distinct !DIGlobalVariable(name: "Three32", scope: !12, file: !7, line: 4, type: !16, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
!23 = distinct !DILifetime(object: !24, location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i64)), argObjects: {!3})
!24 = distinct !DIGlobalVariable(name: "Four64", scope: !25, file: !7, line: 12, type: !9, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
!25 = distinct !DISubprogram(name: "bye", scope: !7, file: !7, line: 11, type: !13, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !26)
!26 = !{!27}
!27 = !DILocalVariable(name: "B", arg: 1, scope: !25, file: !7, line: 11, type: !15, memorySpace: DW_MSPACE_LLVM_private)
!28 = distinct !DILifetime(object: !29, location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i32)), argObjects: {!4})
!29 = distinct !DIGlobalVariable(name: "Five32", scope: !25, file: !7, line: 13, type: !16, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
!30 = distinct !DILifetime(object: !31, location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i32)), argObjects: {!5})
!31 = distinct !DIGlobalVariable(name: "Six32", scope: !25, file: !7, line: 14, type: !16, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
!32 = !{i32 1, !"amdgpu_code_object_version", i32 400}
!33 = !{i32 7, !"Dwarf Version", i32 5}
!34 = !{i32 2, !"Debug Info Version", i32 4}
!35 = !{i32 1, !"wchar_size", i32 4}
!36 = !{i32 8, !"PIC Level", i32 1}
!37 = !{i32 7, !"frame-pointer", i32 2}
!38 = !{i32 1, i32 2}
!39 = !{i32 2, i32 0}
!40 = !{!"clang version 17.0.0 (ssh://kzhuravl@gerrit-git.amd.com:29418/lightning/ec/llvm-project 76d7dcab493592ef3d677380bd7bb72eba837b6d)"}
!41 = !{!"AMD clang version 16.0.0 (ssh://gerritgit/lightning/ec/llvm-project amd-mainline-open 23023 c6f1813f2f3ef5d65f01bbe5114b758a0523e094)"}
!42 = !{i32 1}
!43 = !{!"none"}
!44 = !{!"int*"}
!45 = !{!""}
!46 = distinct !DILifetime(object: !18, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(1))))
!47 = !DILocation(line: 1, column: 31, scope: !12)
!48 = !DILocation(line: 6, column: 11, scope: !12)
!49 = !DILocation(line: 6, column: 9, scope: !12)
!50 = !DILocation(line: 7, column: 17, scope: !12)
!51 = !DILocation(line: 7, column: 11, scope: !12)
!52 = !DILocation(line: 7, column: 9, scope: !12)
!53 = !DILocation(line: 8, column: 13, scope: !12)
!54 = !DILocation(line: 8, column: 11, scope: !12)
!55 = !DILocation(line: 9, column: 1, scope: !12)
!56 = distinct !DILifetime(object: !27, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr addrspace(1))))
!57 = !DILocation(line: 11, column: 29, scope: !25)
!58 = !DILocation(line: 16, column: 18, scope: !25)
!59 = !DILocation(line: 16, column: 12, scope: !25)
!60 = !DILocation(line: 16, column: 10, scope: !25)
!61 = !DILocation(line: 17, column: 12, scope: !25)
!62 = !DILocation(line: 17, column: 10, scope: !25)
!63 = !DILocation(line: 18, column: 11, scope: !25)
!64 = !DILocation(line: 18, column: 9, scope: !25)
!65 = !DILocation(line: 19, column: 1, scope: !25)
