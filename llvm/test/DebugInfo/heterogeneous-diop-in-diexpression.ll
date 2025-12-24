; RUN: opt -S -passes=verify < %s | FileCheck %s
; RUN: llc --filetype=obj --relocation-model=pic -fast-isel=false < %s | llvm-dwarfdump -v -debug-info - | FileCheck --check-prefix=DWARF %s
; RUN: llc --filetype=obj --relocation-model=pic -fast-isel=true < %s | llvm-dwarfdump -v -debug-info - | FileCheck --check-prefix=DWARF %s

; TODO: Test for global isel

; DWARF: DW_TAG_variable
; DWARF: DW_AT_name [DW_FORM_strx1] (indexed ([[#%x,]]) string = "glob")
; DWARF: DW_AT_location [DW_FORM_exprloc] (DW_OP_addrx 0x0, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

; DWARF: DW_TAG_variable
; DWARF: DW_AT_name [DW_FORM_strx1] (indexed ([[#%x,]]) string = "glob_fragmented")
; DWARF: DW_AT_location [DW_FORM_exprloc] (DW_OP_addrx 0x1, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_piece 0x2, DW_OP_addrx 0x2, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_piece 0x2)

; DWARF: DW_TAG_variable
; DWARF: DW_AT_location [DW_FORM_loclistx] (indexed (0x[[#%x,]]) loclist = 0x[[#%x,]]:
; DWARF:    [0x[[#%x,]], 0x[[#%x,]]) ".text": DW_OP_reg6 RBP, DW_OP_deref_size 0x8, DW_OP_consts -4, DW_OP_plus, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address
; DWARF: DW_AT_name [DW_FORM_strx1] (indexed ([[#%x,]]) string = "var")

; DWARF: DW_TAG_variable
; DWARF: DW_AT_location [DW_FORM_loclistx] (indexed (0x[[#%x,]]) loclist = 0x[[#%x,]]:
; DWARF:    [0x[[#%x,]], 0x[[#%x,]]) ".text": DW_OP_reg6 RBP, DW_OP_deref_size 0x8, DW_OP_consts -8, DW_OP_plus, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address
; DWARF:    [0x[[#%x,]], 0x[[#%x,]]) ".text": DW_OP_reg6 RBP, DW_OP_deref_size 0x8, DW_OP_consts -8, DW_OP_plus, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_piece 0x2, DW_OP_reg6 RBP, DW_OP_deref_size 0x8, DW_OP_consts -6, DW_OP_plus, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_piece 0x2)
; DWARF: DW_AT_name [DW_FORM_strx1] (indexed ([[#%x,]]) string = "var_fragmented")

; ModuleID = '<stdin>'
source_filename = "<stdin>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @ex()

; CHECK: @glob = {{.*}}, !dbg ![[#GLOB_GVE:]]
@glob = global i32 42, align 4, !dbg !0

; CHECK: @glob_fragmented.lo = {{.*}}, !dbg ![[#GLOB_FRAGMENTED_LO_GVE:]]
@glob_fragmented.lo = global i16 42, align 2, !dbg !23
; CHECK: @glob_fragmented.hi = {{.*}}, !dbg ![[#GLOB_FRAGMENTED_HI_GVE:]]
@glob_fragmented.hi = global i16 42, align 2, !dbg !24

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @func() #0 !dbg !13 {
entry:
  %var = alloca i32, align 4
  ; CHECK: #dbg_value(!DIArgList(ptr %var), ![[#]], !DIExpression(DIOpArg(0, ptr), DIOpFragment(1, 2), DIOpDeref(i32)),
    #dbg_value(!DIArgList(ptr %var), !18, !DIExpression(DIOpArg(0, ptr), DIOpFragment(1, 2), DIOpDeref(i32)), !19)
  ; CHECK: #dbg_value(ptr %var, ![[#VAR:]], !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)),
    #dbg_value(ptr %var, !18, !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)), !19)
  %var_fragmented.lo = alloca i16, align 2
  %var_fragmented.hi = alloca i16, align 2
  ; CHECK: #dbg_value(ptr %var_fragmented.lo, ![[#VAR_FRAGMENTED:]], !DIExpression(DIOpArg(0, ptr), DIOpDeref(i16), DIOpFragment(0, 16)),
    #dbg_value(ptr %var_fragmented.lo, !22, !DIExpression(DIOpArg(0, ptr), DIOpDeref(i16), DIOpFragment(0, 16)), !19)
  call void @ex()
  ; CHECK: #dbg_value(ptr %var_fragmented.hi, ![[#VAR_FRAGMENTED]], !DIExpression(DIOpArg(0, ptr), DIOpDeref(i16), DIOpFragment(16, 16)),
    #dbg_value(ptr %var_fragmented.hi, !22, !DIExpression(DIOpArg(0, ptr), DIOpDeref(i16), DIOpFragment(16, 16)), !19)
  ret void, !dbg !20
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "target-cpu"="x86-64" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!5, !6, !7, !8, !9, !10, !11}
!llvm.ident = !{!12}

; CHECK-DAG: ![[#GLOB_GVE]] = !DIGlobalVariableExpression(var: ![[#GLOB:]], expr: !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)))
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)))
; CHECK-DAG: ![[#GLOB]] = distinct !DIGlobalVariable(name: "glob",
!1 = distinct !DIGlobalVariable(name: "glob", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 19.0.0git (git@github.com:slinder1/llvm-project.git e4263955383c3e364bd752d02fc44cf5f22143ef)", isOptimized: false, runtimeVersion: 0, globals: !21, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "-", directory: "/home/slinder1/llvm-project/main", checksumkind: CSK_MD5, checksum: "9e51994790e4105fa7153a61c95a824f")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !{i32 7, !"Dwarf Version", i32 5}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 8, !"PIC Level", i32 2}
!9 = !{i32 7, !"PIE Level", i32 2}
!10 = !{i32 7, !"uwtable", i32 2}
!11 = !{i32 7, !"frame-pointer", i32 2}
!12 = !{!"clang version 19.0.0git (git@github.com:slinder1/llvm-project.git e4263955383c3e364bd752d02fc44cf5f22143ef)"}
!13 = distinct !DISubprogram(name: "func", scope: !14, file: !14, line: 15, type: !15, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
!14 = !DIFile(filename: "<stdin>", directory: "/home/slinder1/llvm-project/main", checksumkind: CSK_MD5, checksum: "9e51994790e4105fa7153a61c95a824f")
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{}
; CHECK-DAG: ![[#VAR]] = !DILocalVariable(name: "var",
!18 = !DILocalVariable(name: "var", scope: !13, file: !14, line: 16, type: !4)
!19 = !DILocation(line: 16, column: 9, scope: !13)
!20 = !DILocation(line: 17, column: 1, scope: !13)
!21 = !{!0, !23, !24}
; CHECK-DAG: ![[#VAR_FRAGMENTED]] = !DILocalVariable(name: "var_fragmented",
!22 = !DILocalVariable(name: "var_fragmented", scope: !13, file: !14, line: 16, type: !4)
; CHECK-DAG: ![[#GLOB_FRAGMENTED_LO_GVE]] = !DIGlobalVariableExpression(var: ![[#GLOB_FRAGMENTED:]], expr: !DIExpression(DIOpArg(0, ptr), DIOpDeref(i16), DIOpFragment(0, 16)))
!23 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression(DIOpArg(0, ptr), DIOpDeref(i16), DIOpFragment(0, 16)))
; CHECK-DAG: ![[#GLOB_FRAGMENTED_HI_GVE]] = !DIGlobalVariableExpression(var: ![[#GLOB_FRAGMENTED]], expr: !DIExpression(DIOpArg(0, ptr), DIOpDeref(i16), DIOpFragment(16, 16)))
!24 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression(DIOpArg(0, ptr), DIOpDeref(i16), DIOpFragment(16, 16)))
; CHECK-DAG: ![[#GLOB_FRAGMENTED]] = distinct !DIGlobalVariable(name: "glob_fragmented",
!25 = distinct !DIGlobalVariable(name: "glob_fragmented", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
