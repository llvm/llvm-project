; RUN: opt -S -O0 < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-RETAINED %s
; RUN: opt -S -O1 < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-STRIPPED %s
; RUN: opt -S -O2 < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-STRIPPED %s
; RUN: opt -S -O3 < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-STRIPPED %s
; RUN: opt -S -disable-heterogeneous-debug-verify -O3 < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-RETAINED %s
; RUN: llc -stop-before=amdgpu-isel -O0 < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-RETAINED %s
; RUN: llc -stop-before=amdgpu-isel -O1 < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-STRIPPED %s
; RUN: llc -stop-before=amdgpu-isel -O2 < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-STRIPPED %s
; RUN: llc -stop-before=amdgpu-isel -O3 < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-STRIPPED %s
; RUN: llc -disable-heterogeneous-debug-verify -stop-before=amdgpu-isel -O3 < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-RETAINED %s
; RUN: opt -S -passes="default<O0>" < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-RETAINED %s
; RUN: opt -S -passes="default<O1>" < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-STRIPPED %s
; RUN: opt -S -passes="default<O2>" < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-STRIPPED %s
; RUN: opt -S -passes="default<O3>" < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-STRIPPED %s
; RUN: opt -S -disable-heterogeneous-debug-verify -passes="default<O3>" < %s 2>&1 | FileCheck --check-prefixes=PRE-ISEL-COMMON,PRE-ISEL-RETAINED %s
; RUN: llc -O0 --mcpu=gfx900 --filetype=obj < %s 2>/dev/null | llvm-dwarfdump --all - 2>/dev/null | FileCheck --check-prefixes=DWARFDUMP-COMMON,DWARFDUMP-RETAINED %s
; RUN: llc -O1 --mcpu=gfx900 --filetype=obj < %s 2>/dev/null | llvm-dwarfdump --all - 2>/dev/null | FileCheck --check-prefixes=DWARFDUMP-COMMON,DWARFDUMP-STRIPPED %s
; RUN: llc -O2 --mcpu=gfx900 --filetype=obj < %s 2>/dev/null | llvm-dwarfdump --all - 2>/dev/null | FileCheck --check-prefixes=DWARFDUMP-COMMON,DWARFDUMP-STRIPPED %s
; RUN: llc -O3 --mcpu=gfx900 --filetype=obj < %s 2>/dev/null | llvm-dwarfdump --all - 2>/dev/null | FileCheck --check-prefixes=DWARFDUMP-COMMON,DWARFDUMP-STRIPPED %s
; RUN: llc -disable-heterogeneous-debug-verify -O3 --mcpu=gfx900 --filetype=obj < %s 2>/dev/null | llvm-dwarfdump --all - 2>/dev/null | FileCheck --check-prefixes=DWARFDUMP-COMMON,DWARFDUMP-RETAINED %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; PRE-ISEL-RETAINED: @global = protected local_unnamed_addr addrspace(1) externally_initialized global i32 0, align 4, !dbg.def !0
; PRE-ISEL-STRIPPED: @global = protected local_unnamed_addr addrspace(1) externally_initialized global i32 0, align 4{{$}}
@global = protected local_unnamed_addr addrspace(1) externally_initialized global i32 0, align 4, !dbg.def !0

define hidden void @_Z6kernelv() #0 !dbg !14 {
entry:
  %local = alloca i32, align 4, addrspace(5)
  %local.ascast = addrspacecast ptr addrspace(5) %local to ptr
  ; PRE-ISEL-RETAINED: call void @llvm.dbg.def(metadata !19, metadata ptr addrspace(5) %local), !dbg ![[#DEF_LOC:]]
  ; PRE-ISEL-STRIPPED-NOT: call void @llvm.dbg.def(metadata
  call void @llvm.dbg.def(metadata !19, metadata ptr addrspace(5) %local), !dbg !20
  ; PRE-ISEL-COMMON: ret void, !dbg ![[#RET_LOC:]]
  ret void, !dbg !21
}

declare void @llvm.dbg.def(metadata, metadata) #1

attributes #0 = { convergent mustprogress noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx900" "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!1}
; PRE-ISEL-RETAINED: !llvm.dbg.retainedNodes = !{![[#]]}
; PRE-ISEL-STRIPPED-NOT: !llvm.dbg.retainedNodes
!llvm.dbg.retainedNodes = !{!3}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12}
!llvm.ident = !{!13}


; PRE-ISEL-RETAINED: ![[#GLOBAL_FRAGMENT:]] = distinct !DIFragment()
; PRE-ISEL-STRIPPED-NOT: ![[#]] = distinct !DIFragment()
; PRE-ISEL-COMMON: distinct !DICompileUnit
; PRE-ISEL-STRIPPED-SAME: globals: ![[#GLOBAL_EXPR_ARRAY:]]
; PRE-ISEL-STRIPPED-DAG: ![[#GLOBAL_EXPR_ARRAY]] = !{![[#GLOBAL_EXPR:]]}
; PRE-ISEL-STRIPPED-DAG: ![[#GLOBAL_EXPR]] = !DIGlobalVariableExpression(var: ![[#GLOBAL_VARIABLE:]], expr: !DIExpression())
!0 = distinct !DIFragment()
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "-", directory: "/")
; PRE-ISEL-RETAINED: ![[#]] = distinct !DILifetime(object: ![[#GLOBAL_VARIABLE:]], location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(i32)), argObjects: {![[#GLOBAL_FRAGMENT]]})
; PRE-ISEL-STRIPPED-NOT: ![[#]] = distinct !DILifetime{{.*}}
!3 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpArg(0, ptr addrspace(1)), DIOpDeref(i32)), argObjects: {!0})
; PRE-ISEL-COMMON: ![[#GLOBAL_VARIABLE]] = distinct !DIGlobalVariable(name: "global",
!4 = distinct !DIGlobalVariable(name: "global", scope: !1, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
!5 = !DIFile(filename: "<stdin>", directory: "/")
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 1, !"amdgpu_code_object_version", i32 400}
!8 = !{i32 7, !"Dwarf Version", i32 5}
; PRE-ISEL-RETAINED: ![[#]] = !{i32 2, !"Debug Info Version", i32 4}
; PRE-ISEL-STRIPPED: ![[#]] = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 2, !"Debug Info Version", i32 4}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 8, !"PIC Level", i32 2}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{!"clang"}
; PRE-ISEL-COMMON: ![[#]] = distinct !DISubprogram(name: "kernel"{{.*}}retainedNodes: ![[#RETAINED_NODES:]])
!14 = distinct !DISubprogram(name: "kernel", linkageName: "_Z6kernelv", scope: !5, file: !5, line: 1, type: !15, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
; PRE-ISEL-COMMON: ![[#RETAINED_NODES]] = !{![[#]]}
!17 = !{!18}
; PRE-ISEL-COMMON: ![[#LOCAL_VARIABLE:]] = !DILocalVariable(name: "local",
!18 = !DILocalVariable(name: "local", scope: !14, file: !5, line: 1, type: !6)
; PRE-ISEL-RETAINED: ![[#]] = distinct !DILifetime(object: ![[#LOCAL_VARIABLE:]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
; PRE-ISEL-STRIPPED-NOT: ![[#]] = distinct !DILifetime{{.*}}
!19 = distinct !DILifetime(object: !18, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
; PRE-ISEL-RETAINED: ![[#DEF_LOC]] = !DILocation(line: 1, column: 81,
; PRE-ISEL-STRIPPED-NOT: ![[#]] = !DILocation(line: 1, column: 81,
!20 = !DILocation(line: 1, column: 81, scope: !14)
; PRE-ISEL-COMMON: ![[#RET_LOC]] = !DILocation(line: 1, column: 88,
!21 = !DILocation(line: 1, column: 88, scope: !14)

; DWARFDUMP-COMMON: 0x[[#%x,]]: DW_TAG_variable
; DWARFDUMP-COMMON:   DW_AT_name      ("global")
; DWARFDUMP-COMMON:   DW_AT_LLVM_memory_space (DW_MSPACE_LLVM_global)
; DWARFDUMP-RETAINED:   DW_AT_location  ({{.+}})
; DWARFDUMP-STRIPPED-NOT:   DW_AT_location  ({{.+}})

; DWARFDUMP-COMMON: 0x[[#%x,]]: DW_TAG_variable
; DWARFDUMP-RETIANED:   DW_AT_location ({{.+}})
; DWARFDUMP-STRIPPED-NOT:   DW_AT_location  ({{.+}})
; DWARFDUMP-COMMON:   DW_AT_name ("local")

; DWARFDUMP-COMMON: .debug_line contents:
; DWARFDUMP-COMMON-NEXT: debug_line[0x00000000]
; DWARFDUMP-COMMON-NEXT: Line table prologue:
; DWARFDUMP-COMMON-NEXT:     total_length: 0x00000048
; DWARFDUMP-COMMON-NEXT:           format: DWARF32
; DWARFDUMP-COMMON-NEXT:          version: 5
; DWARFDUMP-COMMON-NEXT:     address_size: 8
; DWARFDUMP-COMMON-NEXT:  seg_select_size: 0
; DWARFDUMP-COMMON-NEXT:  prologue_length: 0x0000002a
; DWARFDUMP-COMMON-NEXT:  min_inst_length: 4
; DWARFDUMP-COMMON-NEXT: max_ops_per_inst: 1
; DWARFDUMP-COMMON-NEXT:  default_is_stmt: 1
; DWARFDUMP-COMMON-NEXT:        line_base: -5
; DWARFDUMP-COMMON-NEXT:       line_range: 14
; DWARFDUMP-COMMON-NEXT:      opcode_base: 13
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_copy] = 0
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_advance_pc] = 1
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_advance_line] = 1
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_set_file] = 1
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_set_column] = 1
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_negate_stmt] = 0
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_set_basic_block] = 0
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_const_add_pc] = 0
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 1
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_set_prologue_end] = 0
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_set_epilogue_begin] = 0
; DWARFDUMP-COMMON-NEXT: standard_opcode_lengths[DW_LNS_set_isa] = 1
; DWARFDUMP-COMMON-NEXT: include_directories[  0] = "/"
; DWARFDUMP-COMMON-NEXT: file_names[  0]:
; DWARFDUMP-COMMON-NEXT:            name: "-"
; DWARFDUMP-COMMON-NEXT:       dir_index: 0
; DWARFDUMP-COMMON-NEXT: file_names[  1]:
; DWARFDUMP-COMMON-NEXT:            name: "<stdin>"
; DWARFDUMP-COMMON-NEXT:       dir_index: 0
; DWARFDUMP-COMMON-EMPTY:
; DWARFDUMP-COMMON-NEXT: Address            Line   Column File   ISA Discriminator OpIndex  Flags
; DWARFDUMP-COMMON-NEXT: ------------------ ------ ------ ------ --- -------------  ------- -------------
; DWARFDUMP-COMMON-NEXT: 0x[[#%x,]]              1      0      1   0             0 0  is_stmt
; DWARFDUMP-COMMON-NEXT: 0x[[#%x,]]              1     88      1   0             0 0  is_stmt prologue_end epilogue_begin
; DWARFDUMP-COMMON-NEXT: 0x[[#%x,]]              1     88      1   0             0 0  is_stmt end_sequence
