; RUN: sed -e "s,SRC_COMPDIR,%/p/Inputs,g" %s > %t.ll
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj -O0 -o %t.o %t.ll
; RUN: llvm-objdump --triple=amdgcn-amd-amdhsa --mcpu=gfx802 -d -l %t.o | FileCheck --check-prefix=LINE %t.ll
; RUN: llvm-objdump --triple=amdgcn-amd-amdhsa --mcpu=gfx802 -d -S %t.o | FileCheck --check-prefix=SOURCE %t.ll

; Prologue.
; LINE:      source_lines_test{{>?}}:
; LINE-NEXT: ; source_lines_test():
; LINE-NEXT: ; {{.*}}source-lines.cl:1
; Kernel.
; LINE: v_mov_b32_e32 v{{[0-9]+}}, 0x777
; LINE: ; {{.*}}source-lines.cl:2
; LINE: v_mov_b32_e32 v{{[0-9]+}}, 0x888
; LINE: ; {{.*}}source-lines.cl:3
; LINE: ; {{.*}}source-lines.cl:4
; LINE: v_add_u32_e64
; LINE: ; {{.*}}source-lines.cl:5
; LINE: flat_store_dword
; Epilogue.
; LINE:      ; {{.*}}source-lines.cl:6
; LINE-NEXT: s_endpgm

; Prologue.
; SOURCE:      source_lines_test{{>?}}:
; SOURCE-NEXT: ; kernel void source_lines_test(global int *Out) {
; Kernel.
; SOURCE: v_mov_b32_e32 v{{[0-9]+}}, 0x777
; SOURCE: ; int var0 = 0x777;
; SOURCE: v_mov_b32_e32 v{{[0-9]+}}, 0x888
; SOURCE: ; int var1 = 0x888;
; SOURCE: ; int var2 = var0 + var1;
; SOURCE: v_add_u32_e64
; SOURCE: ; *Out = var2;
; SOURCE: flat_store_dword
; Epilogue.
; SOURCE:      ; }
; SOURCE-NEXT: s_endpgm

; ModuleID = 'source-lines.cl'
source_filename = "source-lines.cl"
target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: noinline nounwind
define amdgpu_kernel void @source_lines_test(i32 addrspace(1)* %Out) #0 !dbg !7 !kernel_arg_addr_space !12 !kernel_arg_access_qual !13 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !15 {
entry:
  %Out.addr = alloca i32 addrspace(1)*, align 4, addrspace(5)
  %var0 = alloca i32, align 4, addrspace(5)
  %var1 = alloca i32, align 4, addrspace(5)
  %var2 = alloca i32, align 4, addrspace(5)
  store i32 addrspace(1)* %Out, i32 addrspace(1)* addrspace(5)* %Out.addr, align 4
  call void @llvm.dbg.declare(metadata i32 addrspace(1)* addrspace(5)* %Out.addr, metadata !16, metadata !17), !dbg !18
  call void @llvm.dbg.declare(metadata i32 addrspace(5)* %var0, metadata !19, metadata !17), !dbg !20
  store i32 1911, i32 addrspace(5)* %var0, align 4, !dbg !20
  call void @llvm.dbg.declare(metadata i32 addrspace(5)* %var1, metadata !21, metadata !17), !dbg !22
  store i32 2184, i32 addrspace(5)* %var1, align 4, !dbg !22
  call void @llvm.dbg.declare(metadata i32 addrspace(5)* %var2, metadata !23, metadata !17), !dbg !24
  %0 = load i32, i32 addrspace(5)* %var0, align 4, !dbg !25
  %1 = load i32, i32 addrspace(5)* %var1, align 4, !dbg !26
  %add = add nsw i32 %0, %1, !dbg !27
  store i32 %add, i32 addrspace(5)* %var2, align 4, !dbg !24
  %2 = load i32, i32 addrspace(5)* %var2, align 4, !dbg !28
  %3 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(5)* %Out.addr, align 4, !dbg !29
  store i32 %2, i32 addrspace(1)* %3, align 4, !dbg !30
  ret void, !dbg !31
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!opencl.ocl.version = !{!3}
!llvm.module.flags = !{!4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "source-lines.cl", directory: "SRC_COMPDIR")
!2 = !{}
!3 = !{i32 1, i32 0}
!4 = !{i32 2, !"Dwarf Version", i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!"clang version 5.0.0"}
!7 = distinct !DISubprogram(name: "source_lines_test", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{i32 1}
!13 = !{!"none"}
!14 = !{!"int*"}
!15 = !{!""}
!16 = !DILocalVariable(name: "Out", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!17 = !DIExpression()
!18 = !DILocation(line: 1, column: 43, scope: !7)
!19 = !DILocalVariable(name: "var0", scope: !7, file: !1, line: 2, type: !11)
!20 = !DILocation(line: 2, column: 7, scope: !7)
!21 = !DILocalVariable(name: "var1", scope: !7, file: !1, line: 3, type: !11)
!22 = !DILocation(line: 3, column: 7, scope: !7)
!23 = !DILocalVariable(name: "var2", scope: !7, file: !1, line: 4, type: !11)
!24 = !DILocation(line: 4, column: 7, scope: !7)
!25 = !DILocation(line: 4, column: 14, scope: !7)
!26 = !DILocation(line: 4, column: 21, scope: !7)
!27 = !DILocation(line: 4, column: 19, scope: !7)
!28 = !DILocation(line: 5, column: 10, scope: !7)
!29 = !DILocation(line: 5, column: 4, scope: !7)
!30 = !DILocation(line: 5, column: 8, scope: !7)
!31 = !DILocation(line: 6, column: 1, scope: !7)
