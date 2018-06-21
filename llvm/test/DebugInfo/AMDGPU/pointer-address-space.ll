; RUN: llc -O0 -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s

; LLVM IR generated with the following command and OpenCL source:
;
; $clang -cl-std=CL2.0 -g -O0 -target amdgcn-amd-amdhsa -S -emit-llvm <path-to-file>
;
; kernel void kernel1() {
;   global int *FuncVar0 = 0;
;   constant int *FuncVar1 = 0;
;   local int *FuncVar2 = 0;
;   private int *FuncVar3 = 0;
;   int *FuncVar4 = 0;
; }

; CHECK:      DW_AT_name {{.*}}"FuncVar0"
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + 0x{{[a-f0-9]+}} => {0x[[NONE:[a-f0-9]+]]}

; CHECK:      DW_AT_name {{.*}}"FuncVar1"
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + 0x{{[a-f0-9]+}} => {0x[[NONE]]}

; CHECK:      DW_AT_name {{.*}}"FuncVar2"
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line
; CHECK-NEXT:      DW_AT_type [DW_FORM_ref4] (cu + 0x{{[a-f0-9]+}} => {0x[[LOCAL:[a-f0-9]+]]}

; CHECK:      DW_AT_name {{.*}}"FuncVar3"
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + 0x{{[a-f0-9]+}} => {0x[[PRIVATE:[a-f0-9]+]]}

; CHECK:      DW_AT_name {{.*}}"FuncVar4"
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + 0x{{[a-f0-9]+}} => {0x[[NONE]]}

; CHECK:      0x[[NONE]]: DW_TAG_pointer_type
; CHECK-NEXT:               DW_AT_type
; CHECK-NOT:                DW_AT_address_class

; CHECK:      0x[[LOCAL]]: DW_TAG_pointer_type
; CHECK-NEXT:                DW_AT_type
; CHECK-NEXT:                DW_AT_address_class [DW_FORM_data4] (0x00000002)

; CHECK:      0x[[PRIVATE]]: DW_TAG_pointer_type
; CHECK-NEXT:                  DW_AT_type
; CHECK-NEXT:                  DW_AT_address_class [DW_FORM_data4] (0x00000001)

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define amdgpu_kernel void @kernel1() !dbg !7 {
entry:
  %FuncVar0 = alloca i32 addrspace(1)*, align 4
  %FuncVar1 = alloca i32 addrspace(2)*, align 4
  %FuncVar2 = alloca i32 addrspace(3)*, align 4
  %FuncVar3 = alloca i32*, align 4
  %FuncVar4 = alloca i32 addrspace(4)*, align 4
  call void @llvm.dbg.declare(metadata i32 addrspace(1)** %FuncVar0, metadata !10, metadata !13), !dbg !14
  store i32 addrspace(1)* null, i32 addrspace(1)** %FuncVar0, align 4, !dbg !14
  call void @llvm.dbg.declare(metadata i32 addrspace(2)** %FuncVar1, metadata !15, metadata !13), !dbg !16
  store i32 addrspace(2)* null, i32 addrspace(2)** %FuncVar1, align 4, !dbg !16
  call void @llvm.dbg.declare(metadata i32 addrspace(3)** %FuncVar2, metadata !17, metadata !13), !dbg !19
  store i32 addrspace(3)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(3)*), i32 addrspace(3)** %FuncVar2, align 4, !dbg !19
  call void @llvm.dbg.declare(metadata i32** %FuncVar3, metadata !20, metadata !13), !dbg !22
  store i32* addrspacecast (i32 addrspace(4)* null to i32*), i32** %FuncVar3, align 4, !dbg !22
  call void @llvm.dbg.declare(metadata i32 addrspace(4)** %FuncVar4, metadata !23, metadata !13), !dbg !24
  store i32 addrspace(4)* null, i32 addrspace(4)** %FuncVar4, align 4, !dbg !24
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}
!opencl.ocl.version = !{!3}
!llvm.module.flags = !{!4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "pointer-address-space.ll", directory: "/some/random/directory")
!2 = !{}
!3 = !{i32 2, i32 0}
!4 = !{i32 2, !"Dwarf Version", i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!""}
!7 = distinct !DISubprogram(name: "kernel1", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "FuncVar0", scope: !7, file: !1, line: 2, type: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIExpression()
!14 = !DILocation(line: 2, column: 15, scope: !7)
!15 = !DILocalVariable(name: "FuncVar1", scope: !7, file: !1, line: 3, type: !11)
!16 = !DILocation(line: 3, column: 17, scope: !7)
!17 = !DILocalVariable(name: "FuncVar2", scope: !7, file: !1, line: 4, type: !18)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 32, dwarfAddressSpace: 2)
!19 = !DILocation(line: 4, column: 14, scope: !7)
!20 = !DILocalVariable(name: "FuncVar3", scope: !7, file: !1, line: 5, type: !21)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 32, dwarfAddressSpace: 1)
!22 = !DILocation(line: 5, column: 16, scope: !7)
!23 = !DILocalVariable(name: "FuncVar4", scope: !7, file: !1, line: 6, type: !11)
!24 = !DILocation(line: 6, column: 8, scope: !7)
!25 = !DILocation(line: 7, column: 1, scope: !7)
