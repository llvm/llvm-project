; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"
; PR 9879

; CHECK: ##DEBUG_VALUE: tid <-
%0 = type { i8*, i8*, i8*, i8*, i32 }

@sgv = internal addrspace(2) constant [1 x i8] zeroinitializer
@fgv = internal addrspace(2) constant [1 x i8] zeroinitializer
@lvgv = internal constant [0 x i8*] zeroinitializer
@llvm.global.annotations = appending global [1 x %0] [%0 { i8* bitcast (void (i32 addrspace(1)*)* @__OpenCL_nbt02_kernel to i8*), i8* addrspacecast ([1 x i8] addrspace(2)* @sgv to i8*), i8* addrspacecast ([1 x i8] addrspace(2)* @fgv to i8*), i8* bitcast ([0 x i8*]* @lvgv to i8*), i32 0 }], section "llvm.metadata"

define void @__OpenCL_nbt02_kernel(i32 addrspace(1)* %ip) nounwind !dbg !0 {
entry:
  call void @llvm.dbg.value(metadata i32 addrspace(1)* %ip, i64 0, metadata !8, metadata !DIExpression()), !dbg !9
  %0 = call <4 x i32> @__amdil_get_local_id_int() nounwind
  %1 = extractelement <4 x i32> %0, i32 0
  br label %2

; <label>:2                                       ; preds = %entry
  %3 = phi i32 [ %1, %entry ]
  br label %4

; <label>:4                                       ; preds = %2
  %5 = phi i32 [ %3, %2 ]
  br label %get_local_id.exit

get_local_id.exit:                                ; preds = %4
  %6 = phi i32 [ %5, %4 ]
  call void @llvm.dbg.value(metadata i32 %6, i64 0, metadata !10, metadata !DIExpression()), !dbg !12
  %7 = call <4 x i32> @__amdil_get_global_id_int() nounwind, !dbg !12
  %8 = extractelement <4 x i32> %7, i32 0, !dbg !12
  br label %9

; <label>:9                                       ; preds = %get_local_id.exit
  %10 = phi i32 [ %8, %get_local_id.exit ]
  br label %11

; <label>:11                                      ; preds = %9
  %12 = phi i32 [ %10, %9 ]
  br label %get_global_id.exit

get_global_id.exit:                               ; preds = %11
  %13 = phi i32 [ %12, %11 ]
  call void @llvm.dbg.value(metadata i32 %13, i64 0, metadata !13, metadata !DIExpression()), !dbg !14
  %14 = call <4 x i32> @__amdil_get_local_size_int() nounwind
  %15 = extractelement <4 x i32> %14, i32 0
  br label %16

; <label>:16                                      ; preds = %get_global_id.exit
  %17 = phi i32 [ %15, %get_global_id.exit ]
  br label %18

; <label>:18                                      ; preds = %16
  %19 = phi i32 [ %17, %16 ]
  br label %get_local_size.exit

get_local_size.exit:                              ; preds = %18
  %20 = phi i32 [ %19, %18 ]
  call void @llvm.dbg.value(metadata i32 %20, i64 0, metadata !15, metadata !DIExpression()), !dbg !16
  %tmp5 = add i32 %6, %13, !dbg !17
  %tmp7 = add i32 %tmp5, %20, !dbg !17
  store i32 %tmp7, i32 addrspace(1)* %ip, align 4, !dbg !17
  br label %return, !dbg !17

return:                                           ; preds = %get_local_size.exit
  ret void, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare <4 x i32> @__amdil_get_local_size_int() nounwind

declare <4 x i32> @__amdil_get_local_id_int() nounwind

declare <4 x i32> @__amdil_get_global_id_int() nounwind

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!22}

!0 = distinct !DISubprogram(name: "__OpenCL_nbt02_kernel", linkageName: "__OpenCL_nbt02_kernel", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !2, file: !20, scope: !1, type: !3)
!1 = !DIFile(filename: "OCLlLwTXZ.cl", directory: "/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "clc", isOptimized: false, emissionKind: FullDebug, file: !20, enums: !21, retainedTypes: !21, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{null, !5}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align:64, scope: !2, baseType: !6)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint", file: !20, scope: !2, baseType: !7)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!8 = !DILocalVariable(name: "ip", line: 1, arg: 1, scope: !0, file: !1, type: !5)
!9 = !DILocation(line: 1, column: 32, scope: !0)
!10 = !DILocalVariable(name: "tid", line: 3, scope: !11, file: !1, type: !6)
!11 = distinct !DILexicalBlock(line: 2, column: 1, file: !1, scope: !0)
!12 = !DILocation(line: 5, column: 24, scope: !11)
!13 = !DILocalVariable(name: "gid", line: 3, scope: !11, file: !1, type: !6)
!14 = !DILocation(line: 6, column: 25, scope: !11)
!15 = !DILocalVariable(name: "lsz", line: 3, scope: !11, file: !1, type: !6)
!16 = !DILocation(line: 7, column: 26, scope: !11)
!17 = !DILocation(line: 9, column: 24, scope: !11)
!18 = !DILocation(line: 10, column: 1, scope: !0)
!20 = !DIFile(filename: "OCLlLwTXZ.cl", directory: "/tmp")
!21 = !{}
!22 = !{i32 1, !"Debug Info Version", i32 3}
