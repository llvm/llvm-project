; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji < %s | FileCheck %s

; Make sure we do not crash during scheduling when DBG_VALUE is the first
; instruction in the basic block.

; LLVM IR generated with the following command and OpenCL source:
;
; $clang -cl-std=CL2.0 -g -O2 -target amdgcn-amd-amdhsa -S -emit-llvm <path-to-file>
;
; kernel void kernel1(global int *A, global int *B) {
;   if (*A == 1) {
;     *B = 12;
;   }
;   if (*A == 2) {
;     *B = 13;
;   }
; }

declare void @llvm.dbg.value(metadata, metadata, metadata)

; CHECK-LABEL: {{^}}kernel1:
define amdgpu_kernel void @kernel1(
    i32 addrspace(1)* nocapture readonly %A,
    i32 addrspace(1)* nocapture %B) !dbg !7  {
entry:
  tail call void @llvm.dbg.value(metadata i32 addrspace(1)* %A, metadata !13, metadata !19), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 addrspace(1)* %B, metadata !14, metadata !19), !dbg !21
  %0 = load i32, i32 addrspace(1)* %A, align 4, !dbg !22, !tbaa !24
  %cmp = icmp eq i32 %0, 1, !dbg !28
  br i1 %cmp, label %if.then, label %if.end, !dbg !29

if.then:                                          ; preds = %entry
  store i32 12, i32 addrspace(1)* %B, align 4, !dbg !30, !tbaa !24
  %.pr = load i32, i32 addrspace(1)* %A, align 4, !dbg !32, !tbaa !24
  br label %if.end, !dbg !34

if.end:                                           ; preds = %if.then, %entry
  %1 = phi i32 [ %.pr, %if.then ], [ %0, %entry ], !dbg !32
  %cmp1 = icmp eq i32 %1, 2, !dbg !35
  br i1 %cmp1, label %if.then2, label %if.end3, !dbg !36

if.then2:                                         ; preds = %if.end
  store i32 13, i32 addrspace(1)* %B, align 4, !dbg !37, !tbaa !24
  br label %if.end3, !dbg !39

if.end3:                                          ; preds = %if.then2, %if.end
  ret void, !dbg !40
}

!llvm.dbg.cu = !{!0}
!opencl.ocl.version = !{!3}
!llvm.module.flags = !{!4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "dbg-value-sched-crash.cl", directory: "/some/random/directory")
!2 = !{}
!3 = !{i32 2, i32 0}
!4 = !{i32 2, !"Dwarf Version", i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!"clang version 4.0 "}
!7 = distinct !DISubprogram(name: "kernel1", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "A", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "B", arg: 2, scope: !7, file: !1, line: 1, type: !10)
!15 = !{i32 1, i32 1}
!16 = !{!"none", !"none"}
!17 = !{!"int*", !"int*"}
!18 = !{!"", !""}
!19 = !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)
!20 = !DILocation(line: 1, column: 33, scope: !7)
!21 = !DILocation(line: 1, column: 48, scope: !7)
!22 = !DILocation(line: 2, column: 7, scope: !23)
!23 = distinct !DILexicalBlock(scope: !7, file: !1, line: 2, column: 7)
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C/C++ TBAA"}
!28 = !DILocation(line: 2, column: 10, scope: !23)
!29 = !DILocation(line: 2, column: 7, scope: !7)
!30 = !DILocation(line: 3, column: 8, scope: !31)
!31 = distinct !DILexicalBlock(scope: !23, file: !1, line: 2, column: 16)
!32 = !DILocation(line: 5, column: 7, scope: !33)
!33 = distinct !DILexicalBlock(scope: !7, file: !1, line: 5, column: 7)
!34 = !DILocation(line: 4, column: 3, scope: !31)
!35 = !DILocation(line: 5, column: 10, scope: !33)
!36 = !DILocation(line: 5, column: 7, scope: !7)
!37 = !DILocation(line: 6, column: 8, scope: !38)
!38 = distinct !DILexicalBlock(scope: !33, file: !1, line: 5, column: 16)
!39 = !DILocation(line: 7, column: 3, scope: !38)
!40 = !DILocation(line: 8, column: 1, scope: !7)
