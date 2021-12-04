; REQUIRES: x86-registered-target
; RUN: llc -filetype=obj -mtriple=x86_64-unknown-linux-gnu -o %t %s
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; Source:
;   #define __tag1 __attribute__((btf_decl_tag("tag1")))
;   #define __tag2 __attribute__((btf_decl_tag("tag2")))
;
;   struct t1 {
;     int a __tag1 __tag2;
;   } __tag1 __tag2;
;
;   int g1 __tag1 __tag2;
;
;   int __tag1 __tag2 foo(struct t1 *arg __tag1 __tag2) {
;     return arg->a;
;   }
; Compilation flag:
;   clang -target x86_64 -g -S -emit-llvm t.c

%struct.t1 = type { i32 }

@g1 = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @foo(%struct.t1* %arg) #0 !dbg !16 {
entry:
  %arg.addr = alloca %struct.t1*, align 8
  store %struct.t1* %arg, %struct.t1** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.t1** %arg.addr, metadata !23, metadata !DIExpression()), !dbg !24
  %0 = load %struct.t1*, %struct.t1** %arg.addr, align 8, !dbg !25
  %a = getelementptr inbounds %struct.t1, %struct.t1* %0, i32 0, i32 0, !dbg !26
  %1 = load i32, i32* %a, align 4, !dbg !26
  ret i32 %1, !dbg !27
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g1", scope: !2, file: !3, line: 8, type: !6, isLocal: false, isDefinition: true, annotations: !7)

; CHECK:       DW_TAG_variable
; CHECK-NEXT:     DW_AT_name      ("g1")
; CHECK:          DW_TAG_LLVM_annotation
; CHECK-NEXT:       DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:       DW_AT_const_value     ("tag1")
; CHECK-EMPTY:
; CHECK-NEXT:     DW_TAG_LLVM_annotation
; CHECK-NEXT:       DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:       DW_AT_const_value     ("tag2")
; CHECK-EMPTY:
; CHECK-NEXT:     NULL

!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 305231a4f71b68945b4dd92925c76ff49e377c86)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp/home/yhs/work/tests/llvm/btf_tag")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{!8, !9}
!8 = !{!"btf_decl_tag", !"tag1"}
!9 = !{!"btf_decl_tag", !"tag2"}
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"uwtable", i32 1}
!14 = !{i32 7, !"frame-pointer", i32 2}
!15 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 305231a4f71b68945b4dd92925c76ff49e377c86)"}
!16 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 10, type: !17, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4, annotations: !7)

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_name      ("foo")
; CHECK:        DW_TAG_formal_parameter
; CHECK:          DW_TAG_LLVM_annotation
; CHECK-NEXT:       DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:       DW_AT_const_value     ("tag1")
; CHECK-EMPTY:
; CHECK-NEXT:     DW_TAG_LLVM_annotation
; CHECK-NEXT:       DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:       DW_AT_const_value     ("tag2")
; CHECK-EMPTY:
; CHECK-NEXT:     NULL
; CHECK-EMPTY:
; CHECK-NEXT:   DW_TAG_LLVM_annotation
; CHECK-NEXT:     DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:     DW_AT_const_value     ("tag1")
; CHECK-EMPTY:
; CHECK-NEXT:   DW_TAG_LLVM_annotation
; CHECK-NEXT:     DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:     DW_AT_const_value     ("tag2")
; CHECK-EMPTY:
; CHECK-NEXT:   NULL

!17 = !DISubroutineType(types: !18)
!18 = !{!6, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !3, line: 4, size: 32, elements: !21, annotations: !7)
!21 = !{!22}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !20, file: !3, line: 5, baseType: !6, size: 32, annotations: !7)

; CHECK:      DW_TAG_structure_type
; CHECK-NEXT:   DW_AT_name      ("t1")
; CHECK:        DW_TAG_member
; CHECK-NEXT:     DW_AT_name      ("a")
; CHECK:          DW_TAG_LLVM_annotation
; CHECK-NEXT:       DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:       DW_AT_const_value     ("tag1")
; CHECK-EMPTY:
; CHECK-NEXT:     DW_TAG_LLVM_annotation
; CHECK-NEXT:       DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:       DW_AT_const_value     ("tag2")
; CHECK-EMPTY:
; CHECK-NEXT:     NULL
; CHECK-EMPTY:
; CHECK-NEXT:   DW_TAG_LLVM_annotation
; CHECK-NEXT:     DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:     DW_AT_const_value     ("tag1")
; CHECK-EMPTY:
; CHECK-NEXT:   DW_TAG_LLVM_annotation
; CHECK-NEXT:     DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:     DW_AT_const_value     ("tag2")
; CHECK-EMPTY:
; CHECK-NEXT:   NULL

!23 = !DILocalVariable(name: "arg", arg: 1, scope: !16, file: !3, line: 10, type: !19, annotations: !7)
!24 = !DILocation(line: 10, column: 48, scope: !16)
!25 = !DILocation(line: 11, column: 10, scope: !16)
!26 = !DILocation(line: 11, column: 15, scope: !16)
!27 = !DILocation(line: 11, column: 3, scope: !16)
