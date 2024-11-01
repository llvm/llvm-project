; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   #define __tag(x) __attribute__((btf_decl_tag(x)))
;
;   extern void foo(int x __tag("x_tag"), int y __tag("y_tag")) __tag("foo_tag");
;
;   void root(void) {
;     foo(0, 0);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c


; Function Attrs: nounwind
define dso_local void @root() local_unnamed_addr #0 !dbg !7 {
entry:
  tail call void @foo(i32 noundef 0, i32 noundef 0) #2, !dbg !12
  ret void, !dbg !13
}

declare !dbg !14 dso_local void @foo(i32 noundef, i32 noundef) local_unnamed_addr #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 16.0.0 (https://github.com/llvm/llvm-project.git 603e8490729e477680f0bc8284e136ceeb66e7f4)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "fake-file-name.c", directory: "fake-directory", checksumkind: CSK_MD5, checksum: "00000000000000000000000000000000")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 16.0.0 (https://github.com/llvm/llvm-project.git 603e8490729e477680f0bc8284e136ceeb66e7f4)"}
!7 = distinct !DISubprogram(name: "root", scope: !8, file: !8, line: 5, type: !9, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DIFile(filename: "fake-file-name.c", directory: "fake-directory", checksumkind: CSK_MD5, checksum: "00000000000000000000000000000000")
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !{}
!12 = !DILocation(line: 6, column: 3, scope: !7)
!13 = !DILocation(line: 7, column: 1, scope: !7)
!14 = !DISubprogram(name: "foo", scope: !8, file: !8, line: 3, type: !15, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !18, annotations: !25)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17, !17}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !{!19, !22}
!19 = !DILocalVariable(name: "x", arg: 1, scope: !14, file: !8, line: 3, type: !17, annotations: !20)
!20 = !{!21}
!21 = !{!"btf_decl_tag", !"x_tag"}
!22 = !DILocalVariable(name: "y", arg: 2, scope: !14, file: !8, line: 3, type: !17, annotations: !23)
!23 = !{!24}
!24 = !{!"btf_decl_tag", !"y_tag"}
!25 = !{!26}
!26 = !{!"btf_decl_tag", !"foo_tag"}

; CHECK: 	.long	0                               # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT: 	.long	218103808                       # 0xd000000
; CHECK-NEXT: 	.long	0
; CHECK-NEXT: 	.long	1                               # BTF_KIND_FUNC(id = 2)
; CHECK-NEXT: 	.long	201326593                       # 0xc000001
; CHECK-NEXT: 	.long	1
; CHECK-NEXT: 	.long	0                               # BTF_KIND_FUNC_PROTO(id = 3)
; CHECK-NEXT: 	.long	218103810                       # 0xd000002
; CHECK-NEXT: 	.long	0
; CHECK-NEXT: 	.long	0
; CHECK-NEXT: 	.long	4
; CHECK-NEXT: 	.long	0
; CHECK-NEXT: 	.long	4
; CHECK-NEXT: 	.long	44                              # BTF_KIND_INT(id = 4)
; CHECK-NEXT: 	.long	16777216                        # 0x1000000
; CHECK-NEXT: 	.long	4
; CHECK-NEXT: 	.long	16777248                        # 0x1000020
; CHECK-NEXT: 	.long	48                              # BTF_KIND_FUNC(id = 5)
; CHECK-NEXT: 	.long	201326594                       # 0xc000002
; CHECK-NEXT: 	.long	3
; CHECK-NEXT: 	.long	52                              # BTF_KIND_DECL_TAG(id = 6)
; CHECK-NEXT: 	.long	285212672                       # 0x11000000
; CHECK-NEXT: 	.long	5
; CHECK-NEXT: 	.long	0
; CHECK-NEXT: 	.long	58                              # BTF_KIND_DECL_TAG(id = 7)
; CHECK-NEXT: 	.long	285212672                       # 0x11000000
; CHECK-NEXT: 	.long	5
; CHECK-NEXT: 	.long	1
; CHECK-NEXT: 	.long	64                              # BTF_KIND_DECL_TAG(id = 8)
; CHECK-NEXT: 	.long	285212672                       # 0x11000000
; CHECK-NEXT: 	.long	5
; CHECK-NEXT: 	.long	4294967295

; CHECK:	.ascii	"int"                           # string offset=44
; CHECK:	.ascii	"foo"                           # string offset=48
; CHECK: 	.ascii	"x_tag"                         # string offset=52
; CHECK: 	.ascii	"y_tag"                         # string offset=58
; CHECK: 	.ascii	"foo_tag"                       # string offset=64
