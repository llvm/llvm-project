; RUN: llc -mtriple=x86_64-linux -split-dwarf-cross-cu-references -split-dwarf-file=foo.dwo -filetype=obj -o %t < %s
; RUN: llvm-objdump -r %t | FileCheck %s
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck --check-prefix=ALL --check-prefix=INFO --check-prefix=DWO --check-prefix=CROSS %s
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck --check-prefix=ALL --check-prefix=INFO %s

; RUN: llc -mtriple=x86_64-linux -split-dwarf-file=foo.dwo -filetype=obj -o %t < %s
; RUN: llvm-objdump -r %t | FileCheck %s
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck --check-prefix=ALL --check-prefix=DWO --check-prefix=NOCROSS %s
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck --check-prefix=ALL --check-prefix=INFO %s

; Testing cross-CU references for types, subprograms, and variables
; Built from code something like this:
; foo.cpp:
;   struct t1 { int i; };
;   void f();
;   __attribute__((always_inline)) void f1(t1 t) {
;     f();
;   }
;   void foo(t1 t) {
;     f1(t);
;   }
; bar.cpp:
;   struct t1 { int i; };
;   void f1(t1);
;   void bar(t1 t) {
;     f1(t);
;   }
; $ clang++-tot -emit-llvm -S {foo,bar}.cpp -g
; $ llvm-link-tot {foo,bar}.ll -S -o foobar.ll
; $ clang++-tot -emit-llvm foobar.ll -o foobar.opt.ll -S -c
;
; Then manually removing the original f1 definition, to simplify the DWARF a bit
; (so it only has the inlined definitions, no concrete definition)

; Check that:
; * no relocations are emitted for the debug_info.dwo section no matter what
; * one debug_info->debug_info relocation in debug_info no matter what (for
;   split dwarf inlining)
; * debug_info uses relocations and ref_addr no matter what
; * debug_info.dwo uses relocations for types as well as abstract subprograms
;   and variables when -split-dwarf-cross-cu-references is used
; * debug_info.dwo contains duplicate types, abstract subprograms and abstract
;   variables otherwise to avoid the need for cross-cu references

; DWO: .debug_info.dwo contents:
; CHECK-NOT: .rel{{a?}}.debug_info.dwo
; CHECK: RELOCATION RECORDS FOR [.rel{{a?}}.debug_info]:
; CHECK-NOT: RELOCATION RECORDS
; Expect one relocation in debug_info, from the inlined f1 in foo to its
; abstract origin in bar
; CHECK: R_X86_64_32 .debug_info
; CHECK-NOT: RELOCATION RECORDS
; CHECK-NOT: .debug_info
; CHECK: RELOCATION RECORDS
; CHECK-NOT: .rel{{a?}}.debug_info.dwo

; ALL: Compile Unit
; ALL: DW_TAG_compile_unit
; DWO:   DW_AT_name {{.*}} "foo.cpp"
; ALL: 0x[[F1:.*]]: DW_TAG_subprogram
; ALL:     DW_AT_name {{.*}} "f1"
; DWO: 0x[[F1T:.*]]: DW_TAG_formal_parameter
; DWO:       DW_AT_name {{.*}} "t"
; DWO:       DW_AT_type [DW_FORM_ref4] {{.*}}{0x[[T1:.*]]}
; DWO:     NULL
; DWO: 0x[[T1]]: DW_TAG_structure_type
; DWO:     DW_AT_name {{.*}} "t1"
; ALL:   DW_TAG_subprogram
; ALL:     DW_AT_name {{.*}} "foo"
; DWO:     DW_TAG_formal_parameter
; DWO:       DW_AT_name {{.*}} "t"
; DWO:       DW_AT_type [DW_FORM_ref4] {{.*}}{0x[[T1]]}
; ALL:     DW_TAG_inlined_subroutine
; ALL:       DW_AT_abstract_origin [DW_FORM_ref4] {{.*}}{0x[[F1]]}
; DWO:       DW_TAG_formal_parameter
; DWO:         DW_AT_abstract_origin [DW_FORM_ref4] {{.*}}{0x[[F1T]]}

; ALL: Compile Unit
; ALL: DW_TAG_compile_unit
; DWO:   DW_AT_name {{.*}} "bar.cpp"
; NOCROSS: 0x[[BAR_F1:.*]]: DW_TAG_subprogram
; NOCROSS: DW_AT_name {{.*}} "f1"
; NOCROSS: 0x[[BAR_F1T:.*]]: DW_TAG_formal_parameter
; NOCROSS:   DW_AT_name {{.*}} "t"
; NOCROSS:   DW_AT_type [DW_FORM_ref4] {{.*}}{0x[[BAR_T1:.*]]}
; NOCROSS: NULL
; NOCROSS: 0x[[BAR_T1]]: DW_TAG_structure_type
; NOCROSS: DW_AT_name {{.*}} "t1"
; ALL:   DW_TAG_subprogram
; ALL:     DW_AT_name {{.*}} "bar"
; DWO:     DW_TAG_formal_parameter
; DWO:       DW_AT_name {{.*}} "t"
; CROSS:     DW_AT_type [DW_FORM_ref_addr] (0x00000000[[T1]]
; NOCROSS:   DW_AT_type [DW_FORM_ref4] {{.*}}{0x[[BAR_T1]]}
; ALL:     DW_TAG_inlined_subroutine
; INFO:     DW_AT_abstract_origin [DW_FORM_ref_addr] (0x00000000[[F1]]
; NOCROSS:   DW_AT_abstract_origin [DW_FORM_ref4] {{.*}}{0x[[BAR_F1]]}
; DWO:       DW_TAG_formal_parameter
; CROSS:       DW_AT_abstract_origin [DW_FORM_ref_addr] (0x00000000[[F1T]]
; NOCROSS:     DW_AT_abstract_origin [DW_FORM_ref4] {{.*}}{0x[[BAR_F1T]]

%struct.t1 = type { i32 }

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_Z1fv() #2

; Function Attrs: noinline uwtable
define void @_Z3foo2t1(i32 %t.coerce) #3 !dbg !20 {
entry:
  %t.i = alloca %struct.t1, align 4
  call void @llvm.dbg.declare(metadata %struct.t1* %t.i, metadata !15, metadata !16), !dbg !21
  %t = alloca %struct.t1, align 4
  %agg.tmp = alloca %struct.t1, align 4
  %coerce.dive = getelementptr inbounds %struct.t1, %struct.t1* %t, i32 0, i32 0
  store i32 %t.coerce, i32* %coerce.dive, align 4
  call void @llvm.dbg.declare(metadata %struct.t1* %t, metadata !23, metadata !16), !dbg !24
  %0 = bitcast %struct.t1* %agg.tmp to i8*, !dbg !25
  %1 = bitcast %struct.t1* %t to i8*, !dbg !25
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 %1, i64 4, i1 false), !dbg !25
  %coerce.dive1 = getelementptr inbounds %struct.t1, %struct.t1* %agg.tmp, i32 0, i32 0, !dbg !26
  %2 = load i32, i32* %coerce.dive1, align 4, !dbg !26
  %coerce.dive.i = getelementptr inbounds %struct.t1, %struct.t1* %t.i, i32 0, i32 0
  store i32 %2, i32* %coerce.dive.i, align 4
  call void @_Z1fv(), !dbg !27
  ret void, !dbg !28
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #4

; Function Attrs: noinline uwtable
define void @_Z3bar2t1(i32 %t.coerce) #3 !dbg !29 {
entry:
  %t.i = alloca %struct.t1, align 4
  call void @llvm.dbg.declare(metadata %struct.t1* %t.i, metadata !15, metadata !16), !dbg !30
  %t = alloca %struct.t1, align 4
  %agg.tmp = alloca %struct.t1, align 4
  %coerce.dive = getelementptr inbounds %struct.t1, %struct.t1* %t, i32 0, i32 0
  store i32 %t.coerce, i32* %coerce.dive, align 4
  call void @llvm.dbg.declare(metadata %struct.t1* %t, metadata !32, metadata !16), !dbg !33
  %0 = bitcast %struct.t1* %agg.tmp to i8*, !dbg !34
  %1 = bitcast %struct.t1* %t to i8*, !dbg !34
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 %1, i64 4, i1 false), !dbg !34
  %coerce.dive1 = getelementptr inbounds %struct.t1, %struct.t1* %agg.tmp, i32 0, i32 0, !dbg !35
  %2 = load i32, i32* %coerce.dive1, align 4, !dbg !35
  %coerce.dive.i = getelementptr inbounds %struct.t1, %struct.t1* %t.i, i32 0, i32 0
  store i32 %2, i32* %coerce.dive.i, align 4
  call void @_Z1fv(), !dbg !36
  ret void, !dbg !37
}

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 302809) (llvm/trunk 302815)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: true)
!1 = !DIFile(filename: "foo.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 5.0.0 (trunk 302809) (llvm/trunk 302815)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: true)
!4 = !DIFile(filename: "bar.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!5 = !{!"clang version 5.0.0 (trunk 302809) (llvm/trunk 302815)"}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f12t1", scope: !1, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !1, line: 1, size: 32, elements: !12, identifier: "_ZTS2t1")
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !11, file: !1, line: 1, baseType: !14, size: 32)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocalVariable(name: "t", arg: 1, scope: !8, file: !1, line: 3, type: !11)
!16 = !DIExpression()
!17 = !DILocation(line: 3, column: 43, scope: !8)
!18 = !DILocation(line: 4, column: 3, scope: !8)
!19 = !DILocation(line: 5, column: 1, scope: !8)
!20 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foo2t1", scope: !1, file: !1, line: 6, type: !9, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 3, column: 43, scope: !8, inlinedAt: !22)
!22 = distinct !DILocation(line: 7, column: 3, scope: !20)
!23 = !DILocalVariable(name: "t", arg: 1, scope: !20, file: !1, line: 6, type: !11)
!24 = !DILocation(line: 6, column: 13, scope: !20)
!25 = !DILocation(line: 7, column: 6, scope: !20)
!26 = !DILocation(line: 7, column: 3, scope: !20)
!27 = !DILocation(line: 4, column: 3, scope: !8, inlinedAt: !22)
!28 = !DILocation(line: 8, column: 1, scope: !20)
!29 = distinct !DISubprogram(name: "bar", linkageName: "_Z3bar2t1", scope: !4, file: !4, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !3, retainedNodes: !2)
!30 = !DILocation(line: 3, column: 43, scope: !8, inlinedAt: !31)
!31 = distinct !DILocation(line: 4, column: 3, scope: !29)
!32 = !DILocalVariable(name: "t", arg: 1, scope: !29, file: !4, line: 3, type: !11)
!33 = !DILocation(line: 3, column: 13, scope: !29)
!34 = !DILocation(line: 4, column: 6, scope: !29)
!35 = !DILocation(line: 4, column: 3, scope: !29)
!36 = !DILocation(line: 4, column: 3, scope: !8, inlinedAt: !31)
!37 = !DILocation(line: 5, column: 1, scope: !29)
