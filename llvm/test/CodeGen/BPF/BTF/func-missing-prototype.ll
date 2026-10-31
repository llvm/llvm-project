; RUN: llc < %s | FileCheck %s --implicit-check-not="BTF_KIND_FUNC(id"
;
; Save the source as bpf_prog.cpp to reproduce the initializer name below.
;
; Source code:
;   int type_id = __builtin_btf_type_id(0, 1);
;   extern "C" int get_type_id() { return type_id; }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm bpf_prog.cpp -o bpf_prog.ll
;   llc -filetype=obj bpf_prog.ll -o bpf_prog.o
;
; Clang generates an initializer for type_id with a linkage name but no
; source prototype. Keep its line and CO-RE records and the getter's FUNC.

; CHECK-LABEL: _GLOBAL__sub_I_bpf_prog.cpp:
; CHECK:       [[INIT:\.Lfunc_begin[0-9]+]]:
; CHECK:       # BTF_KIND_FUNC_PROTO(id = [[PROTO:[0-9]+]])
; CHECK:       .long [[NAME:[1-9][0-9]*]] # BTF_KIND_FUNC(id =
; CHECK-NEXT:  .long {{[0-9]+}}
; CHECK-NEXT:  .long [[PROTO]]
; CHECK:       .ascii "get_type_id" # string offset=[[NAME]]
; CHECK:       .section .BTF.ext
; CHECK:       .long 32
; CHECK-NEXT:  .long 0
; CHECK-NEXT:  .long 0
; CHECK:       # LineInfo section
; CHECK:       # Line 2 Col
; CHECK:       .long [[INIT]]
; CHECK:       # Field reloc section
; CHECK-NEXT:  .long 1

target triple = "bpf"

@type_id = global i32 0
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_bpf_prog.cpp, ptr null }]
@"llvm.btf_type_id.0$7" = external global i64, !llvm.preserve.access.index !3 #0

define i32 @get_type_id() !dbg !4 {
  %id = load i32, ptr @type_id, !dbg !6
  ret i32 %id, !dbg !6
}

define internal void @_GLOBAL__sub_I_bpf_prog.cpp() !dbg !7 {
  %id = load i64, ptr @"llvm.btf_type_id.0$7"
  %value = call i64 @llvm.bpf.passthrough.i64.i64(i32 0, i64 %id)
  %narrow = trunc i64 %value to i32, !dbg !9
  store i32 %narrow, ptr @type_id, !dbg !10
  ret void
}

declare i64 @llvm.bpf.passthrough.i64.i64(i32, i64)
attributes #0 = { "btf_type_id" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "bpf_prog.cpp", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = distinct !DISubprogram(name: "get_type_id", scope: !1, file: !1, line: 2, type: !5, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !{!3})
!6 = !DILocation(line: 2, column: 39, scope: !4)
!7 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_bpf_prog.cpp", scope: !1, file: !1, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!8 = !DISubroutineType(types: !{})
!9 = !DILocation(line: 1, column: 15, scope: !11, inlinedAt: !12)
!10 = !DILocation(line: 0, scope: !11, inlinedAt: !12)
!11 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !1, file: !1, type: !13, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!12 = !DILocation(line: 0, scope: !7)
!13 = !DISubroutineType(types: !{null})
