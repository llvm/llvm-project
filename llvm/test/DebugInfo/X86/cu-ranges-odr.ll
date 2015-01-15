; RUN: llc -split-dwarf=Enable -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s
; RUN: llvm-readobj --relocations %t | FileCheck --check-prefix=CHECK-RELOCS %s

; From:
; class A {
; public:
;   A(int i = 0) : a(i) {}
; private:
;   int a;
; };
;
; A a;

; With function sections enabled make sure that we have a DW_AT_ranges attribute.
; CHECK: DW_AT_ranges

; Check that we have a relocation against the .debug_ranges section.
; CHECK-RELOCS: R_X86_64_32 .debug_ranges 0x0

%class.A = type { i32 }

@a = global %class.A zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  call void @_ZN1AC2Ei(%class.A* @a, i32 0), !dbg !26
  ret void, !dbg !26
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN1AC2Ei(%class.A* %this, i32 %i) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %i.addr = alloca i32, align 4
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !27, metadata !{!"0x102"}), !dbg !29
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !30, metadata !{!"0x102"}), !dbg !31
  %this1 = load %class.A** %this.addr
  %a = getelementptr inbounds %class.A* %this1, i32 0, i32 0, !dbg !31
  %0 = load i32* %i.addr, align 4, !dbg !31
  store i32 %0, i32* %a, align 4, !dbg !31
  ret void, !dbg !31
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !32
  ret void, !dbg !32
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23, !24}
!llvm.ident = !{!25}

!0 = !{!"0x11\004\00clang version 3.5 (trunk 199923) (llvm/trunk 199940)\000\00\000\00\001", !1, !2, !3, !13, !21, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/baz.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"baz.cpp", !"/usr/local/google/home/echristo/tmp"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2\00A\001\0032\0032\000\000\000", !1, null, null, !5, null, null, !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = !{!6, !8}
!6 = !{!"0xd\00a\005\0032\0032\000\001", !1, !"_ZTS1A", !7} ; [ DW_TAG_member ] [a] [line 5, size 32, align 32, offset 0] [private] [from int]
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = !{!"0x2e\00A\00A\00\003\000\000\000\006\00256\000\003", !1, !"_ZTS1A", !9, null, null, null, i32 0, !12} ; [ DW_TAG_subprogram ] [line 3] [A]
!9 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = !{null, !11, !7}
!11 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!12 = !{i32 786468}
!13 = !{!14, !18, !19}
!14 = !{!"0x2e\00__cxx_global_var_init\00__cxx_global_var_init\00\008\001\001\000\006\00256\000\008", !1, !15, !16, null, void ()* @__cxx_global_var_init, null, null, !2} ; [ DW_TAG_subprogram ] [line 8] [local] [def] [__cxx_global_var_init]
!15 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/baz.cpp]
!16 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !17, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = !{null}
!18 = !{!"0x2e\00A\00A\00_ZN1AC2Ei\003\000\001\000\006\00256\000\003", !1, !"_ZTS1A", !9, null, void (%class.A*, i32)* @_ZN1AC2Ei, null, !8, !2} ; [ DW_TAG_subprogram ] [line 3] [def] [A]
!19 = !{!"0x2e\00\00\00_GLOBAL__I_a\003\001\001\000\006\0064\000\003", !1, !15, !20, null, void ()* @_GLOBAL__I_a, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [local] [def]
!20 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = !{!22}
!22 = !{!"0x34\00a\00a\00\008\000\001", null, !15, !4, %class.A* @a, null} ; [ DW_TAG_variable ] [a] [line 8] [def]
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 1, !"Debug Info Version", i32 2}
!25 = !{!"clang version 3.5 (trunk 199923) (llvm/trunk 199940)"}
!26 = !MDLocation(line: 8, scope: !14)
!27 = !{!"0x101\00this\0016777216\001088", !18, null, !28} ; [ DW_TAG_arg_variable ] [this] [line 0]
!28 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!29 = !MDLocation(line: 0, scope: !18)
!30 = !{!"0x101\00i\0033554435\000", !18, !15, !7} ; [ DW_TAG_arg_variable ] [i] [line 3]
!31 = !MDLocation(line: 3, scope: !18)
!32 = !MDLocation(line: 3, scope: !19)
