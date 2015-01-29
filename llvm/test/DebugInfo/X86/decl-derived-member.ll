; REQUIRES: object-emission

; RUN: llc -mtriple x86_64-pc-linux -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Testcase from:
; struct base {
;  virtual ~base();
; };
; typedef base base_type;
; struct foo {
;  base_type b;
; };
; foo f;

; Where member b should be seen as a field at an offset and not a bitfield.

; CHECK: DW_TAG_member
; CHECK: DW_AT_name{{.*}}"b"
; CHECK-NOT: DW_AT_bit_offset

%struct.foo = type { %struct.base }
%struct.base = type { i32 (...)** }

$_ZN3fooC2Ev = comdat any

$_ZN3fooD2Ev = comdat any

$_ZN4baseC2Ev = comdat any

@f = global %struct.foo zeroinitializer, align 8
@__dso_handle = external global i8
@_ZTV4base = external unnamed_addr constant [4 x i8*]
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_decl_derived_member.cpp, i8* null }]

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  call void @_ZN3fooC2Ev(%struct.foo* @f) #2, !dbg !33
  %0 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.foo*)* @_ZN3fooD2Ev to void (i8*)*), i8* bitcast (%struct.foo* @f to i8*), i8* @__dso_handle) #2, !dbg !33
  ret void, !dbg !33
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN3fooC2Ev(%struct.foo* %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr, metadata !34, metadata !36), !dbg !37
  %this1 = load %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo* %this1, i32 0, i32 0, !dbg !38
  call void @_ZN4baseC2Ev(%struct.base* %b) #2, !dbg !38
  ret void, !dbg !38
}

; Function Attrs: inlinehint uwtable
define linkonce_odr void @_ZN3fooD2Ev(%struct.foo* %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr, metadata !39, metadata !36), !dbg !40
  %this1 = load %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo* %this1, i32 0, i32 0, !dbg !41
  call void @_ZN4baseD1Ev(%struct.base* %b), !dbg !41
  ret void, !dbg !43
}

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN4baseC2Ev(%struct.base* %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %struct.base*, align 8
  store %struct.base* %this, %struct.base** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.base** %this.addr, metadata !44, metadata !36), !dbg !46
  %this1 = load %struct.base** %this.addr
  %0 = bitcast %struct.base* %this1 to i32 (...)***, !dbg !47
  store i32 (...)** bitcast (i8** getelementptr inbounds ([4 x i8*]* @_ZTV4base, i64 0, i64 2) to i32 (...)**), i32 (...)*** %0, !dbg !47
  ret void, !dbg !47
}

declare void @_ZN4baseD1Ev(%struct.base*) #4

define internal void @_GLOBAL__sub_I_decl_derived_member.cpp() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !48
  ret void
}

attributes #0 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!30, !31}
!llvm.ident = !{!32}

!0 = !{!"0x11\004\00clang version 3.7.0 (trunk 227104) (llvm/trunk 227103)\000\00\000\00\001", !1, !2, !3, !9, !28, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/decl-derived-member.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"decl-derived-member.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4, !8}
!4 = !{!"0x13\00foo\005\0064\0064\000\000\000", !1, null, null, !5, null, null, !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo] [line 5, size 64, align 64, offset 0] [def] [from ]
!5 = !{!6}
!6 = !{!"0xd\00b\006\0064\0064\000\000", !1, !"_ZTS3foo", !7} ; [ DW_TAG_member ] [b] [line 6, size 64, align 64, offset 0] [from base_type]
!7 = !{!"0x16\00base_type\004\000\000\000\000", !1, null, !"_ZTS4base"} ; [ DW_TAG_typedef ] [base_type] [line 4, size 0, align 0, offset 0] [from _ZTS4base]
!8 = !{!"0x13\00base\001\000\000\000\004\000", !1, null, null, null, null, null, !"_ZTS4base"} ; [ DW_TAG_structure_type ] [base] [line 1, size 0, align 0, offset 0] [decl] [from ]
!9 = !{!10, !14, !19, !24, !26}
!10 = !{!"0x2e\00__cxx_global_var_init\00__cxx_global_var_init\00\008\001\001\000\000\00256\000\008", !1, !11, !12, null, void ()* @__cxx_global_var_init, null, null, !2} ; [ DW_TAG_subprogram ] [line 8] [local] [def] [__cxx_global_var_init]
!11 = !{!"0x29", !1}                              ; [ DW_TAG_file_type ] [/tmp/dbginfo/decl-derived-member.cpp]
!12 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = !{null}
!14 = !{!"0x2e\00foo\00foo\00_ZN3fooC2Ev\005\000\001\000\000\00320\000\005", !1, !"_ZTS3foo", !15, null, void (%struct.foo*)* @_ZN3fooC2Ev, null, !18, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [foo]
!15 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{null, !17}
!17 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS3foo]
!18 = !{!"0x2e\00foo\00foo\00\000\000\000\000\000\00320\000\000", null, !"_ZTS3foo", !15, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 0] [foo]
!19 = !{!"0x2e\00base\00base\00_ZN4baseC2Ev\001\000\001\000\000\00320\000\001", !1, !"_ZTS4base", !20, null, void (%struct.base*)* @_ZN4baseC2Ev, null, !23, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [base]
!20 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !21, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = !{null, !22}
!22 = !{!"0xf\00\000\0064\0064\000\001088\00", null, null, !"_ZTS4base"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS4base]
!23 = !{!"0x2e\00base\00base\00\000\000\000\000\000\00320\000\000", null, !"_ZTS4base", !20, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 0] [base]
!24 = !{!"0x2e\00~foo\00~foo\00_ZN3fooD2Ev\005\000\001\000\000\00320\000\005", !1, !"_ZTS3foo", !15, null, void (%struct.foo*)* @_ZN3fooD2Ev, null, !25, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [~foo]
!25 = !{!"0x2e\00~foo\00~foo\00\000\000\000\000\000\00320\000\000", null, !"_ZTS3foo", !15, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 0] [~foo]
!26 = !{!"0x2e\00\00\00_GLOBAL__sub_I_decl_derived_member.cpp\000\001\001\000\000\0064\000\000", !1, !11, !27, null, void ()* @_GLOBAL__sub_I_decl_derived_member.cpp, null, null, !2} ; [ DW_TAG_subprogram ] [line 0] [local] [def]
!27 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = !{!29}
!29 = !{!"0x34\00f\00f\00\008\000\001", null, !11, !"_ZTS3foo", %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 8] [def]
!30 = !{i32 2, !"Dwarf Version", i32 4}
!31 = !{i32 2, !"Debug Info Version", i32 2}
!32 = !{!"clang version 3.7.0 (trunk 227104) (llvm/trunk 227103)"}
!33 = !MDLocation(line: 8, column: 5, scope: !10)
!34 = !{!"0x101\00this\0016777216\001088", !14, null, !35} ; [ DW_TAG_arg_variable ] [this] [line 0]
!35 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS3foo]
!36 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!37 = !MDLocation(line: 0, scope: !14)
!38 = !MDLocation(line: 5, column: 8, scope: !14)
!39 = !{!"0x101\00this\0016777216\001088", !24, null, !35} ; [ DW_TAG_arg_variable ] [this] [line 0]
!40 = !MDLocation(line: 0, scope: !24)
!41 = !MDLocation(line: 5, column: 8, scope: !42)
!42 = !{!"0xb\005\008\002", !1, !24}              ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/decl-derived-member.cpp]
!43 = !MDLocation(line: 5, column: 8, scope: !24)
!44 = !{!"0x101\00this\0016777216\001088", !19, null, !45} ; [ DW_TAG_arg_variable ] [this] [line 0]
!45 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS4base"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS4base]
!46 = !MDLocation(line: 0, scope: !19)
!47 = !MDLocation(line: 1, column: 8, scope: !19)
!48 = !MDLocation(line: 0, scope: !26)
