; REQUIRES: object-emission

; RUN: llc -mtriple x86_64-pc-linux -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Testcase from:
; struct base {
;  virtual ~base();
; };
; struct foo {
;  base b;
; };
; foo f;

; Where member b should be seen as a field at an offset and not a bitfield.

; CHECK: DW_TAG_member
; CHECK: DW_AT_name{{.*}}"b"
; CHECK-NOT: DW_AT_bit_offset

%struct.foo = type { %struct.base }
%struct.base = type { i32 (...)** }
@f = global %struct.foo zeroinitializer, align 8
@__dso_handle = external global i8
@_ZTV4base = external unnamed_addr constant [4 x i8*]
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  call void @_ZN3fooC2Ev(%struct.foo* @f) #2, !dbg !35
  %0 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.foo*)* @_ZN3fooD2Ev to void (i8*)*), i8* bitcast (%struct.foo* @f to i8*), i8* @__dso_handle) #2, !dbg !35
  ret void, !dbg !35
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN3fooC2Ev(%struct.foo* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr, metadata !36, metadata !{!"0x102"}), !dbg !38
  %this1 = load %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo* %this1, i32 0, i32 0, !dbg !39
  call void @_ZN4baseC2Ev(%struct.base* %b) #2, !dbg !39
  ret void, !dbg !39
}

; Function Attrs: inlinehint uwtable
define linkonce_odr void @_ZN3fooD2Ev(%struct.foo* %this) unnamed_addr #1 align 2 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr, metadata !40, metadata !{!"0x102"}), !dbg !41
  %this1 = load %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo* %this1, i32 0, i32 0, !dbg !42
  call void @_ZN4baseD1Ev(%struct.base* %b), !dbg !42
  ret void, !dbg !44
}

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

declare void @_ZN4baseD1Ev(%struct.base*) #4

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN4baseC2Ev(%struct.base* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.base*, align 8
  store %struct.base* %this, %struct.base** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.base** %this.addr, metadata !45, metadata !{!"0x102"}), !dbg !47
  %this1 = load %struct.base** %this.addr
  %0 = bitcast %struct.base* %this1 to i8***, !dbg !48
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV4base, i64 0, i64 2), i8*** %0, !dbg !48
  ret void, !dbg !48
}

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !49
  ret void, !dbg !49
}

attributes #0 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!32, !33}
!llvm.ident = !{!34}

!0 = !{!"0x11\004\00clang version 3.5.0 (trunk 203673) (llvm/trunk 203681)\000\00\000\00\001", !1, !2, !3, !8, !30, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/foo.cc] [DW_LANG_C_plus_plus]
!1 = !{!"foo.cc", !"/usr/local/google/home/echristo"}
!2 = !{}
!3 = !{!4, !7}
!4 = !{!"0x13\00foo\005\0064\0064\000\000\000", !1, null, null, !5, null, null, !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo] [line 5, size 64, align 64, offset 0] [def] [from ]
!5 = !{!6}
!6 = !{!"0xd\00b\006\0064\0064\000\000", !1, !"_ZTS3foo", !"_ZTS4base"} ; [ DW_TAG_member ] [b] [line 6, size 64, align 64, offset 0] [from _ZTS4base]
!7 = !{!"0x13\00base\001\000\000\000\004\000", !1, null, null, null, null, null, !"_ZTS4base"} ; [ DW_TAG_structure_type ] [base] [line 1, size 0, align 0, offset 0] [decl] [from ]
!8 = !{!9, !13, !19, !22, !28}
!9 = !{!"0x2e\00__cxx_global_var_init\00__cxx_global_var_init\00\009\001\001\000\006\00256\000\009", !1, !10, !11, null, void ()* @__cxx_global_var_init, null, null, !2} ; [ DW_TAG_subprogram ] [line 9] [local] [def] [__cxx_global_var_init]
!10 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/foo.cc]
!11 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{null}
!13 = !{!"0x2e\00~foo\00~foo\00_ZN3fooD2Ev\005\000\001\000\006\00320\000\005", !1, !"_ZTS3foo", !14, null, void (%struct.foo*)* @_ZN3fooD2Ev, null, !17, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [~foo]
!14 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = !{null, !16}
!16 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS3foo]
!17 = !{!"0x2e\00~foo\00~foo\00\000\000\000\000\006\00320\000\000", null, !"_ZTS3foo", !14, null, null, null, i32 0, !18} ; [ DW_TAG_subprogram ] [line 0] [~foo]
!18 = !{i32 786468}
!19 = !{!"0x2e\00foo\00foo\00_ZN3fooC2Ev\005\000\001\000\006\00320\000\005", !1, !"_ZTS3foo", !14, null, void (%struct.foo*)* @_ZN3fooC2Ev, null, !20, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [foo]
!20 = !{!"0x2e\00foo\00foo\00\000\000\000\000\006\00320\000\000", null, !"_ZTS3foo", !14, null, null, null, i32 0, !21} ; [ DW_TAG_subprogram ] [line 0] [foo]
!21 = !{i32 786468}
!22 = !{!"0x2e\00base\00base\00_ZN4baseC2Ev\001\000\001\000\006\00320\000\001", !1, !"_ZTS4base", !23, null, void (%struct.base*)* @_ZN4baseC2Ev, null, !26, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [base]
!23 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !24, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!24 = !{null, !25}
!25 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS4base"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS4base]
!26 = !{!"0x2e\00base\00base\00\000\000\000\000\006\00320\000\000", null, !"_ZTS4base", !23, null, null, null, i32 0, !27} ; [ DW_TAG_subprogram ] [line 0] [base]
!27 = !{i32 786468}
!28 = !{!"0x2e\00\00\00_GLOBAL__I_a\001\001\001\000\006\0064\000\001", !1, !10, !29, null, void ()* @_GLOBAL__I_a, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [local] [def]
!29 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = !{!31}
!31 = !{!"0x34\00f\00f\00\009\000\001", null, !10, !4, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 9] [def]
!32 = !{i32 2, !"Dwarf Version", i32 4}
!33 = !{i32 1, !"Debug Info Version", i32 2}
!34 = !{!"clang version 3.5.0 (trunk 203673) (llvm/trunk 203681)"}
!35 = !MDLocation(line: 9, scope: !9)
!36 = !{!"0x101\00this\0016777216\001088", !19, null, !37} ; [ DW_TAG_arg_variable ] [this] [line 0]
!37 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS3foo]
!38 = !MDLocation(line: 0, scope: !19)
!39 = !MDLocation(line: 5, scope: !19)
!40 = !{!"0x101\00this\0016777216\001088", !13, null, !37} ; [ DW_TAG_arg_variable ] [this] [line 0]
!41 = !MDLocation(line: 0, scope: !13)
!42 = !MDLocation(line: 5, scope: !43)
!43 = !{!"0xb\005\000\000", !1, !13} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/foo.cc]
!44 = !MDLocation(line: 5, scope: !13)
!45 = !{!"0x101\00this\0016777216\001088", !22, null, !46} ; [ DW_TAG_arg_variable ] [this] [line 0]
!46 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS4base"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS4base]
!47 = !MDLocation(line: 0, scope: !22)
!48 = !MDLocation(line: 1, scope: !22)
!49 = !MDLocation(line: 1, scope: !28)
