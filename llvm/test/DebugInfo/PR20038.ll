; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; IR generated from clang -O0 with:
; struct C {
;   ~C();
; };
; extern bool b;
; void fun4() { b && (C(), 1); }
; __attribute__((always_inline)) C::~C() { }

; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "C"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "~C"

; CHECK:  DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_ZN1CD1Ev"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:  DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "this"

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "fun4"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "_ZN1CD1Ev"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_abstract_origin {{.*}} "this"

; FIXME: D2 is actually inlined into D1 but doesn't show up here, possibly due
; to there being no work in D2 (calling another member function from the dtor
; causes D2 to show up, calling a free function doesn't).

; CHECK-NOT: DW_TAG
; CHECK:     NULL
; CHECK-NOT: DW_TAG
; CHECK:   NULL

%struct.C = type { i8 }

@b = external global i8

; Function Attrs: nounwind
define void @_Z4fun4v() #0 {
entry:
  %this.addr.i.i = alloca %struct.C*, align 8, !dbg !21
  %this.addr.i = alloca %struct.C*, align 8, !dbg !22
  %agg.tmp.ensured = alloca %struct.C, align 1
  %cleanup.cond = alloca i1
  %0 = load i8* @b, align 1, !dbg !24
  %tobool = trunc i8 %0 to i1, !dbg !24
  store i1 false, i1* %cleanup.cond
  br i1 %tobool, label %land.rhs, label %land.end, !dbg !24

land.rhs:                                         ; preds = %entry
  store i1 true, i1* %cleanup.cond, !dbg !25
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %1 = phi i1 [ false, %entry ], [ true, %land.rhs ]
  %cleanup.is_active = load i1* %cleanup.cond, !dbg !27
  br i1 %cleanup.is_active, label %cleanup.action, label %cleanup.done, !dbg !27

cleanup.action:                                   ; preds = %land.end
  store %struct.C* %agg.tmp.ensured, %struct.C** %this.addr.i, align 8, !dbg !22
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr.i, metadata !29, metadata !{!"0x102"}), !dbg !31
  %this1.i = load %struct.C** %this.addr.i, !dbg !22
  store %struct.C* %this1.i, %struct.C** %this.addr.i.i, align 8, !dbg !21
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr.i.i, metadata !32, metadata !{!"0x102"}), !dbg !33
  %this1.i.i = load %struct.C** %this.addr.i.i, !dbg !21
  br label %cleanup.done, !dbg !22

cleanup.done:                                     ; preds = %cleanup.action, %land.end
  ret void, !dbg !34
}

; Function Attrs: alwaysinline nounwind
define void @_ZN1CD1Ev(%struct.C* %this) unnamed_addr #1 align 2 {
entry:
  %this.addr.i = alloca %struct.C*, align 8, !dbg !37
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !29, metadata !{!"0x102"}), !dbg !38
  %this1 = load %struct.C** %this.addr
  store %struct.C* %this1, %struct.C** %this.addr.i, align 8, !dbg !37
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr.i, metadata !32, metadata !{!"0x102"}), !dbg !39
  %this1.i = load %struct.C** %this.addr.i, !dbg !37
  ret void, !dbg !37
}

; Function Attrs: alwaysinline nounwind
define void @_ZN1CD2Ev(%struct.C* %this) unnamed_addr #1 align 2 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !32, metadata !{!"0x102"}), !dbg !40
  %this1 = load %struct.C** %this.addr
  ret void, !dbg !41
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !3, !11, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/<stdin>] [DW_LANG_C_plus_plus]
!1 = !{!"<stdin>", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00C\001\008\008\000\000\000", !5, null, null, !6, null, null, !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = !{!"PR20038.cpp", !"/tmp/dbginfo"}
!6 = !{!7}
!7 = !{!"0x2e\00~C\00~C\00\002\000\000\000\006\00256\000\002", !5, !"_ZTS1C", !8, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 2] [~C]
!8 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !9, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = !{null, !10}
!10 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1C]
!11 = !{!12, !16, !17}
!12 = !{!"0x2e\00fun4\00fun4\00_Z4fun4v\005\000\001\000\006\00256\000\005", !5, !13, !14, null, void ()* @_Z4fun4v, null, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [fun4]
!13 = !{!"0x29", !5}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/PR20038.cpp]
!14 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = !{null}
!16 = !{!"0x2e\00~C\00~C\00_ZN1CD2Ev\006\000\001\000\006\00256\000\006", !5, !"_ZTS1C", !8, null, void (%struct.C*)* @_ZN1CD2Ev, null, !7, !2} ; [ DW_TAG_subprogram ] [line 6] [def] [~C]
!17 = !{!"0x2e\00~C\00~C\00_ZN1CD1Ev\006\000\001\000\006\00256\000\006", !5, !"_ZTS1C", !8, null, void (%struct.C*)* @_ZN1CD1Ev, null, !7, !2} ; [ DW_TAG_subprogram ] [line 6] [def] [~C]
!18 = !{i32 2, !"Dwarf Version", i32 4}
!19 = !{i32 2, !"Debug Info Version", i32 2}
!20 = !{!"clang version 3.5.0 "}
!21 = !MDLocation(line: 6, scope: !17, inlinedAt: !22)
!22 = !MDLocation(line: 5, scope: !23)
!23 = !{!"0xb\005\000\003", !5, !12} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/PR20038.cpp]
!24 = !MDLocation(line: 5, scope: !12)
!25 = !MDLocation(line: 5, scope: !26)
!26 = !{!"0xb\005\000\001", !5, !12} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/PR20038.cpp]
!27 = !MDLocation(line: 5, scope: !28)
!28 = !{!"0xb\005\000\002", !5, !12} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/PR20038.cpp]
!29 = !{!"0x101\00this\0016777216\001088", !17, null, !30} ; [ DW_TAG_arg_variable ] [this] [line 0]
!30 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1C]
!31 = !MDLocation(line: 0, scope: !17, inlinedAt: !22)
!32 = !{!"0x101\00this\0016777216\001088", !16, null, !30} ; [ DW_TAG_arg_variable ] [this] [line 0]
!33 = !MDLocation(line: 0, scope: !16, inlinedAt: !21)
!34 = !MDLocation(line: 5, scope: !35)
!35 = !{!"0xb\005\000\005", !5, !36} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/PR20038.cpp]
!36 = !{!"0xb\005\000\004", !5, !12} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/PR20038.cpp]
!37 = !MDLocation(line: 6, scope: !17)
!38 = !MDLocation(line: 0, scope: !17)
!39 = !MDLocation(line: 0, scope: !16, inlinedAt: !37)
!40 = !MDLocation(line: 0, scope: !16)
!41 = !MDLocation(line: 6, scope: !16)
