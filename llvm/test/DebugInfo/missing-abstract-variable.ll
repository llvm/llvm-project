; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; The formal parameter 'b' for Function 'x' when inlined within 'a' is lost on
; mips and powerpc64 (and on x86_64 at at least -O2). Presumably this is a
; SelectionDAG issue (do mips/powerpc64 use FastISel?).
; XFAIL: mips, powerpc64, s390x

; Build from the following source with clang -O2.

; The important details are that 'x's abstract definition is first built during
; the definition of 'b', where the parameter to 'x' is constant and so 'x's 's'
; variable is optimized away. No abstract definition DIE for 's' is constructed.
; Then, during 'a' emission, the abstract DbgVariable for 's' is created, but
; the abstract DIE isn't (since the abstract definition for 'b' is already
; built). This results in 's' inlined in 'a' being emitted with its name, line,
; file there, rather than referencing an abstract definition.

; extern int t;
;
; void f(int);
;
; inline void x(bool b) {
;   if (b) {
;     int s = t;
;     f(s);
;   }
;   f(0);
; }
;
; void b() {
;   x(false);
; }
;
; void a(bool u) {
;   x(u);
; }

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "x"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "b"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:       DW_TAG_lexical_block
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:         DW_AT_name {{.*}} "s"

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "b"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "_Z1xb"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_abstract_origin {{.*}} "b"
; Notice 'x's local variable 's' is missing. Not necessarily a bug here,
; since it's been optimized entirely away and it should be described in
; abstract subprogram.
; CHECK-NOT: DW_TAG
; CHECK: NULL
; CHECK-NOT: DW_TAG
; CHECK: NULL

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "a"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "_Z1xb"
; CHECK-NOT: {{DW_TAG|NULL}}
; FIXME: This formal parameter goes missing at least at -O2 (& on
; mips/powerpc), maybe before that. Perhaps SelectionDAG is to blame (and
; fastisel succeeds).
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_abstract_origin {{.*}} "b"

; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_lexical_block
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:       DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:         DW_AT_abstract_origin {{.*}} "s"

@t = external global i32

; Function Attrs: uwtable
define void @_Z1bv() #0 {
entry:
  tail call void @llvm.dbg.value(metadata i1 false, i64 0, metadata !25, metadata !{!"0x102"}), !dbg !27
  tail call void @_Z1fi(i32 0), !dbg !28
  ret void, !dbg !29
}

; Function Attrs: uwtable
define void @_Z1ab(i1 zeroext %u) #0 {
entry:
  tail call void @llvm.dbg.value(metadata i1 %u, i64 0, metadata !13, metadata !{!"0x102"}), !dbg !30
  tail call void @llvm.dbg.value(metadata i1 %u, i64 0, metadata !31, metadata !{!"0x102"}), !dbg !33
  br i1 %u, label %if.then.i, label %_Z1xb.exit, !dbg !34

if.then.i:                                        ; preds = %entry
  %0 = load i32* @t, align 4, !dbg !35, !tbaa !36
  tail call void @llvm.dbg.value(metadata i32 %0, i64 0, metadata !40, metadata !{!"0x102"}), !dbg !35
  tail call void @_Z1fi(i32 %0), !dbg !41
  br label %_Z1xb.exit, !dbg !42

_Z1xb.exit:                                       ; preds = %entry, %if.then.i
  tail call void @_Z1fi(i32 0), !dbg !43
  ret void, !dbg !44
}

declare void @_Z1fi(i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = !{!"0x11\004\00clang version 3.5.0 \001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/missing-abstract-variables.cc] [DW_LANG_C_plus_plus]
!1 = !{!"missing-abstract-variables.cc", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4, !8, !14}
!4 = !{!"0x2e\00b\00b\00_Z1bv\0013\000\001\000\006\00256\001\0013", !1, !5, !6, null, void ()* @_Z1bv, null, null, !2} ; [ DW_TAG_subprogram ] [line 13] [def] [b]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/missing-abstract-variables.cc]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{!"0x2e\00a\00a\00_Z1ab\0017\000\001\000\006\00256\001\0017", !1, !5, !9, null, void (i1)* @_Z1ab, null, null, !12} ; [ DW_TAG_subprogram ] [line 17] [def] [a]
!9 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = !{null, !11}
!11 = !{!"0x24\00bool\000\008\008\000\000\002", null, null} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!12 = !{!13}
!13 = !{!"0x101\00u\0016777233\000", !8, !5, !11} ; [ DW_TAG_arg_variable ] [u] [line 17]
!14 = !{!"0x2e\00x\00x\00_Z1xb\005\000\001\000\006\00256\001\005", !1, !5, !9, null, null, null, null, !15} ; [ DW_TAG_subprogram ] [line 5] [def] [x]
!15 = !{!16, !17}
!16 = !{!"0x101\00b\0016777221\000", !14, !5, !11} ; [ DW_TAG_arg_variable ] [b] [line 5]
!17 = !{!"0x100\00s\007\000", !18, !5, !20} ; [ DW_TAG_auto_variable ] [s] [line 7]
!18 = !{!"0xb\006\000\000", !1, !19} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/missing-abstract-variables.cc]
!19 = !{!"0xb\006\000\000", !1, !14} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/missing-abstract-variables.cc]
!20 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 2}
!23 = !{!"clang version 3.5.0 "}
!24 = !{i1 false}
!25 = !{!"0x101\00b\0016777221\000", !14, !5, !11, !26} ; [ DW_TAG_arg_variable ] [b] [line 5]
!26 = !MDLocation(line: 14, scope: !4)
!27 = !MDLocation(line: 5, scope: !14, inlinedAt: !26)
!28 = !MDLocation(line: 10, scope: !14, inlinedAt: !26)
!29 = !MDLocation(line: 15, scope: !4)
!30 = !MDLocation(line: 17, scope: !8)
!31 = !{!"0x101\00b\0016777221\000", !14, !5, !11, !32} ; [ DW_TAG_arg_variable ] [b] [line 5]
!32 = !MDLocation(line: 18, scope: !8)
!33 = !MDLocation(line: 5, scope: !14, inlinedAt: !32)
!34 = !MDLocation(line: 6, scope: !19, inlinedAt: !32)
!35 = !MDLocation(line: 7, scope: !18, inlinedAt: !32)
!36 = !{!37, !37, i64 0}
!37 = !{!"int", !38, i64 0}
!38 = !{!"omnipotent char", !39, i64 0}
!39 = !{!"Simple C/C++ TBAA"}
!40 = !{!"0x100\00s\007\000", !18, !5, !20, !32} ; [ DW_TAG_auto_variable ] [s] [line 7]
!41 = !MDLocation(line: 8, scope: !18, inlinedAt: !32)
!42 = !MDLocation(line: 9, scope: !18, inlinedAt: !32)
!43 = !MDLocation(line: 10, scope: !14, inlinedAt: !32)
!44 = !MDLocation(line: 19, scope: !8)
