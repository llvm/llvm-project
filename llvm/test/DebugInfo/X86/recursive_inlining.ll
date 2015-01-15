; REQUIRES: object-emission

; RUN: %llc_dwarf -filetype=obj -O0 < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; This isn't a very pretty test case - I imagine there might be other ways to
; tickle the optimizers into producing the desired code, but I haven't found
; them.

; The issue is when a function is inlined into itself, the inlined argument
; accidentally overwrote the concrete argument and was lost.

; IR generated from the following source compiled with clang -g:
; void fn1(void *);
; void fn2(int, int, int, int);
; void fn3();
; void fn8();
; struct C {
;   int b;
;   void m_fn2() {
;     fn8();
;     if (b) fn2(0, 0, 0, 0);
;     fn3();
;   }
; };
; C *x;
; inline void fn7() {}
; void fn6() {
;   fn8();
;   x->m_fn2();
;   fn7();
; }
; void fn3() { fn6(); }
; void fn4() { x->m_fn2(); }
; void fn5() { x->m_fn2(); }

; The definition of C and declaration of C::m_fn2
; CHECK: DW_TAG_structure_type
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: DW_TAG_member
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "m_fn2"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[M_FN2_THIS_DECL:.*]]:     DW_TAG_formal_parameter

; The abstract definition of C::m_fn2
; CHECK: [[M_FN2_ABS_DEF:.*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_specification {{.*}} "_ZN1C5m_fn2Ev"
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_inline
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[M_FN2_THIS_ABS_DEF:.*]]:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "this"

; Skip some other functions
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram

; The concrete definition of C::m_fn2
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_abstract_origin {{.*}} {[[M_FN2_ABS_DEF]]} "_ZN1C5m_fn2Ev"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} {[[M_FN2_THIS_ABS_DEF]]}
; CHECK-NOT: {{DW_TAG|NULL}}
; Inlined fn3:
; CHECK:     DW_TAG_inlined_subroutine
; CHECK-NOT: {{DW_TAG|NULL}}
; Inlined fn6:
; CHECK:       DW_TAG_inlined_subroutine
; CHECK-NOT: {{DW_TAG|NULL}}
; Inlined C::m_fn2:
; CHECK:         DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:           DW_AT_abstract_origin {{.*}} {[[M_FN2_ABS_DEF]]} "_ZN1C5m_fn2Ev"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:           DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:              DW_AT_abstract_origin {{.*}} {[[M_FN2_THIS_ABS_DEF]]}



%struct.C = type { i32 }

@x = global %struct.C* null, align 8

; Function Attrs: nounwind
define void @_Z3fn6v() #0 {
entry:
  tail call void @_Z3fn8v() #3, !dbg !31
  %0 = load %struct.C** @x, align 8, !dbg !32, !tbaa !33
  tail call void @llvm.dbg.value(metadata %struct.C* %0, i64 0, metadata !37, metadata !{!"0x102"}) #3, !dbg !38
  tail call void @_Z3fn8v() #3, !dbg !39
  %b.i = getelementptr inbounds %struct.C* %0, i64 0, i32 0, !dbg !40
  %1 = load i32* %b.i, align 4, !dbg !40, !tbaa !42
  %tobool.i = icmp eq i32 %1, 0, !dbg !40
  br i1 %tobool.i, label %_ZN1C5m_fn2Ev.exit, label %if.then.i, !dbg !40

if.then.i:                                        ; preds = %entry
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !45
  br label %_ZN1C5m_fn2Ev.exit, !dbg !45

_ZN1C5m_fn2Ev.exit:                               ; preds = %entry, %if.then.i
  tail call void @_Z3fn3v() #3, !dbg !47
  ret void, !dbg !48
}

declare void @_Z3fn8v() #1

; Function Attrs: nounwind
define linkonce_odr void @_ZN1C5m_fn2Ev(%struct.C* nocapture readonly %this) #0 align 2 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.C* %this, i64 0, metadata !24, metadata !{!"0x102"}), !dbg !49
  tail call void @_Z3fn8v() #3, !dbg !50
  %b = getelementptr inbounds %struct.C* %this, i64 0, i32 0, !dbg !51
  %0 = load i32* %b, align 4, !dbg !51, !tbaa !42
  %tobool = icmp eq i32 %0, 0, !dbg !51
  br i1 %tobool, label %if.end, label %if.then, !dbg !51

if.then:                                          ; preds = %entry
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !52
  br label %if.end, !dbg !52

if.end:                                           ; preds = %entry, %if.then
  tail call void @_Z3fn8v() #3, !dbg !53
  %1 = load %struct.C** @x, align 8, !dbg !56, !tbaa !33
  tail call void @llvm.dbg.value(metadata %struct.C* %1, i64 0, metadata !57, metadata !{!"0x102"}) #3, !dbg !58
  tail call void @_Z3fn8v() #3, !dbg !59
  %b.i.i = getelementptr inbounds %struct.C* %1, i64 0, i32 0, !dbg !60
  %2 = load i32* %b.i.i, align 4, !dbg !60, !tbaa !42
  %tobool.i.i = icmp eq i32 %2, 0, !dbg !60
  br i1 %tobool.i.i, label %_Z3fn6v.exit, label %if.then.i.i, !dbg !60

if.then.i.i:                                      ; preds = %if.end
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !61
  br label %_Z3fn6v.exit, !dbg !61

_Z3fn6v.exit:                                     ; preds = %if.end, %if.then.i.i
  tail call void @_Z3fn3v() #3, !dbg !62
  ret void, !dbg !63
}

; Function Attrs: nounwind
define void @_Z3fn3v() #0 {
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %tailrecurse.backedge, %entry
  tail call void @_Z3fn8v() #3, !dbg !64
  %0 = load %struct.C** @x, align 8, !dbg !66, !tbaa !33
  tail call void @llvm.dbg.value(metadata %struct.C* %0, i64 0, metadata !67, metadata !{!"0x102"}) #3, !dbg !68
  tail call void @_Z3fn8v() #3, !dbg !69
  %b.i.i = getelementptr inbounds %struct.C* %0, i64 0, i32 0, !dbg !70
  %1 = load i32* %b.i.i, align 4, !dbg !70, !tbaa !42
  %tobool.i.i = icmp eq i32 %1, 0, !dbg !70
  br i1 %tobool.i.i, label %tailrecurse.backedge, label %if.then.i.i, !dbg !70

tailrecurse.backedge:                             ; preds = %tailrecurse, %if.then.i.i
  br label %tailrecurse

if.then.i.i:                                      ; preds = %tailrecurse
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !71
  br label %tailrecurse.backedge, !dbg !71
}

; Function Attrs: nounwind
define void @_Z3fn4v() #0 {
entry:
  %0 = load %struct.C** @x, align 8, !dbg !72, !tbaa !33
  tail call void @_ZN1C5m_fn2Ev(%struct.C* %0), !dbg !72
  ret void, !dbg !72
}

; Function Attrs: nounwind
define void @_Z3fn5v() #0 {
entry:
  %0 = load %struct.C** @x, align 8, !dbg !73, !tbaa !33
  tail call void @_ZN1C5m_fn2Ev(%struct.C* %0), !dbg !73
  ret void, !dbg !73
}

declare void @_Z3fn2iiii(i32, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28, !29}
!llvm.ident = !{!30}

!0 = !{!"0x11\004\00clang version 3.6.0 \001\00\000\00\001", !1, !2, !3, !13, !26, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce/<stdin>] [DW_LANG_C_plus_plus]
!1 = !{!"<stdin>", !"/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00C\005\0032\0032\000\000\000", !5, null, null, !6, null, null, !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] [line 5, size 32, align 32, offset 0] [def] [from ]
!5 = !{!"recursive_inlining.cpp", !"/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce"}
!6 = !{!7, !9}
!7 = !{!"0xd\00b\006\0032\0032\000\000", !5, !"_ZTS1C", !8} ; [ DW_TAG_member ] [b] [line 6, size 32, align 32, offset 0] [from int]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0x2e\00m_fn2\00m_fn2\00_ZN1C5m_fn2Ev\007\000\000\000\006\00256\001\007", !5, !"_ZTS1C", !10, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 7] [m_fn2]
!10 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = !{null, !12}
!12 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1C]
!13 = !{!14, !18, !19, !20, !21, !22}
!14 = !{!"0x2e\00fn6\00fn6\00_Z3fn6v\0015\000\001\000\006\00256\001\0015", !5, !15, !16, null, void ()* @_Z3fn6v, null, null, !2} ; [ DW_TAG_subprogram ] [line 15] [def] [fn6]
!15 = !{!"0x29", !5}         ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce/recursive_inlining.cpp]
!16 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !17, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = !{null}
!18 = !{!"0x2e\00fn3\00fn3\00_Z3fn3v\0020\000\001\000\006\00256\001\0020", !5, !15, !16, null, void ()* @_Z3fn3v, null, null, !2} ; [ DW_TAG_subprogram ] [line 20] [def] [fn3]
!19 = !{!"0x2e\00fn4\00fn4\00_Z3fn4v\0021\000\001\000\006\00256\001\0021", !5, !15, !16, null, void ()* @_Z3fn4v, null, null, !2} ; [ DW_TAG_subprogram ] [line 21] [def] [fn4]
!20 = !{!"0x2e\00fn5\00fn5\00_Z3fn5v\0022\000\001\000\006\00256\001\0022", !5, !15, !16, null, void ()* @_Z3fn5v, null, null, !2} ; [ DW_TAG_subprogram ] [line 22] [def] [fn5]
!21 = !{!"0x2e\00fn7\00fn7\00_Z3fn7v\0014\000\001\000\006\00256\001\0014", !5, !15, !16, null, null, null, null, !2} ; [ DW_TAG_subprogram ] [line 14] [def] [fn7]
!22 = !{!"0x2e\00m_fn2\00m_fn2\00_ZN1C5m_fn2Ev\007\000\001\000\006\00256\001\007", !5, !"_ZTS1C", !10, null, void (%struct.C*)* @_ZN1C5m_fn2Ev, null, !9, !23} ; [ DW_TAG_subprogram ] [line 7] [def] [m_fn2]
!23 = !{!24}
!24 = !{!"0x101\00this\0016777216\001088", !22, null, !25} ; [ DW_TAG_arg_variable ] [this] [line 0]
!25 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1C]
!26 = !{!27}
!27 = !{!"0x34\00x\00x\00\0013\000\001", null, !15, !25, %struct.C** @x, null} ; [ DW_TAG_variable ] [x] [line 13] [def]
!28 = !{i32 2, !"Dwarf Version", i32 4}
!29 = !{i32 2, !"Debug Info Version", i32 2}
!30 = !{!"clang version 3.6.0 "}
!31 = !MDLocation(line: 16, scope: !14)
!32 = !MDLocation(line: 17, scope: !14)
!33 = !{!34, !34, i64 0}
!34 = !{!"any pointer", !35, i64 0}
!35 = !{!"omnipotent char", !36, i64 0}
!36 = !{!"Simple C/C++ TBAA"}
!37 = !{!"0x101\00this\0016777216\001088", !22, null, !25, !32} ; [ DW_TAG_arg_variable ] [this] [line 0]
!38 = !MDLocation(line: 0, scope: !22, inlinedAt: !32)
!39 = !MDLocation(line: 8, scope: !22, inlinedAt: !32)
!40 = !MDLocation(line: 9, scope: !41, inlinedAt: !32)
!41 = !{!"0xb\009\000\000", !5, !22} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce/recursive_inlining.cpp]
!42 = !{!43, !44, i64 0}
!43 = !{!"_ZTS1C", !44, i64 0}
!44 = !{!"int", !35, i64 0}
!45 = !MDLocation(line: 9, scope: !46, inlinedAt: !32)
!46 = !{!"0xb\009\000\001", !5, !41} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce/recursive_inlining.cpp]
!47 = !MDLocation(line: 10, scope: !22, inlinedAt: !32)
!48 = !MDLocation(line: 19, scope: !14)
!49 = !MDLocation(line: 0, scope: !22)
!50 = !MDLocation(line: 8, scope: !22)
!51 = !MDLocation(line: 9, scope: !41)
!52 = !MDLocation(line: 9, scope: !46)
!53 = !MDLocation(line: 16, scope: !14, inlinedAt: !54)
!54 = !MDLocation(line: 20, scope: !18, inlinedAt: !55)
!55 = !MDLocation(line: 10, scope: !22)
!56 = !MDLocation(line: 17, scope: !14, inlinedAt: !54)
!57 = !{!"0x101\00this\0016777216\001088", !22, null, !25, !56} ; [ DW_TAG_arg_variable ] [this] [line 0]
!58 = !MDLocation(line: 0, scope: !22, inlinedAt: !56)
!59 = !MDLocation(line: 8, scope: !22, inlinedAt: !56)
!60 = !MDLocation(line: 9, scope: !41, inlinedAt: !56)
!61 = !MDLocation(line: 9, scope: !46, inlinedAt: !56)
!62 = !MDLocation(line: 10, scope: !22, inlinedAt: !56)
!63 = !MDLocation(line: 11, scope: !22)
!64 = !MDLocation(line: 16, scope: !14, inlinedAt: !65)
!65 = !MDLocation(line: 20, scope: !18)
!66 = !MDLocation(line: 17, scope: !14, inlinedAt: !65)
!67 = !{!"0x101\00this\0016777216\001088", !22, null, !25, !66} ; [ DW_TAG_arg_variable ] [this] [line 0]
!68 = !MDLocation(line: 0, scope: !22, inlinedAt: !66)
!69 = !MDLocation(line: 8, scope: !22, inlinedAt: !66)
!70 = !MDLocation(line: 9, scope: !41, inlinedAt: !66)
!71 = !MDLocation(line: 9, scope: !46, inlinedAt: !66)
!72 = !MDLocation(line: 21, scope: !19)
!73 = !MDLocation(line: 22, scope: !20)
