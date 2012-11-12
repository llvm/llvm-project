; RUN: llc -O0 -fast-isel=false < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"
;Radar 9321650

;CHECK: ##DEBUG_VALUE: my_a 

%class.A = type { i32, i32, i32, i32 }

define void @_Z3fooi(%class.A* sret %agg.result, i32 %i) ssp {
entry:
  %i.addr = alloca i32, align 4
  %j = alloca i32, align 4
  %nrvo = alloca i1
  %cleanup.dest.slot = alloca i32
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !26), !dbg !27
  call void @llvm.dbg.declare(metadata !{i32* %j}, metadata !28), !dbg !30
  store i32 0, i32* %j, align 4, !dbg !31
  %tmp = load i32* %i.addr, align 4, !dbg !32
  %cmp = icmp eq i32 %tmp, 42, !dbg !32
  br i1 %cmp, label %if.then, label %if.end, !dbg !32

if.then:                                          ; preds = %entry
  %tmp1 = load i32* %i.addr, align 4, !dbg !33
  %add = add nsw i32 %tmp1, 1, !dbg !33
  store i32 %add, i32* %j, align 4, !dbg !33
  br label %if.end, !dbg !35

if.end:                                           ; preds = %if.then, %entry
  store i1 false, i1* %nrvo, !dbg !36
  call void @llvm.dbg.declare(metadata !{%class.A* %agg.result}, metadata !37), !dbg !39
  %tmp2 = load i32* %j, align 4, !dbg !40
  %x = getelementptr inbounds %class.A* %agg.result, i32 0, i32 0, !dbg !40
  store i32 %tmp2, i32* %x, align 4, !dbg !40
  store i1 true, i1* %nrvo, !dbg !41
  store i32 1, i32* %cleanup.dest.slot
  %nrvo.val = load i1* %nrvo, !dbg !42
  br i1 %nrvo.val, label %nrvo.skipdtor, label %nrvo.unused, !dbg !42

nrvo.unused:                                      ; preds = %if.end
  call void @_ZN1AD1Ev(%class.A* %agg.result), !dbg !42
  br label %nrvo.skipdtor, !dbg !42

nrvo.skipdtor:                                    ; preds = %nrvo.unused, %if.end
  ret void, !dbg !42
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN1AD1Ev(%class.A* %this) unnamed_addr ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !43), !dbg !44
  %this1 = load %class.A** %this.addr
  call void @_ZN1AD2Ev(%class.A* %this1)
  ret void, !dbg !45
}

define linkonce_odr void @_ZN1AD2Ev(%class.A* %this) unnamed_addr nounwind ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !46), !dbg !47
  %this1 = load %class.A** %this.addr
  %x = getelementptr inbounds %class.A* %this1, i32 0, i32 0, !dbg !48
  store i32 1, i32* %x, align 4, !dbg !48
  ret void, !dbg !48
}

!llvm.dbg.sp = !{!0, !10, !14, !19, !22, !25}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"~A", metadata !"~A", metadata !"", metadata !3, i32 2, metadata !11, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589826, metadata !2, metadata !"A", metadata !3, i32 2, i64 128, i64 32, i32 0, i32 0, null, metadata !4, i32 0, null, null} ; [ DW_TAG_class_type ]
!2 = metadata !{i32 589841, i32 0, i32 4, metadata !"a.cc", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 130127)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589865, metadata !"a.cc", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !5, metadata !7, metadata !8, metadata !9, metadata !0, metadata !10, metadata !14}
!5 = metadata !{i32 589837, metadata !3, metadata !"x", metadata !3, i32 2, i64 32, i64 32, i64 0, i32 0, metadata !6} ; [ DW_TAG_member ]
!6 = metadata !{i32 589860, metadata !2, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 589837, metadata !3, metadata !"y", metadata !3, i32 2, i64 32, i64 32, i64 32, i32 0, metadata !6} ; [ DW_TAG_member ]
!8 = metadata !{i32 589837, metadata !3, metadata !"z", metadata !3, i32 2, i64 32, i64 32, i64 64, i32 0, metadata !6} ; [ DW_TAG_member ]
!9 = metadata !{i32 589837, metadata !3, metadata !"o", metadata !3, i32 2, i64 32, i64 32, i64 96, i32 0, metadata !6} ; [ DW_TAG_member ]
!10 = metadata !{i32 589870, i32 0, metadata !1, metadata !"A", metadata !"A", metadata !"", metadata !3, i32 2, metadata !11, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 589845, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !12, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!12 = metadata !{null, metadata !13}
!13 = metadata !{i32 589839, metadata !2, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !1} ; [ DW_TAG_pointer_type ]
!14 = metadata !{i32 589870, i32 0, metadata !1, metadata !"A", metadata !"A", metadata !"", metadata !3, i32 2, metadata !15, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null} ; [ DW_TAG_subprogram ]
!15 = metadata !{i32 589845, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !16, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!16 = metadata !{null, metadata !13, metadata !17}
!17 = metadata !{i32 589840, metadata !2, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !18} ; [ DW_TAG_reference_type ]
!18 = metadata !{i32 589862, metadata !2, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !1} ; [ DW_TAG_const_type ]
!19 = metadata !{i32 589870, i32 0, metadata !3, metadata !"foo", metadata !"foo", metadata !"_Z3fooi", metadata !3, i32 4, metadata !20, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (%class.A*, i32)* @_Z3fooi, null, null} ; [ DW_TAG_subprogram ]
!20 = metadata !{i32 589845, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !21, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!21 = metadata !{metadata !1}
!22 = metadata !{i32 589870, i32 0, metadata !3, metadata !"~A", metadata !"~A", metadata !"_ZN1AD1Ev", metadata !3, i32 2, metadata !23, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (%class.A*)* @_ZN1AD1Ev, null, null} ; [ DW_TAG_subprogram ]
!23 = metadata !{i32 589845, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !24, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!24 = metadata !{null}
!25 = metadata !{i32 589870, i32 0, metadata !3, metadata !"~A", metadata !"~A", metadata !"_ZN1AD2Ev", metadata !3, i32 2, metadata !23, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (%class.A*)* @_ZN1AD2Ev, null, null} ; [ DW_TAG_subprogram ]
!26 = metadata !{i32 590081, metadata !19, metadata !"i", metadata !3, i32 16777220, metadata !6, i32 0} ; [ DW_TAG_arg_variable ]
!27 = metadata !{i32 4, i32 11, metadata !19, null}
!28 = metadata !{i32 590080, metadata !29, metadata !"j", metadata !3, i32 5, metadata !6, i32 0} ; [ DW_TAG_auto_variable ]
!29 = metadata !{i32 589835, metadata !19, i32 4, i32 14, metadata !3, i32 0} ; [ DW_TAG_lexical_block ]
!30 = metadata !{i32 5, i32 7, metadata !29, null}
!31 = metadata !{i32 5, i32 12, metadata !29, null}
!32 = metadata !{i32 6, i32 3, metadata !29, null}
!33 = metadata !{i32 7, i32 5, metadata !34, null}
!34 = metadata !{i32 589835, metadata !29, i32 6, i32 16, metadata !3, i32 1} ; [ DW_TAG_lexical_block ]
!35 = metadata !{i32 8, i32 3, metadata !34, null}
!36 = metadata !{i32 9, i32 9, metadata !29, null}
!37 = metadata !{i32 590080, metadata !29, metadata !"my_a", metadata !3, i32 9, metadata !38, i32 0} ; [ DW_TAG_auto_variable ]
!38 = metadata !{i32 589840, metadata !2, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !1} ; [ DW_TAG_reference_type ]
!39 = metadata !{i32 9, i32 5, metadata !29, null}
!40 = metadata !{i32 10, i32 3, metadata !29, null}
!41 = metadata !{i32 11, i32 3, metadata !29, null}
!42 = metadata !{i32 12, i32 1, metadata !29, null}
!43 = metadata !{i32 590081, metadata !22, metadata !"this", metadata !3, i32 16777218, metadata !13, i32 64} ; [ DW_TAG_arg_variable ]
!44 = metadata !{i32 2, i32 47, metadata !22, null}
!45 = metadata !{i32 2, i32 61, metadata !22, null}
!46 = metadata !{i32 590081, metadata !25, metadata !"this", metadata !3, i32 16777218, metadata !13, i32 64} ; [ DW_TAG_arg_variable ]
!47 = metadata !{i32 2, i32 47, metadata !25, null}
!48 = metadata !{i32 2, i32 54, metadata !49, null}
!49 = metadata !{i32 589835, metadata !25, i32 2, i32 52, metadata !3, i32 2} ; [ DW_TAG_lexical_block ]
