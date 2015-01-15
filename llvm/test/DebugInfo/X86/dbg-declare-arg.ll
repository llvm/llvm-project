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
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !26, metadata !{!"0x102"}), !dbg !27
  call void @llvm.dbg.declare(metadata i32* %j, metadata !28, metadata !{!"0x102"}), !dbg !30
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
  call void @llvm.dbg.declare(metadata %class.A* %agg.result, metadata !37, metadata !{!"0x102"}), !dbg !39
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

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN1AD1Ev(%class.A* %this) unnamed_addr ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !43, metadata !{!"0x102"}), !dbg !44
  %this1 = load %class.A** %this.addr
  call void @_ZN1AD2Ev(%class.A* %this1)
  ret void, !dbg !45
}

define linkonce_odr void @_ZN1AD2Ev(%class.A* %this) unnamed_addr nounwind ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !46, metadata !{!"0x102"}), !dbg !47
  %this1 = load %class.A** %this.addr
  %x = getelementptr inbounds %class.A* %this1, i32 0, i32 0, !dbg !48
  store i32 1, i32* %x, align 4, !dbg !48
  ret void, !dbg !48
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!52}

!0 = !{!"0x2e\00~A\00~A\00\002\000\000\000\006\00256\000\000", !51, !1, !11, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x2\00A\002\00128\0032\000\000\000", !51, !2, null, !4, null, null, null} ; [ DW_TAG_class_type ] [A] [line 2, size 128, align 32, offset 0] [def] [from ]
!2 = !{!"0x11\004\00clang version 3.0 (trunk 130127)\000\00\000\00\001", !51, !24, !24, !50, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x29", !51} ; [ DW_TAG_file_type ]
!4 = !{!5, !7, !8, !9, !0, !10, !14}
!5 = !{!"0xd\00x\002\0032\0032\000\000", !51, !3, !6} ; [ DW_TAG_member ]
!6 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!7 = !{!"0xd\00y\002\0032\0032\0032\000", !51, !3, !6} ; [ DW_TAG_member ]
!8 = !{!"0xd\00z\002\0032\0032\0064\000", !51, !3, !6} ; [ DW_TAG_member ]
!9 = !{!"0xd\00o\002\0032\0032\0096\000", !51, !3, !6} ; [ DW_TAG_member ]
!10 = !{!"0x2e\00A\00A\00\002\000\000\000\006\00320\000\000", !51, !1, !11, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!11 = !{!"0x15\00\000\000\000\000\000\000", !51, !3, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{null, !13}
!13 = !{!"0xf\00\000\0064\0064\000\0064", !2, null, !1} ; [ DW_TAG_pointer_type ]
!14 = !{!"0x2e\00A\00A\00\002\000\000\000\006\00320\000\000", !51, !1, !15, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!15 = !{!"0x15\00\000\000\000\000\000\000", !51, !3, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{null, !13, !17}
!17 = !{!"0x10\00\000\000\000\000\000", null, !2, !18} ; [ DW_TAG_reference_type ]
!18 = !{!"0x26\00\000\000\000\000\000", !2, null, !1} ; [ DW_TAG_const_type ]
!19 = !{!"0x2e\00foo\00foo\00_Z3fooi\004\000\001\000\006\00256\000\000", !51, !3, !20, null, void (%class.A*, i32)* @_Z3fooi, null, null, null} ; [ DW_TAG_subprogram ] [line 4] [def] [scope 0] [foo]
!20 = !{!"0x15\00\000\000\000\000\000\000", !51, !3, null, !21, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = !{!1}
!22 = !{!"0x2e\00~A\00~A\00_ZN1AD1Ev\002\000\001\000\006\00256\000\000", !51, !3, !23, null, void (%class.A*)* @_ZN1AD1Ev, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [~A]
!23 = !{!"0x15\00\000\000\000\000\000\000", !51, !3, null, !24, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!24 = !{null}
!25 = !{!"0x2e\00~A\00~A\00_ZN1AD2Ev\002\000\001\000\006\00256\000\000", !51, !3, !23, null, void (%class.A*)* @_ZN1AD2Ev, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [~A]
!26 = !{!"0x101\00i\0016777220\000", !19, !3, !6} ; [ DW_TAG_arg_variable ]
!27 = !MDLocation(line: 4, column: 11, scope: !19)
!28 = !{!"0x100\00j\005\000", !29, !3, !6} ; [ DW_TAG_auto_variable ]
!29 = !{!"0xb\004\0014\000", !51, !19} ; [ DW_TAG_lexical_block ]
!30 = !MDLocation(line: 5, column: 7, scope: !29)
!31 = !MDLocation(line: 5, column: 12, scope: !29)
!32 = !MDLocation(line: 6, column: 3, scope: !29)
!33 = !MDLocation(line: 7, column: 5, scope: !34)
!34 = !{!"0xb\006\0016\001", !51, !29} ; [ DW_TAG_lexical_block ]
!35 = !MDLocation(line: 8, column: 3, scope: !34)
!36 = !MDLocation(line: 9, column: 9, scope: !29)
!37 = !{!"0x100\00my_a\009\000", !29, !3, !38} ; [ DW_TAG_auto_variable ]
!38 = !{!"0x10\00\000\000\000\000\000", !2, null, !1} ; [ DW_TAG_reference_type ]
!39 = !MDLocation(line: 9, column: 5, scope: !29)
!40 = !MDLocation(line: 10, column: 3, scope: !29)
!41 = !MDLocation(line: 11, column: 3, scope: !29)
!42 = !MDLocation(line: 12, column: 1, scope: !29)
!43 = !{!"0x101\00this\0016777218\0064", !22, !3, !13} ; [ DW_TAG_arg_variable ]
!44 = !MDLocation(line: 2, column: 47, scope: !22)
!45 = !MDLocation(line: 2, column: 61, scope: !22)
!46 = !{!"0x101\00this\0016777218\0064", !25, !3, !13} ; [ DW_TAG_arg_variable ]
!47 = !MDLocation(line: 2, column: 47, scope: !25)
!48 = !MDLocation(line: 2, column: 54, scope: !49)
!49 = !{!"0xb\002\0052\002", !51, !25} ; [ DW_TAG_lexical_block ]
!50 = !{!19, !22, !25}
!51 = !{!"a.cc", !"/private/tmp"}
!52 = !{i32 1, !"Debug Info Version", i32 2}
