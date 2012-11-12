; RUN: llc -mtriple=x86_64-apple-macosx10.7 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

; CHECK: b_ref
; CHECK-NOT: AT_bit_size

%struct.bar = type { %struct.baz, %struct.baz* }
%struct.baz = type { i32 }

define i32 @main(i32 %argc, i8** %argv) uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %myBar = alloca %struct.bar, align 8
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %argc.addr}, metadata !49), !dbg !50
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8*** %argv.addr}, metadata !51), !dbg !52
  call void @llvm.dbg.declare(metadata !{%struct.bar* %myBar}, metadata !53), !dbg !55
  call void @_ZN3barC1Ei(%struct.bar* %myBar, i32 1), !dbg !56
  ret i32 0, !dbg !57
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN3barC1Ei(%struct.bar* %this, i32 %x) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca %struct.bar*, align 8
  %x.addr = alloca i32, align 4
  store %struct.bar* %this, %struct.bar** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.bar** %this.addr}, metadata !58), !dbg !59
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr}, metadata !60), !dbg !61
  %this1 = load %struct.bar** %this.addr
  %0 = load i32* %x.addr, align 4, !dbg !62
  call void @_ZN3barC2Ei(%struct.bar* %this1, i32 %0), !dbg !62
  ret void, !dbg !62
}

define linkonce_odr void @_ZN3barC2Ei(%struct.bar* %this, i32 %x) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca %struct.bar*, align 8
  %x.addr = alloca i32, align 4
  store %struct.bar* %this, %struct.bar** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.bar** %this.addr}, metadata !63), !dbg !64
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr}, metadata !65), !dbg !66
  %this1 = load %struct.bar** %this.addr
  %b = getelementptr inbounds %struct.bar* %this1, i32 0, i32 0, !dbg !67
  %0 = load i32* %x.addr, align 4, !dbg !67
  call void @_ZN3bazC1Ei(%struct.baz* %b, i32 %0), !dbg !67
  %1 = getelementptr inbounds %struct.bar* %this1, i32 0, i32 1, !dbg !67
  %b2 = getelementptr inbounds %struct.bar* %this1, i32 0, i32 0, !dbg !67
  store %struct.baz* %b2, %struct.baz** %1, align 8, !dbg !67
  ret void, !dbg !68
}

define linkonce_odr void @_ZN3bazC1Ei(%struct.baz* %this, i32 %a) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca %struct.baz*, align 8
  %a.addr = alloca i32, align 4
  store %struct.baz* %this, %struct.baz** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.baz** %this.addr}, metadata !70), !dbg !71
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !72), !dbg !73
  %this1 = load %struct.baz** %this.addr
  %0 = load i32* %a.addr, align 4, !dbg !74
  call void @_ZN3bazC2Ei(%struct.baz* %this1, i32 %0), !dbg !74
  ret void, !dbg !74
}

define linkonce_odr void @_ZN3bazC2Ei(%struct.baz* %this, i32 %a) unnamed_addr nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca %struct.baz*, align 8
  %a.addr = alloca i32, align 4
  store %struct.baz* %this, %struct.baz** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.baz** %this.addr}, metadata !75), !dbg !76
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !77), !dbg !78
  %this1 = load %struct.baz** %this.addr
  %h = getelementptr inbounds %struct.baz* %this1, i32 0, i32 0, !dbg !79
  %0 = load i32* %a.addr, align 4, !dbg !79
  store i32 %0, i32* %h, align 4, !dbg !79
  ret void, !dbg !80
}

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 4, metadata !"main.cpp", metadata !"/Users/echristo/tmp/bad-struct-ref", metadata !"clang version 3.1 (trunk 146596)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !3, metadata !27, metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5, metadata !9}
!5 = metadata !{i32 720898, null, metadata !"bar", metadata !6, i32 9, i64 128, i64 64, i32 0, i32 0, null, metadata !7, i32 0, null, null} ; [ DW_TAG_class_type ]
!6 = metadata !{i32 720937, metadata !"main.cpp", metadata !"/Users/echristo/tmp/bad-struct-ref", null} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !8, metadata !19, metadata !21}
!8 = metadata !{i32 720909, metadata !5, metadata !"b", metadata !6, i32 11, i64 32, i64 32, i64 0, i32 0, metadata !9} ; [ DW_TAG_member ]
!9 = metadata !{i32 720898, null, metadata !"baz", metadata !6, i32 3, i64 32, i64 32, i32 0, i32 0, null, metadata !10, i32 0, null, null} ; [ DW_TAG_class_type ]
!10 = metadata !{metadata !11, metadata !13}
!11 = metadata !{i32 720909, metadata !9, metadata !"h", metadata !6, i32 5, i64 32, i64 32, i64 0, i32 0, metadata !12} ; [ DW_TAG_member ]
!12 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!13 = metadata !{i32 720942, i32 0, metadata !9, metadata !"baz", metadata !"baz", metadata !"", metadata !6, i32 6, metadata !14, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !17} ; [ DW_TAG_subprogram ]
!14 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !15, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!15 = metadata !{null, metadata !16, metadata !12}
!16 = metadata !{i32 720911, i32 0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !9} ; [ DW_TAG_pointer_type ]
!17 = metadata !{metadata !18}
!18 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!19 = metadata !{i32 720909, metadata !5, metadata !"b_ref", metadata !6, i32 12, i64 64, i64 64, i64 64, i32 0, metadata !20} ; [ DW_TAG_member ]
!20 = metadata !{i32 720912, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_reference_type ]
!21 = metadata !{i32 720942, i32 0, metadata !5, metadata !"bar", metadata !"bar", metadata !"", metadata !6, i32 13, metadata !22, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !25} ; [ DW_TAG_subprogram ]
!22 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !23, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!23 = metadata !{null, metadata !24, metadata !12}
!24 = metadata !{i32 720911, i32 0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !5} ; [ DW_TAG_pointer_type ]
!25 = metadata !{metadata !26}
!26 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!27 = metadata !{metadata !28}
!28 = metadata !{metadata !29, metadata !37, metadata !40, metadata !43, metadata !46}
!29 = metadata !{i32 720942, i32 0, metadata !6, metadata !"main", metadata !"main", metadata !"", metadata !6, i32 17, metadata !30, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, i32 (i32, i8**)* @main, null, null, metadata !35} ; [ DW_TAG_subprogram ]
!30 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !31, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!31 = metadata !{metadata !12, metadata !12, metadata !32}
!32 = metadata !{i32 720911, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !33} ; [ DW_TAG_pointer_type ]
!33 = metadata !{i32 720911, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !34} ; [ DW_TAG_pointer_type ]
!34 = metadata !{i32 720932, null, metadata !"char", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!35 = metadata !{metadata !36}
!36 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!37 = metadata !{i32 720942, i32 0, null, metadata !"bar", metadata !"bar", metadata !"_ZN3barC1Ei", metadata !6, i32 13, metadata !22, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (%struct.bar*, i32)* @_ZN3barC1Ei, null, metadata !21, metadata !38} ; [ DW_TAG_subprogram ]
!38 = metadata !{metadata !39}
!39 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!40 = metadata !{i32 720942, i32 0, null, metadata !"bar", metadata !"bar", metadata !"_ZN3barC2Ei", metadata !6, i32 13, metadata !22, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (%struct.bar*, i32)* @_ZN3barC2Ei, null, metadata !21, metadata !41} ; [ DW_TAG_subprogram ]
!41 = metadata !{metadata !42}
!42 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!43 = metadata !{i32 720942, i32 0, null, metadata !"baz", metadata !"baz", metadata !"_ZN3bazC1Ei", metadata !6, i32 6, metadata !14, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (%struct.baz*, i32)* @_ZN3bazC1Ei, null, metadata !13, metadata !44} ; [ DW_TAG_subprogram ]
!44 = metadata !{metadata !45}
!45 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!46 = metadata !{i32 720942, i32 0, null, metadata !"baz", metadata !"baz", metadata !"_ZN3bazC2Ei", metadata !6, i32 6, metadata !14, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (%struct.baz*, i32)* @_ZN3bazC2Ei, null, metadata !13, metadata !47} ; [ DW_TAG_subprogram ]
!47 = metadata !{metadata !48}
!48 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!49 = metadata !{i32 721153, metadata !29, metadata !"argc", metadata !6, i32 16777232, metadata !12, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!50 = metadata !{i32 16, i32 14, metadata !29, null}
!51 = metadata !{i32 721153, metadata !29, metadata !"argv", metadata !6, i32 33554448, metadata !32, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!52 = metadata !{i32 16, i32 27, metadata !29, null}
!53 = metadata !{i32 721152, metadata !54, metadata !"myBar", metadata !6, i32 18, metadata !5, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!54 = metadata !{i32 720907, metadata !29, i32 17, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!55 = metadata !{i32 18, i32 9, metadata !54, null}
!56 = metadata !{i32 18, i32 17, metadata !54, null}
!57 = metadata !{i32 19, i32 5, metadata !54, null}
!58 = metadata !{i32 721153, metadata !37, metadata !"this", metadata !6, i32 16777229, metadata !24, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!59 = metadata !{i32 13, i32 5, metadata !37, null}
!60 = metadata !{i32 721153, metadata !37, metadata !"x", metadata !6, i32 33554445, metadata !12, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!61 = metadata !{i32 13, i32 13, metadata !37, null}
!62 = metadata !{i32 13, i32 34, metadata !37, null}
!63 = metadata !{i32 721153, metadata !40, metadata !"this", metadata !6, i32 16777229, metadata !24, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!64 = metadata !{i32 13, i32 5, metadata !40, null}
!65 = metadata !{i32 721153, metadata !40, metadata !"x", metadata !6, i32 33554445, metadata !12, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!66 = metadata !{i32 13, i32 13, metadata !40, null}
!67 = metadata !{i32 13, i32 33, metadata !40, null}
!68 = metadata !{i32 13, i32 34, metadata !69, null}
!69 = metadata !{i32 720907, metadata !40, i32 13, i32 33, metadata !6, i32 1} ; [ DW_TAG_lexical_block ]
!70 = metadata !{i32 721153, metadata !43, metadata !"this", metadata !6, i32 16777222, metadata !16, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!71 = metadata !{i32 6, i32 5, metadata !43, null}
!72 = metadata !{i32 721153, metadata !43, metadata !"a", metadata !6, i32 33554438, metadata !12, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!73 = metadata !{i32 6, i32 13, metadata !43, null}
!74 = metadata !{i32 6, i32 24, metadata !43, null}
!75 = metadata !{i32 721153, metadata !46, metadata !"this", metadata !6, i32 16777222, metadata !16, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!76 = metadata !{i32 6, i32 5, metadata !46, null}
!77 = metadata !{i32 721153, metadata !46, metadata !"a", metadata !6, i32 33554438, metadata !12, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!78 = metadata !{i32 6, i32 13, metadata !46, null}
!79 = metadata !{i32 6, i32 23, metadata !46, null}
!80 = metadata !{i32 6, i32 24, metadata !81, null}
!81 = metadata !{i32 720907, metadata !46, i32 6, i32 23, metadata !6, i32 2} ; [ DW_TAG_lexical_block ]
