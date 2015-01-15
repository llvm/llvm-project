; RUN: llc -mtriple=x86_64-apple-macosx10.7 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

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
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !49, metadata !{!"0x102"}), !dbg !50
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !51, metadata !{!"0x102"}), !dbg !52
  call void @llvm.dbg.declare(metadata %struct.bar* %myBar, metadata !53, metadata !{!"0x102"}), !dbg !55
  call void @_ZN3barC1Ei(%struct.bar* %myBar, i32 1), !dbg !56
  ret i32 0, !dbg !57
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN3barC1Ei(%struct.bar* %this, i32 %x) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca %struct.bar*, align 8
  %x.addr = alloca i32, align 4
  store %struct.bar* %this, %struct.bar** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.bar** %this.addr, metadata !58, metadata !{!"0x102"}), !dbg !59
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !60, metadata !{!"0x102"}), !dbg !61
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
  call void @llvm.dbg.declare(metadata %struct.bar** %this.addr, metadata !63, metadata !{!"0x102"}), !dbg !64
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !65, metadata !{!"0x102"}), !dbg !66
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
  call void @llvm.dbg.declare(metadata %struct.baz** %this.addr, metadata !70, metadata !{!"0x102"}), !dbg !71
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !72, metadata !{!"0x102"}), !dbg !73
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
  call void @llvm.dbg.declare(metadata %struct.baz** %this.addr, metadata !75, metadata !{!"0x102"}), !dbg !76
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !77, metadata !{!"0x102"}), !dbg !78
  %this1 = load %struct.baz** %this.addr
  %h = getelementptr inbounds %struct.baz* %this1, i32 0, i32 0, !dbg !79
  %0 = load i32* %a.addr, align 4, !dbg !79
  store i32 %0, i32* %h, align 4, !dbg !79
  ret void, !dbg !80
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!83}

!0 = !{!"0x11\004\00clang version 3.1 (trunk 146596)\000\00\000\00\000", !82, !1, !3, !27, !1,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5, !9}
!5 = !{!"0x2\00bar\009\00128\0064\000\000\000", !82, null, null, !7, null, null, null} ; [ DW_TAG_class_type ] [bar] [line 9, size 128, align 64, offset 0] [def] [from ]
!6 = !{!"0x29", !82} ; [ DW_TAG_file_type ]
!7 = !{!8, !19, !21}
!8 = !{!"0xd\00b\0011\0032\0032\000\000", !82, !5, !9} ; [ DW_TAG_member ]
!9 = !{!"0x2\00baz\003\0032\0032\000\000\000", !82, null, null, !10, null, null, null} ; [ DW_TAG_class_type ] [baz] [line 3, size 32, align 32, offset 0] [def] [from ]
!10 = !{!11, !13}
!11 = !{!"0xd\00h\005\0032\0032\000\000", !82, !9, !12} ; [ DW_TAG_member ]
!12 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!13 = !{!"0x2e\00baz\00baz\00\006\000\000\000\006\00256\000\000", !82, !9, !14, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ]
!14 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = !{null, !16, !12}
!16 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !9} ; [ DW_TAG_pointer_type ]
!19 = !{!"0xd\00b_ref\0012\0064\0064\0064\000", !82, !5, !20} ; [ DW_TAG_member ]
!20 = !{!"0x10\00\000\000\000\000\000", null, null, !9} ; [ DW_TAG_reference_type ]
!21 = !{!"0x2e\00bar\00bar\00\0013\000\000\000\006\00256\000\000", !82, !5, !22, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ]
!22 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !23, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = !{null, !24, !12}
!24 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !5} ; [ DW_TAG_pointer_type ]
!27 = !{!29, !37, !40, !43, !46}
!29 = !{!"0x2e\00main\00main\00\0017\000\001\000\006\00256\000\000", !82, !6, !30, null, i32 (i32, i8**)* @main, null, null, null} ; [ DW_TAG_subprogram ] [line 17] [def] [scope 0] [main]
!30 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !31, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!31 = !{!12, !12, !32}
!32 = !{!"0xf\00\000\0064\0064\000\000", null, null, !33} ; [ DW_TAG_pointer_type ]
!33 = !{!"0xf\00\000\0064\0064\000\000", null, null, !34} ; [ DW_TAG_pointer_type ]
!34 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!35 = !{!36}
!36 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!37 = !{!"0x2e\00bar\00bar\00_ZN3barC1Ei\0013\000\001\000\006\00256\000\000", !82, null, !22, null, void (%struct.bar*, i32)* @_ZN3barC1Ei, null, !21, null} ; [ DW_TAG_subprogram ] [line 13] [def] [scope 0] [bar]
!38 = !{!39}
!39 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!40 = !{!"0x2e\00bar\00bar\00_ZN3barC2Ei\0013\000\001\000\006\00256\000\000", !82, null, !22, null, void (%struct.bar*, i32)* @_ZN3barC2Ei, null, !21, null} ; [ DW_TAG_subprogram ] [line 13] [def] [scope 0] [bar]
!41 = !{!42}
!42 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!43 = !{!"0x2e\00baz\00baz\00_ZN3bazC1Ei\006\000\001\000\006\00256\000\000", !82, null, !14, null, void (%struct.baz*, i32)* @_ZN3bazC1Ei, null, !13, null} ; [ DW_TAG_subprogram ] [line 6] [def] [scope 0] [baz]
!44 = !{!45}
!45 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!46 = !{!"0x2e\00baz\00baz\00_ZN3bazC2Ei\006\000\001\000\006\00256\000\000", !82, null, !14, null, void (%struct.baz*, i32)* @_ZN3bazC2Ei, null, !13, null} ; [ DW_TAG_subprogram ] [line 6] [def] [scope 0] [baz]
!49 = !{!"0x101\00argc\0016777232\000", !29, !6, !12} ; [ DW_TAG_arg_variable ]
!50 = !MDLocation(line: 16, column: 14, scope: !29)
!51 = !{!"0x101\00argv\0033554448\000", !29, !6, !32} ; [ DW_TAG_arg_variable ]
!52 = !MDLocation(line: 16, column: 27, scope: !29)
!53 = !{!"0x100\00myBar\0018\000", !54, !6, !5} ; [ DW_TAG_auto_variable ]
!54 = !{!"0xb\0017\001\000", !82, !29} ; [ DW_TAG_lexical_block ]
!55 = !MDLocation(line: 18, column: 9, scope: !54)
!56 = !MDLocation(line: 18, column: 17, scope: !54)
!57 = !MDLocation(line: 19, column: 5, scope: !54)
!58 = !{!"0x101\00this\0016777229\0064", !37, !6, !24} ; [ DW_TAG_arg_variable ]
!59 = !MDLocation(line: 13, column: 5, scope: !37)
!60 = !{!"0x101\00x\0033554445\000", !37, !6, !12} ; [ DW_TAG_arg_variable ]
!61 = !MDLocation(line: 13, column: 13, scope: !37)
!62 = !MDLocation(line: 13, column: 34, scope: !37)
!63 = !{!"0x101\00this\0016777229\0064", !40, !6, !24} ; [ DW_TAG_arg_variable ]
!64 = !MDLocation(line: 13, column: 5, scope: !40)
!65 = !{!"0x101\00x\0033554445\000", !40, !6, !12} ; [ DW_TAG_arg_variable ]
!66 = !MDLocation(line: 13, column: 13, scope: !40)
!67 = !MDLocation(line: 13, column: 33, scope: !40)
!68 = !MDLocation(line: 13, column: 34, scope: !69)
!69 = !{!"0xb\0013\0033\001", !82, !40} ; [ DW_TAG_lexical_block ]
!70 = !{!"0x101\00this\0016777222\0064", !43, !6, !16} ; [ DW_TAG_arg_variable ]
!71 = !MDLocation(line: 6, column: 5, scope: !43)
!72 = !{!"0x101\00a\0033554438\000", !43, !6, !12} ; [ DW_TAG_arg_variable ]
!73 = !MDLocation(line: 6, column: 13, scope: !43)
!74 = !MDLocation(line: 6, column: 24, scope: !43)
!75 = !{!"0x101\00this\0016777222\0064", !46, !6, !16} ; [ DW_TAG_arg_variable ]
!76 = !MDLocation(line: 6, column: 5, scope: !46)
!77 = !{!"0x101\00a\0033554438\000", !46, !6, !12} ; [ DW_TAG_arg_variable ]
!78 = !MDLocation(line: 6, column: 13, scope: !46)
!79 = !MDLocation(line: 6, column: 23, scope: !46)
!80 = !MDLocation(line: 6, column: 24, scope: !81)
!81 = !{!"0xb\006\0023\002", !82, !46} ; [ DW_TAG_lexical_block ]
!82 = !{!"main.cpp", !"/Users/echristo/tmp/bad-struct-ref"}
!83 = !{i32 1, !"Debug Info Version", i32 2}
