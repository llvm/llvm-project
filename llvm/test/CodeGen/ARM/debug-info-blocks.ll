; RUN: llc -O0 < %s | FileCheck %s
; CHECK: @DEBUG_VALUE: foobar_func_block_invoke_0:mydata <- [SP+{{[0-9]+}}]
; Radar 9331779
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios"

%0 = type opaque
%1 = type { [4 x i32] }
%2 = type <{ i8*, i32, i32, i8*, %struct.Re*, i8*, %3*, %struct.my_struct* }>
%3 = type opaque
%struct.CP = type { float, float }
%struct.CR = type { %struct.CP, %struct.CP }
%struct.Re = type { i32, i32 }
%struct.__block_byref_mydata = type { i8*, %struct.__block_byref_mydata*, i32, i32, i8*, i8*, %0* }
%struct.my_struct = type opaque

@"\01L_OBJC_SELECTOR_REFERENCES_13" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"OBJC_IVAR_$_MyWork._bounds" = external hidden global i32, section "__DATA, __objc_const", align 4
@"OBJC_IVAR_$_MyWork._data" = external hidden global i32, section "__DATA, __objc_const", align 4
@"\01L_OBJC_SELECTOR_REFERENCES_222" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i8* @objc_msgSend(i8*, i8*, ...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define hidden void @foobar_func_block_invoke_0(i8* %.block_descriptor, %0* %loadedMydata, [4 x i32] %bounds.coerce0, [4 x i32] %data.coerce0) ssp {
  %1 = alloca %0*, align 4
  %bounds = alloca %struct.CR, align 4
  %data = alloca %struct.CR, align 4
  call void @llvm.dbg.value(metadata i8* %.block_descriptor, i64 0, metadata !27, metadata !{!"0x102"}), !dbg !129
  store %0* %loadedMydata, %0** %1, align 4
  call void @llvm.dbg.declare(metadata %0** %1, metadata !130, metadata !{!"0x102"}), !dbg !131
  %2 = bitcast %struct.CR* %bounds to %1*
  %3 = getelementptr %1* %2, i32 0, i32 0
  store [4 x i32] %bounds.coerce0, [4 x i32]* %3
  call void @llvm.dbg.declare(metadata %struct.CR* %bounds, metadata !132, metadata !{!"0x102"}), !dbg !133
  %4 = bitcast %struct.CR* %data to %1*
  %5 = getelementptr %1* %4, i32 0, i32 0
  store [4 x i32] %data.coerce0, [4 x i32]* %5
  call void @llvm.dbg.declare(metadata %struct.CR* %data, metadata !134, metadata !{!"0x102"}), !dbg !135
  %6 = bitcast i8* %.block_descriptor to %2*
  %7 = getelementptr inbounds %2* %6, i32 0, i32 6
  call void @llvm.dbg.declare(metadata %2* %6, metadata !136, metadata !163), !dbg !137
  call void @llvm.dbg.declare(metadata %2* %6, metadata !138, metadata !164), !dbg !137
  call void @llvm.dbg.declare(metadata %2* %6, metadata !139, metadata !165), !dbg !140
  %8 = load %0** %1, align 4, !dbg !141
  %9 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_13", !dbg !141
  %10 = bitcast %0* %8 to i8*, !dbg !141
  %11 = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %10, i8* %9), !dbg !141
  %12 = bitcast i8* %11 to %0*, !dbg !141
  %13 = getelementptr inbounds %2* %6, i32 0, i32 5, !dbg !141
  %14 = load i8** %13, !dbg !141
  %15 = bitcast i8* %14 to %struct.__block_byref_mydata*, !dbg !141
  %16 = getelementptr inbounds %struct.__block_byref_mydata* %15, i32 0, i32 1, !dbg !141
  %17 = load %struct.__block_byref_mydata** %16, !dbg !141
  %18 = getelementptr inbounds %struct.__block_byref_mydata* %17, i32 0, i32 6, !dbg !141
  store %0* %12, %0** %18, align 4, !dbg !141
  %19 = getelementptr inbounds %2* %6, i32 0, i32 6, !dbg !143
  %20 = load %3** %19, align 4, !dbg !143
  %21 = load i32* @"OBJC_IVAR_$_MyWork._data", !dbg !143
  %22 = bitcast %3* %20 to i8*, !dbg !143
  %23 = getelementptr inbounds i8* %22, i32 %21, !dbg !143
  %24 = bitcast i8* %23 to %struct.CR*, !dbg !143
  %25 = bitcast %struct.CR* %24 to i8*, !dbg !143
  %26 = bitcast %struct.CR* %data to i8*, !dbg !143
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %25, i8* %26, i32 16, i32 4, i1 false), !dbg !143
  %27 = getelementptr inbounds %2* %6, i32 0, i32 6, !dbg !144
  %28 = load %3** %27, align 4, !dbg !144
  %29 = load i32* @"OBJC_IVAR_$_MyWork._bounds", !dbg !144
  %30 = bitcast %3* %28 to i8*, !dbg !144
  %31 = getelementptr inbounds i8* %30, i32 %29, !dbg !144
  %32 = bitcast i8* %31 to %struct.CR*, !dbg !144
  %33 = bitcast %struct.CR* %32 to i8*, !dbg !144
  %34 = bitcast %struct.CR* %bounds to i8*, !dbg !144
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %33, i8* %34, i32 16, i32 4, i1 false), !dbg !144
  %35 = getelementptr inbounds %2* %6, i32 0, i32 6, !dbg !145
  %36 = load %3** %35, align 4, !dbg !145
  %37 = getelementptr inbounds %2* %6, i32 0, i32 5, !dbg !145
  %38 = load i8** %37, !dbg !145
  %39 = bitcast i8* %38 to %struct.__block_byref_mydata*, !dbg !145
  %40 = getelementptr inbounds %struct.__block_byref_mydata* %39, i32 0, i32 1, !dbg !145
  %41 = load %struct.__block_byref_mydata** %40, !dbg !145
  %42 = getelementptr inbounds %struct.__block_byref_mydata* %41, i32 0, i32 6, !dbg !145
  %43 = load %0** %42, align 4, !dbg !145
  %44 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_222", !dbg !145
  %45 = bitcast %3* %36 to i8*, !dbg !145
  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %0*)*)(i8* %45, i8* %44, %0* %43), !dbg !145
  ret void, !dbg !146
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!162}

!0 = !{!"0x11\0016\00Apple clang version 2.1\000\00\002\00\001", !153, !147, !26, !148, null, null} ; [ DW_TAG_compile_unit ]
!1 = !{!"0x4\00\00248\0032\0032\000\000\000", !160, !0, null, !3, null, null, null} ; [ DW_TAG_enumeration_type ] [line 248, size 32, align 32, offset 0] [def] [from ]
!2 = !{!"0x29", !160} ; [ DW_TAG_file_type ]
!3 = !{!4}
!4 = !{!"0x28\00Ver1\000"} ; [ DW_TAG_enumerator ]
!5 = !{!"0x4\00Mode\0079\0032\0032\000\000\000", !160, !0, null, !7, null, null, null} ; [ DW_TAG_enumeration_type ] [Mode] [line 79, size 32, align 32, offset 0] [def] [from ]
!6 = !{!"0x29", !161} ; [ DW_TAG_file_type ]
!7 = !{!8}
!8 = !{!"0x28\00One\000"} ; [ DW_TAG_enumerator ]
!9 = !{!"0x4\00\0015\0032\0032\000\000\000", !149, !0, null, !11, null, null, null} ; [ DW_TAG_enumeration_type ] [line 15, size 32, align 32, offset 0] [def] [from ]
!10 = !{!"0x29", !149} ; [ DW_TAG_file_type ]
!11 = !{!12, !13}
!12 = !{!"0x28\00Unknown\000"} ; [ DW_TAG_enumerator ]
!13 = !{!"0x28\00Known\001"} ; [ DW_TAG_enumerator ]
!14 = !{!"0x4\00\0020\0032\0032\000\000\000", !150, !0, null, !16, null, null, null} ; [ DW_TAG_enumeration_type ] [line 20, size 32, align 32, offset 0] [def] [from ]
!15 = !{!"0x29", !150} ; [ DW_TAG_file_type ]
!16 = !{!17, !18}
!17 = !{!"0x28\00Single\000"} ; [ DW_TAG_enumerator ]
!18 = !{!"0x28\00Double\001"} ; [ DW_TAG_enumerator ]
!19 = !{!"0x4\00\0014\0032\0032\000\000\000", !151, !0, null, !21, null, null, null} ; [ DW_TAG_enumeration_type ] [line 14, size 32, align 32, offset 0] [def] [from ]
!20 = !{!"0x29", !151} ; [ DW_TAG_file_type ]
!21 = !{!22}
!22 = !{!"0x28\00Eleven\000"} ; [ DW_TAG_enumerator ]
!23 = !{!"0x2e\00foobar_func_block_invoke_0\00foobar_func_block_invoke_0\00\00609\001\001\000\006\00256\000\00609", !152, !24, !25, null, void (i8*, %0*, [4 x i32], [4 x i32])* @foobar_func_block_invoke_0, null, null, null} ; [ DW_TAG_subprogram ] [line 609] [local] [def] [foobar_func_block_invoke_0]
!24 = !{!"0x29", !152} ; [ DW_TAG_file_type ]
!25 = !{!"0x15\00\000\000\000\000\000\000", !152, !24, null, !26, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!26 = !{null}
!27 = !{!"0x101\00.block_descriptor\0016777825\0064", !23, !24, !28} ; [ DW_TAG_arg_variable ]
!28 = !{!"0xf\00\000\0032\000\000\000", null, !0, !29} ; [ DW_TAG_pointer_type ]
!29 = !{!"0x13\00__block_literal_14\00609\00256\0032\000\000\000", !152, !24, null, !30, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_14] [line 609, size 256, align 32, offset 0] [def] [from ]
!30 = !{!31, !33, !35, !36, !37, !48, !89, !124}
!31 = !{!"0xd\00__isa\00609\0032\0032\000\000", !152, !24, !32} ; [ DW_TAG_member ]
!32 = !{!"0xf\00\000\0032\0032\000\000", null, !0, null} ; [ DW_TAG_pointer_type ]
!33 = !{!"0xd\00__flags\00609\0032\0032\0032\000", !152, !24, !34} ; [ DW_TAG_member ]
!34 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !0} ; [ DW_TAG_base_type ]
!35 = !{!"0xd\00__reserved\00609\0032\0032\0064\000", !152, !24, !34} ; [ DW_TAG_member ]
!36 = !{!"0xd\00__FuncPtr\00609\0032\0032\0096\000", !152, !24, !32} ; [ DW_TAG_member ]
!37 = !{!"0xd\00__descriptor\00609\0032\0032\00128\000", !152, !24, !38} ; [ DW_TAG_member ]
!38 = !{!"0xf\00\000\0032\0032\000\000", null, !0, !39} ; [ DW_TAG_pointer_type ]
!39 = !{!"0x13\00__block_descriptor_withcopydispose\00307\00128\0032\000\000\000", !153, !0, null, !41, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor_withcopydispose] [line 307, size 128, align 32, offset 0] [def] [from ]
!40 = !{!"0x29", !153} ; [ DW_TAG_file_type ]
!41 = !{!42, !44, !45, !47}
!42 = !{!"0xd\00reserved\00307\0032\0032\000\000", !153, !40, !43} ; [ DW_TAG_member ]
!43 = !{!"0x24\00long unsigned int\000\0032\0032\000\000\007", null, !0} ; [ DW_TAG_base_type ]
!44 = !{!"0xd\00Size\00307\0032\0032\0032\000", !153, !40, !43} ; [ DW_TAG_member ]
!45 = !{!"0xd\00CopyFuncPtr\00307\0032\0032\0064\000", !153, !40, !46} ; [ DW_TAG_member ]
!46 = !{!"0xf\00\000\0032\0032\000\000", null, !0, !32} ; [ DW_TAG_pointer_type ]
!47 = !{!"0xd\00DestroyFuncPtr\00307\0032\0032\0096\000", !153, !40, !46} ; [ DW_TAG_member ]
!48 = !{!"0xd\00mydata\00609\0032\0032\00160\000", !152, !24, !49} ; [ DW_TAG_member ]
!49 = !{!"0xf\00\000\0032\000\000\000", null, !0, !50} ; [ DW_TAG_pointer_type ]
!50 = !{!"0x13\00\000\00224\000\000\0016\000", !152, !24, null, !51, null, null, null} ; [ DW_TAG_structure_type ] [line 0, size 224, align 0, offset 0] [def] [from ]
!51 = !{!52, !53, !54, !55, !56, !57, !58}
!52 = !{!"0xd\00__isa\000\0032\0032\000\000", !152, !24, !32} ; [ DW_TAG_member ]
!53 = !{!"0xd\00__forwarding\000\0032\0032\0032\000", !152, !24, !32} ; [ DW_TAG_member ]
!54 = !{!"0xd\00__flags\000\0032\0032\0064\000", !152, !24, !34} ; [ DW_TAG_member ]
!55 = !{!"0xd\00__size\000\0032\0032\0096\000", !152, !24, !34} ; [ DW_TAG_member ]
!56 = !{!"0xd\00__copy_helper\000\0032\0032\00128\000", !152, !24, !32} ; [ DW_TAG_member ]
!57 = !{!"0xd\00__destroy_helper\000\0032\0032\00160\000", !152, !24, !32} ; [ DW_TAG_member ]
!58 = !{!"0xd\00mydata\000\0032\0032\00192\000", !152, !24, !59} ; [ DW_TAG_member ]
!59 = !{!"0xf\00\000\0032\0032\000\000", null, !0, !60} ; [ DW_TAG_pointer_type ]
!60 = !{!"0x13\00UIMydata\0026\00128\0032\000\000\0016", !154, !24, null, !62, null, null, null} ; [ DW_TAG_structure_type ] [UIMydata] [line 26, size 128, align 32, offset 0] [def] [from ]
!61 = !{!"0x29", !154} ; [ DW_TAG_file_type ]
!62 = !{!63, !71, !75, !79}
!63 = !{!"0x1c\00\000\000\000\000\000", !60, null, !64} ; [ DW_TAG_inheritance ]
!64 = !{!"0x13\00NSO\0066\0032\0032\000\000\0016", !155, !40, null, !66, null, null, null} ; [ DW_TAG_structure_type ] [NSO] [line 66, size 32, align 32, offset 0] [def] [from ]
!65 = !{!"0x29", !155} ; [ DW_TAG_file_type ]
!66 = !{!67}
!67 = !{!"0xd\00isa\0067\0032\0032\000\002", !155, !65, !68, !"", !"", !"", i32 0} ; [ DW_TAG_member ]
!68 = !{!"0x16\00Class\00197\000\000\000\000", !153, !0, !69} ; [ DW_TAG_typedef ]
!69 = !{!"0xf\00\000\0032\0032\000\000", null, !0, !70} ; [ DW_TAG_pointer_type ]
!70 = !{!"0x13\00objc_class\000\000\000\000\004\000", !153, !0, null, null, null, null, null} ; [ DW_TAG_structure_type ] [objc_class] [line 0, size 0, align 0, offset 0] [decl] [from ]
!71 = !{!"0xd\00_mydataRef\0028\0032\0032\0032\000", !154, !61, !72, !"", !"", !"", i32 0} ; [ DW_TAG_member ]
!72 = !{!"0x16\00CFTypeRef\00313\000\000\000\000", !152, !0, !73} ; [ DW_TAG_typedef ]
!73 = !{!"0xf\00\000\0032\0032\000\000", null, !0, !74} ; [ DW_TAG_pointer_type ]
!74 = !{!"0x26\00\000\000\000\000\000", null, !0, null} ; [ DW_TAG_const_type ]
!75 = !{!"0xd\00_scale\0029\0032\0032\0064\000", !154, !61, !76, !"", !"", !"", i32 0} ; [ DW_TAG_member ]
!76 = !{!"0x16\00Float\0089\000\000\000\000", !156, !0, !78} ; [ DW_TAG_typedef ]
!77 = !{!"0x29", !156} ; [ DW_TAG_file_type ]
!78 = !{!"0x24\00float\000\0032\0032\000\000\004", null, !0} ; [ DW_TAG_base_type ]
!79 = !{!"0xd\00_mydataFlags\0037\008\008\0096\000", !154, !61, !80, !"", !"", !"", i32 0} ; [ DW_TAG_member ]
!80 = !{!"0x13\00\0030\008\008\000\000\000", !154, !0, null, !81, null, null, null} ; [ DW_TAG_structure_type ] [line 30, size 8, align 8, offset 0] [def] [from ]
!81 = !{!82, !84, !85, !86, !87, !88}
!82 = !{!"0xd\00named\0031\001\0032\000\000", !154, !61, !83} ; [ DW_TAG_member ]
!83 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, !0} ; [ DW_TAG_base_type ]
!84 = !{!"0xd\00mydataO\0032\003\0032\001\000", !154, !61, !83} ; [ DW_TAG_member ]
!85 = !{!"0xd\00cached\0033\001\0032\004\000", !154, !61, !83} ; [ DW_TAG_member ]
!86 = !{!"0xd\00hasBeenCached\0034\001\0032\005\000", !154, !61, !83} ; [ DW_TAG_member ]
!87 = !{!"0xd\00hasPattern\0035\001\0032\006\000", !154, !61, !83} ; [ DW_TAG_member ]
!88 = !{!"0xd\00isCIMydata\0036\001\0032\007\000", !154, !61, !83} ; [ DW_TAG_member ]
!89 = !{!"0xd\00self\00609\0032\0032\00192\000", !152, !24, !90} ; [ DW_TAG_member ]
!90 = !{!"0xf\00\000\0032\0032\000\000", null, !0, !91} ; [ DW_TAG_pointer_type ]
!91 = !{!"0x13\00MyWork\0036\00384\0032\000\000\0016", !152, !40, null, !92, null, null, null} ; [ DW_TAG_structure_type ] [MyWork] [line 36, size 384, align 32, offset 0] [def] [from ]
!92 = !{!93, !98, !101, !107, !123}
!93 = !{!"0x1c\00\000\000\000\000\000", !152, !91, !94} ; [ DW_TAG_inheritance ]
!94 = !{!"0x13\00twork\0043\0032\0032\000\000\0016", !157, !40, null, !96, null, null, null} ; [ DW_TAG_structure_type ] [twork] [line 43, size 32, align 32, offset 0] [def] [from ]
!95 = !{!"0x29", !157} ; [ DW_TAG_file_type ]
!96 = !{!97}
!97 = !{!"0x1c\00\000\000\000\000\000", !94, null, !64} ; [ DW_TAG_inheritance ]
!98 = !{!"0xd\00_itemID\0038\0064\0032\0032\001", !152, !24, !99, !"", !"", !"", i32 0} ; [ DW_TAG_member ]
!99 = !{!"0x16\00uint64_t\0055\000\000\000\000", !153, !0, !100} ; [ DW_TAG_typedef ]
!100 = !{!"0x24\00long long unsigned int\000\0064\0032\000\000\007", null, !0} ; [ DW_TAG_base_type ]
!101 = !{!"0xd\00_library\0039\0032\0032\0096\001", !152, !24, !102, !"", !"", !"", i32 0} ; [ DW_TAG_member ]
!102 = !{!"0xf\00\000\0032\0032\000\000", null, !0, !103} ; [ DW_TAG_pointer_type ]
!103 = !{!"0x13\00MyLibrary2\0022\0032\0032\000\000\0016", !158, !40, null, !105, null, null, null} ; [ DW_TAG_structure_type ] [MyLibrary2] [line 22, size 32, align 32, offset 0] [def] [from ]
!104 = !{!"0x29", !158} ; [ DW_TAG_file_type ]
!105 = !{!106}
!106 = !{!"0x1c\00\000\000\000\000\000", !103, null, !64} ; [ DW_TAG_inheritance ]
!107 = !{!"0xd\00_bounds\0040\00128\0032\00128\001", !152, !24, !108, !"", !"", !"", i32 0} ; [ DW_TAG_member ]
!108 = !{!"0x16\00CR\0033\000\000\000\000", !153, !0, !109} ; [ DW_TAG_typedef ]
!109 = !{!"0x13\00CR\0029\00128\0032\000\000\000", !156, !0, null, !110, null, null, null} ; [ DW_TAG_structure_type ] [CR] [line 29, size 128, align 32, offset 0] [def] [from ]
!110 = !{!111, !117}
!111 = !{!"0xd\00origin\0030\0064\0032\000\000", !156, !77, !112} ; [ DW_TAG_member ]
!112 = !{!"0x16\00CP\0017\000\000\000\000", !156, !0, !113} ; [ DW_TAG_typedef ]
!113 = !{!"0x13\00CP\0013\0064\0032\000\000\000", !156, !0, null, !114, null, null, null} ; [ DW_TAG_structure_type ] [CP] [line 13, size 64, align 32, offset 0] [def] [from ]
!114 = !{!115, !116}
!115 = !{!"0xd\00x\0014\0032\0032\000\000", !156, !77, !76} ; [ DW_TAG_member ]
!116 = !{!"0xd\00y\0015\0032\0032\0032\000", !156, !77, !76} ; [ DW_TAG_member ]
!117 = !{!"0xd\00size\0031\0064\0032\0064\000", !156, !77, !118} ; [ DW_TAG_member ]
!118 = !{!"0x16\00Size\0025\000\000\000\000", !156, !0, !119} ; [ DW_TAG_typedef ]
!119 = !{!"0x13\00Size\0021\0064\0032\000\000\000", !156, !0, null, !120, null, null, null} ; [ DW_TAG_structure_type ] [Size] [line 21, size 64, align 32, offset 0] [def] [from ]
!120 = !{!121, !122}
!121 = !{!"0xd\00width\0022\0032\0032\000\000", !156, !77, !76} ; [ DW_TAG_member ]
!122 = !{!"0xd\00height\0023\0032\0032\0032\000", !156, !77, !76} ; [ DW_TAG_member ]
!123 = !{!"0xd\00_data\0040\00128\0032\00256\001", !152, !24, !108, !"", !"", !"", i32 0} ; [ DW_TAG_member ]
!124 = !{!"0xd\00semi\00609\0032\0032\00224\000", !152, !24, !125} ; [ DW_TAG_member ]
!125 = !{!"0x16\00d_t\0035\000\000\000\000", !152, !0, !126} ; [ DW_TAG_typedef ]
!126 = !{!"0xf\00\000\0032\0032\000\000", null, !0, !127} ; [ DW_TAG_pointer_type ]
!127 = !{!"0x13\00my_struct\0049\000\000\000\004\000", !159, !0, null, null, null, null, null} ; [ DW_TAG_structure_type ] [my_struct] [line 49, size 0, align 0, offset 0] [decl] [from ]
!128 = !{!"0x29", !159} ; [ DW_TAG_file_type ]
!129 = !MDLocation(line: 609, column: 144, scope: !23)
!130 = !{!"0x101\00loadedMydata\0033555041\000", !23, !24, !59} ; [ DW_TAG_arg_variable ]
!131 = !MDLocation(line: 609, column: 155, scope: !23)
!132 = !{!"0x101\00bounds\0050332257\000", !23, !24, !108} ; [ DW_TAG_arg_variable ]
!133 = !MDLocation(line: 609, column: 175, scope: !23)
!134 = !{!"0x101\00data\0067109473\000", !23, !24, !108} ; [ DW_TAG_arg_variable ]
!135 = !MDLocation(line: 609, column: 190, scope: !23)
!136 = !{!"0x100\00mydata\00604\000", !23, !24, !50} ; [ DW_TAG_auto_variable ]
!137 = !MDLocation(line: 604, column: 49, scope: !23)
!138 = !{!"0x100\00self\00604\000", !23, !40, !90} ; [ DW_TAG_auto_variable ]
!139 = !{!"0x100\00semi\00607\000", !23, !24, !125} ; [ DW_TAG_auto_variable ]
!140 = !MDLocation(line: 607, column: 30, scope: !23)
!141 = !MDLocation(line: 610, column: 17, scope: !142)
!142 = !{!"0xb\00609\00200\0094", !152, !23} ; [ DW_TAG_lexical_block ]
!143 = !MDLocation(line: 611, column: 17, scope: !142)
!144 = !MDLocation(line: 612, column: 17, scope: !142)
!145 = !MDLocation(line: 613, column: 17, scope: !142)
!146 = !MDLocation(line: 615, column: 13, scope: !142)
!147 = !{!1, !1, !5, !5, !9, !14, !19, !19, !14, !14, !14, !19, !19, !19}
!148 = !{!23}
!149 = !{!"header3.h", !"/Volumes/Sandbox/llvm"}
!150 = !{!"Private.h", !"/Volumes/Sandbox/llvm"}
!151 = !{!"header4.h", !"/Volumes/Sandbox/llvm"}
!152 = !{!"MyLibrary.m", !"/Volumes/Sandbox/llvm"}
!153 = !{!"MyLibrary.i", !"/Volumes/Sandbox/llvm"}
!154 = !{!"header11.h", !"/Volumes/Sandbox/llvm"}
!155 = !{!"NSO.h", !"/Volumes/Sandbox/llvm"}
!156 = !{!"header12.h", !"/Volumes/Sandbox/llvm"}
!157 = !{!"header13.h", !"/Volumes/Sandbox/llvm"}
!158 = !{!"header14.h", !"/Volumes/Sandbox/llvm"}
!159 = !{!"header15.h", !"/Volumes/Sandbox/llvm"}
!160 = !{!"header.h", !"/Volumes/Sandbox/llvm"}
!161 = !{!"header2.h", !"/Volumes/Sandbox/llvm"}
!162 = !{i32 1, !"Debug Info Version", i32 2}
!163 = !{!"0x102\0034\0020\006\0034\004\006\0034\0024"} ; [ DW_TAG_expression ] [DW_OP_plus 20 DW_OP_deref DW_OP_plus 4 DW_OP_deref DW_OP_plus 24]
!164 = !{!"0x102\0034\0024"} ; [ DW_TAG_expression ] [DW_OP_plus 24]
!165 = !{!"0x102\0034\0028"} ; [ DW_TAG_expression ] [DW_OP_plus 28]
