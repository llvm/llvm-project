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

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare i8* @objc_msgSend(i8*, i8*, ...)

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define hidden void @foobar_func_block_invoke_0(i8* %.block_descriptor, %0* %loadedMydata, [4 x i32] %bounds.coerce0, [4 x i32] %data.coerce0) ssp {
  %1 = alloca %0*, align 4
  %bounds = alloca %struct.CR, align 4
  %data = alloca %struct.CR, align 4
  call void @llvm.dbg.value(metadata !{i8* %.block_descriptor}, i64 0, metadata !27), !dbg !129
  store %0* %loadedMydata, %0** %1, align 4
  call void @llvm.dbg.declare(metadata !{%0** %1}, metadata !130), !dbg !131
  %2 = bitcast %struct.CR* %bounds to %1*
  %3 = getelementptr %1* %2, i32 0, i32 0
  store [4 x i32] %bounds.coerce0, [4 x i32]* %3
  call void @llvm.dbg.declare(metadata !{%struct.CR* %bounds}, metadata !132), !dbg !133
  %4 = bitcast %struct.CR* %data to %1*
  %5 = getelementptr %1* %4, i32 0, i32 0
  store [4 x i32] %data.coerce0, [4 x i32]* %5
  call void @llvm.dbg.declare(metadata !{%struct.CR* %data}, metadata !134), !dbg !135
  %6 = bitcast i8* %.block_descriptor to %2*
  %7 = getelementptr inbounds %2* %6, i32 0, i32 6
  call void @llvm.dbg.declare(metadata !{%2* %6}, metadata !136), !dbg !137
  call void @llvm.dbg.declare(metadata !{%2* %6}, metadata !138), !dbg !137
  call void @llvm.dbg.declare(metadata !{%2* %6}, metadata !139), !dbg !140
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

!0 = metadata !{i32 786449, metadata !153, i32 16, metadata !"Apple clang version 2.1", i1 false, metadata !"", i32 2, metadata !147, metadata !26, metadata !148, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 786436, metadata !160, metadata !0, metadata !"", i32 248, i64 32, i64 32, i32 0, i32 0, null, metadata !3, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [line 248, size 32, align 32, offset 0] [def] [from ]
!2 = metadata !{i32 786473, metadata !160} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786472, metadata !"Ver1", i64 0} ; [ DW_TAG_enumerator ]
!5 = metadata !{i32 786436, metadata !160, metadata !0, metadata !"Mode", i32 79, i64 32, i64 32, i32 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [Mode] [line 79, size 32, align 32, offset 0] [def] [from ]
!6 = metadata !{i32 786473, metadata !161} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 786472, metadata !"One", i64 0} ; [ DW_TAG_enumerator ]
!9 = metadata !{i32 786436, metadata !149, metadata !0, metadata !"", i32 15, i64 32, i64 32, i32 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [line 15, size 32, align 32, offset 0] [def] [from ]
!10 = metadata !{i32 786473, metadata !149} ; [ DW_TAG_file_type ]
!11 = metadata !{metadata !12, metadata !13}
!12 = metadata !{i32 786472, metadata !"Unknown", i64 0} ; [ DW_TAG_enumerator ]
!13 = metadata !{i32 786472, metadata !"Known", i64 1} ; [ DW_TAG_enumerator ]
!14 = metadata !{i32 786436, metadata !150, metadata !0, metadata !"", i32 20, i64 32, i64 32, i32 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [line 20, size 32, align 32, offset 0] [def] [from ]
!15 = metadata !{i32 786473, metadata !150} ; [ DW_TAG_file_type ]
!16 = metadata !{metadata !17, metadata !18}
!17 = metadata !{i32 786472, metadata !"Single", i64 0} ; [ DW_TAG_enumerator ]
!18 = metadata !{i32 786472, metadata !"Double", i64 1} ; [ DW_TAG_enumerator ]
!19 = metadata !{i32 786436, metadata !151, metadata !0, metadata !"", i32 14, i64 32, i64 32, i32 0, i32 0, null, metadata !21, i32 0, null, null, null} ; [ DW_TAG_enumeration_type ] [line 14, size 32, align 32, offset 0] [def] [from ]
!20 = metadata !{i32 786473, metadata !151} ; [ DW_TAG_file_type ]
!21 = metadata !{metadata !22}
!22 = metadata !{i32 786472, metadata !"Eleven", i64 0} ; [ DW_TAG_enumerator ]
!23 = metadata !{i32 786478, metadata !152, metadata !24, metadata !"foobar_func_block_invoke_0", metadata !"foobar_func_block_invoke_0", metadata !"", i32 609, metadata !25, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8*, %0*, [4 x i32], [4 x i32])* @foobar_func_block_invoke_0, null, null, null, i32 609} ; [ DW_TAG_subprogram ] [line 609] [local] [def] [foobar_func_block_invoke_0]
!24 = metadata !{i32 786473, metadata !152} ; [ DW_TAG_file_type ]
!25 = metadata !{i32 786453, metadata !152, metadata !24, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !26, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!26 = metadata !{null}
!27 = metadata !{i32 786689, metadata !23, metadata !".block_descriptor", metadata !24, i32 16777825, metadata !28, i32 64, null} ; [ DW_TAG_arg_variable ]
!28 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 0, i64 0, i32 0, metadata !29} ; [ DW_TAG_pointer_type ]
!29 = metadata !{i32 786451, metadata !152, metadata !24, metadata !"__block_literal_14", i32 609, i64 256, i64 32, i32 0, i32 0, null, metadata !30, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_14] [line 609, size 256, align 32, offset 0] [def] [from ]
!30 = metadata !{metadata !31, metadata !33, metadata !35, metadata !36, metadata !37, metadata !48, metadata !89, metadata !124}
!31 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__isa", i32 609, i64 32, i64 32, i64 0, i32 0, metadata !32} ; [ DW_TAG_member ]
!32 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ]
!33 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__flags", i32 609, i64 32, i64 32, i64 32, i32 0, metadata !34} ; [ DW_TAG_member ]
!34 = metadata !{i32 786468, null, metadata !0, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!35 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__reserved", i32 609, i64 32, i64 32, i64 64, i32 0, metadata !34} ; [ DW_TAG_member ]
!36 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__FuncPtr", i32 609, i64 32, i64 32, i64 96, i32 0, metadata !32} ; [ DW_TAG_member ]
!37 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__descriptor", i32 609, i64 32, i64 32, i64 128, i32 0, metadata !38} ; [ DW_TAG_member ]
!38 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !39} ; [ DW_TAG_pointer_type ]
!39 = metadata !{i32 786451, metadata !153, metadata !0, metadata !"__block_descriptor_withcopydispose", i32 307, i64 128, i64 32, i32 0, i32 0, null, metadata !41, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor_withcopydispose] [line 307, size 128, align 32, offset 0] [def] [from ]
!40 = metadata !{i32 786473, metadata !153} ; [ DW_TAG_file_type ]
!41 = metadata !{metadata !42, metadata !44, metadata !45, metadata !47}
!42 = metadata !{i32 786445, metadata !153, metadata !40, metadata !"reserved", i32 307, i64 32, i64 32, i64 0, i32 0, metadata !43} ; [ DW_TAG_member ]
!43 = metadata !{i32 786468, null, metadata !0, metadata !"long unsigned int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!44 = metadata !{i32 786445, metadata !153, metadata !40, metadata !"Size", i32 307, i64 32, i64 32, i64 32, i32 0, metadata !43} ; [ DW_TAG_member ]
!45 = metadata !{i32 786445, metadata !153, metadata !40, metadata !"CopyFuncPtr", i32 307, i64 32, i64 32, i64 64, i32 0, metadata !46} ; [ DW_TAG_member ]
!46 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !32} ; [ DW_TAG_pointer_type ]
!47 = metadata !{i32 786445, metadata !153, metadata !40, metadata !"DestroyFuncPtr", i32 307, i64 32, i64 32, i64 96, i32 0, metadata !46} ; [ DW_TAG_member ]
!48 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"mydata", i32 609, i64 32, i64 32, i64 160, i32 0, metadata !49} ; [ DW_TAG_member ]
!49 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 0, i64 0, i32 0, metadata !50} ; [ DW_TAG_pointer_type ]
!50 = metadata !{i32 786451, metadata !152, metadata !24, metadata !"", i32 0, i64 224, i64 0, i32 0, i32 16, null, metadata !51, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [line 0, size 224, align 0, offset 0] [def] [from ]
!51 = metadata !{metadata !52, metadata !53, metadata !54, metadata !55, metadata !56, metadata !57, metadata !58}
!52 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__isa", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !32} ; [ DW_TAG_member ]
!53 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__forwarding", i32 0, i64 32, i64 32, i64 32, i32 0, metadata !32} ; [ DW_TAG_member ]
!54 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__flags", i32 0, i64 32, i64 32, i64 64, i32 0, metadata !34} ; [ DW_TAG_member ]
!55 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__size", i32 0, i64 32, i64 32, i64 96, i32 0, metadata !34} ; [ DW_TAG_member ]
!56 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__copy_helper", i32 0, i64 32, i64 32, i64 128, i32 0, metadata !32} ; [ DW_TAG_member ]
!57 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"__destroy_helper", i32 0, i64 32, i64 32, i64 160, i32 0, metadata !32} ; [ DW_TAG_member ]
!58 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"mydata", i32 0, i64 32, i64 32, i64 192, i32 0, metadata !59} ; [ DW_TAG_member ]
!59 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !60} ; [ DW_TAG_pointer_type ]
!60 = metadata !{i32 786451, metadata !154, metadata !24, metadata !"UIMydata", i32 26, i64 128, i64 32, i32 0, i32 0, null, metadata !62, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [UIMydata] [line 26, size 128, align 32, offset 0] [def] [from ]
!61 = metadata !{i32 786473, metadata !154} ; [ DW_TAG_file_type ]
!62 = metadata !{metadata !63, metadata !71, metadata !75, metadata !79}
!63 = metadata !{i32 786460, metadata !60, null, metadata !61, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !64} ; [ DW_TAG_inheritance ]
!64 = metadata !{i32 786451, metadata !155, metadata !40, metadata !"NSO", i32 66, i64 32, i64 32, i32 0, i32 0, null, metadata !66, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [NSO] [line 66, size 32, align 32, offset 0] [def] [from ]
!65 = metadata !{i32 786473, metadata !155} ; [ DW_TAG_file_type ]
!66 = metadata !{metadata !67}
!67 = metadata !{i32 786445, metadata !155, metadata !65, metadata !"isa", i32 67, i64 32, i64 32, i64 0, i32 2, metadata !68, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!68 = metadata !{i32 786454, metadata !153, metadata !0, metadata !"Class", i32 197, i64 0, i64 0, i64 0, i32 0, metadata !69} ; [ DW_TAG_typedef ]
!69 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !70} ; [ DW_TAG_pointer_type ]
!70 = metadata !{i32 786451, metadata !153, metadata !0, metadata !"objc_class", i32 0, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [objc_class] [line 0, size 0, align 0, offset 0] [decl] [from ]
!71 = metadata !{i32 786445, metadata !154, metadata !61, metadata !"_mydataRef", i32 28, i64 32, i64 32, i64 32, i32 0, metadata !72, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!72 = metadata !{i32 786454, metadata !152, metadata !0, metadata !"CFTypeRef", i32 313, i64 0, i64 0, i64 0, i32 0, metadata !73} ; [ DW_TAG_typedef ]
!73 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !74} ; [ DW_TAG_pointer_type ]
!74 = metadata !{i32 786470, null, metadata !0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null} ; [ DW_TAG_const_type ]
!75 = metadata !{i32 786445, metadata !154, metadata !61, metadata !"_scale", i32 29, i64 32, i64 32, i64 64, i32 0, metadata !76, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!76 = metadata !{i32 786454, metadata !156, metadata !0, metadata !"Float", i32 89, i64 0, i64 0, i64 0, i32 0, metadata !78} ; [ DW_TAG_typedef ]
!77 = metadata !{i32 786473, metadata !156} ; [ DW_TAG_file_type ]
!78 = metadata !{i32 786468, null, metadata !0, metadata !"float", i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!79 = metadata !{i32 786445, metadata !154, metadata !61, metadata !"_mydataFlags", i32 37, i64 8, i64 8, i64 96, i32 0, metadata !80, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!80 = metadata !{i32 786451, metadata !154, metadata !0, metadata !"", i32 30, i64 8, i64 8, i32 0, i32 0, null, metadata !81, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [line 30, size 8, align 8, offset 0] [def] [from ]
!81 = metadata !{metadata !82, metadata !84, metadata !85, metadata !86, metadata !87, metadata !88}
!82 = metadata !{i32 786445, metadata !154, metadata !61, metadata !"named", i32 31, i64 1, i64 32, i64 0, i32 0, metadata !83} ; [ DW_TAG_member ]
!83 = metadata !{i32 786468, null, metadata !0, metadata !"unsigned int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!84 = metadata !{i32 786445, metadata !154, metadata !61, metadata !"mydataO", i32 32, i64 3, i64 32, i64 1, i32 0, metadata !83} ; [ DW_TAG_member ]
!85 = metadata !{i32 786445, metadata !154, metadata !61, metadata !"cached", i32 33, i64 1, i64 32, i64 4, i32 0, metadata !83} ; [ DW_TAG_member ]
!86 = metadata !{i32 786445, metadata !154, metadata !61, metadata !"hasBeenCached", i32 34, i64 1, i64 32, i64 5, i32 0, metadata !83} ; [ DW_TAG_member ]
!87 = metadata !{i32 786445, metadata !154, metadata !61, metadata !"hasPattern", i32 35, i64 1, i64 32, i64 6, i32 0, metadata !83} ; [ DW_TAG_member ]
!88 = metadata !{i32 786445, metadata !154, metadata !61, metadata !"isCIMydata", i32 36, i64 1, i64 32, i64 7, i32 0, metadata !83} ; [ DW_TAG_member ]
!89 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"self", i32 609, i64 32, i64 32, i64 192, i32 0, metadata !90} ; [ DW_TAG_member ]
!90 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !91} ; [ DW_TAG_pointer_type ]
!91 = metadata !{i32 786451, metadata !152, metadata !40, metadata !"MyWork", i32 36, i64 384, i64 32, i32 0, i32 0, null, metadata !92, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [MyWork] [line 36, size 384, align 32, offset 0] [def] [from ]
!92 = metadata !{metadata !93, metadata !98, metadata !101, metadata !107, metadata !123}
!93 = metadata !{i32 786460, metadata !152, metadata !91, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !94} ; [ DW_TAG_inheritance ]
!94 = metadata !{i32 786451, metadata !157, metadata !40, metadata !"twork", i32 43, i64 32, i64 32, i32 0, i32 0, null, metadata !96, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [twork] [line 43, size 32, align 32, offset 0] [def] [from ]
!95 = metadata !{i32 786473, metadata !157} ; [ DW_TAG_file_type ]
!96 = metadata !{metadata !97}
!97 = metadata !{i32 786460, metadata !94, null, metadata !95, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !64} ; [ DW_TAG_inheritance ]
!98 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"_itemID", i32 38, i64 64, i64 32, i64 32, i32 1, metadata !99, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!99 = metadata !{i32 786454, metadata !153, metadata !0, metadata !"uint64_t", i32 55, i64 0, i64 0, i64 0, i32 0, metadata !100} ; [ DW_TAG_typedef ]
!100 = metadata !{i32 786468, null, metadata !0, metadata !"long long unsigned int", i32 0, i64 64, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!101 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"_library", i32 39, i64 32, i64 32, i64 96, i32 1, metadata !102, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!102 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !103} ; [ DW_TAG_pointer_type ]
!103 = metadata !{i32 786451, metadata !158, metadata !40, metadata !"MyLibrary2", i32 22, i64 32, i64 32, i32 0, i32 0, null, metadata !105, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [MyLibrary2] [line 22, size 32, align 32, offset 0] [def] [from ]
!104 = metadata !{i32 786473, metadata !158} ; [ DW_TAG_file_type ]
!105 = metadata !{metadata !106}
!106 = metadata !{i32 786460, metadata !103, null, metadata !104, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !64} ; [ DW_TAG_inheritance ]
!107 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"_bounds", i32 40, i64 128, i64 32, i64 128, i32 1, metadata !108, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!108 = metadata !{i32 786454, metadata !153, metadata !0, metadata !"CR", i32 33, i64 0, i64 0, i64 0, i32 0, metadata !109} ; [ DW_TAG_typedef ]
!109 = metadata !{i32 786451, metadata !156, metadata !0, metadata !"CR", i32 29, i64 128, i64 32, i32 0, i32 0, null, metadata !110, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [CR] [line 29, size 128, align 32, offset 0] [def] [from ]
!110 = metadata !{metadata !111, metadata !117}
!111 = metadata !{i32 786445, metadata !156, metadata !77, metadata !"origin", i32 30, i64 64, i64 32, i64 0, i32 0, metadata !112} ; [ DW_TAG_member ]
!112 = metadata !{i32 786454, metadata !156, metadata !0, metadata !"CP", i32 17, i64 0, i64 0, i64 0, i32 0, metadata !113} ; [ DW_TAG_typedef ]
!113 = metadata !{i32 786451, metadata !156, metadata !0, metadata !"CP", i32 13, i64 64, i64 32, i32 0, i32 0, null, metadata !114, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [CP] [line 13, size 64, align 32, offset 0] [def] [from ]
!114 = metadata !{metadata !115, metadata !116}
!115 = metadata !{i32 786445, metadata !156, metadata !77, metadata !"x", i32 14, i64 32, i64 32, i64 0, i32 0, metadata !76} ; [ DW_TAG_member ]
!116 = metadata !{i32 786445, metadata !156, metadata !77, metadata !"y", i32 15, i64 32, i64 32, i64 32, i32 0, metadata !76} ; [ DW_TAG_member ]
!117 = metadata !{i32 786445, metadata !156, metadata !77, metadata !"size", i32 31, i64 64, i64 32, i64 64, i32 0, metadata !118} ; [ DW_TAG_member ]
!118 = metadata !{i32 786454, metadata !156, metadata !0, metadata !"Size", i32 25, i64 0, i64 0, i64 0, i32 0, metadata !119} ; [ DW_TAG_typedef ]
!119 = metadata !{i32 786451, metadata !156, metadata !0, metadata !"Size", i32 21, i64 64, i64 32, i32 0, i32 0, null, metadata !120, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [Size] [line 21, size 64, align 32, offset 0] [def] [from ]
!120 = metadata !{metadata !121, metadata !122}
!121 = metadata !{i32 786445, metadata !156, metadata !77, metadata !"width", i32 22, i64 32, i64 32, i64 0, i32 0, metadata !76} ; [ DW_TAG_member ]
!122 = metadata !{i32 786445, metadata !156, metadata !77, metadata !"height", i32 23, i64 32, i64 32, i64 32, i32 0, metadata !76} ; [ DW_TAG_member ]
!123 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"_data", i32 40, i64 128, i64 32, i64 256, i32 1, metadata !108, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!124 = metadata !{i32 786445, metadata !152, metadata !24, metadata !"semi", i32 609, i64 32, i64 32, i64 224, i32 0, metadata !125} ; [ DW_TAG_member ]
!125 = metadata !{i32 786454, metadata !152, metadata !0, metadata !"d_t", i32 35, i64 0, i64 0, i64 0, i32 0, metadata !126} ; [ DW_TAG_typedef ]
!126 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !127} ; [ DW_TAG_pointer_type ]
!127 = metadata !{i32 786451, metadata !159, metadata !0, metadata !"my_struct", i32 49, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [my_struct] [line 49, size 0, align 0, offset 0] [decl] [from ]
!128 = metadata !{i32 786473, metadata !159} ; [ DW_TAG_file_type ]
!129 = metadata !{i32 609, i32 144, metadata !23, null}
!130 = metadata !{i32 786689, metadata !23, metadata !"loadedMydata", metadata !24, i32 33555041, metadata !59, i32 0, null} ; [ DW_TAG_arg_variable ]
!131 = metadata !{i32 609, i32 155, metadata !23, null}
!132 = metadata !{i32 786689, metadata !23, metadata !"bounds", metadata !24, i32 50332257, metadata !108, i32 0, null} ; [ DW_TAG_arg_variable ]
!133 = metadata !{i32 609, i32 175, metadata !23, null}
!134 = metadata !{i32 786689, metadata !23, metadata !"data", metadata !24, i32 67109473, metadata !108, i32 0, null} ; [ DW_TAG_arg_variable ]
!135 = metadata !{i32 609, i32 190, metadata !23, null}
!136 = metadata !{i32 786688, metadata !23, metadata !"mydata", metadata !24, i32 604, metadata !50, i32 0, null, i64 1, i64 20, i64 2, i64 1, i64 4, i64 2, i64 1, i64 24} ; [ DW_TAG_auto_variable ]
!137 = metadata !{i32 604, i32 49, metadata !23, null}
!138 = metadata !{i32 786688, metadata !23, metadata !"self", metadata !40, i32 604, metadata !90, i32 0, null, i64 1, i64 24} ; [ DW_TAG_auto_variable ]
!139 = metadata !{i32 786688, metadata !23, metadata !"semi", metadata !24, i32 607, metadata !125, i32 0, null, i64 1, i64 28} ; [ DW_TAG_auto_variable ]
!140 = metadata !{i32 607, i32 30, metadata !23, null}
!141 = metadata !{i32 610, i32 17, metadata !142, null}
!142 = metadata !{i32 786443, metadata !152, metadata !23, i32 609, i32 200, i32 94} ; [ DW_TAG_lexical_block ]
!143 = metadata !{i32 611, i32 17, metadata !142, null}
!144 = metadata !{i32 612, i32 17, metadata !142, null}
!145 = metadata !{i32 613, i32 17, metadata !142, null}
!146 = metadata !{i32 615, i32 13, metadata !142, null}
!147 = metadata !{metadata !1, metadata !1, metadata !5, metadata !5, metadata !9, metadata !14, metadata !19, metadata !19, metadata !14, metadata !14, metadata !14, metadata !19, metadata !19, metadata !19}
!148 = metadata !{metadata !23}
!149 = metadata !{metadata !"header3.h", metadata !"/Volumes/Sandbox/llvm"}
!150 = metadata !{metadata !"Private.h", metadata !"/Volumes/Sandbox/llvm"}
!151 = metadata !{metadata !"header4.h", metadata !"/Volumes/Sandbox/llvm"}
!152 = metadata !{metadata !"MyLibrary.m", metadata !"/Volumes/Sandbox/llvm"}
!153 = metadata !{metadata !"MyLibrary.i", metadata !"/Volumes/Sandbox/llvm"}
!154 = metadata !{metadata !"header11.h", metadata !"/Volumes/Sandbox/llvm"}
!155 = metadata !{metadata !"NSO.h", metadata !"/Volumes/Sandbox/llvm"}
!156 = metadata !{metadata !"header12.h", metadata !"/Volumes/Sandbox/llvm"}
!157 = metadata !{metadata !"header13.h", metadata !"/Volumes/Sandbox/llvm"}
!158 = metadata !{metadata !"header14.h", metadata !"/Volumes/Sandbox/llvm"}
!159 = metadata !{metadata !"header15.h", metadata !"/Volumes/Sandbox/llvm"}
!160 = metadata !{metadata !"header.h", metadata !"/Volumes/Sandbox/llvm"}
!161 = metadata !{metadata !"header2.h", metadata !"/Volumes/Sandbox/llvm"}
!162 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
