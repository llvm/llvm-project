; RUN: llc -O0 < %s | FileCheck %s
; CHECK: @DEBUG_VALUE: mydata <- [sp+#{{[0-9]+}}]+#0
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
!llvm.dbg.enum = !{!1, !1, !5, !5, !9, !14, !19, !19, !14, !14, !14, !19, !19, !19}
!llvm.dbg.sp = !{!23}

!0 = metadata !{i32 589841, i32 0, i32 16, metadata !"MyLibrary.i", metadata !"/Volumes/Sandbox/llvm", metadata !"Apple clang version 2.1", i1 true, i1 false, metadata !"", i32 2} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 589828, metadata !0, metadata !"", metadata !2, i32 248, i64 32, i64 32, i32 0, i32 0, i32 0, metadata !3, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
!2 = metadata !{i32 589865, metadata !"header.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !4}
!4 = metadata !{i32 589864, metadata !"Ver1", i64 0} ; [ DW_TAG_enumerator ]
!5 = metadata !{i32 589828, metadata !0, metadata !"Mode", metadata !6, i32 79, i64 32, i64 32, i32 0, i32 0, i32 0, metadata !7, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
!6 = metadata !{i32 589865, metadata !"header2.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 589864, metadata !"One", i64 0} ; [ DW_TAG_enumerator ]
!9 = metadata !{i32 589828, metadata !0, metadata !"", metadata !10, i32 15, i64 32, i64 32, i32 0, i32 0, i32 0, metadata !11, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
!10 = metadata !{i32 589865, metadata !"header3.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!11 = metadata !{metadata !12, metadata !13}
!12 = metadata !{i32 589864, metadata !"Unknown", i64 0} ; [ DW_TAG_enumerator ]
!13 = metadata !{i32 589864, metadata !"Known", i64 1} ; [ DW_TAG_enumerator ]
!14 = metadata !{i32 589828, metadata !0, metadata !"", metadata !15, i32 20, i64 32, i64 32, i32 0, i32 0, i32 0, metadata !16, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
!15 = metadata !{i32 589865, metadata !"Private.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!16 = metadata !{metadata !17, metadata !18}
!17 = metadata !{i32 589864, metadata !"Single", i64 0} ; [ DW_TAG_enumerator ]
!18 = metadata !{i32 589864, metadata !"Double", i64 1} ; [ DW_TAG_enumerator ]
!19 = metadata !{i32 589828, metadata !0, metadata !"", metadata !20, i32 14, i64 32, i64 32, i32 0, i32 0, i32 0, metadata !21, i32 0, i32 0} ; [ DW_TAG_enumeration_type ]
!20 = metadata !{i32 589865, metadata !"header4.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!21 = metadata !{metadata !22}
!22 = metadata !{i32 589864, metadata !"Eleven", i64 0} ; [ DW_TAG_enumerator ]
!23 = metadata !{i32 589870, i32 0, metadata !24, metadata !"foobar_func_block_invoke_0", metadata !"foobar_func_block_invoke_0", metadata !"", metadata !24, i32 609, metadata !25, i1 true, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (i8*, %0*, [4 x i32], [4 x i32])* @foobar_func_block_invoke_0, null, null} ; [ DW_TAG_subprogram ]
!24 = metadata !{i32 589865, metadata !"MyLibrary.m", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!25 = metadata !{i32 589845, metadata !24, metadata !"", metadata !24, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !26, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!26 = metadata !{null}
!27 = metadata !{i32 590081, metadata !23, metadata !".block_descriptor", metadata !24, i32 16777825, metadata !28, i32 64} ; [ DW_TAG_arg_variable ]
!28 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 0, i64 0, i32 0, metadata !29} ; [ DW_TAG_pointer_type ]
!29 = metadata !{i32 589843, metadata !24, metadata !"__block_literal_14", metadata !24, i32 609, i64 256, i64 32, i32 0, i32 0, i32 0, metadata !30, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!30 = metadata !{metadata !31, metadata !33, metadata !35, metadata !36, metadata !37, metadata !48, metadata !89, metadata !124}
!31 = metadata !{i32 589837, metadata !24, metadata !"__isa", metadata !24, i32 609, i64 32, i64 32, i64 0, i32 0, metadata !32} ; [ DW_TAG_member ]
!32 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ]
!33 = metadata !{i32 589837, metadata !24, metadata !"__flags", metadata !24, i32 609, i64 32, i64 32, i64 32, i32 0, metadata !34} ; [ DW_TAG_member ]
!34 = metadata !{i32 589860, metadata !0, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!35 = metadata !{i32 589837, metadata !24, metadata !"__reserved", metadata !24, i32 609, i64 32, i64 32, i64 64, i32 0, metadata !34} ; [ DW_TAG_member ]
!36 = metadata !{i32 589837, metadata !24, metadata !"__FuncPtr", metadata !24, i32 609, i64 32, i64 32, i64 96, i32 0, metadata !32} ; [ DW_TAG_member ]
!37 = metadata !{i32 589837, metadata !24, metadata !"__descriptor", metadata !24, i32 609, i64 32, i64 32, i64 128, i32 0, metadata !38} ; [ DW_TAG_member ]
!38 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !39} ; [ DW_TAG_pointer_type ]
!39 = metadata !{i32 589843, metadata !0, metadata !"__block_descriptor_withcopydispose", metadata !40, i32 307, i64 128, i64 32, i32 0, i32 0, i32 0, metadata !41, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!40 = metadata !{i32 589865, metadata !"MyLibrary.i", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!41 = metadata !{metadata !42, metadata !44, metadata !45, metadata !47}
!42 = metadata !{i32 589837, metadata !40, metadata !"reserved", metadata !40, i32 307, i64 32, i64 32, i64 0, i32 0, metadata !43} ; [ DW_TAG_member ]
!43 = metadata !{i32 589860, metadata !0, metadata !"long unsigned int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!44 = metadata !{i32 589837, metadata !40, metadata !"Size", metadata !40, i32 307, i64 32, i64 32, i64 32, i32 0, metadata !43} ; [ DW_TAG_member ]
!45 = metadata !{i32 589837, metadata !40, metadata !"CopyFuncPtr", metadata !40, i32 307, i64 32, i64 32, i64 64, i32 0, metadata !46} ; [ DW_TAG_member ]
!46 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !32} ; [ DW_TAG_pointer_type ]
!47 = metadata !{i32 589837, metadata !40, metadata !"DestroyFuncPtr", metadata !40, i32 307, i64 32, i64 32, i64 96, i32 0, metadata !46} ; [ DW_TAG_member ]
!48 = metadata !{i32 589837, metadata !24, metadata !"mydata", metadata !24, i32 609, i64 32, i64 32, i64 160, i32 0, metadata !49} ; [ DW_TAG_member ]
!49 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 0, i64 0, i32 0, metadata !50} ; [ DW_TAG_pointer_type ]
!50 = metadata !{i32 589843, metadata !24, metadata !"", metadata !24, i32 0, i64 224, i64 0, i32 0, i32 16, i32 0, metadata !51, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!51 = metadata !{metadata !52, metadata !53, metadata !54, metadata !55, metadata !56, metadata !57, metadata !58}
!52 = metadata !{i32 589837, metadata !24, metadata !"__isa", metadata !24, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !32} ; [ DW_TAG_member ]
!53 = metadata !{i32 589837, metadata !24, metadata !"__forwarding", metadata !24, i32 0, i64 32, i64 32, i64 32, i32 0, metadata !32} ; [ DW_TAG_member ]
!54 = metadata !{i32 589837, metadata !24, metadata !"__flags", metadata !24, i32 0, i64 32, i64 32, i64 64, i32 0, metadata !34} ; [ DW_TAG_member ]
!55 = metadata !{i32 589837, metadata !24, metadata !"__size", metadata !24, i32 0, i64 32, i64 32, i64 96, i32 0, metadata !34} ; [ DW_TAG_member ]
!56 = metadata !{i32 589837, metadata !24, metadata !"__copy_helper", metadata !24, i32 0, i64 32, i64 32, i64 128, i32 0, metadata !32} ; [ DW_TAG_member ]
!57 = metadata !{i32 589837, metadata !24, metadata !"__destroy_helper", metadata !24, i32 0, i64 32, i64 32, i64 160, i32 0, metadata !32} ; [ DW_TAG_member ]
!58 = metadata !{i32 589837, metadata !24, metadata !"mydata", metadata !24, i32 0, i64 32, i64 32, i64 192, i32 0, metadata !59} ; [ DW_TAG_member ]
!59 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !60} ; [ DW_TAG_pointer_type ]
!60 = metadata !{i32 589843, metadata !24, metadata !"UIMydata", metadata !61, i32 26, i64 128, i64 32, i32 0, i32 0, i32 0, metadata !62, i32 16, i32 0} ; [ DW_TAG_structure_type ]
!61 = metadata !{i32 589865, metadata !"header11.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!62 = metadata !{metadata !63, metadata !71, metadata !75, metadata !79}
!63 = metadata !{i32 589852, metadata !60, null, metadata !61, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !64} ; [ DW_TAG_inheritance ]
!64 = metadata !{i32 589843, metadata !40, metadata !"NSO", metadata !65, i32 66, i64 32, i64 32, i32 0, i32 0, i32 0, metadata !66, i32 16, i32 0} ; [ DW_TAG_structure_type ]
!65 = metadata !{i32 589865, metadata !"NSO.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!66 = metadata !{metadata !67}
!67 = metadata !{i32 589837, metadata !65, metadata !"isa", metadata !65, i32 67, i64 32, i64 32, i64 0, i32 2, metadata !68, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!68 = metadata !{i32 589846, metadata !0, metadata !"Class", metadata !40, i32 197, i64 0, i64 0, i64 0, i32 0, metadata !69} ; [ DW_TAG_typedef ]
!69 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !70} ; [ DW_TAG_pointer_type ]
!70 = metadata !{i32 589843, metadata !0, metadata !"objc_class", metadata !40, i32 0, i64 0, i64 0, i32 0, i32 4, i32 0, null, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!71 = metadata !{i32 589837, metadata !61, metadata !"_mydataRef", metadata !61, i32 28, i64 32, i64 32, i64 32, i32 0, metadata !72, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!72 = metadata !{i32 589846, metadata !0, metadata !"CFTypeRef", metadata !24, i32 313, i64 0, i64 0, i64 0, i32 0, metadata !73} ; [ DW_TAG_typedef ]
!73 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !74} ; [ DW_TAG_pointer_type ]
!74 = metadata !{i32 589862, metadata !0, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, null} ; [ DW_TAG_const_type ]
!75 = metadata !{i32 589837, metadata !61, metadata !"_scale", metadata !61, i32 29, i64 32, i64 32, i64 64, i32 0, metadata !76, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!76 = metadata !{i32 589846, metadata !0, metadata !"Float", metadata !77, i32 89, i64 0, i64 0, i64 0, i32 0, metadata !78} ; [ DW_TAG_typedef ]
!77 = metadata !{i32 589865, metadata !"header12.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!78 = metadata !{i32 589860, metadata !0, metadata !"float", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!79 = metadata !{i32 589837, metadata !61, metadata !"_mydataFlags", metadata !61, i32 37, i64 8, i64 8, i64 96, i32 0, metadata !80, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!80 = metadata !{i32 589843, metadata !0, metadata !"", metadata !61, i32 30, i64 8, i64 8, i32 0, i32 0, i32 0, metadata !81, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!81 = metadata !{metadata !82, metadata !84, metadata !85, metadata !86, metadata !87, metadata !88}
!82 = metadata !{i32 589837, metadata !61, metadata !"named", metadata !61, i32 31, i64 1, i64 32, i64 0, i32 0, metadata !83} ; [ DW_TAG_member ]
!83 = metadata !{i32 589860, metadata !0, metadata !"unsigned int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!84 = metadata !{i32 589837, metadata !61, metadata !"mydataO", metadata !61, i32 32, i64 3, i64 32, i64 1, i32 0, metadata !83} ; [ DW_TAG_member ]
!85 = metadata !{i32 589837, metadata !61, metadata !"cached", metadata !61, i32 33, i64 1, i64 32, i64 4, i32 0, metadata !83} ; [ DW_TAG_member ]
!86 = metadata !{i32 589837, metadata !61, metadata !"hasBeenCached", metadata !61, i32 34, i64 1, i64 32, i64 5, i32 0, metadata !83} ; [ DW_TAG_member ]
!87 = metadata !{i32 589837, metadata !61, metadata !"hasPattern", metadata !61, i32 35, i64 1, i64 32, i64 6, i32 0, metadata !83} ; [ DW_TAG_member ]
!88 = metadata !{i32 589837, metadata !61, metadata !"isCIMydata", metadata !61, i32 36, i64 1, i64 32, i64 7, i32 0, metadata !83} ; [ DW_TAG_member ]
!89 = metadata !{i32 589837, metadata !24, metadata !"self", metadata !24, i32 609, i64 32, i64 32, i64 192, i32 0, metadata !90} ; [ DW_TAG_member ]
!90 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !91} ; [ DW_TAG_pointer_type ]
!91 = metadata !{i32 589843, metadata !40, metadata !"MyWork", metadata !24, i32 36, i64 384, i64 32, i32 0, i32 0, i32 0, metadata !92, i32 16, i32 0} ; [ DW_TAG_structure_type ]
!92 = metadata !{metadata !93, metadata !98, metadata !101, metadata !107, metadata !123}
!93 = metadata !{i32 589852, metadata !91, null, metadata !24, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !94} ; [ DW_TAG_inheritance ]
!94 = metadata !{i32 589843, metadata !40, metadata !"twork", metadata !95, i32 43, i64 32, i64 32, i32 0, i32 0, i32 0, metadata !96, i32 16, i32 0} ; [ DW_TAG_structure_type ]
!95 = metadata !{i32 589865, metadata !"header13.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!96 = metadata !{metadata !97}
!97 = metadata !{i32 589852, metadata !94, null, metadata !95, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !64} ; [ DW_TAG_inheritance ]
!98 = metadata !{i32 589837, metadata !24, metadata !"_itemID", metadata !24, i32 38, i64 64, i64 32, i64 32, i32 1, metadata !99, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!99 = metadata !{i32 589846, metadata !0, metadata !"uint64_t", metadata !40, i32 55, i64 0, i64 0, i64 0, i32 0, metadata !100} ; [ DW_TAG_typedef ]
!100 = metadata !{i32 589860, metadata !0, metadata !"long long unsigned int", null, i32 0, i64 64, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!101 = metadata !{i32 589837, metadata !24, metadata !"_library", metadata !24, i32 39, i64 32, i64 32, i64 96, i32 1, metadata !102, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!102 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !103} ; [ DW_TAG_pointer_type ]
!103 = metadata !{i32 589843, metadata !40, metadata !"MyLibrary2", metadata !104, i32 22, i64 32, i64 32, i32 0, i32 0, i32 0, metadata !105, i32 16, i32 0} ; [ DW_TAG_structure_type ]
!104 = metadata !{i32 589865, metadata !"header14.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!105 = metadata !{metadata !106}
!106 = metadata !{i32 589852, metadata !103, null, metadata !104, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !64} ; [ DW_TAG_inheritance ]
!107 = metadata !{i32 589837, metadata !24, metadata !"_bounds", metadata !24, i32 40, i64 128, i64 32, i64 128, i32 1, metadata !108, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!108 = metadata !{i32 589846, metadata !0, metadata !"CR", metadata !40, i32 33, i64 0, i64 0, i64 0, i32 0, metadata !109} ; [ DW_TAG_typedef ]
!109 = metadata !{i32 589843, metadata !0, metadata !"CR", metadata !77, i32 29, i64 128, i64 32, i32 0, i32 0, i32 0, metadata !110, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!110 = metadata !{metadata !111, metadata !117}
!111 = metadata !{i32 589837, metadata !77, metadata !"origin", metadata !77, i32 30, i64 64, i64 32, i64 0, i32 0, metadata !112} ; [ DW_TAG_member ]
!112 = metadata !{i32 589846, metadata !0, metadata !"CP", metadata !77, i32 17, i64 0, i64 0, i64 0, i32 0, metadata !113} ; [ DW_TAG_typedef ]
!113 = metadata !{i32 589843, metadata !0, metadata !"CP", metadata !77, i32 13, i64 64, i64 32, i32 0, i32 0, i32 0, metadata !114, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!114 = metadata !{metadata !115, metadata !116}
!115 = metadata !{i32 589837, metadata !77, metadata !"x", metadata !77, i32 14, i64 32, i64 32, i64 0, i32 0, metadata !76} ; [ DW_TAG_member ]
!116 = metadata !{i32 589837, metadata !77, metadata !"y", metadata !77, i32 15, i64 32, i64 32, i64 32, i32 0, metadata !76} ; [ DW_TAG_member ]
!117 = metadata !{i32 589837, metadata !77, metadata !"size", metadata !77, i32 31, i64 64, i64 32, i64 64, i32 0, metadata !118} ; [ DW_TAG_member ]
!118 = metadata !{i32 589846, metadata !0, metadata !"Size", metadata !77, i32 25, i64 0, i64 0, i64 0, i32 0, metadata !119} ; [ DW_TAG_typedef ]
!119 = metadata !{i32 589843, metadata !0, metadata !"Size", metadata !77, i32 21, i64 64, i64 32, i32 0, i32 0, i32 0, metadata !120, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!120 = metadata !{metadata !121, metadata !122}
!121 = metadata !{i32 589837, metadata !77, metadata !"width", metadata !77, i32 22, i64 32, i64 32, i64 0, i32 0, metadata !76} ; [ DW_TAG_member ]
!122 = metadata !{i32 589837, metadata !77, metadata !"height", metadata !77, i32 23, i64 32, i64 32, i64 32, i32 0, metadata !76} ; [ DW_TAG_member ]
!123 = metadata !{i32 589837, metadata !24, metadata !"_data", metadata !24, i32 40, i64 128, i64 32, i64 256, i32 1, metadata !108, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!124 = metadata !{i32 589837, metadata !24, metadata !"semi", metadata !24, i32 609, i64 32, i64 32, i64 224, i32 0, metadata !125} ; [ DW_TAG_member ]
!125 = metadata !{i32 589846, metadata !0, metadata !"d_t", metadata !24, i32 35, i64 0, i64 0, i64 0, i32 0, metadata !126} ; [ DW_TAG_typedef ]
!126 = metadata !{i32 589839, metadata !0, metadata !"", null, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !127} ; [ DW_TAG_pointer_type ]
!127 = metadata !{i32 589843, metadata !0, metadata !"my_struct", metadata !128, i32 49, i64 0, i64 0, i32 0, i32 4, i32 0, null, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!128 = metadata !{i32 589865, metadata !"header15.h", metadata !"/Volumes/Sandbox/llvm", metadata !0} ; [ DW_TAG_file_type ]
!129 = metadata !{i32 609, i32 144, metadata !23, null}
!130 = metadata !{i32 590081, metadata !23, metadata !"loadedMydata", metadata !24, i32 33555041, metadata !59, i32 0} ; [ DW_TAG_arg_variable ]
!131 = metadata !{i32 609, i32 155, metadata !23, null}
!132 = metadata !{i32 590081, metadata !23, metadata !"bounds", metadata !24, i32 50332257, metadata !108, i32 0} ; [ DW_TAG_arg_variable ]
!133 = metadata !{i32 609, i32 175, metadata !23, null}
!134 = metadata !{i32 590081, metadata !23, metadata !"data", metadata !24, i32 67109473, metadata !108, i32 0} ; [ DW_TAG_arg_variable ]
!135 = metadata !{i32 609, i32 190, metadata !23, null}
!136 = metadata !{i32 590080, metadata !23, metadata !"mydata", metadata !24, i32 604, metadata !50, i32 0, i64 1, i64 20, i64 2, i64 1, i64 4, i64 2, i64 1, i64 24} ; [ DW_TAG_auto_variable ]
!137 = metadata !{i32 604, i32 49, metadata !23, null}
!138 = metadata !{i32 590080, metadata !23, metadata !"self", metadata !40, i32 604, metadata !90, i32 0, i64 1, i64 24} ; [ DW_TAG_auto_variable ]
!139 = metadata !{i32 590080, metadata !23, metadata !"semi", metadata !24, i32 607, metadata !125, i32 0, i64 1, i64 28} ; [ DW_TAG_auto_variable ]
!140 = metadata !{i32 607, i32 30, metadata !23, null}
!141 = metadata !{i32 610, i32 17, metadata !142, null}
!142 = metadata !{i32 589835, metadata !23, i32 609, i32 200, metadata !24, i32 94} ; [ DW_TAG_lexical_block ]
!143 = metadata !{i32 611, i32 17, metadata !142, null}
!144 = metadata !{i32 612, i32 17, metadata !142, null}
!145 = metadata !{i32 613, i32 17, metadata !142, null}
!146 = metadata !{i32 615, i32 13, metadata !142, null}
