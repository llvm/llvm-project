;; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
;; RUN: llvm-dwarfdump --show-children %t | FileCheck --check-prefix=DWARF %s

; ModuleID = 'swift-move-function-debuginfo.ll'
source_filename = "swift-move-function-dbginfo.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx11.0.0"

%T4main5KlassC = type <{ %swift.refcounted }>
%swift.refcounted = type { %swift.type*, i64 }
%swift.type = type { i64 }
%swift.metadata_response = type { %swift.type*, i64 }
%swift.opaque = type opaque
%swift.vwtable = type { i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, i64, i32, i32 }

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #0

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare hidden swiftcc %T4main5KlassC* @"$s4main5KlassCACycfC"(%swift.type* swiftself) #2

;; DWARF: DW_AT_linkage_name{{.*}}("$s4main17copyableValueTestyyF")
;; DWARF-NEXT: DW_AT_name ("copyableValueTest")
;; DWARF-NEXT: DW_AT_decl_file
;; DWARF-NEXT: DW_AT_decl_line
;; DWARF-NEXT: DW_AT_type
;; DWARF-NEXT: DW_AT_external	(true)
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg
;; DWARF-NEXT: DW_AT_name	("m")
;; DWARF-NEXT: DW_AT_decl_file	(
;; DWARF-NEXT: DW_AT_decl_line	(
;; DWARF-NEXT: DW_AT_type	(
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg
;; DWARF-NEXT: DW_AT_name	("k")
;; DWARF-NEXT: DW_AT_decl_file	(
;; DWARF-NEXT: DW_AT_decl_line	(
;; DWARF-NEXT: DW_AT_type	(
define swiftcc void @"$s4main17copyableValueTestyyF"() #2 !dbg !42 {
entry:
  %k.debug = alloca %T4main5KlassC*, align 8
  %0 = bitcast %T4main5KlassC** %k.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 8, i1 false)
  %m.debug = alloca %T4main5KlassC*, align 8
  call void @llvm.dbg.declare(metadata %T4main5KlassC** %m.debug, metadata !52, metadata !DIExpression()), !dbg !53
  %1 = bitcast %T4main5KlassC** %m.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  %2 = call swiftcc %swift.metadata_response @"$s4main5KlassCMa"(i64 0) #7, !dbg !54
  %3 = extractvalue %swift.metadata_response %2, 0, !dbg !54
  %4 = call swiftcc %T4main5KlassC* @"$s4main5KlassCACycfC"(%swift.type* swiftself %3), !dbg !54
  store %T4main5KlassC* %4, %T4main5KlassC** %k.debug, align 8, !dbg !55
  call void @llvm.dbg.declare(metadata %T4main5KlassC** %k.debug, metadata !48, metadata !DIExpression()), !dbg !56
  br label %entry.split, !dbg !57

entry.split:                                      ; preds = %entry
  %5 = getelementptr inbounds %T4main5KlassC, %T4main5KlassC* %4, i32 0, i32 0, i32 0, !dbg !57
  %6 = load %swift.type*, %swift.type** %5, align 8, !dbg !57
  %7 = bitcast %swift.type* %6 to void (%T4main5KlassC*)**, !dbg !57
  %8 = getelementptr inbounds void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %7, i64 10, !dbg !57
  %9 = load void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %8, align 8, !dbg !57, !invariant.load !46
  call swiftcc void %9(%T4main5KlassC* swiftself %4), !dbg !57
  %10 = bitcast %T4main5KlassC* %4 to %swift.refcounted*, !dbg !58
  %11 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %10) #5, !dbg !58
  call void @llvm.dbg.value(metadata %T4main5KlassC* undef, metadata !48, metadata !DIExpression()), !dbg !56
  store %T4main5KlassC* %4, %T4main5KlassC** %m.debug, align 8, !dbg !55
  %12 = getelementptr inbounds %T4main5KlassC, %T4main5KlassC* %4, i32 0, i32 0, i32 0, !dbg !59
  %13 = load %swift.type*, %swift.type** %12, align 8, !dbg !59
  %14 = bitcast %swift.type* %13 to void (%T4main5KlassC*)**, !dbg !59
  %15 = getelementptr inbounds void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %14, i64 10, !dbg !59
  %16 = load void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %15, align 8, !dbg !59, !invariant.load !46
  call swiftcc void %16(%T4main5KlassC* swiftself %4), !dbg !59
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %4) #5, !dbg !60
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %4) #5, !dbg !60
  ret void, !dbg !60
}

; Function Attrs: noinline nounwind readnone
declare swiftcc %swift.metadata_response @"$s4main5KlassCMa"(i64) #3

; Function Attrs: nounwind willreturn
declare %swift.refcounted* @swift_retain(%swift.refcounted* returned) #4

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @swift_release(%swift.refcounted*) #5

;; DWARF: DW_AT_linkage_name{{.*}}("$s4main15copyableArgTestyyAA5KlassCnF")
;; DWARF-NEXT: DW_AT_name ("copyableArgTest")
;; DWARF-NEXT: DW_AT_decl_file
;; DWARF-NEXT: DW_AT_decl_line
;; DWARF-NEXT: DW_AT_type
;; DWARF-NEXT: DW_AT_external	(true)
;;
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg
;; DWARF-NEXT: DW_AT_name	("k")
;; DWARF-NEXT: DW_AT_decl_file	(
;; DWARF-NEXT: DW_AT_decl_line	(
;; DWARF-NEXT: DW_AT_type	(
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg
;; DWARF-NEXT: DW_AT_name	("m")
;; DWARF-NEXT: DW_AT_decl_file	(
;; DWARF-NEXT: DW_AT_decl_line	(
;; DWARF-NEXT: DW_AT_type	(
define swiftcc void @"$s4main15copyableArgTestyyAA5KlassCnF"(%T4main5KlassC* %0) #2 !dbg !61 {
entry:
  %k.debug = alloca %T4main5KlassC*, align 8
  %1 = bitcast %T4main5KlassC** %k.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  %m.debug = alloca %T4main5KlassC*, align 8
  call void @llvm.dbg.declare(metadata %T4main5KlassC** %m.debug, metadata !66, metadata !DIExpression()), !dbg !68
  %2 = bitcast %T4main5KlassC** %m.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 8, i1 false)
  store %T4main5KlassC* %0, %T4main5KlassC** %k.debug, align 8, !dbg !69
  call void @llvm.dbg.declare(metadata %T4main5KlassC** %k.debug, metadata !65, metadata !DIExpression()), !dbg !70
  br label %entry.split, !dbg !71

entry.split:                                      ; preds = %entry
  %3 = getelementptr inbounds %T4main5KlassC, %T4main5KlassC* %0, i32 0, i32 0, i32 0, !dbg !71
  %4 = load %swift.type*, %swift.type** %3, align 8, !dbg !71
  %5 = bitcast %swift.type* %4 to void (%T4main5KlassC*)**, !dbg !71
  %6 = getelementptr inbounds void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %5, i64 10, !dbg !71
  %7 = load void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %6, align 8, !dbg !71, !invariant.load !46
  call swiftcc void %7(%T4main5KlassC* swiftself %0), !dbg !71
  %8 = bitcast %T4main5KlassC* %0 to %swift.refcounted*, !dbg !72
  %9 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %8) #5, !dbg !72
  call void @llvm.dbg.value(metadata %T4main5KlassC* undef, metadata !65, metadata !DIExpression()), !dbg !70
  store %T4main5KlassC* %0, %T4main5KlassC** %m.debug, align 8, !dbg !73
  %10 = getelementptr inbounds %T4main5KlassC, %T4main5KlassC* %0, i32 0, i32 0, i32 0, !dbg !74
  %11 = load %swift.type*, %swift.type** %10, align 8, !dbg !74
  %12 = bitcast %swift.type* %11 to void (%T4main5KlassC*)**, !dbg !74
  %13 = getelementptr inbounds void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %12, i64 10, !dbg !74
  %14 = load void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %13, align 8, !dbg !74, !invariant.load !46
  call swiftcc void %14(%T4main5KlassC* swiftself %0), !dbg !74
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %0) #5, !dbg !75
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %0) #5, !dbg !75
  ret void, !dbg !75
}

;; DWARF: DW_AT_linkage_name	("$s4main15copyableVarTestyyF")
;; DWARF-NEXT: DW_AT_name	("copyableVarTest")
;; DWARF-NEXT: DW_AT_decl_file	(
;; DWARF-NEXT: DW_AT_decl_line	(
;; DWARF-NEXT: DW_AT_type	(
;; DWARF-NEXT: DW_AT_external	(
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg -24)
;; DWARF-NEXT: DW_AT_name	("m")
;; DWARF-NEXT: DW_AT_decl_file	(
;; DWARF-NEXT: DW_AT_decl_line	(
;; DWARF-NEXT: DW_AT_type	(
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg
;; DWARF-NEXT: DW_AT_name	("k")
;; DWARF-NEXT: DW_AT_decl_file	(
;; DWARF-NEXT: DW_AT_decl_line	(
;; DWARF-NEXT: DW_AT_type	(
define swiftcc void @"$s4main15copyableVarTestyyF"() #2 !dbg !76 {
entry:
  %k = alloca %T4main5KlassC*, align 8
  %0 = bitcast %T4main5KlassC** %k to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 8, i1 false)
  %m.debug = alloca %T4main5KlassC*, align 8
  call void @llvm.dbg.declare(metadata %T4main5KlassC** %m.debug, metadata !80, metadata !DIExpression()), !dbg !81
  %1 = bitcast %T4main5KlassC** %m.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  %2 = bitcast %T4main5KlassC** %k to i8*, !dbg !82
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %2), !dbg !82
  call void @llvm.dbg.declare(metadata %T4main5KlassC** %k, metadata !78, metadata !DIExpression()), !dbg !85
  br label %entry.split1, !dbg !86

entry.split1:                                     ; preds = %entry
  %3 = call swiftcc %swift.metadata_response @"$s4main5KlassCMa"(i64 0) #7, !dbg !86
  %4 = extractvalue %swift.metadata_response %3, 0, !dbg !86
  %5 = call swiftcc %T4main5KlassC* @"$s4main5KlassCACycfC"(%swift.type* swiftself %4), !dbg !86
  %6 = bitcast %T4main5KlassC* %5 to %swift.refcounted*, !dbg !86
  %7 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %6) #5, !dbg !86
  store %T4main5KlassC* %5, %T4main5KlassC** %k, align 8, !dbg !86
  %8 = getelementptr inbounds %T4main5KlassC, %T4main5KlassC* %5, i32 0, i32 0, i32 0, !dbg !87
  %9 = load %swift.type*, %swift.type** %8, align 8, !dbg !87
  %10 = bitcast %swift.type* %9 to void (%T4main5KlassC*)**, !dbg !87
  %11 = getelementptr inbounds void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %10, i64 10, !dbg !87
  %12 = load void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %11, align 8, !dbg !87, !invariant.load !46
  call swiftcc void %12(%T4main5KlassC* swiftself %5), !dbg !87
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %5) #5, !dbg !87
  %13 = load %T4main5KlassC*, %T4main5KlassC** %k, align 8, !dbg !88
  call void @llvm.dbg.value(metadata %T4main5KlassC** undef, metadata !78, metadata !DIExpression()), !dbg !85
  store %T4main5KlassC* %13, %T4main5KlassC** %m.debug, align 8, !dbg !89
  %14 = getelementptr inbounds %T4main5KlassC, %T4main5KlassC* %13, i32 0, i32 0, i32 0, !dbg !90
  %15 = load %swift.type*, %swift.type** %14, align 8, !dbg !90
  %16 = bitcast %swift.type* %15 to void (%T4main5KlassC*)**, !dbg !90
  %17 = getelementptr inbounds void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %16, i64 10, !dbg !90
  %18 = load void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %17, align 8, !dbg !90, !invariant.load !46
  call swiftcc void %18(%T4main5KlassC* swiftself %13), !dbg !90
  %19 = call swiftcc %T4main5KlassC* @"$s4main5KlassCACycfC"(%swift.type* swiftself %4), !dbg !91
  %20 = bitcast %T4main5KlassC* %19 to %swift.refcounted*, !dbg !92
  %21 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %20) #5, !dbg !92
  store %T4main5KlassC* %19, %T4main5KlassC** %k, align 8, !dbg !92
  call void @llvm.dbg.declare(metadata %T4main5KlassC** %k, metadata !78, metadata !DIExpression()), !dbg !85
  br label %entry.split, !dbg !93

entry.split:                                      ; preds = %entry.split1
  %22 = getelementptr inbounds %T4main5KlassC, %T4main5KlassC* %19, i32 0, i32 0, i32 0, !dbg !93
  %23 = load %swift.type*, %swift.type** %22, align 8, !dbg !93
  %24 = bitcast %swift.type* %23 to void (%T4main5KlassC*)**, !dbg !93
  %25 = getelementptr inbounds void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %24, i64 10, !dbg !93
  %26 = load void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %25, align 8, !dbg !93, !invariant.load !46
  call swiftcc void %26(%T4main5KlassC* swiftself %19), !dbg !93
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %19) #5, !dbg !94
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %13) #5, !dbg !94
  %toDestroy = load %T4main5KlassC*, %T4main5KlassC** %k, align 8, !dbg !94
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %toDestroy) #5, !dbg !94
  %27 = bitcast %T4main5KlassC** %k to i8*, !dbg !94
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %27), !dbg !94
  ret void, !dbg !94
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #6

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #6

;; DWARF: DW_AT_linkage_name	("$s4main18copyableVarArgTestyyAA5KlassCzF")
;; DWARF-NEXT: DW_AT_name	("copyableVarArgTest")
;; DWARF-NEXT: DW_AT_decl_file	(
;; DWARF-NEXT: DW_AT_decl_line	(
;; DWARF-NEXT: DW_AT_type	(
;; DWARF-NEXT: DW_AT_external	(
;;
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg
;; DWARF-NEXT: DW_AT_name	("k")
;; DWARF-NEXT: DW_AT_decl_file	(
;; DWARF-NEXT: DW_AT_decl_line	(
;; DWARF-NEXT: DW_AT_type	(
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg -24)
;; DWARF-NEXT: DW_AT_name	("m")
;; DWARF-NEXT: DW_AT_decl_file	(
;; DWARF-NEXT: DW_AT_decl_line	(
;; DWARF-NEXT: DW_AT_type	(
define swiftcc void @"$s4main18copyableVarArgTestyyAA5KlassCzF"(%T4main5KlassC** nocapture dereferenceable(8) %0) #2 !dbg !95 {
entry:
  %k.debug = alloca %T4main5KlassC**, align 8
  %1 = bitcast %T4main5KlassC*** %k.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  %m.debug = alloca %T4main5KlassC*, align 8
  call void @llvm.dbg.declare(metadata %T4main5KlassC** %m.debug, metadata !98, metadata !DIExpression()), !dbg !100
  %2 = bitcast %T4main5KlassC** %m.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 8, i1 false)
  %3 = bitcast %T4main5KlassC*** %k.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  store %T4main5KlassC** %0, %T4main5KlassC*** %k.debug, align 8, !dbg !101
  call void @llvm.dbg.declare(metadata %T4main5KlassC*** %k.debug, metadata !97, metadata !DIExpression(DW_OP_deref)), !dbg !102
  br label %entry.split1, !dbg !103

entry.split1:                                     ; preds = %entry
  %4 = load %T4main5KlassC*, %T4main5KlassC** %0, align 8, !dbg !103
  %5 = bitcast %T4main5KlassC* %4 to %swift.refcounted*, !dbg !103
  %6 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %5) #5, !dbg !103
  %7 = getelementptr inbounds %T4main5KlassC, %T4main5KlassC* %4, i32 0, i32 0, i32 0, !dbg !104
  %8 = load %swift.type*, %swift.type** %7, align 8, !dbg !104
  %9 = bitcast %swift.type* %8 to void (%T4main5KlassC*)**, !dbg !104
  %10 = getelementptr inbounds void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %9, i64 10, !dbg !104
  %11 = load void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %10, align 8, !dbg !104, !invariant.load !46
  call swiftcc void %11(%T4main5KlassC* swiftself %4), !dbg !104
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %4) #5, !dbg !104
  %12 = load %T4main5KlassC*, %T4main5KlassC** %0, align 8, !dbg !105
  call void @llvm.dbg.value(metadata %T4main5KlassC** undef, metadata !97, metadata !DIExpression(DW_OP_deref)), !dbg !102
  store %T4main5KlassC* %12, %T4main5KlassC** %m.debug, align 8, !dbg !106
  %13 = getelementptr inbounds %T4main5KlassC, %T4main5KlassC* %12, i32 0, i32 0, i32 0, !dbg !107
  %14 = load %swift.type*, %swift.type** %13, align 8, !dbg !107
  %15 = bitcast %swift.type* %14 to void (%T4main5KlassC*)**, !dbg !107
  %16 = getelementptr inbounds void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %15, i64 10, !dbg !107
  %17 = load void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %16, align 8, !dbg !107, !invariant.load !46
  call swiftcc void %17(%T4main5KlassC* swiftself %12), !dbg !107
  %18 = call swiftcc %swift.metadata_response @"$s4main5KlassCMa"(i64 0) #7, !dbg !108
  %19 = extractvalue %swift.metadata_response %18, 0, !dbg !108
  %20 = call swiftcc %T4main5KlassC* @"$s4main5KlassCACycfC"(%swift.type* swiftself %19), !dbg !108
  store %T4main5KlassC* %20, %T4main5KlassC** %0, align 8, !dbg !109
  store %T4main5KlassC** %0, %T4main5KlassC*** %k.debug, align 8, !dbg !101
  call void @llvm.dbg.declare(metadata %T4main5KlassC*** %k.debug, metadata !97, metadata !DIExpression(DW_OP_deref)), !dbg !102
  br label %entry.split, !dbg !110

entry.split:                                      ; preds = %entry.split1
  %21 = load %T4main5KlassC*, %T4main5KlassC** %0, align 8, !dbg !110
  %22 = bitcast %T4main5KlassC* %21 to %swift.refcounted*, !dbg !110
  %23 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %22) #5, !dbg !110
  %24 = getelementptr inbounds %T4main5KlassC, %T4main5KlassC* %21, i32 0, i32 0, i32 0, !dbg !111
  %25 = load %swift.type*, %swift.type** %24, align 8, !dbg !111
  %26 = bitcast %swift.type* %25 to void (%T4main5KlassC*)**, !dbg !111
  %27 = getelementptr inbounds void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %26, i64 10, !dbg !111
  %28 = load void (%T4main5KlassC*)*, void (%T4main5KlassC*)** %27, align 8, !dbg !111, !invariant.load !46
  call swiftcc void %28(%T4main5KlassC* swiftself %21), !dbg !111
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %21) #5, !dbg !112
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4main5KlassC*)*)(%T4main5KlassC* %12) #5, !dbg !112
  ret void, !dbg !112
}

;; DWARF: DW_AT_linkage_name   ("$s4main20addressOnlyValueTestyyxAA1PRzlF")
;; DWARF-NEXT: DW_AT_name      ("addressOnlyValueTest")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
;; DWARF-NEXT: DW_AT_external  (
;;
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location  (
;; DWARF-NEXT: DW_AT_name      ("x")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location  (
;; DWARF-NEXT: DW_AT_name      ("$\317\204_0_0")
;; DWARF-NEXT: DW_AT_type      (
;; DWARF-NEXT: DW_AT_artificial        (true)
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location  (
;; DWARF-NEXT: DW_AT_name      ("m")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg
;; DWARF-NEXT: DW_AT_name      ("k")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
define swiftcc void @"$s4main20addressOnlyValueTestyyxAA1PRzlF"(%swift.opaque* noalias nocapture %0, %swift.type* %T, i8** %T.P) #2 !dbg !113 {
entry:
  %T1 = alloca %swift.type*, align 8
  call void @llvm.dbg.declare(metadata %swift.type** %T1, metadata !118, metadata !DIExpression()), !dbg !127
  %m.debug = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i8** %m.debug, metadata !122, metadata !DIExpression(DW_OP_deref)), !dbg !128
  %1 = bitcast i8** %m.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  %x.debug = alloca %swift.opaque*, align 8
  call void @llvm.dbg.declare(metadata %swift.opaque** %x.debug, metadata !125, metadata !DIExpression(DW_OP_deref)), !dbg !129
  %2 = bitcast %swift.opaque** %x.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 8, i1 false)
  %k.debug = alloca %swift.opaque*, align 8
  %3 = bitcast %swift.opaque** %k.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  store %swift.type* %T, %swift.type** %T1, align 8
  %4 = bitcast %swift.type* %T to i8***, !dbg !130
  %5 = getelementptr inbounds i8**, i8*** %4, i64 -1, !dbg !130
  %T.valueWitnesses = load i8**, i8*** %5, align 8, !dbg !130, !invariant.load !46, !dereferenceable !132
  %6 = bitcast i8** %T.valueWitnesses to %swift.vwtable*, !dbg !130
  %7 = getelementptr inbounds %swift.vwtable, %swift.vwtable* %6, i32 0, i32 8, !dbg !130
  %size = load i64, i64* %7, align 8, !dbg !130, !invariant.load !46
  %8 = alloca i8, i64 %size, align 16, !dbg !130
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %8), !dbg !130
  %9 = bitcast i8* %8 to %swift.opaque*, !dbg !130
  %m = alloca i8, i64 %size, align 16, !dbg !130
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %m), !dbg !130
  %10 = bitcast i8* %m to %swift.opaque*, !dbg !130
  store i8* %m, i8** %m.debug, align 8, !dbg !133
  %11 = alloca i8, i64 %size, align 16, !dbg !130
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %11), !dbg !130
  %12 = bitcast i8* %11 to %swift.opaque*, !dbg !130
  store %swift.opaque* %0, %swift.opaque** %x.debug, align 8, !dbg !127
  store %swift.opaque* %12, %swift.opaque** %k.debug, align 8, !dbg !133
  call void @llvm.dbg.declare(metadata %swift.opaque** %k.debug, metadata !126, metadata !DIExpression()), !dbg !134
  br label %13, !dbg !135

13:                                               ; preds = %entry
  %14 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 2, !dbg !135
  %15 = load i8*, i8** %14, align 8, !dbg !135, !invariant.load !46
  %initializeWithCopy = bitcast i8* %15 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !135
  %16 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %12, %swift.opaque* noalias %0, %swift.type* %T) #5, !dbg !135
  %17 = getelementptr inbounds i8*, i8** %T.P, i32 2, !dbg !136
  %18 = load i8*, i8** %17, align 8, !dbg !136, !invariant.load !46
  %19 = bitcast i8* %18 to void (%swift.opaque*, %swift.type*, i8**)*, !dbg !136
  call swiftcc void %19(%swift.opaque* noalias nocapture swiftself %12, %swift.type* %T, i8** %T.P), !dbg !136
  %20 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %9, %swift.opaque* noalias %12, %swift.type* %T) #5, !dbg !137
  %21 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 4, !dbg !138
  %22 = load i8*, i8** %21, align 8, !dbg !138, !invariant.load !46
  %initializeWithTake = bitcast i8* %22 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !138
  %23 = call %swift.opaque* %initializeWithTake(%swift.opaque* noalias %10, %swift.opaque* noalias %12, %swift.type* %T) #5, !dbg !138
  call void @llvm.dbg.value(metadata %swift.opaque* undef, metadata !126, metadata !DIExpression()), !dbg !134
  %24 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 1, !dbg !138
  %25 = load i8*, i8** %24, align 8, !dbg !138, !invariant.load !46
  %destroy = bitcast i8* %25 to void (%swift.opaque*, %swift.type*)*, !dbg !138
  call void %destroy(%swift.opaque* noalias %9, %swift.type* %T) #5, !dbg !138
  %26 = getelementptr inbounds i8*, i8** %T.P, i32 2, !dbg !139
  %27 = load i8*, i8** %26, align 8, !dbg !139, !invariant.load !46
  %28 = bitcast i8* %27 to void (%swift.opaque*, %swift.type*, i8**)*, !dbg !139
  call swiftcc void %28(%swift.opaque* noalias nocapture swiftself %10, %swift.type* %T, i8** %T.P), !dbg !139
  call void %destroy(%swift.opaque* noalias %10, %swift.type* %T) #5, !dbg !140
  %29 = bitcast %swift.opaque* %12 to i8*, !dbg !140
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %29), !dbg !140
  %30 = bitcast %swift.opaque* %10 to i8*, !dbg !140
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %30), !dbg !140
  %31 = bitcast %swift.opaque* %9 to i8*, !dbg !140
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %31), !dbg !140
  ret void, !dbg !140
}

;; DWARF: DW_AT_linkage_name   ("$s4main23addressOnlyValueArgTestyyxnAA1PRzlF")
;; DWARF-NEXT: DW_AT_name      ("addressOnlyValueArgTest")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
;; DWARF-NEXT: DW_AT_external  (
;;
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg
;; DWARF-NEXT: DW_AT_name      ("k")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location  (
;; DWARF-NEXT: DW_AT_name      ("$\317\204_0_0")
;; DWARF-NEXT: DW_AT_type      (
;; DWARF-NEXT: DW_AT_artificial        (true)
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location  (
;; DWARF-NEXT: DW_AT_name      ("m")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
define swiftcc void @"$s4main23addressOnlyValueArgTestyyxnAA1PRzlF"(%swift.opaque* noalias nocapture %0, %swift.type* %T, i8** %T.P) #2 !dbg !141 {
entry:
  %T1 = alloca %swift.type*, align 8
  call void @llvm.dbg.declare(metadata %swift.type** %T1, metadata !143, metadata !DIExpression()), !dbg !147
  %m.debug = alloca i8*, align 8
  call void @llvm.dbg.declare(metadata i8** %m.debug, metadata !144, metadata !DIExpression(DW_OP_deref)), !dbg !148
  %1 = bitcast i8** %m.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  %k.debug = alloca %swift.opaque*, align 8
  %2 = bitcast %swift.opaque** %k.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 8, i1 false)
  store %swift.type* %T, %swift.type** %T1, align 8
  %3 = bitcast %swift.type* %T to i8***, !dbg !149
  %4 = getelementptr inbounds i8**, i8*** %3, i64 -1, !dbg !149
  %T.valueWitnesses = load i8**, i8*** %4, align 8, !dbg !149, !invariant.load !46, !dereferenceable !132
  %5 = bitcast i8** %T.valueWitnesses to %swift.vwtable*, !dbg !149
  %6 = getelementptr inbounds %swift.vwtable, %swift.vwtable* %5, i32 0, i32 8, !dbg !149
  %size = load i64, i64* %6, align 8, !dbg !149, !invariant.load !46
  %7 = alloca i8, i64 %size, align 16, !dbg !149
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %7), !dbg !149
  %8 = bitcast i8* %7 to %swift.opaque*, !dbg !149
  %m = alloca i8, i64 %size, align 16, !dbg !149
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %m), !dbg !149
  %9 = bitcast i8* %m to %swift.opaque*, !dbg !149
  store i8* %m, i8** %m.debug, align 8, !dbg !151
  store %swift.opaque* %0, %swift.opaque** %k.debug, align 8, !dbg !147
  call void @llvm.dbg.declare(metadata %swift.opaque** %k.debug, metadata !146, metadata !DIExpression(DW_OP_deref)), !dbg !152
  br label %entry.split, !dbg !153

entry.split:                                      ; preds = %entry
  %10 = getelementptr inbounds i8*, i8** %T.P, i32 2, !dbg !153
  %11 = load i8*, i8** %10, align 8, !dbg !153, !invariant.load !46
  %12 = bitcast i8* %11 to void (%swift.opaque*, %swift.type*, i8**)*, !dbg !153
  call swiftcc void %12(%swift.opaque* noalias nocapture swiftself %0, %swift.type* %T, i8** %T.P), !dbg !153
  %13 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 2, !dbg !154
  %14 = load i8*, i8** %13, align 8, !dbg !154, !invariant.load !46
  %initializeWithCopy = bitcast i8* %14 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !154
  %15 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %8, %swift.opaque* noalias %0, %swift.type* %T) #5, !dbg !154
  %16 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 4, !dbg !155
  %17 = load i8*, i8** %16, align 8, !dbg !155, !invariant.load !46
  %initializeWithTake = bitcast i8* %17 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !155
  %18 = call %swift.opaque* %initializeWithTake(%swift.opaque* noalias %9, %swift.opaque* noalias %0, %swift.type* %T) #5, !dbg !155
  call void @llvm.dbg.value(metadata %swift.opaque* undef, metadata !146, metadata !DIExpression(DW_OP_deref)), !dbg !152
  %19 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 1, !dbg !155
  %20 = load i8*, i8** %19, align 8, !dbg !155, !invariant.load !46
  %destroy = bitcast i8* %20 to void (%swift.opaque*, %swift.type*)*, !dbg !155
  call void %destroy(%swift.opaque* noalias %8, %swift.type* %T) #5, !dbg !155
  %21 = getelementptr inbounds i8*, i8** %T.P, i32 2, !dbg !156
  %22 = load i8*, i8** %21, align 8, !dbg !156, !invariant.load !46
  %23 = bitcast i8* %22 to void (%swift.opaque*, %swift.type*, i8**)*, !dbg !156
  call swiftcc void %23(%swift.opaque* noalias nocapture swiftself %9, %swift.type* %T, i8** %T.P), !dbg !156
  call void %destroy(%swift.opaque* noalias %9, %swift.type* %T) #5, !dbg !157
  %24 = bitcast %swift.opaque* %9 to i8*, !dbg !157
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %24), !dbg !157
  %25 = bitcast %swift.opaque* %8 to i8*, !dbg !157
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %25), !dbg !157
  ret void, !dbg !157
}

;; DWARF: DW_AT_linkage_name   ("$s4main18addressOnlyVarTestyyxAA1PRzlF")
;; DWARF-NEXT: DW_AT_name      ("addressOnlyVarTest")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
;; DWARF-NEXT: DW_AT_external  (
;;
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location  (
;; DWARF-NEXT: DW_AT_name      ("x")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location  (
;; DWARF-NEXT: DW_AT_name      ("$\317\204_0_0")
;; DWARF-NEXT: DW_AT_type      (
;; DWARF-NEXT: DW_AT_artificial        (true)
;;
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg
;; DWARF-NEXT: DW_AT_name      ("k")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
define swiftcc void @"$s4main18addressOnlyVarTestyyxAA1PRzlF"(%swift.opaque* noalias nocapture %0, %swift.type* %T, i8** %T.P) #2 !dbg !158 {
entry:
  %T1 = alloca %swift.type*, align 8
  call void @llvm.dbg.declare(metadata %swift.type** %T1, metadata !160, metadata !DIExpression()), !dbg !164
  %x.debug = alloca %swift.opaque*, align 8
  call void @llvm.dbg.declare(metadata %swift.opaque** %x.debug, metadata !161, metadata !DIExpression(DW_OP_deref)), !dbg !165
  %1 = bitcast %swift.opaque** %x.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  %k.debug = alloca %swift.opaque*, align 8
  %2 = bitcast %swift.opaque** %k.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 8, i1 false)
  %3 = bitcast %swift.opaque** %k.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  store %swift.type* %T, %swift.type** %T1, align 8
  %4 = bitcast %swift.type* %T to i8***, !dbg !166
  %5 = getelementptr inbounds i8**, i8*** %4, i64 -1, !dbg !166
  %T.valueWitnesses = load i8**, i8*** %5, align 8, !dbg !166, !invariant.load !46, !dereferenceable !132
  %6 = bitcast i8** %T.valueWitnesses to %swift.vwtable*, !dbg !166
  %7 = getelementptr inbounds %swift.vwtable, %swift.vwtable* %6, i32 0, i32 8, !dbg !166
  %size = load i64, i64* %7, align 8, !dbg !166, !invariant.load !46
  %8 = alloca i8, i64 %size, align 16, !dbg !166
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %8), !dbg !166
  %9 = bitcast i8* %8 to %swift.opaque*, !dbg !166
  %10 = alloca i8, i64 %size, align 16, !dbg !166
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %10), !dbg !166
  %11 = bitcast i8* %10 to %swift.opaque*, !dbg !166
  %12 = alloca i8, i64 %size, align 16, !dbg !166
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %12), !dbg !166
  %13 = bitcast i8* %12 to %swift.opaque*, !dbg !166
  store %swift.opaque* %0, %swift.opaque** %x.debug, align 8, !dbg !164
  store %swift.opaque* %13, %swift.opaque** %k.debug, align 8, !dbg !168
  call void @llvm.dbg.declare(metadata %swift.opaque** %k.debug, metadata !162, metadata !DIExpression()), !dbg !169
  br label %14, !dbg !170

14:                                               ; preds = %entry
  %15 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 2, !dbg !170
  %16 = load i8*, i8** %15, align 8, !dbg !170, !invariant.load !46
  %initializeWithCopy = bitcast i8* %16 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !170
  %17 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %13, %swift.opaque* noalias %0, %swift.type* %T) #5, !dbg !170
  %18 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %11, %swift.opaque* noalias %13, %swift.type* %T) #5, !dbg !171
  %19 = getelementptr inbounds i8*, i8** %T.P, i32 2, !dbg !172
  %20 = load i8*, i8** %19, align 8, !dbg !172, !invariant.load !46
  %21 = bitcast i8* %20 to void (%swift.opaque*, %swift.type*, i8**)*, !dbg !172
  call swiftcc void %21(%swift.opaque* noalias nocapture swiftself %11, %swift.type* %T, i8** %T.P), !dbg !172
  %22 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 1, !dbg !172
  %23 = load i8*, i8** %22, align 8, !dbg !172, !invariant.load !46
  %destroy = bitcast i8* %23 to void (%swift.opaque*, %swift.type*)*, !dbg !172
  call void %destroy(%swift.opaque* noalias %11, %swift.type* %T) #5, !dbg !172
  %24 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %9, %swift.opaque* noalias %13, %swift.type* %T) #5, !dbg !173
  %25 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 4, !dbg !174
  %26 = load i8*, i8** %25, align 8, !dbg !174, !invariant.load !46
  %initializeWithTake = bitcast i8* %26 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !174
  %27 = call %swift.opaque* %initializeWithTake(%swift.opaque* noalias %11, %swift.opaque* noalias %13, %swift.type* %T) #5, !dbg !174
  call void @llvm.dbg.value(metadata %swift.opaque* undef, metadata !162, metadata !DIExpression()), !dbg !169
  call void %destroy(%swift.opaque* noalias %9, %swift.type* %T) #5, !dbg !174
  %28 = getelementptr inbounds i8*, i8** %T.P, i32 2, !dbg !175
  %29 = load i8*, i8** %28, align 8, !dbg !175, !invariant.load !46
  %30 = bitcast i8* %29 to void (%swift.opaque*, %swift.type*, i8**)*, !dbg !175
  call swiftcc void %30(%swift.opaque* noalias nocapture swiftself %11, %swift.type* %T, i8** %T.P), !dbg !175
  %31 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %9, %swift.opaque* noalias %0, %swift.type* %T) #5, !dbg !176
  %32 = call %swift.opaque* %initializeWithTake(%swift.opaque* noalias %13, %swift.opaque* noalias %9, %swift.type* %T) #5, !dbg !177
  store %swift.opaque* %13, %swift.opaque** %k.debug, align 8, !dbg !168
  call void @llvm.dbg.declare(metadata %swift.opaque** %k.debug, metadata !162, metadata !DIExpression()), !dbg !169
  br label %.split, !dbg !178

.split:                                           ; preds = %14
  %33 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %9, %swift.opaque* noalias %13, %swift.type* %T) #5, !dbg !178
  %34 = getelementptr inbounds i8*, i8** %T.P, i32 2, !dbg !179
  %35 = load i8*, i8** %34, align 8, !dbg !179, !invariant.load !46
  %36 = bitcast i8* %35 to void (%swift.opaque*, %swift.type*, i8**)*, !dbg !179
  call swiftcc void %36(%swift.opaque* noalias nocapture swiftself %9, %swift.type* %T, i8** %T.P), !dbg !179
  call void %destroy(%swift.opaque* noalias %9, %swift.type* %T) #5, !dbg !180
  call void %destroy(%swift.opaque* noalias %11, %swift.type* %T) #5, !dbg !180
  call void %destroy(%swift.opaque* noalias %13, %swift.type* %T) #5, !dbg !180
  %37 = bitcast %swift.opaque* %13 to i8*, !dbg !180
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %37), !dbg !180
  %38 = bitcast %swift.opaque* %11 to i8*, !dbg !180
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %38), !dbg !180
  %39 = bitcast %swift.opaque* %9 to i8*, !dbg !180
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %39), !dbg !180
  ret void, !dbg !180
}

;; DWARF: DW_AT_linkage_name   ("$s4main21addressOnlyVarArgTestyyxz_xtAA1PRzlF")
;; DWARF-NEXT: DW_AT_name      ("addressOnlyVarArgTest")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
;; DWARF-NEXT: DW_AT_external  (
;;
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location	(DW_OP_fbreg
;; DWARF-NEXT: DW_AT_name      ("k")
;; DWARF-NEXT: DW_AT_decl_file (
;; DWARF-NEXT: DW_AT_decl_line (
;; DWARF-NEXT: DW_AT_type      (
define swiftcc void @"$s4main21addressOnlyVarArgTestyyxz_xtAA1PRzlF"(%swift.opaque* nocapture %0, %swift.opaque* noalias nocapture %1, %swift.type* %T, i8** %T.P) #2 !dbg !181 {
entry:
  %T1 = alloca %swift.type*, align 8
  call void @llvm.dbg.declare(metadata %swift.type** %T1, metadata !185, metadata !DIExpression()), !dbg !188
  %k.debug = alloca %swift.opaque*, align 8
  %2 = bitcast %swift.opaque** %k.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %2, i8 0, i64 8, i1 false)
  %x.debug = alloca %swift.opaque*, align 8
  call void @llvm.dbg.declare(metadata %swift.opaque** %x.debug, metadata !187, metadata !DIExpression(DW_OP_deref)), !dbg !189
  %3 = bitcast %swift.opaque** %x.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  %4 = bitcast %swift.opaque** %k.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %4, i8 0, i64 8, i1 false)
  store %swift.type* %T, %swift.type** %T1, align 8
  %5 = bitcast %swift.type* %T to i8***, !dbg !190
  %6 = getelementptr inbounds i8**, i8*** %5, i64 -1, !dbg !190
  %T.valueWitnesses = load i8**, i8*** %6, align 8, !dbg !190, !invariant.load !46, !dereferenceable !132
  %7 = bitcast i8** %T.valueWitnesses to %swift.vwtable*, !dbg !190
  %8 = getelementptr inbounds %swift.vwtable, %swift.vwtable* %7, i32 0, i32 8, !dbg !190
  %size = load i64, i64* %8, align 8, !dbg !190, !invariant.load !46
  %9 = alloca i8, i64 %size, align 16, !dbg !190
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %9), !dbg !190
  %10 = bitcast i8* %9 to %swift.opaque*, !dbg !190
  %11 = alloca i8, i64 %size, align 16, !dbg !190
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %11), !dbg !190
  %12 = bitcast i8* %11 to %swift.opaque*, !dbg !190
  store %swift.opaque* %0, %swift.opaque** %k.debug, align 8, !dbg !188
  call void @llvm.dbg.declare(metadata %swift.opaque** %k.debug, metadata !186, metadata !DIExpression(DW_OP_deref)), !dbg !193
  br label %entry.split2, !dbg !188

entry.split2:                                     ; preds = %entry
  store %swift.opaque* %1, %swift.opaque** %x.debug, align 8, !dbg !188
  %13 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 2, !dbg !194
  %14 = load i8*, i8** %13, align 8, !dbg !194, !invariant.load !46
  %initializeWithCopy = bitcast i8* %14 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !194
  %15 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %12, %swift.opaque* noalias %0, %swift.type* %T) #5, !dbg !194
  %16 = getelementptr inbounds i8*, i8** %T.P, i32 2, !dbg !195
  %17 = load i8*, i8** %16, align 8, !dbg !195, !invariant.load !46
  %18 = bitcast i8* %17 to void (%swift.opaque*, %swift.type*, i8**)*, !dbg !195
  call swiftcc void %18(%swift.opaque* noalias nocapture swiftself %12, %swift.type* %T, i8** %T.P), !dbg !195
  %19 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 1, !dbg !195
  %20 = load i8*, i8** %19, align 8, !dbg !195, !invariant.load !46
  %destroy = bitcast i8* %20 to void (%swift.opaque*, %swift.type*)*, !dbg !195
  call void %destroy(%swift.opaque* noalias %12, %swift.type* %T) #5, !dbg !195
  %21 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %10, %swift.opaque* noalias %0, %swift.type* %T) #5, !dbg !196
  %22 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 4, !dbg !197
  %23 = load i8*, i8** %22, align 8, !dbg !197, !invariant.load !46
  %initializeWithTake = bitcast i8* %23 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !197
  %24 = call %swift.opaque* %initializeWithTake(%swift.opaque* noalias %12, %swift.opaque* noalias %0, %swift.type* %T) #5, !dbg !197
  call void @llvm.dbg.value(metadata %swift.opaque* undef, metadata !186, metadata !DIExpression(DW_OP_deref)), !dbg !193
  call void %destroy(%swift.opaque* noalias %10, %swift.type* %T) #5, !dbg !197
  %25 = getelementptr inbounds i8*, i8** %T.P, i32 2, !dbg !198
  %26 = load i8*, i8** %25, align 8, !dbg !198, !invariant.load !46
  %27 = bitcast i8* %26 to void (%swift.opaque*, %swift.type*, i8**)*, !dbg !198
  call swiftcc void %27(%swift.opaque* noalias nocapture swiftself %12, %swift.type* %T, i8** %T.P), !dbg !198
  %28 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %10, %swift.opaque* noalias %1, %swift.type* %T) #5, !dbg !199
  %29 = call %swift.opaque* %initializeWithTake(%swift.opaque* noalias %0, %swift.opaque* noalias %10, %swift.type* %T) #5, !dbg !200
  store %swift.opaque* %0, %swift.opaque** %k.debug, align 8, !dbg !188
  call void @llvm.dbg.declare(metadata %swift.opaque** %k.debug, metadata !186, metadata !DIExpression(DW_OP_deref)), !dbg !193
  br label %entry.split, !dbg !201

entry.split:                                      ; preds = %entry.split2
  %30 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %10, %swift.opaque* noalias %0, %swift.type* %T) #5, !dbg !201
  %31 = getelementptr inbounds i8*, i8** %T.P, i32 2, !dbg !202
  %32 = load i8*, i8** %31, align 8, !dbg !202, !invariant.load !46
  %33 = bitcast i8* %32 to void (%swift.opaque*, %swift.type*, i8**)*, !dbg !202
  call swiftcc void %33(%swift.opaque* noalias nocapture swiftself %10, %swift.type* %T, i8** %T.P), !dbg !202
  call void %destroy(%swift.opaque* noalias %10, %swift.type* %T) #5, !dbg !203
  call void %destroy(%swift.opaque* noalias %12, %swift.type* %T) #5, !dbg !203
  %34 = bitcast %swift.opaque* %12 to i8*, !dbg !203
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %34), !dbg !203
  %35 = bitcast %swift.opaque* %10 to i8*, !dbg !203
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %35), !dbg !203
  ret void, !dbg !203
}

attributes #0 = { argmemonly nofree nounwind willreturn writeonly }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #3 = { noinline nounwind readnone "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind willreturn }
attributes #5 = { nounwind }
attributes #6 = { argmemonly nofree nosync nounwind willreturn }
attributes #7 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!swift.module.flags = !{!11}
!llvm.asan.globals = !{!12, !13, !14, !15, !16, !17, !18, !19, !20, !21}
!llvm.module.flags = !{!22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36}
!llvm.linker.options = !{!37, !38, !39, !40, !41}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Swift version 5.7-dev (LLVM 7abf5772a7e9e08, Swift c57bc03b341eaae)", isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, imports: !2)
!1 = !DIFile(filename: "move_function_dbginfo.swift", directory: "/Volumes/Data/work/solon/swift/test/DebugInfo")
!2 = !{!3, !5, !7, !9}
!3 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !4, file: !1)
!4 = !DIModule(scope: null, name: "main")
!5 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !6, file: !1)
!6 = !DIModule(scope: null, name: "Swift", includePath: "/Volumes/Data/work/solon/build/Ninja-RelWithDebInfoAssert/swift-macosx-x86_64/lib/swift/macosx/Swift.swiftmodule/x86_64-apple-macos.swiftmodule")
!7 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !8, file: !1)
!8 = !DIModule(scope: null, name: "_Concurrency", includePath: "/Volumes/Data/work/solon/build/Ninja-RelWithDebInfoAssert/swift-macosx-x86_64/lib/swift/macosx/_Concurrency.swiftmodule/x86_64-apple-macos.swiftmodule")
!9 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !10, file: !1)
!10 = !DIModule(scope: null, name: "SwiftOnoneSupport", includePath: "/Volumes/Data/work/solon/build/Ninja-RelWithDebInfoAssert/swift-macosx-x86_64/lib/swift/macosx/SwiftOnoneSupport.swiftmodule/x86_64-apple-macos.swiftmodule")
!11 = !{!"standard-library", i1 false}
!12 = distinct !{null, null, null, i1 false, i1 true}
!13 = distinct !{null, null, null, i1 false, i1 true}
!14 = distinct !{null, null, null, i1 false, i1 true}
!15 = distinct !{null, null, null, i1 false, i1 true}
!16 = distinct !{null, null, null, i1 false, i1 true}
!17 = distinct !{null, null, null, i1 false, i1 true}
!18 = distinct !{null, null, null, i1 false, i1 true}
!19 = distinct !{null, null, null, i1 false, i1 true}
!20 = distinct !{null, null, null, i1 false, i1 true}
!21 = distinct !{null, null, null, i1 false, i1 true}
!22 = !{i32 1, !"Objective-C Version", i32 2}
!23 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!24 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!25 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!26 = !{i32 1, !"Objective-C Class Properties", i32 64}
!27 = !{i32 7, !"Dwarf Version", i32 4}
!28 = !{i32 2, !"Debug Info Version", i32 3}
!29 = !{i32 1, !"wchar_size", i32 4}
!30 = !{i32 7, !"PIC Level", i32 2}
!31 = !{i32 7, !"uwtable", i32 1}
!32 = !{i32 7, !"frame-pointer", i32 2}
!33 = !{i32 1, !"Swift Version", i32 7}
!34 = !{i32 1, !"Swift ABI Version", i32 7}
!35 = !{i32 1, !"Swift Major Version", i8 5}
!36 = !{i32 1, !"Swift Minor Version", i8 7}
!37 = !{!"-lswiftSwiftOnoneSupport"}
!38 = !{!"-lswiftCore"}
!39 = !{!"-lswift_Concurrency"}
!40 = !{!"-lobjc"}
!41 = !{!"-lswiftCompatibilityConcurrency"}
!42 = distinct !DISubprogram(name: "copyableValueTest", linkageName: "$s4main17copyableValueTestyyF", scope: !4, file: !1, line: 81, type: !43, scopeLine: 81, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !47)
!43 = !DISubroutineType(types: !44)
!44 = !{!45}
!45 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sytD", file: !1, elements: !46, runtimeLang: DW_LANG_Swift, identifier: "$sytD")
!46 = !{}
!47 = !{!48, !52}
!48 = !DILocalVariable(name: "k", scope: !49, file: !1, line: 82, type: !50)
!49 = distinct !DILexicalBlock(scope: !42, file: !1, line: 81, column: 33)
!50 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !51)
!51 = !DICompositeType(tag: DW_TAG_structure_type, name: "Klass", scope: !4, file: !1, size: 64, elements: !46, runtimeLang: DW_LANG_Swift, identifier: "$s4main5KlassCD")
!52 = !DILocalVariable(name: "m", scope: !49, file: !1, line: 84, type: !50)
!53 = !DILocation(line: 84, column: 9, scope: !49)
!54 = !DILocation(line: 82, column: 13, scope: !49)
!55 = !DILocation(line: 0, scope: !49)
!56 = !DILocation(line: 82, column: 9, scope: !49)
!57 = !DILocation(line: 83, column: 7, scope: !49)
!58 = !DILocation(line: 84, column: 19, scope: !49)
!59 = !DILocation(line: 85, column: 7, scope: !49)
!60 = !DILocation(line: 86, column: 1, scope: !49)
!61 = distinct !DISubprogram(name: "copyableArgTest", linkageName: "$s4main15copyableArgTestyyAA5KlassCnF", scope: !4, file: !1, line: 128, type: !62, scopeLine: 128, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !64)
!62 = !DISubroutineType(types: !63)
!63 = !{!45, !51}
!64 = !{!65, !66}
!65 = !DILocalVariable(name: "k", arg: 1, scope: !61, file: !1, line: 128, type: !50)
!66 = !DILocalVariable(name: "m", scope: !67, file: !1, line: 130, type: !50)
!67 = distinct !DILexicalBlock(scope: !61, file: !1, line: 128, column: 49)
!68 = !DILocation(line: 130, column: 9, scope: !67)
!69 = !DILocation(line: 0, scope: !61)
!70 = !DILocation(line: 128, column: 29, scope: !61)
!71 = !DILocation(line: 129, column: 7, scope: !67)
!72 = !DILocation(line: 130, column: 19, scope: !67)
!73 = !DILocation(line: 0, scope: !67)
!74 = !DILocation(line: 131, column: 7, scope: !67)
!75 = !DILocation(line: 132, column: 1, scope: !67)
!76 = distinct !DISubprogram(name: "copyableVarTest", linkageName: "$s4main15copyableVarTestyyF", scope: !4, file: !1, line: 169, type: !43, scopeLine: 169, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !77)
!77 = !{!78, !80}
!78 = !DILocalVariable(name: "k", scope: !79, file: !1, line: 170, type: !51)
!79 = distinct !DILexicalBlock(scope: !76, file: !1, line: 169, column: 31)
!80 = !DILocalVariable(name: "m", scope: !79, file: !1, line: 172, type: !50)
!81 = !DILocation(line: 172, column: 9, scope: !79)
!82 = !DILocation(line: 0, scope: !83)
!83 = !DILexicalBlockFile(scope: !79, file: !84, discriminator: 0)
!84 = !DIFile(filename: "<compiler-generated>", directory: "")
!85 = !DILocation(line: 170, column: 9, scope: !79)
!86 = !DILocation(line: 170, column: 13, scope: !79)
!87 = !DILocation(line: 171, column: 7, scope: !79)
!88 = !DILocation(line: 172, column: 13, scope: !79)
!89 = !DILocation(line: 0, scope: !79)
!90 = !DILocation(line: 173, column: 7, scope: !79)
!91 = !DILocation(line: 174, column: 9, scope: !79)
!92 = !DILocation(line: 174, column: 7, scope: !79)
!93 = !DILocation(line: 175, column: 7, scope: !79)
!94 = !DILocation(line: 176, column: 1, scope: !79)
!95 = distinct !DISubprogram(name: "copyableVarArgTest", linkageName: "$s4main18copyableVarArgTestyyAA5KlassCzF", scope: !4, file: !1, line: 213, type: !62, scopeLine: 213, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !96)
!96 = !{!97, !98}
!97 = !DILocalVariable(name: "k", arg: 1, scope: !95, file: !1, line: 213, type: !51)
!98 = !DILocalVariable(name: "m", scope: !99, file: !1, line: 215, type: !50)
!99 = distinct !DILexicalBlock(scope: !95, file: !1, line: 213, column: 50)
!100 = !DILocation(line: 215, column: 9, scope: !99)
!101 = !DILocation(line: 0, scope: !95)
!102 = !DILocation(line: 213, column: 32, scope: !95)
!103 = !DILocation(line: 214, column: 5, scope: !99)
!104 = !DILocation(line: 214, column: 7, scope: !99)
!105 = !DILocation(line: 215, column: 13, scope: !99)
!106 = !DILocation(line: 0, scope: !99)
!107 = !DILocation(line: 216, column: 7, scope: !99)
!108 = !DILocation(line: 217, column: 9, scope: !99)
!109 = !DILocation(line: 217, column: 7, scope: !99)
!110 = !DILocation(line: 218, column: 5, scope: !99)
!111 = !DILocation(line: 218, column: 7, scope: !99)
!112 = !DILocation(line: 219, column: 1, scope: !99)
!113 = distinct !DISubprogram(name: "addressOnlyValueTest", linkageName: "$s4main20addressOnlyValueTestyyxAA1PRzlF", scope: !4, file: !1, line: 265, type: !114, scopeLine: 265, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !117)
!114 = !DISubroutineType(types: !115)
!115 = !{!45, !116}
!116 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sxD", file: !1, runtimeLang: DW_LANG_Swift, identifier: "$sxD")
!117 = !{!118, !122, !125, !126}
!118 = !DILocalVariable(name: "$\CF\84_0_0", scope: !113, file: !1, type: !119, flags: DIFlagArtificial)
!119 = !DIDerivedType(tag: DW_TAG_typedef, name: "T", scope: !120, file: !84, baseType: !121)
!120 = !DIModule(scope: null, name: "Builtin")
!121 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "$sBpD", baseType: null, size: 64)
!122 = !DILocalVariable(name: "m", scope: !123, file: !1, line: 268, type: !124)
!123 = distinct !DILexicalBlock(scope: !113, file: !1, line: 265, column: 49)
!124 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !116)
!125 = !DILocalVariable(name: "x", arg: 1, scope: !113, file: !1, line: 265, type: !124)
!126 = !DILocalVariable(name: "k", scope: !123, file: !1, line: 266, type: !124)
!127 = !DILocation(line: 0, scope: !113)
!128 = !DILocation(line: 268, column: 9, scope: !123)
!129 = !DILocation(line: 265, column: 41, scope: !113)
!130 = !DILocation(line: 0, scope: !131)
!131 = !DILexicalBlockFile(scope: !123, file: !84, discriminator: 0)
!132 = !{i64 96}
!133 = !DILocation(line: 0, scope: !123)
!134 = !DILocation(line: 266, column: 9, scope: !123)
!135 = !DILocation(line: 266, column: 13, scope: !123)
!136 = !DILocation(line: 267, column: 7, scope: !123)
!137 = !DILocation(line: 268, column: 19, scope: !123)
!138 = !DILocation(line: 268, column: 13, scope: !123)
!139 = !DILocation(line: 269, column: 7, scope: !123)
!140 = !DILocation(line: 270, column: 1, scope: !123)
!141 = distinct !DISubprogram(name: "addressOnlyValueArgTest", linkageName: "$s4main23addressOnlyValueArgTestyyxnAA1PRzlF", scope: !4, file: !1, line: 308, type: !114, scopeLine: 308, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !142)
!142 = !{!143, !144, !146}
!143 = !DILocalVariable(name: "$\CF\84_0_0", scope: !141, file: !1, type: !119, flags: DIFlagArtificial)
!144 = !DILocalVariable(name: "m", scope: !145, file: !1, line: 310, type: !124)
!145 = distinct !DILexicalBlock(scope: !141, file: !1, line: 308, column: 60)
!146 = !DILocalVariable(name: "k", arg: 1, scope: !141, file: !1, line: 308, type: !124)
!147 = !DILocation(line: 0, scope: !141)
!148 = !DILocation(line: 310, column: 9, scope: !145)
!149 = !DILocation(line: 0, scope: !150)
!150 = !DILexicalBlockFile(scope: !145, file: !84, discriminator: 0)
!151 = !DILocation(line: 0, scope: !145)
!152 = !DILocation(line: 308, column: 44, scope: !141)
!153 = !DILocation(line: 309, column: 7, scope: !145)
!154 = !DILocation(line: 310, column: 19, scope: !145)
!155 = !DILocation(line: 310, column: 13, scope: !145)
!156 = !DILocation(line: 311, column: 7, scope: !145)
!157 = !DILocation(line: 312, column: 1, scope: !145)
!158 = distinct !DISubprogram(name: "addressOnlyVarTest", linkageName: "$s4main18addressOnlyVarTestyyxAA1PRzlF", scope: !4, file: !1, line: 362, type: !114, scopeLine: 362, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !159)
!159 = !{!160, !161, !162}
!160 = !DILocalVariable(name: "$\CF\84_0_0", scope: !158, file: !1, type: !119, flags: DIFlagArtificial)
!161 = !DILocalVariable(name: "x", arg: 1, scope: !158, file: !1, line: 362, type: !124)
!162 = !DILocalVariable(name: "k", scope: !163, file: !1, line: 363, type: !116)
!163 = distinct !DILexicalBlock(scope: !158, file: !1, line: 362, column: 47)
!164 = !DILocation(line: 0, scope: !158)
!165 = !DILocation(line: 362, column: 39, scope: !158)
!166 = !DILocation(line: 0, scope: !167)
!167 = !DILexicalBlockFile(scope: !163, file: !84, discriminator: 0)
!168 = !DILocation(line: 0, scope: !163)
!169 = !DILocation(line: 363, column: 9, scope: !163)
!170 = !DILocation(line: 363, column: 13, scope: !163)
!171 = !DILocation(line: 364, column: 5, scope: !163)
!172 = !DILocation(line: 364, column: 7, scope: !163)
!173 = !DILocation(line: 365, column: 19, scope: !163)
!174 = !DILocation(line: 365, column: 13, scope: !163)
!175 = !DILocation(line: 366, column: 7, scope: !163)
!176 = !DILocation(line: 367, column: 9, scope: !163)
!177 = !DILocation(line: 367, column: 7, scope: !163)
!178 = !DILocation(line: 368, column: 5, scope: !163)
!179 = !DILocation(line: 368, column: 7, scope: !163)
!180 = !DILocation(line: 369, column: 1, scope: !163)
!181 = distinct !DISubprogram(name: "addressOnlyVarArgTest", linkageName: "$s4main21addressOnlyVarArgTestyyxz_xtAA1PRzlF", scope: !4, file: !1, line: 418, type: !182, scopeLine: 418, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !184)
!182 = !DISubroutineType(types: !183)
!183 = !{!45, !116, !116}
!184 = !{!185, !186, !187}
!185 = !DILocalVariable(name: "$\CF\84_0_0", scope: !181, file: !1, type: !119, flags: DIFlagArtificial)
!186 = !DILocalVariable(name: "k", arg: 1, scope: !181, file: !1, line: 418, type: !116)
!187 = !DILocalVariable(name: "x", arg: 2, scope: !181, file: !1, line: 418, type: !124)
!188 = !DILocation(line: 0, scope: !181)
!189 = !DILocation(line: 418, column: 56, scope: !181)
!190 = !DILocation(line: 0, scope: !191)
!191 = !DILexicalBlockFile(scope: !192, file: !84, discriminator: 0)
!192 = distinct !DILexicalBlock(scope: !181, file: !1, line: 418, column: 64)
!193 = !DILocation(line: 418, column: 42, scope: !181)
!194 = !DILocation(line: 419, column: 5, scope: !192)
!195 = !DILocation(line: 419, column: 7, scope: !192)
!196 = !DILocation(line: 420, column: 19, scope: !192)
!197 = !DILocation(line: 420, column: 13, scope: !192)
!198 = !DILocation(line: 421, column: 7, scope: !192)
!199 = !DILocation(line: 422, column: 9, scope: !192)
!200 = !DILocation(line: 422, column: 7, scope: !192)
!201 = !DILocation(line: 423, column: 5, scope: !192)
!202 = !DILocation(line: 423, column: 7, scope: !192)
!203 = !DILocation(line: 424, column: 1, scope: !192)
