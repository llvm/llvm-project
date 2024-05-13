; ModuleID = '/tmp/odr-lambda-1.ii'

; RUN: llvm-link %s %p/Inputs/odr-lambda-2.ll -S -o - 2>&1 | FileCheck %s

; When materializing the ODR-uniqued types they may be resolved to types from a
; previously loaded module. Don't treat this as an error.
; CHECK-NOT: ignoring invalid debug info
; CHECK: !llvm.dbg.cu = !{!{{[0-9]+}}, !{{[0-9]+}}}
; CHECK: distinct !DICompositeType(tag: DW_TAG_class_type, {{.*}}identifier: "_ZTSZ12consumeError5ErrorEUlvE_")
; CHECK-NOT: identifier: "_ZTSZ12consumeError5ErrorEUlvE_"


; generated from:
; clang++ -x c++ -std=c++14 -fPIC -flto -g -fno-exceptions -fno-rtti
; class Error {};
; template <typename HandlerTs>
; void handleAllErrors(HandlerTs Handlers) {}
; inline void consumeError(Error Err) {
;   handleAllErrors( []() {});
; }
; void ArchiveMemberHeader() 
; {
;   consumeError(Error());
; }

source_filename = "/tmp/odr-lambda-1.ii"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

%class.Error = type { i8 }
%class.anon = type { i8 }

; Function Attrs: noinline nounwind optnone ssp uwtable
define void @_Z19ArchiveMemberHeaderv() #0 !dbg !8 {
entry:
  %agg.tmp = alloca %class.Error, align 1
  call void @_Z12consumeError5Error(), !dbg !11
  ret void, !dbg !12
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr void @_Z12consumeError5Error() #0 !dbg !13 {
entry:
  %Err = alloca %class.Error, align 1
  %agg.tmp = alloca %class.anon, align 1
  call void @llvm.dbg.declare(metadata ptr %Err, metadata !17, metadata !DIExpression()), !dbg !18
  call void @_Z15handleAllErrorsIZ12consumeError5ErrorEUlvE_EvT_(), !dbg !19
  ret void, !dbg !20
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr void @_Z15handleAllErrorsIZ12consumeError5ErrorEUlvE_EvT_() #0 !dbg !21 {
entry:
  %Handlers = alloca %class.anon, align 1
  call void @llvm.dbg.declare(metadata ptr %Handlers, metadata !27, metadata !DIExpression()), !dbg !28
  ret void, !dbg !29
}

attributes #0 = { noinline nounwind optnone ssp uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 315772) (llvm/trunk 315773)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/odr-lambda-1.ii", directory: "/Data")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 (trunk 315772) (llvm/trunk 315773)"}
!8 = distinct !DISubprogram(name: "ArchiveMemberHeader", linkageName: "_Z19ArchiveMemberHeaderv", scope: !1, file: !1, line: 7, type: !9, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 9, column: 3, scope: !8)
!12 = !DILocation(line: 10, column: 1, scope: !8)
!13 = distinct !DISubprogram(name: "consumeError", linkageName: "_Z12consumeError5Error", scope: !1, file: !1, line: 4, type: !14, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16}
!16 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Error", file: !1, line: 1, size: 8, elements: !2, identifier: "_ZTS5Error")
!17 = !DILocalVariable(name: "Err", arg: 1, scope: !13, file: !1, line: 4, type: !16)
!18 = !DILocation(line: 4, column: 32, scope: !13)
!19 = !DILocation(line: 5, column: 3, scope: !13)
!20 = !DILocation(line: 6, column: 1, scope: !13)
!21 = distinct !DISubprogram(name: "handleAllErrors<(lambda at /tmp/odr-lambda-1.ii:5:20)>", linkageName: "_Z15handleAllErrorsIZ12consumeError5ErrorEUlvE_EvT_", scope: !1, file: !1, line: 3, type: !22, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, templateParams: !25, retainedNodes: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !24}
!24 = distinct !DICompositeType(tag: DW_TAG_class_type, scope: !13, file: !1, line: 5, size: 8, elements: !2, identifier: "_ZTSZ12consumeError5ErrorEUlvE_")
!25 = !{!26}
!26 = !DITemplateTypeParameter(name: "HandlerTs", type: !24)
!27 = !DILocalVariable(name: "Handlers", arg: 1, scope: !21, file: !1, line: 3, type: !24)
!28 = !DILocation(line: 3, column: 32, scope: !21)
!29 = !DILocation(line: 3, column: 43, scope: !21)
