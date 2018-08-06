; ModuleID = '/tmp/odr-lambda-2.ii'
; generated from:

; clang++ -x c++ -std=c++14 -fPIC -flto -g -fno-exceptions -fno-rtti
; class Error {};
; template <typename HandlerTs>
; void handleAllErrors( HandlerTs  Handlers) {}
; inline void consumeError(Error Err) {
;   handleAllErrors( []() {});
; }
; int main(int argc, char **argv) {
;   consumeError(Error());
; }

source_filename = "/tmp/odr-lambda-2.ii"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

%class.Error = type { i8 }
%class.anon = type { i8 }

; Function Attrs: noinline norecurse nounwind optnone ssp uwtable
define i32 @main(i32 %argc, i8** %argv) #0 !dbg !8 {
entry:
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %agg.tmp = alloca %class.Error, align 1
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !15, metadata !DIExpression()), !dbg !16
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !17, metadata !DIExpression()), !dbg !18
  call void @_Z12consumeError5Error(), !dbg !19
  ret i32 0, !dbg !20
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr void @_Z12consumeError5Error() #2 !dbg !21 {
entry:
  %Err = alloca %class.Error, align 1
  %agg.tmp = alloca %class.anon, align 1
  call void @llvm.dbg.declare(metadata %class.Error* %Err, metadata !25, metadata !DIExpression()), !dbg !26
  call void @_Z15handleAllErrorsIZ12consumeError5ErrorEUlvE_EvT_(), !dbg !27
  ret void, !dbg !28
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr void @_Z15handleAllErrorsIZ12consumeError5ErrorEUlvE_EvT_() #2 !dbg !29 {
entry:
  %Handlers = alloca %class.anon, align 1
  call void @llvm.dbg.declare(metadata %class.anon* %Handlers, metadata !35, metadata !DIExpression()), !dbg !36
  ret void, !dbg !37
}

attributes #0 = { noinline norecurse nounwind optnone ssp uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline nounwind optnone ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 315772) (llvm/trunk 315773)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/odr-lambda-2.ii", directory: "/Data")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 (trunk 315772) (llvm/trunk 315773)"}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !9, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11, !12}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!15 = !DILocalVariable(name: "argc", arg: 1, scope: !8, file: !1, line: 7, type: !11)
!16 = !DILocation(line: 7, column: 14, scope: !8)
!17 = !DILocalVariable(name: "argv", arg: 2, scope: !8, file: !1, line: 7, type: !12)
!18 = !DILocation(line: 7, column: 27, scope: !8)
!19 = !DILocation(line: 8, column: 3, scope: !8)
!20 = !DILocation(line: 9, column: 1, scope: !8)
!21 = distinct !DISubprogram(name: "consumeError", linkageName: "_Z12consumeError5Error", scope: !1, file: !1, line: 4, type: !22, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !24}
!24 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Error", file: !1, line: 1, size: 8, elements: !2, identifier: "_ZTS5Error")
!25 = !DILocalVariable(name: "Err", arg: 1, scope: !21, file: !1, line: 4, type: !24)
!26 = !DILocation(line: 4, column: 32, scope: !21)
!27 = !DILocation(line: 5, column: 3, scope: !21)
!28 = !DILocation(line: 6, column: 1, scope: !21)
!29 = distinct !DISubprogram(name: "handleAllErrors<(lambda at /tmp/odr-lambda-2.ii:5:20)>", linkageName: "_Z15handleAllErrorsIZ12consumeError5ErrorEUlvE_EvT_", scope: !1, file: !1, line: 3, type: !30, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, templateParams: !33, retainedNodes: !2)
!30 = !DISubroutineType(types: !31)
!31 = !{null, !32}
!32 = distinct !DICompositeType(tag: DW_TAG_class_type, scope: !21, file: !1, line: 5, size: 8, elements: !2, identifier: "_ZTSZ12consumeError5ErrorEUlvE_")
!33 = !{!34}
!34 = !DITemplateTypeParameter(name: "HandlerTs", type: !32)
!35 = !DILocalVariable(name: "Handlers", arg: 1, scope: !29, file: !1, line: 3, type: !32)
!36 = !DILocation(line: 3, column: 34, scope: !29)
!37 = !DILocation(line: 3, column: 45, scope: !29)
