; RUN: llc -O3 < %s | FileCheck %s

; Regression test for https://github.com/llvm/llvm-project/pull/110889#issuecomment-2393405613
; Marking x64 SEH instructions as meta led to cv directives being duplicated, which caused
; `cv_fpo_stackalloc` to be observed after seeing a `cv_fpo_endprologue`, which is an error.

; Generated from the following code:
; int q;
; class b {
; public:
;   b();
; };
; struct G {
;   char n[sizeof(void *)];
;   int *i;
;   int p() const { return n[0] ? *i : 1; }
;   int s() const;
; };
; int G::s() const {
;   q = p();
;   b();
; }
; To reproduce: clang -target i686-w64-mingw32 -w -c repro.cpp -O3 -g -gcodeview  -emit-llvm

; CHECK-LABEL:  __ZNK1G1sEv:
; CHECK:        .cv_fpo_proc    __ZNK1G1sEv 0
; CHECK:        .cv_fpo_stackalloc  4
; CHECK:        .cv_fpo_endprologue
; CHECK-NOT:    .cv_fpo_stackalloc
; CHECK-NOT:    .cv_fpo_endprologue

target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-w64-windows-gnu"

%class.b = type { i8 }

@q = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

; Function Attrs: mustprogress noreturn
define dso_local x86_thiscallcc noundef i32 @_ZNK1G1sEv(ptr nocapture noundef nonnull readonly align 4 dereferenceable(8) %this) local_unnamed_addr #0 align 2 !dbg !13 {
entry:
  %agg.tmp.ensured = alloca %class.b, align 1
    #dbg_value(ptr %this, !30, !DIExpression(), !32)
    #dbg_value(ptr %this, !33, !DIExpression(), !36)
  %0 = load i8, ptr %this, align 4, !dbg !38, !tbaa !39
  %tobool.not.i = icmp eq i8 %0, 0, !dbg !38
  br i1 %tobool.not.i, label %_ZNK1G1pEv.exit, label %cond.true.i, !dbg !38

cond.true.i:                                      ; preds = %entry
  %i.i = getelementptr inbounds nuw i8, ptr %this, i32 4, !dbg !38
  %1 = load ptr, ptr %i.i, align 4, !dbg !38, !tbaa !42
  %2 = load i32, ptr %1, align 4, !dbg !38, !tbaa !45
  br label %_ZNK1G1pEv.exit, !dbg !38

_ZNK1G1pEv.exit:                                  ; preds = %entry, %cond.true.i
  %cond.i = phi i32 [ %2, %cond.true.i ], [ 1, %entry ], !dbg !38
  store i32 %cond.i, ptr @q, align 4, !dbg !47, !tbaa !45
  call x86_thiscallcc void @_ZN1bC1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp.ensured), !dbg !48
  unreachable, !dbg !48
}

declare dso_local x86_thiscallcc void @_ZN1bC1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #1

attributes #0 = { mustprogress noreturn "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "q", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 20.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "repro.cpp", directory: "C:\\llvm", checksumkind: CSK_MD5, checksum: "54362b0cc0bf4b9927aafc8b00498049")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 1, !"NumRegisterParameters", i32 0}
!7 = !{i32 2, !"CodeView", i32 1}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 2}
!10 = !{i32 1, !"MaxTLSAlign", i32 65536}
!11 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!12 = !{!"clang version 20.0.0"}
!13 = distinct !DISubprogram(name: "s", linkageName: "_ZNK1G1sEv", scope: !14, file: !3, line: 12, type: !24, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !28, retainedNodes: !29)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "G", file: !3, line: 6, size: 64, flags: DIFlagTypePassByValue, elements: !15, identifier: "_ZTS1G")
!15 = !{!16, !21, !23, !28}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !14, file: !3, line: 7, baseType: !17, size: 32)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 32, elements: !19)
!18 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!19 = !{!20}
!20 = !DISubrange(count: 4)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !14, file: !3, line: 8, baseType: !22, size: 32, offset: 32)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 32)
!23 = !DISubprogram(name: "p", linkageName: "_ZNK1G1pEv", scope: !14, file: !3, line: 9, type: !24, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!24 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !25)
!25 = !{!5, !26}
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!27 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)
!28 = !DISubprogram(name: "s", linkageName: "_ZNK1G1sEv", scope: !14, file: !3, line: 10, type: !24, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!29 = !{!30}
!30 = !DILocalVariable(name: "this", arg: 1, scope: !13, type: !31, flags: DIFlagArtificial | DIFlagObjectPointer)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 32)
!32 = !DILocation(line: 0, scope: !13)
!33 = !DILocalVariable(name: "this", arg: 1, scope: !34, type: !31, flags: DIFlagArtificial | DIFlagObjectPointer)
!34 = distinct !DISubprogram(name: "p", linkageName: "_ZNK1G1pEv", scope: !14, file: !3, line: 9, type: !24, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !23, retainedNodes: !35)
!35 = !{!33}
!36 = !DILocation(line: 0, scope: !34, inlinedAt: !37)
!37 = distinct !DILocation(line: 13, scope: !13)
!38 = !DILocation(line: 9, scope: !34, inlinedAt: !37)
!39 = !{!40, !40, i64 0}
!40 = !{!"omnipotent char", !41, i64 0}
!41 = !{!"Simple C++ TBAA"}
!42 = !{!43, !44, i64 4}
!43 = !{!"_ZTS1G", !40, i64 0, !44, i64 4}
!44 = !{!"any pointer", !40, i64 0}
!45 = !{!46, !46, i64 0}
!46 = !{!"int", !40, i64 0}
!47 = !DILocation(line: 13, scope: !13)
!48 = !DILocation(line: 14, scope: !13)
