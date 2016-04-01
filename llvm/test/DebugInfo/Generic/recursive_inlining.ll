; REQUIRES: object-emission

; RUN: %llc_dwarf -filetype=obj -O0 < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; This isn't a very pretty test case - I imagine there might be other ways to
; tickle the optimizers into producing the desired code, but I haven't found
; them.

; The issue is when a function is inlined into itself, the inlined argument
; accidentally overwrote the concrete argument and was lost.

; IR generated from the following source compiled with clang -g:
; void fn1(void *);
; void fn2(int, int, int, int);
; void fn3();
; void fn8();
; struct C {
;   int b;
;   void m_fn2() {
;     fn8();
;     if (b) fn2(0, 0, 0, 0);
;     fn3();
;   }
; };
; C *x;
; inline void fn7() {}
; void fn6() {
;   fn8();
;   x->m_fn2();
;   fn7();
; }
; void fn3() { fn6(); }
; void fn4() { x->m_fn2(); }
; void fn5() { x->m_fn2(); }

; The definition of C and declaration of C::m_fn2
; CHECK: DW_TAG_structure_type
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: DW_TAG_member
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[M_FN2_DECL:.*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "m_fn2"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter

; The abstract definition of C::m_fn2
; CHECK: [[M_FN2_ABS_DEF:.*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_specification {{.*}} {[[M_FN2_DECL]]}
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_inline
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[M_FN2_THIS_ABS_DEF:.*]]:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "this"

; Skip some other functions
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram

; The concrete definition of C::m_fn2
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_abstract_origin {{.*}} {[[M_FN2_ABS_DEF]]}
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} {[[M_FN2_THIS_ABS_DEF]]}
; CHECK-NOT: {{DW_TAG|NULL}}
; Inlined fn3:
; CHECK:     DW_TAG_inlined_subroutine
; CHECK-NOT: {{DW_TAG|NULL}}
; Inlined fn6:
; CHECK:       DW_TAG_inlined_subroutine
; CHECK-NOT: {{DW_TAG|NULL}}
; Inlined C::m_fn2:
; CHECK:         DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:           DW_AT_abstract_origin {{.*}} {[[M_FN2_ABS_DEF]]}
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:           DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:              DW_AT_abstract_origin {{.*}} {[[M_FN2_THIS_ABS_DEF]]}



%struct.C = type { i32 }

@x = global %struct.C* null, align 8

; Function Attrs: nounwind
define void @_Z3fn6v() #0 !dbg !14 {
entry:
  tail call void @_Z3fn8v() #3, !dbg !31
  %0 = load %struct.C*, %struct.C** @x, align 8, !dbg !32, !tbaa !33
  tail call void @llvm.dbg.value(metadata %struct.C* %0, i64 0, metadata !37, metadata !DIExpression()) #3, !dbg !38
  tail call void @_Z3fn8v() #3, !dbg !39
  %b.i = getelementptr inbounds %struct.C, %struct.C* %0, i64 0, i32 0, !dbg !40
  %1 = load i32, i32* %b.i, align 4, !dbg !40, !tbaa !42
  %tobool.i = icmp eq i32 %1, 0, !dbg !40
  br i1 %tobool.i, label %_ZN1C5m_fn2Ev.exit, label %if.then.i, !dbg !40

if.then.i:                                        ; preds = %entry
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !45
  br label %_ZN1C5m_fn2Ev.exit, !dbg !45

_ZN1C5m_fn2Ev.exit:                               ; preds = %entry, %if.then.i
  tail call void @_Z3fn3v() #3, !dbg !47
  ret void, !dbg !48
}

declare void @_Z3fn8v() #1

; Function Attrs: nounwind
define linkonce_odr void @_ZN1C5m_fn2Ev(%struct.C* nocapture readonly %this) #0 align 2 !dbg !22 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.C* %this, i64 0, metadata !24, metadata !DIExpression()), !dbg !49
  tail call void @_Z3fn8v() #3, !dbg !50
  %b = getelementptr inbounds %struct.C, %struct.C* %this, i64 0, i32 0, !dbg !51
  %0 = load i32, i32* %b, align 4, !dbg !51, !tbaa !42
  %tobool = icmp eq i32 %0, 0, !dbg !51
  br i1 %tobool, label %if.end, label %if.then, !dbg !51

if.then:                                          ; preds = %entry
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !52
  br label %if.end, !dbg !52

if.end:                                           ; preds = %entry, %if.then
  tail call void @_Z3fn8v() #3, !dbg !53
  %1 = load %struct.C*, %struct.C** @x, align 8, !dbg !56, !tbaa !33
  tail call void @llvm.dbg.value(metadata %struct.C* %1, i64 0, metadata !57, metadata !DIExpression()) #3, !dbg !58
  tail call void @_Z3fn8v() #3, !dbg !59
  %b.i.i = getelementptr inbounds %struct.C, %struct.C* %1, i64 0, i32 0, !dbg !60
  %2 = load i32, i32* %b.i.i, align 4, !dbg !60, !tbaa !42
  %tobool.i.i = icmp eq i32 %2, 0, !dbg !60
  br i1 %tobool.i.i, label %_Z3fn6v.exit, label %if.then.i.i, !dbg !60

if.then.i.i:                                      ; preds = %if.end
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !61
  br label %_Z3fn6v.exit, !dbg !61

_Z3fn6v.exit:                                     ; preds = %if.end, %if.then.i.i
  tail call void @_Z3fn3v() #3, !dbg !62
  ret void, !dbg !63
}

; Function Attrs: nounwind
define void @_Z3fn3v() #0 !dbg !18 {
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %tailrecurse.backedge, %entry
  tail call void @_Z3fn8v() #3, !dbg !64
  %0 = load %struct.C*, %struct.C** @x, align 8, !dbg !66, !tbaa !33
  tail call void @llvm.dbg.value(metadata %struct.C* %0, i64 0, metadata !67, metadata !DIExpression()) #3, !dbg !68
  tail call void @_Z3fn8v() #3, !dbg !69
  %b.i.i = getelementptr inbounds %struct.C, %struct.C* %0, i64 0, i32 0, !dbg !70
  %1 = load i32, i32* %b.i.i, align 4, !dbg !70, !tbaa !42
  %tobool.i.i = icmp eq i32 %1, 0, !dbg !70
  br i1 %tobool.i.i, label %tailrecurse.backedge, label %if.then.i.i, !dbg !70

tailrecurse.backedge:                             ; preds = %tailrecurse, %if.then.i.i
  br label %tailrecurse

if.then.i.i:                                      ; preds = %tailrecurse
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !71
  br label %tailrecurse.backedge, !dbg !71
}

; Function Attrs: nounwind
define void @_Z3fn4v() #0 !dbg !19 {
entry:
  %0 = load %struct.C*, %struct.C** @x, align 8, !dbg !72, !tbaa !33
  tail call void @_ZN1C5m_fn2Ev(%struct.C* %0), !dbg !72
  ret void, !dbg !72
}

; Function Attrs: nounwind
define void @_Z3fn5v() #0 !dbg !20 {
entry:
  %0 = load %struct.C*, %struct.C** @x, align 8, !dbg !73, !tbaa !33
  tail call void @_ZN1C5m_fn2Ev(%struct.C* %0), !dbg !73
  ret void, !dbg !73
}

declare void @_Z3fn2iiii(i32, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28, !29}
!llvm.ident = !{!30}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 ", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, subprograms: !13, globals: !26, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", line: 5, size: 32, align: 32, file: !5, elements: !6, identifier: "_ZTS1C")
!5 = !DIFile(filename: "recursive_inlining.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce")
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 6, size: 32, align: 32, file: !5, scope: !"_ZTS1C", baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DISubprogram(name: "m_fn2", linkageName: "_ZN1C5m_fn2Ev", line: 7, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 7, file: !5, scope: !"_ZTS1C", type: !10)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !"_ZTS1C")
!13 = !{!14, !18, !19, !20, !21, !22}
!14 = distinct !DISubprogram(name: "fn6", linkageName: "_Z3fn6v", line: 15, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 15, file: !5, scope: !15, type: !16, variables: !2)
!15 = !DIFile(filename: "recursive_inlining.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce")
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = distinct !DISubprogram(name: "fn3", linkageName: "_Z3fn3v", line: 20, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 20, file: !5, scope: !15, type: !16, variables: !2)
!19 = distinct !DISubprogram(name: "fn4", linkageName: "_Z3fn4v", line: 21, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 21, file: !5, scope: !15, type: !16, variables: !2)
!20 = distinct !DISubprogram(name: "fn5", linkageName: "_Z3fn5v", line: 22, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 22, file: !5, scope: !15, type: !16, variables: !2)
!21 = distinct !DISubprogram(name: "fn7", linkageName: "_Z3fn7v", line: 14, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 14, file: !5, scope: !15, type: !16, variables: !2)
!22 = distinct !DISubprogram(name: "m_fn2", linkageName: "_ZN1C5m_fn2Ev", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 7, file: !5, scope: !"_ZTS1C", type: !10, declaration: !9, variables: !23)
!23 = !{!24}
!24 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !22, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !"_ZTS1C")
!26 = !{!27}
!27 = !DIGlobalVariable(name: "x", line: 13, isLocal: false, isDefinition: true, scope: null, file: !15, type: !25, variable: %struct.C** @x)
!28 = !{i32 2, !"Dwarf Version", i32 4}
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{!"clang version 3.6.0 "}
!31 = !DILocation(line: 16, scope: !14)
!32 = !DILocation(line: 17, scope: !14)
!33 = !{!34, !34, i64 0}
!34 = !{!"any pointer", !35, i64 0}
!35 = !{!"omnipotent char", !36, i64 0}
!36 = !{!"Simple C/C++ TBAA"}
!37 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !22, type: !25)
!38 = !DILocation(line: 0, scope: !22, inlinedAt: !32)
!39 = !DILocation(line: 8, scope: !22, inlinedAt: !32)
!40 = !DILocation(line: 9, scope: !41, inlinedAt: !32)
!41 = distinct !DILexicalBlock(line: 9, column: 0, file: !5, scope: !22)
!42 = !{!43, !44, i64 0}
!43 = !{!"_ZTS1C", !44, i64 0}
!44 = !{!"int", !35, i64 0}
!45 = !DILocation(line: 9, scope: !46, inlinedAt: !32)
!46 = distinct !DILexicalBlock(line: 9, column: 0, file: !5, scope: !41)
!47 = !DILocation(line: 10, scope: !22, inlinedAt: !32)
!48 = !DILocation(line: 19, scope: !14)
!49 = !DILocation(line: 0, scope: !22)
!50 = !DILocation(line: 8, scope: !22)
!51 = !DILocation(line: 9, scope: !41)
!52 = !DILocation(line: 9, scope: !46)
!53 = !DILocation(line: 16, scope: !14, inlinedAt: !54)
!54 = !DILocation(line: 20, scope: !18, inlinedAt: !55)
!55 = !DILocation(line: 10, scope: !22)
!56 = !DILocation(line: 17, scope: !14, inlinedAt: !54)
!57 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !22, type: !25)
!58 = !DILocation(line: 0, scope: !22, inlinedAt: !56)
!59 = !DILocation(line: 8, scope: !22, inlinedAt: !56)
!60 = !DILocation(line: 9, scope: !41, inlinedAt: !56)
!61 = !DILocation(line: 9, scope: !46, inlinedAt: !56)
!62 = !DILocation(line: 10, scope: !22, inlinedAt: !56)
!63 = !DILocation(line: 11, scope: !22)
!64 = !DILocation(line: 16, scope: !14, inlinedAt: !65)
!65 = !DILocation(line: 20, scope: !18)
!66 = !DILocation(line: 17, scope: !14, inlinedAt: !65)
!67 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !22, type: !25)
!68 = !DILocation(line: 0, scope: !22, inlinedAt: !66)
!69 = !DILocation(line: 8, scope: !22, inlinedAt: !66)
!70 = !DILocation(line: 9, scope: !41, inlinedAt: !66)
!71 = !DILocation(line: 9, scope: !46, inlinedAt: !66)
!72 = !DILocation(line: 21, scope: !19)
!73 = !DILocation(line: 22, scope: !20)
