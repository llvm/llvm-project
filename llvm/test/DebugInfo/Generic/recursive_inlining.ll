; REQUIRES: object-emission

; RUN: %llc_dwarf -filetype=obj -O0 < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s

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

source_filename = "test/DebugInfo/Generic/recursive_inlining.ll"

%struct.C = type { i32 }

@x = global %struct.C* null, align 8, !dbg !0

; Function Attrs: nounwind
define void @_Z3fn6v() #0 !dbg !20 {
entry:
  tail call void @_Z3fn8v() #3, !dbg !23
  %0 = load %struct.C*, %struct.C** @x, align 8, !dbg !24, !tbaa !25
  tail call void @llvm.dbg.value(metadata %struct.C* %0, metadata !29, metadata !32) #3, !dbg !33
  tail call void @_Z3fn8v() #3, !dbg !34
  %b.i = getelementptr inbounds %struct.C, %struct.C* %0, i64 0, i32 0, !dbg !35
  %1 = load i32, i32* %b.i, align 4, !dbg !35, !tbaa !37
  %tobool.i = icmp eq i32 %1, 0, !dbg !35
  br i1 %tobool.i, label %_ZN1C5m_fn2Ev.exit, label %if.then.i, !dbg !35

if.then.i:                                        ; preds = %entry
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !40
  br label %_ZN1C5m_fn2Ev.exit, !dbg !40

_ZN1C5m_fn2Ev.exit:                               ; preds = %if.then.i, %entry
  tail call void @_Z3fn3v() #3, !dbg !42
  ret void, !dbg !43
}

declare void @_Z3fn8v() #1

; Function Attrs: nounwind

define linkonce_odr void @_ZN1C5m_fn2Ev(%struct.C* nocapture readonly %this) #0 align 2 !dbg !30 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.C* %this, metadata !29, metadata !32), !dbg !44
  tail call void @_Z3fn8v() #3, !dbg !45
  %b = getelementptr inbounds %struct.C, %struct.C* %this, i64 0, i32 0, !dbg !46
  %0 = load i32, i32* %b, align 4, !dbg !46, !tbaa !37
  %tobool = icmp eq i32 %0, 0, !dbg !46
  br i1 %tobool, label %if.end, label %if.then, !dbg !46

if.then:                                          ; preds = %entry
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !47
  br label %if.end, !dbg !47

if.end:                                           ; preds = %if.then, %entry
  tail call void @_Z3fn8v() #3, !dbg !48
  %1 = load %struct.C*, %struct.C** @x, align 8, !dbg !52, !tbaa !25
  tail call void @llvm.dbg.value(metadata %struct.C* %1, metadata !29, metadata !32) #3, !dbg !53
  tail call void @_Z3fn8v() #3, !dbg !54
  %b.i.i = getelementptr inbounds %struct.C, %struct.C* %1, i64 0, i32 0, !dbg !55
  %2 = load i32, i32* %b.i.i, align 4, !dbg !55, !tbaa !37
  %tobool.i.i = icmp eq i32 %2, 0, !dbg !55
  br i1 %tobool.i.i, label %_Z3fn6v.exit, label %if.then.i.i, !dbg !55

if.then.i.i:                                      ; preds = %if.end

  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !56
  br label %_Z3fn6v.exit, !dbg !56

_Z3fn6v.exit:                                     ; preds = %if.then.i.i, %if.end
  tail call void @_Z3fn3v() #3, !dbg !57
  ret void, !dbg !58
}

; Function Attrs: nounwind
define void @_Z3fn3v() #0 !dbg !50 {
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %tailrecurse.backedge, %entry
  tail call void @_Z3fn8v() #3, !dbg !59
  %0 = load %struct.C*, %struct.C** @x, align 8, !dbg !61, !tbaa !25
  tail call void @llvm.dbg.value(metadata %struct.C* %0, metadata !29, metadata !32) #3, !dbg !62
  tail call void @_Z3fn8v() #3, !dbg !63
  %b.i.i = getelementptr inbounds %struct.C, %struct.C* %0, i64 0, i32 0, !dbg !64
  %1 = load i32, i32* %b.i.i, align 4, !dbg !64, !tbaa !37
  %tobool.i.i = icmp eq i32 %1, 0, !dbg !64
  br i1 %tobool.i.i, label %tailrecurse.backedge, label %if.then.i.i, !dbg !64

tailrecurse.backedge:                             ; preds = %if.then.i.i, %tailrecurse
  br label %tailrecurse

if.then.i.i:                                      ; preds = %tailrecurse
  tail call void @_Z3fn2iiii(i32 0, i32 0, i32 0, i32 0) #3, !dbg !65
  br label %tailrecurse.backedge, !dbg !65
}

; Function Attrs: nounwind
define void @_Z3fn4v() #0 !dbg !66 {
entry:
  %0 = load %struct.C*, %struct.C** @x, align 8, !dbg !67, !tbaa !25
  tail call void @_ZN1C5m_fn2Ev(%struct.C* %0), !dbg !67
  ret void, !dbg !67
}

; Function Attrs: nounwind
define void @_Z3fn5v() #0 !dbg !68 {
entry:
  %0 = load %struct.C*, %struct.C** @x, align 8, !dbg !69, !tbaa !25
  tail call void @_ZN1C5m_fn2Ev(%struct.C* %0), !dbg !69
  ret void, !dbg !69
}

declare void @_Z3fn2iiii(i32, i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!12}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: null, file: !2, line: 13, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "recursive_inlining.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce")
!3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, align: 64)
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !2, line: 5, size: 32, align: 32, elements: !5, identifier: "_ZTS1C")
!5 = !{!6, !8}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !4, file: !2, line: 6, baseType: !7, size: 32, align: 32)
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DISubprogram(name: "m_fn2", linkageName: "_ZN1C5m_fn2Ev", scope: !4, file: !2, line: 7, type: !9, isLocal: false, isDefinition: false, scopeLine: 7, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!12 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !13, producer: "clang version 3.6.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !14, retainedTypes: !15, globals: !16, imports: !14)
!13 = !DIFile(filename: "<stdin>", directory: "/usr/local/google/home/blaikie/dev/scratch/missing_concrete_variable_on_darwin/reduce")
!14 = !{}
!15 = !{!4}
!16 = !{!0}
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{!"clang version 3.6.0 "}
!20 = distinct !DISubprogram(name: "fn6", linkageName: "_Z3fn6v", scope: !2, file: !2, line: 15, type: !21, isLocal: false, isDefinition: true, scopeLine: 15, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !12, retainedNodes: !14)
!21 = !DISubroutineType(types: !22)
!22 = !{null}
!23 = !DILocation(line: 16, scope: !20)
!24 = !DILocation(line: 17, scope: !20)
!25 = !{!26, !26, i64 0}
!26 = !{!"any pointer", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !DILocalVariable(name: "this", arg: 1, scope: !30, type: !3, flags: DIFlagArtificial | DIFlagObjectPointer)
!30 = distinct !DISubprogram(name: "m_fn2", linkageName: "_ZN1C5m_fn2Ev", scope: !4, file: !2, line: 7, type: !9, isLocal: false, isDefinition: true, scopeLine: 7, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !12, declaration: !8, retainedNodes: !31)
!31 = !{!29}
!32 = !DIExpression()
!33 = !DILocation(line: 0, scope: !30, inlinedAt: !24)
!34 = !DILocation(line: 8, scope: !30, inlinedAt: !24)
!35 = !DILocation(line: 9, scope: !36, inlinedAt: !24)
!36 = distinct !DILexicalBlock(scope: !30, file: !2, line: 9)
!37 = !{!38, !39, i64 0}
!38 = !{!"struct", !39, i64 0}
!39 = !{!"int", !27, i64 0}
!40 = !DILocation(line: 9, scope: !41, inlinedAt: !24)
!41 = distinct !DILexicalBlock(scope: !36, file: !2, line: 9)
!42 = !DILocation(line: 10, scope: !30, inlinedAt: !24)
!43 = !DILocation(line: 19, scope: !20)
!44 = !DILocation(line: 0, scope: !30)
!45 = !DILocation(line: 8, scope: !30)
!46 = !DILocation(line: 9, scope: !36)
!47 = !DILocation(line: 9, scope: !41)
!48 = !DILocation(line: 16, scope: !20, inlinedAt: !49)
!49 = !DILocation(line: 20, scope: !50, inlinedAt: !51)
!50 = distinct !DISubprogram(name: "fn3", linkageName: "_Z3fn3v", scope: !2, file: !2, line: 20, type: !21, isLocal: false, isDefinition: true, scopeLine: 20, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !12, retainedNodes: !14)
!51 = !DILocation(line: 10, scope: !30)
!52 = !DILocation(line: 17, scope: !20, inlinedAt: !49)
!53 = !DILocation(line: 0, scope: !30, inlinedAt: !52)
!54 = !DILocation(line: 8, scope: !30, inlinedAt: !52)
!55 = !DILocation(line: 9, scope: !36, inlinedAt: !52)
!56 = !DILocation(line: 9, scope: !41, inlinedAt: !52)
!57 = !DILocation(line: 10, scope: !30, inlinedAt: !52)
!58 = !DILocation(line: 11, scope: !30)
!59 = !DILocation(line: 16, scope: !20, inlinedAt: !60)
!60 = !DILocation(line: 20, scope: !50)
!61 = !DILocation(line: 17, scope: !20, inlinedAt: !60)
!62 = !DILocation(line: 0, scope: !30, inlinedAt: !61)
!63 = !DILocation(line: 8, scope: !30, inlinedAt: !61)
!64 = !DILocation(line: 9, scope: !36, inlinedAt: !61)
!65 = !DILocation(line: 9, scope: !41, inlinedAt: !61)
!66 = distinct !DISubprogram(name: "fn4", linkageName: "_Z3fn4v", scope: !2, file: !2, line: 21, type: !21, isLocal: false, isDefinition: true, scopeLine: 21, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !12, retainedNodes: !14)
!67 = !DILocation(line: 21, scope: !66)
!68 = distinct !DISubprogram(name: "fn5", linkageName: "_Z3fn5v", scope: !2, file: !2, line: 22, type: !21, isLocal: false, isDefinition: true, scopeLine: 22, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !12, retainedNodes: !14)
!69 = !DILocation(line: 22, scope: !68)

