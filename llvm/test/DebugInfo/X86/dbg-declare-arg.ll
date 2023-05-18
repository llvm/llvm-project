; RUN: llc -O0 -fast-isel=true  -filetype=obj -o - %s | llvm-dwarfdump -v - | FileCheck %s
; RUN: llc --try-experimental-debuginfo-iterators -O0 -fast-isel=true  -filetype=obj -o - %s | llvm-dwarfdump -v - | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"
; rdar://problem/9321650

; C++ source:
; class A { public: int x; int y; int z; int o; ~A() { x = 1; }};
;
; A foo(int i) {
;   int j = 0;
;   if (i == 42) {
;     j = i + 1;
;   };
;   A my_a;
;   my_a.x = j;
;   return my_a;
; }

; CHECK: DW_AT_name {{.*}}"j"
; CHECK: DW_TAG_variable
; CHECK-NEXT:   DW_AT_location [DW_FORM_sec_offset] (
; CHECK-NEXT:     [0x{{.*}}, 0x{{.*}}): DW_OP_breg7 RSP+8, DW_OP_deref)
; CHECK-NEXT:   DW_AT_name {{.*}}"my_a"

%class.A = type { i32, i32, i32, i32 }

define void @_Z3fooi(ptr sret(%class.A) %agg.result, i32 %i) ssp !dbg !19 {
entry:
  %i.addr = alloca i32, align 4
  %j = alloca i32, align 4
  %nrvo = alloca i1
  %cleanup.dest.slot = alloca i32
  store i32 %i, ptr %i.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %i.addr, metadata !26, metadata !DIExpression()), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %j, metadata !28, metadata !DIExpression()), !dbg !30
  store i32 0, ptr %j, align 4, !dbg !31
  %tmp = load i32, ptr %i.addr, align 4, !dbg !32
  %cmp = icmp eq i32 %tmp, 42, !dbg !32
  br i1 %cmp, label %if.then, label %if.end, !dbg !32

if.then:                                          ; preds = %entry
  %tmp1 = load i32, ptr %i.addr, align 4, !dbg !33
  %add = add nsw i32 %tmp1, 1, !dbg !33
  store i32 %add, ptr %j, align 4, !dbg !33
  br label %if.end, !dbg !35

if.end:                                           ; preds = %if.then, %entry
  store i1 false, ptr %nrvo, !dbg !36
  call void @llvm.dbg.declare(metadata ptr %agg.result, metadata !37, metadata !DIExpression()), !dbg !39
  %tmp2 = load i32, ptr %j, align 4, !dbg !40
  store i32 %tmp2, ptr %agg.result, align 4, !dbg !40
  store i1 true, ptr %nrvo, !dbg !41
  store i32 1, ptr %cleanup.dest.slot
  %nrvo.val = load i1, ptr %nrvo, !dbg !42
  br i1 %nrvo.val, label %nrvo.skipdtor, label %nrvo.unused, !dbg !42

nrvo.unused:                                      ; preds = %if.end
  call void @_ZN1AD1Ev(ptr %agg.result), !dbg !42
  br label %nrvo.skipdtor, !dbg !42

nrvo.skipdtor:                                    ; preds = %nrvo.unused, %if.end
  ret void, !dbg !42
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN1AD1Ev(ptr %this) unnamed_addr ssp align 2 !dbg !22 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !43, metadata !DIExpression()), !dbg !44
  %this1 = load ptr, ptr %this.addr
  call void @_ZN1AD2Ev(ptr %this1), !dbg !53
  ret void, !dbg !45
}

define linkonce_odr void @_ZN1AD2Ev(ptr %this) unnamed_addr nounwind ssp align 2 !dbg !25 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !46, metadata !DIExpression()), !dbg !47
  %this1 = load ptr, ptr %this.addr
  store i32 1, ptr %this1, align 4, !dbg !48
  ret void, !dbg !48
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!52}

!0 = !DISubprogram(name: "~A", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, file: !51, scope: !1, type: !11)
!1 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 2, size: 128, align: 32, file: !51, scope: !2, elements: !4)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.0 (trunk 130127)", isOptimized: false, emissionKind: FullDebug, file: !51, enums: !{}, retainedTypes: !{})
!3 = !DIFile(filename: "a.cc", directory: "/private/tmp")
!4 = !{!5, !7, !8, !9, !0, !10, !14}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 2, size: 32, align: 32, file: !51, scope: !3, baseType: !6)
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "y", line: 2, size: 32, align: 32, offset: 32, file: !51, scope: !3, baseType: !6)
!8 = !DIDerivedType(tag: DW_TAG_member, name: "z", line: 2, size: 32, align: 32, offset: 64, file: !51, scope: !3, baseType: !6)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "o", line: 2, size: 32, align: 32, offset: 96, file: !51, scope: !3, baseType: !6)
!10 = !DISubprogram(name: "A", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, file: !51, scope: !1, type: !11)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, file: !3, baseType: !1)
!14 = !DISubprogram(name: "A", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, file: !51, scope: !1, type: !15)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !13, !17}
!17 = !DIDerivedType(tag: DW_TAG_reference_type, scope: !2, baseType: !18)
!18 = !DIDerivedType(tag: DW_TAG_const_type, file: !3, baseType: !1)
!19 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !51, scope: !3, type: !20)
!20 = !DISubroutineType(types: !21)
!21 = !{!1}
!22 = distinct !DISubprogram(name: "~A", linkageName: "_ZN1AD1Ev", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !51, scope: !3, type: !23)
!23 = !DISubroutineType(types: !24)
!24 = !{null}
!25 = distinct !DISubprogram(name: "~A", linkageName: "_ZN1AD2Ev", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !51, scope: !3, type: !23)
!26 = !DILocalVariable(name: "i", line: 4, arg: 1, scope: !19, file: !3, type: !6)
!27 = !DILocation(line: 4, column: 11, scope: !19)
!28 = !DILocalVariable(name: "j", line: 5, scope: !29, file: !3, type: !6)
!29 = distinct !DILexicalBlock(line: 4, column: 14, file: !51, scope: !19)
!30 = !DILocation(line: 5, column: 7, scope: !29)
!31 = !DILocation(line: 5, column: 12, scope: !29)
!32 = !DILocation(line: 6, column: 3, scope: !29)
!33 = !DILocation(line: 7, column: 5, scope: !34)
!34 = distinct !DILexicalBlock(line: 6, column: 16, file: !51, scope: !29)
!35 = !DILocation(line: 8, column: 3, scope: !34)
!36 = !DILocation(line: 9, column: 9, scope: !29)
!37 = !DILocalVariable(name: "my_a", line: 9, scope: !29, file: !3, type: !38)
!38 = !DIDerivedType(tag: DW_TAG_reference_type, file: !3, baseType: !1)
!39 = !DILocation(line: 9, column: 5, scope: !29)
!40 = !DILocation(line: 10, column: 3, scope: !29)
!41 = !DILocation(line: 11, column: 3, scope: !29)
!42 = !DILocation(line: 12, column: 1, scope: !29)
!43 = !DILocalVariable(name: "this", line: 2, arg: 1, flags: DIFlagArtificial, scope: !22, file: !3, type: !13)
!44 = !DILocation(line: 2, column: 47, scope: !22)
!45 = !DILocation(line: 2, column: 61, scope: !22)
!46 = !DILocalVariable(name: "this", line: 2, arg: 1, flags: DIFlagArtificial, scope: !25, file: !3, type: !13)
!47 = !DILocation(line: 2, column: 47, scope: !25)
!48 = !DILocation(line: 2, column: 54, scope: !49)
!49 = distinct !DILexicalBlock(line: 2, column: 52, file: !51, scope: !25)
!51 = !DIFile(filename: "a.cc", directory: "/private/tmp")
!52 = !{i32 1, !"Debug Info Version", i32 3}
!53 = !DILocation(line: 0, scope: !22)
