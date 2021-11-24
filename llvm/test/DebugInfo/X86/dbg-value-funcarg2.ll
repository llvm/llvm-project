; RUN: llc -mtriple=x86_64-unknown-linux-gnu -start-after=codegenprepare -stop-before=finalize-isel -o - %s -experimental-debug-variable-locations=false | FileCheck %s --check-prefixes=CHECK,COMMON
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -start-after=codegenprepare -stop-before=finalize-isel -o - %s -experimental-debug-variable-locations=true | FileCheck %s --check-prefixes=INSTRREF,COMMON

; Test case was generated from the following C code,
; using: clang -g -O1 -S -emit-llvm s.c -o s.ll
;
; struct s { long long int i, j; };
;
; extern void bar(struct s, struct s, struct s);
;
; int f(struct s s1, struct s s2) {
;   volatile struct s tmp = {0};
;   bar(s1, s2, tmp);
;   s1.i = s2.i;
;   s1.j = s2.j;
;   return s1.j + s1.j;
; }

; Catch metadata references for involved variables.
;
; COMMON-DAG: ![[S1:.*]] = !DILocalVariable(name: "s1"
; COMMON-DAG: ![[S2:.*]] = !DILocalVariable(name: "s2"

define dso_local i32 @f(i64 %s1.coerce0, i64 %s1.coerce1, i64 %s2.coerce0, i64 %s2.coerce1) local_unnamed_addr #0 !dbg !7 {
; We expect DBG_VALUE instructions for the arguments at the entry.
; In instr-ref mode, there'll be some DBG_PHIs in there too.
; COMMON-LABEL: name:            f
; COMMON-NOT: DBG_VALUE
; INSTRREF-DAG: DBG_PHI $rcx, 2
; INSTRREF-DAG: DBG_PHI $rdx, 1
; COMMON-DAG: DBG_VALUE $rdi, $noreg, ![[S1]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
; COMMON-DAG: DBG_VALUE $rsi, $noreg, ![[S1]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)
; COMMON-DAG: DBG_VALUE $rdx, $noreg, ![[S2]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
; COMMON-DAG: DBG_VALUE $rcx, $noreg, ![[S2]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)
; COMMON-NOT: DBG_

; Then arguments are copied to virtual registers.
; CHECK-NOT: DBG_VALUE
; CHECK-DAG: %[[R1:.*]]:gr64 = COPY $rcx
; CHECK-DAG: DBG_VALUE %[[R1]], $noreg, ![[S2]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)
; CHECK-DAG: %[[R2:.*]]:gr64 = COPY $rdx
; CHECK-DAG: DBG_VALUE %[[R2]], $noreg, ![[S2]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
; CHECK-DAG: %[[R3:.*]]:gr64 = COPY $rsi
; CHECK-DAG: DBG_VALUE %[[R3]], $noreg, ![[S1]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)
; CHECK-DAG: %[[R4:.*]]:gr64 = COPY $rdi
; CHECK-DAG: DBG_VALUE %[[R4]], $noreg, ![[S1]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
; CHECK-NOT: DBG_VALUE

; We have the call to bar.
; COMMON:     ADJCALLSTACKDOWN
; COMMON:     CALL64pcrel32 @bar

; After the call we expect to find new DBG_VALUE instructions for "s1".
; CHECK:     ADJCALLSTACKUP
; CHECK-NOT: DBG_VALUE
; CHECK-DAG: DBG_VALUE %[[R2]], $noreg, ![[S1]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
; CHECK-DAG: DBG_VALUE %[[R1]], $noreg, ![[S1]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)

;; In instruction referencing mode, we should refer to the instruction number
;; of the earlier DBG_PHIs.
; INSTRREF:     ADJCALLSTACKUP
; INSTRREF-NOT: DBG_
; INSTRREF-DAG: DBG_INSTR_REF 1, 0, ![[S1]], !DIExpression(DW_OP_LLVM_fragment, 0, 64)
; INSTRREF-DAG: DBG_INSTR_REF 2, 0, ![[S1]], !DIExpression(DW_OP_LLVM_fragment, 64, 64)

; And then no more DBG_ instructions before the add.
; COMMON-NOT: DBG_
; COMMON:     ADD32rr

entry:
  %tmp.sroa.0 = alloca i64, align 8
  %tmp.sroa.4 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %tmp.sroa.0, metadata !19, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !21
  call void @llvm.dbg.declare(metadata i64* %tmp.sroa.4, metadata !19, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !21
  call void @llvm.dbg.value(metadata i64 %s1.coerce0, metadata !17, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !22
  call void @llvm.dbg.value(metadata i64 %s1.coerce1, metadata !17, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !22
  call void @llvm.dbg.value(metadata i64 %s2.coerce0, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !23
  call void @llvm.dbg.value(metadata i64 %s2.coerce1, metadata !18, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !23
  %tmp.sroa.0.0..sroa_cast = bitcast i64* %tmp.sroa.0 to i8*, !dbg !24
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %tmp.sroa.0.0..sroa_cast), !dbg !24
  %tmp.sroa.4.0..sroa_cast = bitcast i64* %tmp.sroa.4 to i8*, !dbg !24
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %tmp.sroa.4.0..sroa_cast), !dbg !24
  store volatile i64 0, i64* %tmp.sroa.0, align 8, !dbg !21
  store volatile i64 0, i64* %tmp.sroa.4, align 8, !dbg !21
  tail call void @bar(i64 %s1.coerce0, i64 %s1.coerce1, i64 %s2.coerce0, i64 %s2.coerce1, i64 0, i64 0) #4, !dbg !25
  call void @llvm.dbg.value(metadata i64 %s2.coerce0, metadata !17, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg !22
  call void @llvm.dbg.value(metadata i64 %s2.coerce1, metadata !17, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !22
  %s2.coerce1.tr = trunc i64 %s2.coerce1 to i32, !dbg !26
  %conv = shl i32 %s2.coerce1.tr, 1, !dbg !26
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %tmp.sroa.0.0..sroa_cast), !dbg !27
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %tmp.sroa.4.0..sroa_cast), !dbg !27
  ret i32 %conv, !dbg !28
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #1
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

declare dso_local void @bar(i64, i64, i64, i64, i64, i64) local_unnamed_addr

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "llvm-svn @ 353529", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "s.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"llvm-svn @ 353529"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 1, size: 128, elements: !12)
!12 = !{!13, !15}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !11, file: !1, line: 1, baseType: !14, size: 64)
!14 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !11, file: !1, line: 1, baseType: !14, size: 64, offset: 64)
!16 = !{!17, !18, !19}
!17 = !DILocalVariable(name: "s1", arg: 1, scope: !7, file: !1, line: 5, type: !11)
!18 = !DILocalVariable(name: "s2", arg: 2, scope: !7, file: !1, line: 5, type: !11)
!19 = !DILocalVariable(name: "tmp", scope: !7, file: !1, line: 6, type: !20)
!20 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !11)
!21 = !DILocation(line: 6, column: 21, scope: !7)
!22 = !DILocation(line: 5, column: 16, scope: !7)
!23 = !DILocation(line: 5, column: 29, scope: !7)
!24 = !DILocation(line: 6, column: 3, scope: !7)
!25 = !DILocation(line: 7, column: 3, scope: !7)
!26 = !DILocation(line: 10, column: 10, scope: !7)
!27 = !DILocation(line: 11, column: 1, scope: !7)
!28 = !DILocation(line: 10, column: 3, scope: !7)
