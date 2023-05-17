; RUN: opt -passes=mem2reg -S %s -o - | FileCheck %s --implicit-check-not="call void @llvm.dbg"

; CHECK: llvm.dbg.value(metadata i64 0, metadata ![[#]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32))

;; The store has a debug intrinsic attached to it with a fragment size
;; different to the base alloca debug intrinsic fragment size. Check that
;; mem2reg doesn't think this store is "untagged" for that base variable.  If
;; that were the case mem2reg would insert a dbg.value covering the entire
;; variable, which isn't the right thing to do here. This example looks weird
;; and not particularly compelling, but this was encountered in the wild on
;; "real code".

;; Reduced from this C++ (which itself has been reduced).
;; class a {
;; public:
;;   a(float, float);
;; };
;; class d {
;; protected:
;;   float b[4];
;; 
;; public:
;;   float e() { return b[0]; }
;;   float f() { return b[1]; }
;; };
;; class g : public d {
;; public:
;;   void operator*=(g) {
;;     {
;;       float __attribute__((nodebug)) c = b[2], __attribute__((nodebug)) h = b[0];
;;       b[0] = c;
;;       b[1] = h;
;;     }
;;   }
;; };
;; g get();
;; void i() {
;;   g __attribute__((nodebug)) j = get();
;;   g k = j;
;;   k *= j;
;;   a(k.e(), k.f());
;; }

define dso_local i64 @_Z3funv() #0 !dbg !10 {
entry:
  %retval.sroa.0 = alloca i64, align 8, !DIAssignID !20
  call void @llvm.dbg.assign(metadata i1 undef, metadata !19, metadata !DIExpression(), metadata !20, metadata ptr %retval.sroa.0, metadata !DIExpression()), !dbg !21
  store i64 0, ptr %retval.sroa.0, align 8, !dbg !22, !DIAssignID !23
  call void @llvm.dbg.assign(metadata i64 0, metadata !19, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !23, metadata ptr %retval.sroa.0, metadata !DIExpression()), !dbg !21
  ret i64 0
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 17.0.0"}
!10 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Pair", file: !1, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !14, identifier: "_ZTS4Pair")
!14 = !{!15, !17}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "A", scope: !13, file: !1, line: 1, baseType: !16, size: 32)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "B", scope: !13, file: !1, line: 1, baseType: !16, size: 32, offset: 32)
!18 = !{!19}
!19 = !DILocalVariable(name: "X", scope: !10, file: !1, line: 3, type: !13)
!20 = distinct !DIAssignID()
!21 = !DILocation(line: 0, scope: !10)
!22 = !DILocation(line: 3, column: 8, scope: !10)
!23 = distinct !DIAssignID()
!24 = !DILocation(line: 4, column: 3, scope: !10)
