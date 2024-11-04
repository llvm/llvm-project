; RUN: opt -passes='declare-to-assign,verify' %s -S -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"
; RUN: opt --try-experimental-debuginfo-iterators -passes='declare-to-assign,verify' %s -S -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; This test checks that `trackAssignments` is working correctly by using the
;; pass-wrapper `declare-to-assign`. Each function checks some specific
;; functionality and has its checks inline.

;; $ cat test.cpp
;; struct Inner { int A, B; };
;; struct Outer { Inner A, B; };
;; struct Large { int A[10]; };
;; struct LCopyCtor { int A[4]; LCopyCtor(); LCopyCtor(LCopyCtor const &); };
;; int Value, Index, Cond;
;; Inner InnerA, InnerB;
;; Large L;
;;
;; void zeroInit() { int Z[3] = {0, 0, 0}; }
;;
;; void memcpyInit() { int A[4] = {0, 1, 2, 3}; }
;;
;; void setField() {
;;   Outer O;
;;   O.A.B = Value;
;; }
;;
;; void unknownOffset() {
;;   int A[2];
;;   A[Index] = Value;
;; }
;;
;; Inner sharedAlloca() {
;;   if (Cond) {
;;     Inner A = InnerA;
;;     return A;
;;   } else {
;;     Inner B = InnerB;
;;     return B;
;;   }
;; }
;;
;; Large sret() {
;;   Large X = L;
;;   return X;
;; }
;;
;; void byval(Large X) {}
;;
;; LCopyCtor indirectReturn() {
;;  LCopyCtor R;
;;  return R;
;; }
;;
;; $ clang++ -g test.cpp -o - -emit-llvm -S -Xclang -disable-llvm-passes -O2
;;
;; Then these functions are added manually to the IR using
;;     $ clang++ -g test.cpp -o - -emit-llvm -S -O0
;; and removing the optnone attributes:
;;
;; __attribute__((always_inline))
;; int sqr(int Y) { return Y * Y;  }
;; int fun(int X) { return sqr(X); }


source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Inner = type { i32, i32 }
%struct.Large = type { [10 x i32] }
%struct.Outer = type { %struct.Inner, %struct.Inner }
%struct.LCopyCtor = type { [4 x i32] }

@Value = dso_local global i32 0, align 4, !dbg !0
@Index = dso_local global i32 0, align 4, !dbg !5
@Cond = dso_local global i32 0, align 4, !dbg !8
@InnerA = dso_local global %struct.Inner zeroinitializer, align 4, !dbg !10
@InnerB = dso_local global %struct.Inner zeroinitializer, align 4, !dbg !16
@L = dso_local global %struct.Large zeroinitializer, align 4, !dbg !18
@__const._Z10memcpyInitv.A = private unnamed_addr constant [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16

;; Zero init with a memset.
;;
;;    void zeroInit() { int Z[3] = {0, 0, 0}; }
;;
;; Check that we get two dbg.assign intrinsics. The first linked to the alloca
;; and the second linked to the zero-init-memset, which should have a constant 0
;; for the value component.
define dso_local void @_Z8zeroInitv() #0 !dbg !31 {
; CHECK-LABEL: define dso_local void @_Z8zeroInitv
entry:
  %Z = alloca [3 x i32], align 4
; CHECK:      %Z = alloca [3 x i32], align 4, !DIAssignID ![[ID_0:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_0:[0-9]+]], metadata !DIExpression(), metadata ![[ID_0]], metadata ptr %Z, metadata !DIExpression())
  %0 = bitcast ptr %Z to ptr, !dbg !39
  call void @llvm.lifetime.start.p0i8(i64 12, ptr %0) #5, !dbg !39
  call void @llvm.dbg.declare(metadata ptr %Z, metadata !35, metadata !DIExpression()), !dbg !40
  %1 = bitcast ptr %Z to ptr, !dbg !40
  call void @llvm.memset.p0i8.i64(ptr align 4 %1, i8 0, i64 12, i1 false), !dbg !40
; CHECK:       @llvm.memset{{.*}}, !DIAssignID ![[ID_1:[0-9]+]]
; CHECK-NEXT:  call void @llvm.dbg.assign(metadata i8 0, metadata ![[VAR_0]], metadata !DIExpression(), metadata ![[ID_1]], metadata ptr %1, metadata !DIExpression())
  %2 = bitcast ptr %Z to ptr, !dbg !41
  call void @llvm.lifetime.end.p0i8(i64 12, ptr %2) #5, !dbg !41
  ret void, !dbg !41
}

;; Init with a memcpy (from a global).
;;
;;    void memcpyInit() { int A[4] = {0, 1, 2, 3}; }
;;
;; Check that we get two dbg.assign intrinsics. The first linked to the alloca
;; and the second linked to the initialising memcpy, which should have an Undef
;; value component.
define dso_local void @_Z10memcpyInitv() #0 !dbg !42 {
; CHECK-LABEL: define dso_local void @_Z10memcpyInitv
entry:
  %A = alloca [4 x i32], align 16
; CHECK:      %A = alloca [4 x i32], align 16, !DIAssignID ![[ID_2:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_1:[0-9]+]], metadata !DIExpression(), metadata ![[ID_2]], metadata ptr %A, metadata !DIExpression())
  %0 = bitcast ptr %A to ptr, !dbg !48
  call void @llvm.lifetime.start.p0i8(i64 16, ptr %0) #5, !dbg !48
  call void @llvm.dbg.declare(metadata ptr %A, metadata !44, metadata !DIExpression()), !dbg !49
  %1 = bitcast ptr %A to ptr, !dbg !49
  call void @llvm.memcpy.p0i8.p0i8.i64(ptr align 16 %1, ptr align 16 bitcast (ptr @__const._Z10memcpyInitv.A to ptr), i64 16, i1 false), !dbg !49
; CHECK:       @llvm.memcpy{{.*}}, !DIAssignID ![[ID_3:[0-9]+]]
; CHECK-NEXT:  call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_1]], metadata !DIExpression(), metadata ![[ID_3]], metadata ptr %1, metadata !DIExpression())
  %2 = bitcast ptr %A to ptr, !dbg !50
  call void @llvm.lifetime.end.p0i8(i64 16, ptr %2) #5, !dbg !50
  ret void, !dbg !50
}

;; Assign to field of local variable.
;;
;;    void setField() {
;;      Outer O;
;;      O.A.B = Value;
;;    }
;;
;; Check that we get two dbg.assign intrinsics. The first linked to the alloca
;; and the second linked to the store to O.A.B, which should include the
;; appropriate fragment info (SizeInBits: 32, OffsetInBits: 32) with respect to
;; the base variable.
define dso_local void @_Z8setFieldv() #0 !dbg !51 {
; CHECK-LABEL: define dso_local void @_Z8setFieldv
entry:
  %O = alloca %struct.Outer, align 4
; CHECK:      %O = alloca %struct.Outer, align 4, !DIAssignID ![[ID_4:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_2:[0-9]+]], metadata !DIExpression(), metadata ![[ID_4]], metadata ptr %O, metadata !DIExpression())
  %0 = bitcast ptr %O to ptr, !dbg !58
  call void @llvm.lifetime.start.p0i8(i64 16, ptr %0) #5, !dbg !58
  call void @llvm.dbg.declare(metadata ptr %O, metadata !53, metadata !DIExpression()), !dbg !59
  %1 = load i32, ptr @Value, align 4, !dbg !60, !tbaa !61
  %A = getelementptr inbounds %struct.Outer, ptr %O, i32 0, i32 0, !dbg !65
  %B = getelementptr inbounds %struct.Inner, ptr %A, i32 0, i32 1, !dbg !66
  store i32 %1, ptr %B, align 4, !dbg !67, !tbaa !68
; CHECK:      store i32 %1, ptr %B, align 4,{{.*}}!DIAssignID ![[ID_5:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %1, metadata ![[VAR_2]], metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata ![[ID_5]], metadata ptr %B, metadata !DIExpression())
  %2 = bitcast ptr %O to ptr, !dbg !71
  call void @llvm.lifetime.end.p0i8(i64 16, ptr %2) #5, !dbg !71
  ret void, !dbg !71
}

;; Assign to (statically) unknown offset into a local variable.
;;
;;   void unknownOffset() {
;;     int A[2];
;;     A[Index] = Value;
;;   }
;;
;; Check that we get only one dbg.assign intrinsic and that it is linked to the
;; alloca. The assignment doesn't get a dbg.assign intrinsic because we cannot
;; encode the fragment info for the assignment.
;; Note: This doesn't mean we lose the assignment entirely: the backend will
;; interpret the store (as it sees it) as an assignment.
define dso_local void @_Z13unknownOffsetv() #0 !dbg !72 {
; CHECK-LABEL: define dso_local void @_Z13unknownOffsetv
entry:
  %A = alloca [2 x i32], align 4
; CHECK:      %A = alloca [2 x i32], align 4, !DIAssignID ![[ID_6:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_3:[0-9]+]], metadata !DIExpression(), metadata ![[ID_6]], metadata ptr %A, metadata !DIExpression())
  %0 = bitcast ptr %A to ptr, !dbg !78
  call void @llvm.lifetime.start.p0i8(i64 8, ptr %0) #5, !dbg !78
  call void @llvm.dbg.declare(metadata ptr %A, metadata !74, metadata !DIExpression()), !dbg !79
  %1 = load i32, ptr @Value, align 4, !dbg !80, !tbaa !61
  %2 = load i32, ptr @Index, align 4, !dbg !81, !tbaa !61
  %idxprom = sext i32 %2 to i64, !dbg !82
  %arrayidx = getelementptr inbounds [2 x i32], ptr %A, i64 0, i64 %idxprom, !dbg !82
  store i32 %1, ptr %arrayidx, align 4, !dbg !83, !tbaa !61
  %3 = bitcast ptr %A to ptr, !dbg !84
  call void @llvm.lifetime.end.p0i8(i64 8, ptr %3) #5, !dbg !84
  ret void, !dbg !84
}

;; Assignments to variables which share the same backing storage.
;;
;; Inner sharedAlloca() {
;;   if (Cond) {
;;     Inner A = InnerA;
;;     return A;
;;   } else {
;;     Inner B = InnerB;
;;     return B;
;;   }
;; }
;;
define dso_local i64 @_Z12sharedAllocav() #0 !dbg !85 {
; CHECK-LABEL: define dso_local i64 @_Z12sharedAllocav
entry:
  %retval = alloca %struct.Inner, align 4
; CHECK:      %retval = alloca %struct.Inner, align 4, !DIAssignID ![[ID_7:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_4:[0-9]+]], metadata !DIExpression(), metadata ![[ID_7]], metadata ptr %retval, metadata !DIExpression())
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_5:[0-9]+]], metadata !DIExpression(), metadata ![[ID_7]], metadata ptr %retval, metadata !DIExpression())
  %0 = load i32, ptr @Cond, align 4, !dbg !94, !tbaa !61
  %tobool = icmp ne i32 %0, 0, !dbg !94
  br i1 %tobool, label %if.then, label %if.else, !dbg !95

if.then:                                          ; preds = %entry
; CHECK:      if.then:
  call void @llvm.dbg.declare(metadata ptr %retval, metadata !89, metadata !DIExpression()), !dbg !96
  %1 = bitcast ptr %retval to ptr, !dbg !97
  call void @llvm.memcpy.p0i8.p0i8.i64(ptr align 4 %1, ptr align 4 bitcast (ptr @InnerA to ptr), i64 8, i1 false), !dbg !97, !tbaa.struct !98
; CHECK:      call void @llvm.memcpy{{.*}}, !DIAssignID ![[ID_8:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_4]], metadata !DIExpression(), metadata ![[ID_8]], metadata ptr %1, metadata !DIExpression())
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_5]], metadata !DIExpression(), metadata ![[ID_8]], metadata ptr %1, metadata !DIExpression())
  br label %return, !dbg !99

if.else:                                          ; preds = %entry
; CHECK:      if.else:
  call void @llvm.dbg.declare(metadata ptr %retval, metadata !92, metadata !DIExpression()), !dbg !100
  %2 = bitcast ptr %retval to ptr, !dbg !101
  call void @llvm.memcpy.p0i8.p0i8.i64(ptr align 4 %2, ptr align 4 bitcast (ptr @InnerB to ptr), i64 8, i1 false), !dbg !101, !tbaa.struct !98
; CHECK:      call void @llvm.memcpy{{.*}}, !DIAssignID ![[ID_9:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_4]], metadata !DIExpression(), metadata ![[ID_9]], metadata ptr %2, metadata !DIExpression())
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_5]], metadata !DIExpression(), metadata ![[ID_9]], metadata ptr %2, metadata !DIExpression())
  br label %return, !dbg !102

return:                                           ; preds = %if.else, %if.then
  %3 = bitcast ptr %retval to ptr, !dbg !103
  %4 = load i64, ptr %3, align 4, !dbg !103
  ret i64 %4, !dbg !103
}

;; Caller-allocated memory for a local (sret parameter).
;;
;;    Large sret() {
;;      Large X = L;
;;      return X;
;;    }
;;
;; TODO: Currently not supported by `trackAssignments` (or the rest of the
;; assignment tracking pipeline). In lieu of being able to xfail a part of a
;; test, check that the dbg.declare is preserved so the test fails (and can be
;; updated) when this is fixed.
define dso_local void @_Z4sretv(ptr noalias sret(%struct.Large) align 4 %agg.result) #0 !dbg !104 {
; CHECK-LABEL: define dso_local void @_Z4sretv
entry:
; CHECK: call void @llvm.dbg.declare
  call void @llvm.dbg.declare(metadata ptr %agg.result, metadata !108, metadata !DIExpression()), !dbg !109
  %0 = bitcast ptr %agg.result to ptr, !dbg !110
  call void @llvm.memcpy.p0i8.p0i8.i64(ptr align 4 %0, ptr align 4 bitcast (ptr @L to ptr), i64 40, i1 false), !dbg !110, !tbaa.struct !111
  ret void, !dbg !113
}

;; Caller-allocated memory for a local (byval parameter).
;;
;;    void byval(Large X) {}
;;
;; TODO: See comment for sret parameters above.
define dso_local void @_Z5byval5Large(ptr noundef byval(%struct.Large) align 8 %X) #0 !dbg !114 {
; CHECK-LABEL: define dso_local void @_Z5byval5Large
entry:
; CHECK: llvm.dbg.declare
  call void @llvm.dbg.declare(metadata ptr %X, metadata !118, metadata !DIExpression()), !dbg !119
  ret void, !dbg !120
}

;; Caller-allocated memory for a local (sret parameter) with address stored to
;; local alloca.
;;
;;    LCopyCtor indirectReturn() {
;;      LCopyCtor R;
;;      return R;
;;    }
;;
;; A sret parameter is used here also, but in this case clang emits an alloca
;; to store the passed-in address for the storage for the variable. The
;; dbg.declare for the local R therefore requires a DW_OP_deref expression.
;; TODO: This isn't supported yet, so check the dbg.declare remains.
define dso_local void @_Z14indirectReturnv(ptr noalias sret(%struct.LCopyCtor) align 4 %agg.result) #0 !dbg !121 {
; CHECK-LABEL: define dso_local void @_Z14indirectReturnv
entry:
  %result.ptr = alloca ptr, align 8
  %0 = bitcast ptr %agg.result to ptr
  store ptr %0, ptr %result.ptr, align 8
  call void @llvm.dbg.declare(metadata ptr %result.ptr, metadata !126, metadata !DIExpression(DW_OP_deref)), !dbg !127
; CHECK: call void @llvm.dbg.declare
  call void @_ZN9LCopyCtorC1Ev(ptr noundef nonnull align 4 dereferenceable(16) %agg.result), !dbg !127
  ret void, !dbg !128
}

;; Inlined variable.
;;
;;    __attribute__((always_inline))
;;    int sqr(int Y) { return Y * Y;  }
;;    int fun(int X) { return sqr(X); }
;;
;; Check that dbg.assign intrinsics correctly inherit the !dbg attachment from
;; the dbg.declre.
define dso_local noundef i32 @_Z3funi(i32 noundef %X) !dbg !139 {
; CHECK-LABEL: define dso_local noundef i32 @_Z3funi
entry:
  %Y.addr.i = alloca i32, align 4
; CHECK:      %Y.addr.i = alloca i32, align 4, !DIAssignID ![[ID_10:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_6:[0-9]+]], metadata !DIExpression(), metadata ![[ID_10]], metadata ptr %Y.addr.i, metadata !DIExpression()), !dbg ![[DBG_0:[0-9]+]]
  %X.addr = alloca i32, align 4
; CHECK-NEXT: %X.addr = alloca i32, align 4, !DIAssignID ![[ID_11:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[VAR_7:[0-9]+]], metadata !DIExpression(), metadata ![[ID_11]], metadata ptr %X.addr, metadata !DIExpression()), !dbg ![[DBG_1:[0-9]+]]
  store i32 %X, ptr %X.addr, align 4
; CHECK-NEXT: store i32 %X, ptr %X.addr, align 4, !DIAssignID ![[ID_12:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %X, metadata ![[VAR_7]], metadata !DIExpression(), metadata ![[ID_12]], metadata ptr %X.addr, metadata !DIExpression()), !dbg ![[DBG_1]]
  call void @llvm.dbg.declare(metadata ptr %X.addr, metadata !140, metadata !DIExpression()), !dbg !141
  %0 = load i32, ptr %X.addr, align 4, !dbg !142
  store i32 %0, ptr %Y.addr.i, align 4
; CHECK:      store i32 %0, ptr %Y.addr.i, align 4, !DIAssignID ![[ID_13:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %0, metadata ![[VAR_6]], metadata !DIExpression(), metadata ![[ID_13]], metadata ptr %Y.addr.i, metadata !DIExpression()), !dbg ![[DBG_0]]
  call void @llvm.dbg.declare(metadata ptr %Y.addr.i, metadata !133, metadata !DIExpression()), !dbg !143
  %1 = load i32, ptr %Y.addr.i, align 4, !dbg !145
  %2 = load i32, ptr %Y.addr.i, align 4, !dbg !146
  %mul.i = mul nsw i32 %1, %2, !dbg !147
  ret i32 %mul.i, !dbg !148
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture) #1
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2
declare void @llvm.memset.p0i8.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #3
declare void @llvm.memcpy.p0i8.p0i8.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #4
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture) #1
declare dso_local void @_ZN9LCopyCtorC1Ev(ptr noundef nonnull align 4 dereferenceable(16)) unnamed_addr

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!26, !27, !28, !29, !1000}
!llvm.ident = !{!30}

; CHECK-DAG: ![[VAR_0]] = !DILocalVariable(name: "Z",
; CHECK-DAG: ![[VAR_1]] = !DILocalVariable(name: "A",
; CHECK-DAG: ![[VAR_2]] = !DILocalVariable(name: "O",
; CHECK-DAG: ![[VAR_3]] = !DILocalVariable(name: "A",
; CHECK-DAG: ![[VAR_4]] = !DILocalVariable(name: "B",
; CHECK-DAG: ![[VAR_5]] = !DILocalVariable(name: "A",
; CHECK-DAG: ![[VAR_6]] = !DILocalVariable(name: "Y", arg: 1, scope: ![[SQR:[0-9]+]],
; CHECK-DAG: ![[VAR_7]] = !DILocalVariable(name: "X", arg: 1, scope: ![[FUN:[0-9]+]],
; CHECK-DAG: ![[SQR]] = distinct !DISubprogram(name: "sqr",
; CHECK-DAG: ![[FUN]] = distinct !DISubprogram(name: "fun",
; CHECK-DAG: ![[DBG_0]] = !DILocation(line: 0, scope: ![[SQR]], inlinedAt: ![[SQR_INLINE_SITE:[0-9]+]])
; CHECK-DAG: [[SQR_INLINE_SITE]] = distinct !DILocation(line: 3, column: 25, scope: ![[FUN]])
; CHECK-DAG: ![[DBG_1]] = !DILocation(line: 0, scope: ![[FUN]])

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "Value", scope: !2, file: !3, line: 5, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0, !5, !8, !10, !16, !18}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "Index", scope: !2, file: !3, line: 5, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "Cond", scope: !2, file: !3, line: 5, type: !7, isLocal: false, isDefinition: true)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "InnerA", scope: !2, file: !3, line: 6, type: !12, isLocal: false, isDefinition: true)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Inner", file: !3, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !13, identifier: "_ZTS5Inner")
!13 = !{!14, !15}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "A", scope: !12, file: !3, line: 1, baseType: !7, size: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "B", scope: !12, file: !3, line: 1, baseType: !7, size: 32, offset: 32)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "InnerB", scope: !2, file: !3, line: 6, type: !12, isLocal: false, isDefinition: true)
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = distinct !DIGlobalVariable(name: "L", scope: !2, file: !3, line: 7, type: !20, isLocal: false, isDefinition: true)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Large", file: !3, line: 3, size: 320, flags: DIFlagTypePassByValue, elements: !21, identifier: "_ZTS5Large")
!21 = !{!22}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "A", scope: !20, file: !3, line: 3, baseType: !23, size: 320)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 320, elements: !24)
!24 = !{!25}
!25 = !DISubrange(count: 10)
!26 = !{i32 7, !"Dwarf Version", i32 5}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !{i32 1, !"wchar_size", i32 4}
!29 = !{i32 7, !"uwtable", i32 1}
!30 = !{!"clang version 14.0.0"}
!31 = distinct !DISubprogram(name: "zeroInit", linkageName: "_Z8zeroInitv", scope: !3, file: !3, line: 9, type: !32, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !34)
!32 = !DISubroutineType(types: !33)
!33 = !{null}
!34 = !{!35}
!35 = !DILocalVariable(name: "Z", scope: !31, file: !3, line: 9, type: !36)
!36 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 96, elements: !37)
!37 = !{!38}
!38 = !DISubrange(count: 3)
!39 = !DILocation(line: 9, column: 19, scope: !31)
!40 = !DILocation(line: 9, column: 23, scope: !31)
!41 = !DILocation(line: 9, column: 41, scope: !31)
!42 = distinct !DISubprogram(name: "memcpyInit", linkageName: "_Z10memcpyInitv", scope: !3, file: !3, line: 11, type: !32, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !43)
!43 = !{!44}
!44 = !DILocalVariable(name: "A", scope: !42, file: !3, line: 11, type: !45)
!45 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 128, elements: !46)
!46 = !{!47}
!47 = !DISubrange(count: 4)
!48 = !DILocation(line: 11, column: 21, scope: !42)
!49 = !DILocation(line: 11, column: 25, scope: !42)
!50 = !DILocation(line: 11, column: 46, scope: !42)
!51 = distinct !DISubprogram(name: "setField", linkageName: "_Z8setFieldv", scope: !3, file: !3, line: 13, type: !32, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !52)
!52 = !{!53}
!53 = !DILocalVariable(name: "O", scope: !51, file: !3, line: 14, type: !54)
!54 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Outer", file: !3, line: 2, size: 128, flags: DIFlagTypePassByValue, elements: !55, identifier: "_ZTS5Outer")
!55 = !{!56, !57}
!56 = !DIDerivedType(tag: DW_TAG_member, name: "A", scope: !54, file: !3, line: 2, baseType: !12, size: 64)
!57 = !DIDerivedType(tag: DW_TAG_member, name: "B", scope: !54, file: !3, line: 2, baseType: !12, size: 64, offset: 64)
!58 = !DILocation(line: 14, column: 3, scope: !51)
!59 = !DILocation(line: 14, column: 9, scope: !51)
!60 = !DILocation(line: 15, column: 11, scope: !51)
!61 = !{!62, !62, i64 0}
!62 = !{!"int", !63, i64 0}
!63 = !{!"omnipotent char", !64, i64 0}
!64 = !{!"Simple C++ TBAA"}
!65 = !DILocation(line: 15, column: 5, scope: !51)
!66 = !DILocation(line: 15, column: 7, scope: !51)
!67 = !DILocation(line: 15, column: 9, scope: !51)
!68 = !{!69, !62, i64 4}
!69 = !{!"_ZTS5Outer", !70, i64 0, !70, i64 8}
!70 = !{!"_ZTS5Inner", !62, i64 0, !62, i64 4}
!71 = !DILocation(line: 16, column: 1, scope: !51)
!72 = distinct !DISubprogram(name: "unknownOffset", linkageName: "_Z13unknownOffsetv", scope: !3, file: !3, line: 18, type: !32, scopeLine: 18, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !73)
!73 = !{!74}
!74 = !DILocalVariable(name: "A", scope: !72, file: !3, line: 19, type: !75)
!75 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 64, elements: !76)
!76 = !{!77}
!77 = !DISubrange(count: 2)
!78 = !DILocation(line: 19, column: 3, scope: !72)
!79 = !DILocation(line: 19, column: 7, scope: !72)
!80 = !DILocation(line: 20, column: 14, scope: !72)
!81 = !DILocation(line: 20, column: 5, scope: !72)
!82 = !DILocation(line: 20, column: 3, scope: !72)
!83 = !DILocation(line: 20, column: 12, scope: !72)
!84 = !DILocation(line: 21, column: 1, scope: !72)
!85 = distinct !DISubprogram(name: "sharedAlloca", linkageName: "_Z12sharedAllocav", scope: !3, file: !3, line: 23, type: !86, scopeLine: 23, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !88)
!86 = !DISubroutineType(types: !87)
!87 = !{!12}
!88 = !{!89, !92}
!89 = !DILocalVariable(name: "A", scope: !90, file: !3, line: 25, type: !12)
!90 = distinct !DILexicalBlock(scope: !91, file: !3, line: 24, column: 13)
!91 = distinct !DILexicalBlock(scope: !85, file: !3, line: 24, column: 7)
!92 = !DILocalVariable(name: "B", scope: !93, file: !3, line: 28, type: !12)
!93 = distinct !DILexicalBlock(scope: !91, file: !3, line: 27, column: 10)
!94 = !DILocation(line: 24, column: 7, scope: !91)
!95 = !DILocation(line: 24, column: 7, scope: !85)
!96 = !DILocation(line: 25, column: 11, scope: !90)
!97 = !DILocation(line: 25, column: 15, scope: !90)
!98 = !{i64 0, i64 4, !61, i64 4, i64 4, !61}
!99 = !DILocation(line: 26, column: 5, scope: !90)
!100 = !DILocation(line: 28, column: 11, scope: !93)
!101 = !DILocation(line: 28, column: 15, scope: !93)
!102 = !DILocation(line: 29, column: 5, scope: !93)
!103 = !DILocation(line: 31, column: 1, scope: !85)
!104 = distinct !DISubprogram(name: "sret", linkageName: "_Z4sretv", scope: !3, file: !3, line: 33, type: !105, scopeLine: 33, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !107)
!105 = !DISubroutineType(types: !106)
!106 = !{!20}
!107 = !{!108}
!108 = !DILocalVariable(name: "X", scope: !104, file: !3, line: 34, type: !20)
!109 = !DILocation(line: 34, column: 9, scope: !104)
!110 = !DILocation(line: 34, column: 13, scope: !104)
!111 = !{i64 0, i64 40, !112}
!112 = !{!63, !63, i64 0}
!113 = !DILocation(line: 35, column: 3, scope: !104)
!114 = distinct !DISubprogram(name: "byval", linkageName: "_Z5byval5Large", scope: !3, file: !3, line: 38, type: !115, scopeLine: 38, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !117)
!115 = !DISubroutineType(types: !116)
!116 = !{null, !20}
!117 = !{!118}
!118 = !DILocalVariable(name: "X", arg: 1, scope: !114, file: !3, line: 38, type: !20)
!119 = !DILocation(line: 38, column: 18, scope: !114)
!120 = !DILocation(line: 38, column: 22, scope: !114)
!121 = distinct !DISubprogram(name: "indirectReturn", linkageName: "_Z14indirectReturnv", scope: !2, file: !3, line: 41, type: !122, scopeLine: 41, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !125)
!122 = !DISubroutineType(types: !123)
!123 = !{!124}
!124 = !DICompositeType(tag: DW_TAG_structure_type, name: "LCopyCtor", file: !3, line: 4, size: 128, flags: DIFlagFwdDecl | DIFlagNonTrivial, identifier: "_ZTS9LCopyCtor")
!125 = !{}
!126 = !DILocalVariable(name: "R", scope: !121, file: !3, line: 42, type: !124)
!127 = !DILocation(line: 42, column: 13, scope: !121)
!128 = !DILocation(line: 43, column: 3, scope: !121)
!129 = distinct !DISubprogram(name: "sqr", linkageName: "_Z3sqri", scope: !2, file: !3, line: 2, type: !130, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !132)
!130 = !DISubroutineType(types: !131)
!131 = !{!7, !7}
!132 = !{}
!133 = !DILocalVariable(name: "Y", arg: 1, scope: !129, file: !3, line: 2, type: !7)
!134 = !DILocation(line: 2, column: 13, scope: !129)
!135 = !DILocation(line: 2, column: 25, scope: !129)
!136 = !DILocation(line: 2, column: 29, scope: !129)
!137 = !DILocation(line: 2, column: 27, scope: !129)
!138 = !DILocation(line: 2, column: 18, scope: !129)
!139 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funi", scope: !2, file: !3, line: 3, type: !130, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !132)
!140 = !DILocalVariable(name: "X", arg: 1, scope: !139, file: !3, line: 3, type: !7)
!141 = !DILocation(line: 3, column: 13, scope: !139)
!142 = !DILocation(line: 3, column: 29, scope: !139)
!143 = !DILocation(line: 2, column: 13, scope: !129, inlinedAt: !144)
!144 = distinct !DILocation(line: 3, column: 25, scope: !139)
!145 = !DILocation(line: 2, column: 25, scope: !129, inlinedAt: !144)
!146 = !DILocation(line: 2, column: 29, scope: !129, inlinedAt: !144)
!147 = !DILocation(line: 2, column: 27, scope: !129, inlinedAt: !144)
!148 = !DILocation(line: 3, column: 18, scope: !139)
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
