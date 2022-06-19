; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info %s 2>&1 | FileCheck %s

; %col.ptr.1 and %col.ptr.2 do not alias, if we know that %skip >= 0, because
; the distance between %col.ptr.1 and %col.ptr.2 is %skip + 6 and we load 6
; elements.
define void @test1(double* %ptr, i32 %skip) {
; CHECK-LABEL: Function: test1: 4 pointers, 1 call sites
; CHECK-NEXT:  MustAlias:   <6 x double>* %col.ptr.1, double* %ptr
; CHECK-NEXT:  NoAlias: double* %col.ptr.2, double* %ptr
; CHECK-NEXT:  NoAlias: <6 x double>* %col.ptr.1, double* %col.ptr.2
; CHECK-NEXT:  NoAlias: <6 x double>* %col.ptr.2.cast, double* %ptr
; CHECK-NEXT:  NoAlias: <6 x double>* %col.ptr.1, <6 x double>* %col.ptr.2.cast
; CHECK-NEXT:  MustAlias:   double* %col.ptr.2, <6 x double>* %col.ptr.2.cast
; CHECK-NEXT:  NoModRef:  Ptr: double* %ptr <->  call void @llvm.assume(i1 %gt)
; CHECK-NEXT:  NoModRef:  Ptr: <6 x double>* %col.ptr.1 <->  call void @llvm.assume(i1 %gt)
; CHECK-NEXT:  NoModRef:  Ptr: double* %col.ptr.2   <->  call void @llvm.assume(i1 %gt)
; CHECK-NEXT:  NoModRef:  Ptr: <6 x double>* %col.ptr.2.cast    <->  call void @llvm.assume(i1 %gt)
;
  load double, double* %ptr
  %gt = icmp sgt i32 %skip, -1
  call void @llvm.assume(i1 %gt)
  %stride = add nsw nuw i32 %skip, 6
  %col.ptr.1 = bitcast double* %ptr to <6 x double>*
  %lv.1 = load <6 x double>, <6 x double>* %col.ptr.1, align 8
  %col.ptr.2= getelementptr double, double* %ptr, i32 %stride
  %col.ptr.2.cast = bitcast double* %col.ptr.2 to <6 x double>*
  load double, double* %col.ptr.2
  %lv.2 = load <6 x double>, <6 x double>* %col.ptr.2.cast, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, <6 x double>* %col.ptr.1, align 8
  store <6 x double> %res.2, <6 x double>* %col.ptr.2.cast, align 8
  ret void
}

; Same as @test1, but now we do not have an assume guaranteeing %skip >= 0.
define void @test2(double* %ptr, i32 %skip) {
; CHECK-LABEL: Function: test2: 4 pointers, 0 call sites
; CHECK-NEXT:  MustAlias:   <6 x double>* %col.ptr.1, double* %ptr
; CHECK-NEXT:  MayAlias:    double* %col.ptr.2, double* %ptr
; CHECK-NEXT:  MayAlias:    <6 x double>* %col.ptr.1, double* %col.ptr.2
; CHECK-NEXT:  MayAlias:    <6 x double>* %col.ptr.2.cast, double* %ptr
; CHECK-NEXT:  MayAlias:    <6 x double>* %col.ptr.1, <6 x double>* %col.ptr.2.cast
; CHECK-NEXT:  MustAlias:   double* %col.ptr.2, <6 x double>* %col.ptr.2.cast
;
  load double, double* %ptr
  %stride = add nsw nuw i32 %skip, 6
  %col.ptr.1 = bitcast double* %ptr to <6 x double>*
  %lv.1 = load <6 x double>, <6 x double>* %col.ptr.1, align 8
  %col.ptr.2 = getelementptr double, double* %ptr, i32 %stride
  load double, double* %col.ptr.2
  %col.ptr.2.cast = bitcast double* %col.ptr.2 to <6 x double>*
  %lv.2 = load <6 x double>, <6 x double>* %col.ptr.2.cast, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, <6 x double>* %col.ptr.1, align 8
  store <6 x double> %res.2, <6 x double>* %col.ptr.2.cast, align 8
  ret void
}

; Same as @test1, this time the assume just guarantees %skip > -3, which is
; enough to derive NoAlias for %ptr and %col.ptr.2 (distance is more than 3
; doubles, and we load 1 double), but not %col.ptr.1 and %col.ptr.2 (distance
; is more than 3 doubles, and we load 6 doubles).
define void @test3(double* %ptr, i32 %skip) {
; CHECK-LABEL: Function: test3: 4 pointers, 1 call sites
; CHECK-NEXT:  MustAlias:   <6 x double>* %col.ptr.1, double* %ptr
; CHECK-NEXT:  NoAlias:     double* %col.ptr.2, double* %ptr
; CHECK-NEXT:  MayAlias:    <6 x double>* %col.ptr.1, double* %col.ptr.2
; CHECK-NEXT:  NoAlias:     <6 x double>* %col.ptr.2.cast, double* %ptr
; CHECK-NEXT:  MayAlias:    <6 x double>* %col.ptr.1, <6 x double>* %col.ptr.2.cast
; CHECK-NEXT:  MustAlias:   double* %col.ptr.2, <6 x double>* %col.ptr.2.cast
; CHECK-NEXT:  NoModRef:  Ptr: double* %ptr <->  call void @llvm.assume(i1 %gt)
; CHECK-NEXT:  NoModRef:  Ptr: <6 x double>* %col.ptr.1 <->  call void @llvm.assume(i1 %gt)
; CHECK-NEXT:  NoModRef:  Ptr: double* %col.ptr.2   <->  call void @llvm.assume(i1 %gt)
; CHECK-NEXT:  NoModRef:  Ptr: <6 x double>* %col.ptr.2.cast    <->  call void @llvm.assume(i1 %gt)
;
  load double, double* %ptr
  %gt = icmp sgt i32 %skip, -3
  call void @llvm.assume(i1 %gt)
  %stride = add nsw nuw i32 %skip, 6
  %col.ptr.1 = bitcast double* %ptr to <6 x double>*
  %lv.1 = load <6 x double>, <6 x double>* %col.ptr.1, align 8
  %col.ptr.2 = getelementptr double, double* %ptr, i32 %stride
  load double, double* %col.ptr.2
  %col.ptr.2.cast = bitcast double* %col.ptr.2 to <6 x double>*
  %lv.2 = load <6 x double>, <6 x double>* %col.ptr.2.cast, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, <6 x double>* %col.ptr.1, align 8
  store <6 x double> %res.2, <6 x double>* %col.ptr.2.cast, align 8
  ret void
}

; Same as @test1, but the assume uses the sge predicate for %skip >= 0.
define void @test4(double* %ptr, i32 %skip) {
; CHECK-LABEL: Function: test4: 4 pointers, 1 call sites
; CHECK-NEXT:  MustAlias:   <6 x double>* %col.ptr.1, double* %ptr
; CHECK-NEXT:  NoAlias:     double* %col.ptr.2, double* %ptr
; CHECK-NEXT:  NoAlias:     <6 x double>* %col.ptr.1, double* %col.ptr.2
; CHECK-NEXT:  NoAlias:     <6 x double>* %col.ptr.2.cast, double* %ptr
; CHECK-NEXT:  NoAlias:     <6 x double>* %col.ptr.1, <6 x double>* %col.ptr.2.cast
; CHECK-NEXT:  MustAlias:   double* %col.ptr.2, <6 x double>* %col.ptr.2.cast
; CHECK-NEXT:  NoModRef:  Ptr: double* %ptr <->  call void @llvm.assume(i1 %gt)
; CHECK-NEXT:  NoModRef:  Ptr: <6 x double>* %col.ptr.1 <->  call void @llvm.assume(i1 %gt)
; CHECK-NEXT:  NoModRef:  Ptr: double* %col.ptr.2   <->  call void @llvm.assume(i1 %gt)
; CHECK-NEXT:  NoModRef:  Ptr: <6 x double>* %col.ptr.2.cast    <->  call void @llvm.assume(i1 %gt)
;
  load double, double* %ptr
  %gt = icmp sge i32 %skip, 0
  call void @llvm.assume(i1 %gt)
  %stride = add nsw nuw i32 %skip, 6
  %col.ptr.1 = bitcast double* %ptr to <6 x double>*
  %lv.1 = load <6 x double>, <6 x double>* %col.ptr.1, align 8
  %col.ptr.2 = getelementptr double, double* %ptr, i32 %stride
  load double, double* %col.ptr.2
  %col.ptr.2.cast = bitcast double* %col.ptr.2 to <6 x double>*
  %lv.2 = load <6 x double>, <6 x double>* %col.ptr.2.cast, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, <6 x double>* %col.ptr.1, align 8
  store <6 x double> %res.2, <6 x double>* %col.ptr.2.cast, align 8
  ret void
}

define void @symmetry([0 x i8]* %ptr, i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: Function: symmetry
; CHECK: NoAlias: i8* %gep1, i8* %gep2
;
  %b.cmp = icmp slt i32 %b, 0
  call void @llvm.assume(i1 %b.cmp)
  %gep1 = getelementptr [0 x i8], [0 x i8]* %ptr, i32 %a, i32 %b
  load i8, i8* %gep1
  call void @barrier()
  %c.cmp = icmp sgt i32 %c, -1
  call void @llvm.assume(i1 %c.cmp)
  %c.off = add nuw nsw i32 %c, 1
  %gep2 = getelementptr [0 x i8], [0 x i8]* %ptr, i32 %a, i32 %c.off
  load i8, i8* %gep2
  ret void
}

; %ptr.neg and %ptr.shl may alias, as the shl renders the previously
; non-negative value potentially negative.
define void @shl_of_non_negative(i8* %ptr, i64 %a) {
; CHECK-LABEL: Function: shl_of_non_negative
; CHECK: NoAlias: i8* %ptr.a, i8* %ptr.neg
; CHECK: MayAlias: i8* %ptr.neg, i8* %ptr.shl
  %a.cmp = icmp sge i64 %a, 0
  call void @llvm.assume(i1 %a.cmp)
  %ptr.neg = getelementptr i8, i8* %ptr, i64 -2
  %ptr.a = getelementptr i8, i8* %ptr, i64 %a
  %shl = shl i64 %a, 1
  %ptr.shl = getelementptr i8, i8* %ptr, i64 %shl
  load i8, i8* %ptr.a
  load i8, i8* %ptr.neg
  load i8, i8* %ptr.shl
  ret void
}

; Unlike the previous case, %ptr.neg and %ptr.shl can't alias, because
; shl nsw of non-negative is non-negative.
define void @shl_nsw_of_non_negative(i8* %ptr, i64 %a) {
; CHECK-LABEL: Function: shl_nsw_of_non_negative
; CHECK: NoAlias: i8* %ptr.a, i8* %ptr.neg
; CHECK: NoAlias: i8* %ptr.neg, i8* %ptr.shl
  %a.cmp = icmp sge i64 %a, 0
  call void @llvm.assume(i1 %a.cmp)
  %ptr.neg = getelementptr i8, i8* %ptr, i64 -2
  %ptr.a = getelementptr i8, i8* %ptr, i64 %a
  %shl = shl nsw i64 %a, 1
  %ptr.shl = getelementptr i8, i8* %ptr, i64 %shl
  load i8, i8* %ptr.a
  load i8, i8* %ptr.neg
  load i8, i8* %ptr.shl
  ret void
}

declare void @llvm.assume(i1 %cond)
declare void @barrier()
