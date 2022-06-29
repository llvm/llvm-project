; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info %s 2>&1 | FileCheck %s

; %col.ptr.1 and %col.ptr.2 do not alias, if we know that %skip >= 0, because
; the distance between %col.ptr.1 and %col.ptr.2 is %skip + 6 and we load 6
; elements.
define void @test1(ptr %ptr, i32 %skip) {
; CHECK-LABEL: Function: test1: 2 pointers, 1 call sites
; CHECK-NEXT:  NoAlias: <6 x double>* %col.ptr.2, <6 x double>* %ptr
;
  %gt = icmp sgt i32 %skip, -1
  call void @llvm.assume(i1 %gt)
  %stride = add nsw nuw i32 %skip, 6
  %lv.1 = load <6 x double>, ptr %ptr, align 8
  %col.ptr.2 = getelementptr double, ptr %ptr, i32 %stride
  %lv.2 = load <6 x double>, ptr %col.ptr.2, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, ptr %ptr, align 8
  store <6 x double> %res.2, ptr %col.ptr.2, align 8
  ret void
}

; Same as @test1, but now we do not have an assume guaranteeing %skip >= 0.
define void @test2(ptr %ptr, i32 %skip) {
; CHECK-LABEL: Function: test2: 2 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  <6 x double>* %col.ptr.2, <6 x double>* %ptr
;
  %stride = add nsw nuw i32 %skip, 6
  %lv.1 = load <6 x double>, ptr %ptr, align 8
  %col.ptr.2 = getelementptr double, ptr  %ptr, i32 %stride
  %lv.2 = load <6 x double>, ptr %col.ptr.2, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, ptr %ptr, align 8
  store <6 x double> %res.2, ptr %col.ptr.2, align 8
  ret void
}

; Same as @test1, this time the assume just guarantees %skip > -3, which is
; enough to derive NoAlias for %ptr and %col.ptr.2 (distance is more than 3
; doubles, and we load 1 double), but not %col.ptr.1 and %col.ptr.2 (distance
; is more than 3 doubles, and we load 6 doubles).
define void @test3(ptr %ptr, i32 %skip) {
; CHECK-LABEL: Function: test3: 2 pointers, 1 call sites
; CHECK-NEXT:   MayAlias:	<6 x double>* %col.ptr.2, <6 x double>* %ptr
;
  %gt = icmp sgt i32 %skip, -3
  call void @llvm.assume(i1 %gt)
  %stride = add nsw nuw i32 %skip, 6
  %lv.1 = load <6 x double>, ptr %ptr, align 8
  %col.ptr.2 = getelementptr double, ptr %ptr, i32 %stride
  %lv.2 = load <6 x double>, ptr %col.ptr.2, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, ptr %ptr, align 8
  store <6 x double> %res.2, ptr %col.ptr.2, align 8
  ret void
}

; Same as @test1, but the assume uses the sge predicate for %skip >= 0.
define void @test4(ptr %ptr, i32 %skip) {
; CHECK-LABEL: Function: test4: 2 pointers, 1 call sites
; CHECK-NEXT:    NoAlias:	<6 x double>* %col.ptr.2, <6 x double>* %ptr
;
  %gt = icmp sge i32 %skip, 0
  call void @llvm.assume(i1 %gt)
  %stride = add nsw nuw i32 %skip, 6
  %lv.1 = load <6 x double>, ptr %ptr, align 8
  %col.ptr.2 = getelementptr double, ptr %ptr, i32 %stride
  %lv.2 = load <6 x double>, ptr %col.ptr.2, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, ptr %ptr, align 8
  store <6 x double> %res.2, ptr %col.ptr.2, align 8
  ret void
}

define void @symmetry(ptr %ptr, i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: Function: symmetry
; CHECK: NoAlias: i8* %gep1, i8* %gep2
;
  %b.cmp = icmp slt i32 %b, 0
  call void @llvm.assume(i1 %b.cmp)
  %gep1 = getelementptr [0 x i8], ptr %ptr, i32 %a, i32 %b
  load i8, ptr %gep1
  call void @barrier()
  %c.cmp = icmp sgt i32 %c, -1
  call void @llvm.assume(i1 %c.cmp)
  %c.off = add nuw nsw i32 %c, 1
  %gep2 = getelementptr [0 x i8], ptr %ptr, i32 %a, i32 %c.off
  load i8, ptr %gep2
  ret void
}

; %ptr.neg and %ptr.shl may alias, as the shl renders the previously
; non-negative value potentially negative.
define void @shl_of_non_negative(ptr %ptr, i64 %a) {
; CHECK-LABEL: Function: shl_of_non_negative
; CHECK: NoAlias: i8* %ptr.a, i8* %ptr.neg
; CHECK: MayAlias: i8* %ptr.neg, i8* %ptr.shl
;
  %a.cmp = icmp sge i64 %a, 0
  call void @llvm.assume(i1 %a.cmp)
  %ptr.neg = getelementptr i8, ptr %ptr, i64 -2
  %ptr.a = getelementptr i8, ptr %ptr, i64 %a
  %shl = shl i64 %a, 1
  %ptr.shl = getelementptr i8, ptr %ptr, i64 %shl
  load i8, i8* %ptr.a
  load i8, i8* %ptr.neg
  load i8, i8* %ptr.shl
  ret void
}

; Unlike the previous case, %ptr.neg and %ptr.shl can't alias, because
; shl nsw of non-negative is non-negative.
define void @shl_nsw_of_non_negative(ptr %ptr, i64 %a) {
; CHECK-LABEL: Function: shl_nsw_of_non_negative
; CHECK: NoAlias: i8* %ptr.a, i8* %ptr.neg
; CHECK: NoAlias: i8* %ptr.neg, i8* %ptr.shl
;
  %a.cmp = icmp sge i64 %a, 0
  call void @llvm.assume(i1 %a.cmp)
  %ptr.neg = getelementptr i8, ptr %ptr, i64 -2
  %ptr.a = getelementptr i8, ptr %ptr, i64 %a
  %shl = shl nsw i64 %a, 1
  %ptr.shl = getelementptr i8, ptr %ptr, i64 %shl
  load i8, ptr %ptr.a
  load i8, ptr %ptr.neg
  load i8, ptr %ptr.shl
  ret void
}

define void @test5(ptr %ptr, i32 %stride) {
; CHECK-LABEL: Function: test5: 2 pointers, 1 call sites
; CHECK-NEXT:    MayAlias:   <6 x double>* %col.ptr.2, <6 x double>* %ptr
;
  %gt = icmp sge i32 %stride, 5
  call void @llvm.assume(i1 %gt)
  %lv.1 = load <6 x double>, ptr %ptr, align 8
  %col.ptr.2= getelementptr double, ptr %ptr, i32 %stride
  %lv.2 = load <6 x double>, ptr %col.ptr.2, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, ptr %ptr, align 8
  store <6 x double> %res.2, ptr %col.ptr.2, align 8
  ret void
}

define void @test6(ptr %ptr, i32 %stride) {
; CHECK-LABEL: Function: test6: 2 pointers, 1 call sites
; CHECK-NEXT:    NoAlias:  <6 x double>* %col.ptr.2, <6 x double>* %ptr
;
  %gt = icmp sge i32 %stride, 6
  call void @llvm.assume(i1 %gt)
  %lv.1 = load <6 x double>, ptr %ptr, align 8
  %col.ptr.2= getelementptr double, ptr %ptr, i32 %stride
  %lv.2 = load <6 x double>, ptr %col.ptr.2, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, ptr %ptr, align 8
  store <6 x double> %res.2, ptr %col.ptr.2, align 8
  ret void
}

define void @test7(ptr %ptr, i32 %stride) {
; CHECK-LABEL: Function: test7: 2 pointers, 1 call sites
; CHECK-NEXT:    MayAlias: <6 x double>* %col.ptr.2, <6 x double>* %ptr
;
  %gt = icmp sge i32 %stride, 0
  call void @llvm.assume(i1 %gt)
  %lv.1 = load <6 x double>, ptr %ptr, align 8
  %col.ptr.2= getelementptr double, ptr %ptr, i32 %stride
  %lv.2 = load <6 x double>, ptr %col.ptr.2, align 8
  %res.1 = fadd <6 x double> %lv.1, %lv.1
  %res.2 = fadd <6 x double> %lv.2, %lv.2
  store <6 x double> %res.1, ptr %ptr, align 8
  store <6 x double> %res.2, ptr %col.ptr.2, align 8
  ret void
}

declare void @llvm.assume(i1 %cond)
declare void @barrier()
