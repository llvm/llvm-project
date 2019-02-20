; RUN: opt -codegenprepare -load-store-vectorizer %s -S -o - | FileCheck %s
; RUN: opt                 -load-store-vectorizer %s -S -o - | FileCheck %s
; RUN: opt -codegenprepare -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' %s -S -o - | FileCheck %s
; RUN: opt                 -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' %s -S -o - | FileCheck %s

target triple = "x86_64--"

%union = type { { [4 x [4 x [4 x [16 x float]]]], [4 x [4 x [4 x [16 x float]]]], [10 x [10 x [4 x float]]] } }

@global_pointer = external unnamed_addr global { %union, [2000 x i8] }, align 4

; Function Attrs: convergent nounwind
define void @test(i32 %base) #0 {
; CHECK-LABEL: @test(
; CHECK-NOT: load i32
; CHECK: load <2 x i32>
; CHECK-NOT: load i32
entry:
  %mul331 = and i32 %base, -4
  %add350.4 = add i32 4, %mul331
  %idx351.4 = zext i32 %add350.4 to i64
  %arrayidx352.4 = getelementptr inbounds { %union, [2000 x i8] }, { %union, [2000 x i8] }* @global_pointer, i64 0, i32 0, i32 0, i32 1, i64 0, i64 0, i64 0, i64 %idx351.4
  %tmp296.4 = bitcast float* %arrayidx352.4 to i32*
  %add350.5 = add i32 5, %mul331
  %idx351.5 = zext i32 %add350.5 to i64
  %arrayidx352.5 = getelementptr inbounds { %union, [2000 x i8] }, { %union, [2000 x i8] }* @global_pointer, i64 0, i32 0, i32 0, i32 1, i64 0, i64 0, i64 0, i64 %idx351.5
  %tmp296.5 = bitcast float* %arrayidx352.5 to i32*
  %cnd = icmp ult i32 %base, 1000
  br i1 %cnd, label %loads, label %exit

loads:
  ; If and only if the loads are in a different BB from the GEPs codegenprepare
  ; would try to turn the GEPs into math, which makes LoadStoreVectorizer's job
  ; harder
  %tmp297.4 = load i32, i32* %tmp296.4, align 4, !tbaa !0
  %tmp297.5 = load i32, i32* %tmp296.5, align 4, !tbaa !0
  br label %exit

exit:
  ret void
}

; Function Attrs: convergent nounwind
define void @test.codegenprepared(i32 %base) #0 {
; CHECK-LABEL: @test.codegenprepared(
; CHECK-NOT: load i32
; CHECK: load <2 x i32>
; CHECK-NOT: load i32
entry:
  %mul331 = and i32 %base, -4
  %add350.4 = add i32 4, %mul331
  %idx351.4 = zext i32 %add350.4 to i64
  %add350.5 = add i32 5, %mul331
  %idx351.5 = zext i32 %add350.5 to i64
  %cnd = icmp ult i32 %base, 1000
  br i1 %cnd, label %loads, label %exit

loads:                                            ; preds = %entry
  %sunkaddr = mul i64 %idx351.4, 4
  %sunkaddr1 = getelementptr inbounds i8, i8* bitcast ({ %union, [2000 x i8] }* @global_pointer to i8*), i64 %sunkaddr
  %sunkaddr2 = getelementptr inbounds i8, i8* %sunkaddr1, i64 4096
  %0 = bitcast i8* %sunkaddr2 to i32*
  %tmp297.4 = load i32, i32* %0, align 4, !tbaa !0
  %sunkaddr3 = mul i64 %idx351.5, 4
  %sunkaddr4 = getelementptr inbounds i8, i8* bitcast ({ %union, [2000 x i8] }* @global_pointer to i8*), i64 %sunkaddr3
  %sunkaddr5 = getelementptr inbounds i8, i8* %sunkaddr4, i64 4096
  %1 = bitcast i8* %sunkaddr5 to i32*
  %tmp297.5 = load i32, i32* %1, align 4, !tbaa !0
  br label %exit

exit:                                             ; preds = %loads, %entry
  ret void
}

attributes #0 = { convergent nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}
