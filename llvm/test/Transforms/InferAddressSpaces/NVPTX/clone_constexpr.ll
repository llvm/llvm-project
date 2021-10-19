; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -infer-address-spaces %s | FileCheck %s

%struct.S = type { [5 x i32] }

$g1 = comdat any

@g1 = linkonce_odr addrspace(3) global %struct.S zeroinitializer, comdat, align 4

; CHECK-LABEL: @foo(
; CHECK:  %x0 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2
; CHECK:  %idxprom.i = zext i32 %x0 to i64
; CHECK:  %arrayidx.i = getelementptr %struct.S, %struct.S* addrspacecast (%struct.S addrspace(3)* @g1 to %struct.S*), i64 0, i32 0, i64 %idxprom.i
; CHECK:  tail call void @f1(i32* %arrayidx.i, i32 undef) #0
; CHECK:  %x1 = load i32, i32 addrspace(3)* getelementptr inbounds (%struct.S, %struct.S addrspace(3)* @g1, i64 0, i32 0, i64 0), align 4
; CHECK:  %L.sroa.0.0.insert.ext.i = zext i32 %x1 to i64
; CHECK:  tail call void @f2(i64* null, i64 %L.sroa.0.0.insert.ext.i) #0
; CHECK:  ret void
define void @foo() local_unnamed_addr #0 {
entry:
  %x0 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2
  %idxprom.i = zext i32 %x0 to i64
  %arrayidx.i = getelementptr %struct.S, %struct.S* addrspacecast (%struct.S addrspace(3)* @g1 to %struct.S*), i64 0, i32 0, i64 %idxprom.i
  tail call void @f1(i32* %arrayidx.i, i32 undef) #0
  %x1 = load i32, i32* getelementptr (%struct.S, %struct.S* addrspacecast (%struct.S addrspace(3)* @g1 to %struct.S*), i64 0, i32 0, i64 0), align 4
  %L.sroa.0.0.insert.ext.i = zext i32 %x1 to i64
  tail call void @f2(i64* null, i64 %L.sroa.0.0.insert.ext.i) #0
  ret void
}

declare void @f1(i32*, i32) local_unnamed_addr #0
declare void @f2(i64*, i64) local_unnamed_addr #0
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Make sure we can clone GEP which uses complex constant expressions as indices.
; https://bugs.llvm.org/show_bug.cgi?id=51099
@g2 = internal addrspace(3) global [128 x i8] undef, align 1

; CHECK-LABEL: @complex_ce(
; CHECK: %0 = load float, float addrspace(3)* bitcast
; CHECK-SAME:   i8 addrspace(3)* getelementptr (i8,
; CHECK-SAME:     i8 addrspace(3)* getelementptr inbounds ([128 x i8], [128 x i8] addrspace(3)* @g2, i64 0, i64 0),
; CHECK-SAME:     i64 sub (
; CHECK-SAME        i64 ptrtoint (
; CHECK-SAME          i8 addrspace(3)* getelementptr inbounds ([128 x i8], [128 x i8] addrspace(3)* @g2, i64 0, i64 123) to i64),
; CHECK-SAME:       i64 ptrtoint (
; CHECK-SAME:         i8 addrspace(3)* getelementptr inbounds ([128 x i8], [128 x i8] addrspace(3)* @g2, i64 2, i64 0) to i64)))
; CHECK-SAME:   to float addrspace(3)*)
; Function Attrs: norecurse nounwind
define float @complex_ce(i8* nocapture readnone %a, i8* nocapture readnone %b, i8* nocapture readnone %c) local_unnamed_addr #0 {
entry:
  %0 = load float, float* bitcast (
       i8* getelementptr (
         i8, i8* getelementptr inbounds (
           [128 x i8],
           [128 x i8]* addrspacecast ([128 x i8] addrspace(3)* @g2 to [128 x i8]*),
           i64 0,
           i64 0),
         i64 sub (
           i64 ptrtoint (
             i8* getelementptr inbounds (
               [128 x i8],
               [128 x i8]* addrspacecast ([128 x i8] addrspace(3)* @g2 to [128 x i8]*),
               i64 0,
               i64 123)
             to i64),
           i64 ptrtoint (
             i8* getelementptr inbounds (
               [128 x i8],
               [128 x i8]* addrspacecast ([128 x i8] addrspace(3)* @g2 to [128 x i8]*),
               i64 2,
               i64 0)
             to i64)))
        to float*), align 4
  ret float %0
}



attributes #0 = { convergent nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
