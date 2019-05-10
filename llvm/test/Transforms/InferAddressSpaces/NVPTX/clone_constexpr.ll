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

attributes #0 = { convergent nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
