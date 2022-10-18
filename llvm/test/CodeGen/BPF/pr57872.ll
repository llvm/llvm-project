; RUN: llc < %s -mtriple=bpf-- | FileCheck %s
; XFAIL: *

%struct.event = type { i8, [84 x i8] }

define void @foo(ptr %g) {
entry:
  %event = alloca %struct.event, align 1
  %hostname = getelementptr inbounds %struct.event, ptr %event, i64 0, i32 1
  %0 = load ptr, ptr %g, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(84) %hostname, ptr noundef nonnull align 1 dereferenceable(84) %0, i64 84, i1 false)
  call void @bar(ptr noundef nonnull %event)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2
declare void @bar(ptr noundef)
