; RUN: opt < %s -instcombine -S | FileCheck %s

@test.data = private unnamed_addr addrspace(2) constant [8 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7], align 4

; CHECK-LABEL: test_load
; CHECK: %[[GEP:.*]] = getelementptr [8 x i32], [8 x i32] addrspace(2)* @test.data, i64 0, i64 %x
; CHECK: %{{.*}} = load i32, i32 addrspace(2)* %[[GEP]]
; CHECK-NOT: alloca
; CHECK-NOT: call void @llvm.memcpy.p0i8.p2i8.i64
; CHECK-NOT: addrspacecast
; CHECK-NOT: load i32, i32*
define void @test_load(i32 addrspace(1)* %out, i64 %x) {
entry:
  %data = alloca [8 x i32], align 4
  %0 = bitcast [8 x i32]* %data to i8*
  call void @llvm.memcpy.p0i8.p2i8.i64(i8* %0, i8 addrspace(2)* bitcast ([8 x i32] addrspace(2)* @test.data to i8 addrspace(2)*), i64 32, i32 4, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* %data, i64 0, i64 %x
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %x
  store i32 %1, i32 addrspace(1)* %arrayidx1, align 4
  ret void
}

; CHECK-LABEL: test_load_bitcast_chain
; CHECK: %[[GEP:.*]] = getelementptr [8 x i32], [8 x i32] addrspace(2)* @test.data, i64 0, i64 %x
; CHECK: %{{.*}} = load i32, i32 addrspace(2)* %[[GEP]]
; CHECK-NOT: alloca
; CHECK-NOT: call void @llvm.memcpy.p0i8.p2i8.i64
; CHECK-NOT: addrspacecast
; CHECK-NOT: load i32, i32*
define void @test_load_bitcast_chain(i32 addrspace(1)* %out, i64 %x) {
entry:
  %data = alloca [8 x i32], align 4
  %0 = bitcast [8 x i32]* %data to i8*
  call void @llvm.memcpy.p0i8.p2i8.i64(i8* %0, i8 addrspace(2)* bitcast ([8 x i32] addrspace(2)* @test.data to i8 addrspace(2)*), i64 32, i32 4, i1 false)
  %1 = bitcast i8* %0 to i32*
  %arrayidx = getelementptr inbounds i32, i32* %1, i64 %x
  %2 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %x
  store i32 %2, i32 addrspace(1)* %arrayidx1, align 4
  ret void
}

; CHECK-LABEL: test_call
; CHECK: alloca
; CHECK: call void @llvm.memcpy.p0i8.p2i8.i64
; CHECK-NOT: addrspacecast
; CHECK: call i32 @foo(i32* %{{.*}})
define void @test_call(i32 addrspace(1)* %out, i64 %x) {
entry:
  %data = alloca [8 x i32], align 4
  %0 = bitcast [8 x i32]* %data to i8*
  call void @llvm.memcpy.p0i8.p2i8.i64(i8* %0, i8 addrspace(2)* bitcast ([8 x i32] addrspace(2)* @test.data to i8 addrspace(2)*), i64 32, i32 4, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* %data, i64 0, i64 %x
  %1 = call i32 @foo(i32* %arrayidx)
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %x
  store i32 %1, i32 addrspace(1)* %arrayidx1, align 4
  ret void
}

; CHECK-LABEL: test_load_and_call
; CHECK: alloca
; CHECK: call void @llvm.memcpy.p0i8.p2i8.i64
; CHECK: load i32, i32* %{{.*}}
; CHECK: call i32 @foo(i32* %{{.*}})
; CHECK-NOT: addrspacecast
; CHECK-NOT: load i32, i32 addrspace(2)*
define void @test_load_and_call(i32 addrspace(1)* %out, i64 %x, i64 %y) {
entry:
  %data = alloca [8 x i32], align 4
  %0 = bitcast [8 x i32]* %data to i8*
  call void @llvm.memcpy.p0i8.p2i8.i64(i8* %0, i8 addrspace(2)* bitcast ([8 x i32] addrspace(2)* @test.data to i8 addrspace(2)*), i64 32, i32 4, i1 false)
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* %data, i64 0, i64 %x
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %x
  store i32 %1, i32 addrspace(1)* %arrayidx1, align 4
  %2 = call i32 @foo(i32* %arrayidx)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %y
  store i32 %2, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}


declare void @llvm.memcpy.p0i8.p2i8.i64(i8* nocapture writeonly, i8 addrspace(2)* nocapture readonly, i64, i32, i1)
declare i32 @foo(i32* %x)
