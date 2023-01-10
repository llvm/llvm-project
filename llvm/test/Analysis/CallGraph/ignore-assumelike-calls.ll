; RUN: opt < %s -passes=print-callgraph -disable-output 2>&1 | FileCheck %s
; CHECK: Call graph node <<null function>><<{{.*}}>>  #uses=0
; CHECK-NEXT:   CS<None> calls function 'other_intrinsic_use'
; CHECK-NEXT:   CS<None> calls function 'other_cast_intrinsic_use'
; CHECK-NEXT:   CS<None> calls function 'llvm.lifetime.start.p0'
; CHECK-NEXT:   CS<None> calls function 'llvm.memset.p0.i64'
; CHECK-NEXT:   CS<None> calls function 'llvm.memset.p1.i64'
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'addrspacecast_only'<<{{.*}}>>  #uses=0
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'bitcast_only'<<{{.*}}>>  #uses=0
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'llvm.lifetime.start.p0'<<{{.*}}>>  #uses=3
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'llvm.memset.p0.i64'<<{{.*}}>>  #uses=2
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'llvm.memset.p1.i64'<<{{.*}}>>  #uses=2
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'other_cast_intrinsic_use'<<{{.*}}>>  #uses=1
; CHECK-NEXT:   CS<{{.*}}> calls function 'llvm.memset.p1.i64'
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'other_intrinsic_use'<<{{.*}}>>  #uses=1
; CHECK-NEXT:   CS<{{.*}}> calls function 'llvm.memset.p0.i64'
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'used_by_lifetime'<<{{.*}}>>  #uses=0
; CHECK-NEXT:   CS<{{.*}}> calls function 'llvm.lifetime.start.p0'
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'used_by_lifetime_cast'<<{{.*}}>>  #uses=0
; CHECK-NEXT:   CS<{{.*}}> calls function 'llvm.lifetime.start.p0'
; CHECK-EMPTY:

define internal void @used_by_lifetime() {
entry:
  call void @llvm.lifetime.start.p0(i64 4, ptr @used_by_lifetime)
  ret void
}

define internal void @used_by_lifetime_cast() addrspace(1) {
  call void @llvm.lifetime.start.p0(i64 4, ptr addrspacecast (ptr addrspace(1) @used_by_lifetime_cast to ptr))
  ret void
}

define internal void @bitcast_only() {
entry:
  %c = bitcast ptr @bitcast_only to ptr
  ret void
}

define internal void @addrspacecast_only() addrspace(1) {
entry:
  %c = addrspacecast ptr addrspace(1) @addrspacecast_only to ptr
  ret void
}

define internal void @other_intrinsic_use() {
  call void @llvm.memset.p0.i64(ptr @other_intrinsic_use, i8 0, i64 1024, i1 false)
  ret void
}

define internal void @other_cast_intrinsic_use() {
  call void @llvm.memset.p1.i64(ptr addrspace(1) addrspacecast (ptr @other_cast_intrinsic_use to ptr addrspace(1)), i8 0, i64 1024, i1 false)
  ret void
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)
declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1 immarg)
