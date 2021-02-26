; RUN: opt < %s -print-callgraph -disable-output 2>&1 | FileCheck %s
; CHECK: Call graph node <<null function>><<{{.*}}>>  #uses=0
; CHECK-NEXT:   CS<None> calls function 'cast_only'
; CHECK-NEXT:   CS<None> calls function 'llvm.lifetime.start.p0i8'
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'cast_only'<<{{.*}}>>  #uses=1
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'llvm.lifetime.start.p0i8'<<{{.*}}>>  #uses=1
; CHECK-EMPTY:
; CHECK-NEXT:   Call graph node for function: 'used_by_lifetime'<<{{.*}}>>  #uses=0
; CHECK-EMPTY:

define internal void @used_by_lifetime() {
entry:
  %c = bitcast void()* @used_by_lifetime to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %c)
  ret void
}

define internal void @cast_only() {
entry:
  %c = bitcast void()* @cast_only to i8*
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
