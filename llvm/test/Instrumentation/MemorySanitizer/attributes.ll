; RUN: opt < %s -S -passes='module(msan-module),function(msan)' 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


declare void @a_() sanitize_memory readnone
declare void @b_() sanitize_memory readonly
declare void @c_() sanitize_memory writeonly
declare void @d_(i32* %p) sanitize_memory writeonly argmemonly
declare void @e_() sanitize_memory speculatable

define void @a() sanitize_memory readnone {
entry:
  call void @a_()
  call void @a_() readnone
  ret void
}

define void @b() sanitize_memory readonly {
entry:
  call void @b_()
  call void @b_() readonly
  ret void
}

define void @c() sanitize_memory writeonly {
entry:
  call void @c_()
  call void @c_() writeonly
  ret void
}

define void @d(i32* %p) sanitize_memory writeonly argmemonly {
entry:
  call void @d_(i32* %p)
  call void @d_(i32* %p) writeonly argmemonly
  ret void
}

define void @e() sanitize_memory speculatable {
entry:
  call void @e_()
  ret void
}

; CHECK-NOT: readnone
; CHECK-NOT: readonly
; CHECK-NOT: writeonly
; CHECK-NOT: argmemonly
; CHECK-NOT: speculatable
