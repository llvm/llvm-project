; RUN: llc < %s -mtriple=bpfel -mattr=+alu32 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=bpfeb -mattr=+alu32 -verify-machineinstrs | FileCheck %s
;
; Source Code:
;   struct t {
;     unsigned char a;
;     unsigned char b;
;     unsigned char c;
;   };
;   extern void foo(ptr);
;   int test() {
;     struct t v = {
;       .b = 2,
;     };
;     foo(&v);
;     return 0;
;   }
; Compilation flag:
;  clang -target bpf -O2 -S -emit-llvm t.c

%struct.t = type { i8, i8, i8 }

@__const.test.v = private unnamed_addr constant %struct.t { i8 0, i8 2, i8 0 }, align 1

; Function Attrs: nounwind
define dso_local i32 @test() local_unnamed_addr {
entry:
  %v1 = alloca [3 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 3, ptr nonnull %v1)
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 dereferenceable(3) %v1, ptr nonnull align 1 dereferenceable(3) @__const.test.v, i64 3, i1 false)
  call void @foo(ptr nonnull %v1)
  call void @llvm.lifetime.end.p0(i64 3, ptr nonnull %v1)
  ret i32 0
}
; CHECK-NOT:    w{{[0-9]+}} = *(u16 *)
; CHECK-NOT:    w{{[0-9]+}} = *(u8 *)
; CHECK:        *(u16 *)(r10 - 4) = w{{[0-9]+}}
; CHECK:        *(u8 *)(r10 - 2) = w{{[0-9]+}}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

declare dso_local void @foo(ptr) local_unnamed_addr

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
