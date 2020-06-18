; RUN: llc < %s -march=bpfel -mattr=+alu32 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=bpfeb -mattr=+alu32 -verify-machineinstrs | FileCheck %s
;
; Source Code:
;   struct t {
;     unsigned char a;
;     unsigned char b;
;     unsigned char c;
;   };
;   extern void foo(void *);
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
  %v1.sub = getelementptr inbounds [3 x i8], [3 x i8]* %v1, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 3, i8* nonnull %v1.sub)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 dereferenceable(3) %v1.sub, i8* nonnull align 1 dereferenceable(3) getelementptr inbounds (%struct.t, %struct.t* @__const.test.v, i64 0, i32 0), i64 3, i1 false)
  call void @foo(i8* nonnull %v1.sub)
  call void @llvm.lifetime.end.p0i8(i64 3, i8* nonnull %v1.sub)
  ret i32 0
}
; CHECK-NOT:    w{{[0-9]+}} = *(u16 *)
; CHECK-NOT:    w{{[0-9]+}} = *(u8 *)
; CHECK:        *(u16 *)(r10 - 4) = w{{[0-9]+}}
; CHECK:        *(u8 *)(r10 - 2) = w{{[0-9]+}}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)

declare dso_local void @foo(i8*) local_unnamed_addr

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
