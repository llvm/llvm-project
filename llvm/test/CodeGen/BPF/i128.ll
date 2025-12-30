; RUN: llc -mtriple=bpfel -mcpu=v1 -o - %s | FileCheck %s
; RUN: llc -mtriple=bpfeb -mcpu=v1 -o - %s | FileCheck %s
; Source code:
;   struct ipv6_key_t {
;     unsigned pid;
;     unsigned __int128 saddr;
;     unsigned short lport;
;   };
;
;   extern void test1(ptr);
;   int test(int pid) {
;     struct ipv6_key_t ipv6_key = {.pid = pid};
;     test1(&ipv6_key);
;     return 0;
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm t.c

%struct.ipv6_key_t = type { i32, i128, i16 }

; Function Attrs: nounwind
define dso_local i32 @test(i32 %pid) local_unnamed_addr {
entry:
  %ipv6_key = alloca %struct.ipv6_key_t, align 16
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %ipv6_key)
  call void @llvm.memset.p0.i64(ptr nonnull align 16 dereferenceable(48) %ipv6_key, i8 0, i64 48, i1 false)
  store i32 %pid, ptr %ipv6_key, align 16, !tbaa !2
  call void @test1(ptr nonnull %ipv6_key)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %ipv6_key)
  ret i32 0
}

; CHECK-LABEL: test
; CHECK:       *(u64 *)(r10 - 48) = r{{[0-9]+}}
; CHECK:       *(u32 *)(r10 - 48) = r{{[0-9]+}}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

; Function Attrs: argmemonly nounwind willreturn writeonly
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

declare dso_local void @test1(ptr) local_unnamed_addr

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 55fc7a47f8f18f84b44ff16f4e7a420c0a42ddf1)"}
!2 = !{!3, !4, i64 0}
!3 = !{!"ipv6_key_t", !4, i64 0, !7, i64 16, !8, i64 32}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!"__int128", !5, i64 0}
!8 = !{!"short", !5, i64 0}
