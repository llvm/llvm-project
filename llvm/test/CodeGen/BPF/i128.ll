; RUN: llc -march=bpfel -o - %s | FileCheck %s
; RUN: llc -march=bpfeb -o - %s | FileCheck %s
; Source code:
;   struct ipv6_key_t {
;     unsigned pid;
;     unsigned __int128 saddr;
;     unsigned short lport;
;   };
;
;   extern void test1(void *);
;   int test(int pid) {
;     struct ipv6_key_t ipv6_key = {.pid = pid};
;     test1(&ipv6_key);
;     return 0;
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm t.c

%struct.ipv6_key_t = type { i32, i128, i16 }

; Function Attrs: nounwind
define dso_local i32 @test(i32 %pid) local_unnamed_addr #0 {
entry:
  %ipv6_key = alloca %struct.ipv6_key_t, align 16
  %0 = bitcast %struct.ipv6_key_t* %ipv6_key to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %0) #4
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 dereferenceable(48) %0, i8 0, i64 48, i1 false)
  %pid1 = getelementptr inbounds %struct.ipv6_key_t, %struct.ipv6_key_t* %ipv6_key, i64 0, i32 0
  store i32 %pid, i32* %pid1, align 16, !tbaa !2
  call void @test1(i8* nonnull %0) #4
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %0) #4
  ret i32 0
}

; CHECK-LABEL: test
; CHECK:       *(u64 *)(r10 - 48) = r{{[0-9]+}}
; CHECK:       *(u32 *)(r10 - 48) = r{{[0-9]+}}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #2

declare dso_local void @test1(i8*) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { argmemonly nounwind willreturn writeonly }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

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
