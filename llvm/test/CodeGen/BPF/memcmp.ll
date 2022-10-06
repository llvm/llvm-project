; RUN: llc -march=bpfel < %s | FileCheck %s
; RUN: llc -march=bpfel -mcpu=v3 < %s | FileCheck %s
;
; Source code:
;   /* set aligned 4 to minimize the number of loads */
;   struct build_id {
;     unsigned char id[20];
;   } __attribute__((aligned(4)));
;
;   /* try to compute a local build_id */
;   void bar1(void *);
;
;   /* the global build_id to compare */
;   struct build_id id2;
;
;   int foo()
;   {
;     struct build_id id1;
;
;     bar1(&id1);
;     return __builtin_memcmp(&id1, &id2, sizeof(id1)) == 0;
;   }
; Compilation flags:
;   clang -target bpf -S -O2 t.c -emit-llvm


%struct.build_id = type { [20 x i8] }

@id2 = dso_local global %struct.build_id zeroinitializer, align 4

; Function Attrs: nounwind
define dso_local i32 @foo() local_unnamed_addr #0 {
entry:
  %id11 = alloca [20 x i8], align 4
  %id11.sub = getelementptr inbounds [20 x i8], [20 x i8]* %id11, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 20, i8* nonnull %id11.sub) #4
  call void @bar1(i8* noundef nonnull %id11.sub) #4
  %call = call i32 @memcmp(i8* noundef nonnull dereferenceable(20) %id11.sub, i8* noundef nonnull dereferenceable(20) getelementptr inbounds (%struct.build_id, %struct.build_id* @id2, i64 0, i32 0, i64 0), i64 noundef 20) #4
  %cmp = icmp eq i32 %call, 0
  %conv = zext i1 %cmp to i32
  call void @llvm.lifetime.end.p0i8(i64 20, i8* nonnull %id11.sub) #4
  ret i32 %conv
}

; CHECK-DAG:   *(u32 *)(r1 + 0)
; CHECK-DAG:   *(u32 *)(r1 + 4)
; CHECK-DAG:   *(u32 *)(r10 - 16)
; CHECK-DAG:   *(u32 *)(r10 - 20)
; CHECK-DAG:   *(u32 *)(r10 - 8)
; CHECK-DAG:   *(u32 *)(r10 - 12)
; CHECK-DAG:   *(u32 *)(r1 + 8)
; CHECK-DAG:   *(u32 *)(r1 + 12)
; CHECK-DAG:   *(u32 *)(r2 + 16)
; CHECK-DAG:   *(u32 *)(r10 - 4)

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local void @bar1(i8* noundef) local_unnamed_addr #2

; Function Attrs: argmemonly mustprogress nofree nounwind readonly willreturn
declare dso_local i32 @memcmp(i8* nocapture noundef, i8* nocapture noundef, i64 noundef) local_unnamed_addr #3

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { argmemonly mustprogress nofree nounwind readonly willreturn "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git dea65874b2505f8f5e8e51fd8cad6908feb375ec)"}
