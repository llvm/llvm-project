; RUN: llc -mtriple=bpfel -mattr=+alu32 < %s | FileCheck %s
; Source:
;   struct env_t {
;     unsigned data;
;     unsigned data_end;
;   };
;   extern int work(struct env_t *skb, unsigned offset);
;   int test(struct env_t *skb)
;   {
;     ptr cursor, *data_end;
;     struct env_t *srh, *ip;
;
;     data_end = (ptr)(long)skb->data_end;
;     cursor = (ptr)(long)skb->data;
;
;     ip = cursor; cursor += sizeof(*ip);
;     if ((ptr)ip + sizeof(*ip) > data_end)
;       return 0;
;
;     srh = cursor; cursor += sizeof(*srh);
;     if ((ptr)srh + sizeof(*srh) > data_end)
;       return 0;
;
;     return work(skb, (char *)srh - (char *)(long)skb->data);
;   }
; Compilation flag:
;   clang -target bpf -O2 -emit-llvm -S test.c

%struct.env_t = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @test(ptr %skb) local_unnamed_addr #0 {
entry:
  %data_end1 = getelementptr inbounds %struct.env_t, ptr %skb, i64 0, i32 1
  %0 = load i32, ptr %data_end1, align 4, !tbaa !2
  %conv = zext i32 %0 to i64
  %1 = inttoptr i64 %conv to ptr
  %2 = load i32, ptr %skb, align 4, !tbaa !7
  %conv2 = zext i32 %2 to i64
  %3 = inttoptr i64 %conv2 to ptr
  %add.ptr = getelementptr i8, ptr %3, i64 8
  %cmp = icmp ugt ptr %add.ptr, %1
  %add.ptr6 = getelementptr i8, ptr %3, i64 16
  %cmp7 = icmp ugt ptr %add.ptr6, %1
  %or.cond = or i1 %cmp, %cmp7
  br i1 %or.cond, label %cleanup, label %if.end10

if.end10:                                         ; preds = %entry
  %sub.ptr.lhs.cast = ptrtoint ptr %add.ptr to i64
  %4 = trunc i64 %sub.ptr.lhs.cast to i32
  %conv13 = sub i32 %4, %2
  %call = tail call i32 @work(ptr nonnull %skb, i32 %conv13) #2
  br label %cleanup

cleanup:                                          ; preds = %entry, %if.end10
  %retval.0 = phi i32 [ %call, %if.end10 ], [ 0, %entry ]
  ret i32 %retval.0
}

; CHECK: w{{[0-9]+}} = *(u32 *)(r{{[0-9]+}} + 0)
; CHECK-NOT: w{{[0-9]+}} = w{{[0-9]+}}

declare dso_local i32 @work(ptr, i32) local_unnamed_addr #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 016d3ce1f4b07ee3056f7c10fedb24c441c4870f)"}
!2 = !{!3, !4, i64 4}
!3 = !{!"env_t", !4, i64 0, !4, i64 4}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!3, !4, i64 0}
