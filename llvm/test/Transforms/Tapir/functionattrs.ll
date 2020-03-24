; Check that functionattrs derives the correct memory-access
; properties for a function that uses Tapir instructions.  In this
; case, functionattrs should recognize @fib() as readnone.
;
; RUN: opt < %s -functionattrs -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define i32 @fib(i32 %n) local_unnamed_addr #0 {
; CHECK: define i32 @fib(i32 %n) local_unnamed_addr #[[ATTRNUM:[0-9]+]]
entry:
  %x = alloca i32, align 4
  %syncreg = call token @llvm.syncregion.start()
  %cmp = icmp slt i32 %n, 2
  br i1 %cmp, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  %x.0..sroa_cast = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %x.0..sroa_cast)
  detach within %syncreg, label %det.achd, label %det.cont

det.achd:                                         ; preds = %if.end
  %sub = add nsw i32 %n, -1
  %call = call i32 @fib(i32 %sub)
  store i32 %call, i32* %x, align 4
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %if.end
  %sub1 = add nsw i32 %n, -2
  %call2 = call i32 @fib(i32 %sub1)
  sync within %syncreg, label %sync.continue

sync.continue:                                    ; preds = %det.cont
  %x.0.load = load i32, i32* %x, align 4
  %add = add nsw i32 %x.0.load, %call2
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %x.0..sroa_cast)
  br label %cleanup

cleanup:                                          ; preds = %entry, %sync.continue
  %retval.0 = phi i32 [ %add, %sync.continue ], [ %n, %entry ]
  ret i32 %retval.0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

; CHECK: attributes #[[ATTRNUM]] = {
; CHECK: readnone
; CHECK: attributes
; CHECK-NOT: attributes

