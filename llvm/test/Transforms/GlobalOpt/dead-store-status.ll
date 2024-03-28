; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; When removing the store to @global in @foo, the pass would incorrectly return
; false. This was caught by the pass return status check that is hidden under
; EXPENSIVE_CHECKS.

; CHECK-LABEL: @foo
; CHECK-NEXT: entry:
; CHECK-NEXT: ret i16 undef

@global = internal unnamed_addr global ptr null, align 1

; Function Attrs: nofree noinline norecurse nounwind writeonly
define i16 @foo(i16 %c) local_unnamed_addr #0 {
entry:
  %local1.addr = alloca i16, align 1
  store ptr %local1.addr, ptr @global, align 1
  ret i16 undef
}

; Function Attrs: noinline nounwind writeonly
define i16 @bar() local_unnamed_addr #1 {
entry:
  %local2 = alloca [1 x i16], align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %local2)
  store ptr %local2, ptr @global, align 1
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %local2)
  ret i16 undef
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

attributes #0 = { nofree noinline norecurse nounwind writeonly }
attributes #1 = { noinline nounwind writeonly }
attributes #2 = { argmemonly nounwind willreturn }
