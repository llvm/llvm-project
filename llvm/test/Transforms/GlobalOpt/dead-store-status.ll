; RUN: opt < %s -globalopt -S | FileCheck %s

; When removing the store to @global in @foo, the pass would incorrectly return
; false. This was caught by the pass return status check that is hidden under
; EXPENSIVE_CHECKS.

; CHECK: @global = internal unnamed_addr global i16* null, align 1

; CHECK-LABEL: @foo
; CHECK-NEXT: entry:
; CHECK-NEXT: ret i16 undef

@global = internal unnamed_addr global i16* null, align 1

; Function Attrs: nofree noinline norecurse nounwind writeonly
define i16 @foo(i16 %c) local_unnamed_addr #0 {
entry:
  %local1.addr = alloca i16, align 1
  store i16* %local1.addr, i16** @global, align 1
  ret i16 undef
}

; Function Attrs: noinline nounwind writeonly
define i16 @bar() local_unnamed_addr #1 {
entry:
  %local2 = alloca [1 x i16], align 1
  %0 = bitcast [1 x i16]* %local2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [1 x i16], [1 x i16]* %local2, i16 0, i16 0
  store i16* %arraydecay, i16** @global, align 1
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %0)
  ret i16 undef
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

attributes #0 = { nofree noinline norecurse nounwind writeonly }
attributes #1 = { noinline nounwind writeonly }
attributes #2 = { argmemonly nounwind willreturn }
