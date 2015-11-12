; RUN: opt %loadPolly -polly-codegen < %s
;
; Check we do not crash even though the dead %tmp8 is referenced by a parameter
; and we do not pre-load it (as it is dead).
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.hoge = type { %struct.widget*, i32, i32, i32, %struct.hoge*, %struct.hoge*, %struct.barney, %struct.ham, %struct.wombat }
%struct.widget = type { i32, i32, i32, i32, %struct.quux* }
%struct.quux = type { i8*, i32, i32, i32, %struct.hoge.0 }
%struct.hoge.0 = type { [2 x i64] }
%struct.barney = type { %struct.hoge* }
%struct.ham = type { i32 }
%struct.wombat = type { %struct.hoge** }
%struct.foo = type { %struct.wibble*, %struct.wibble*, i32, i32, i32, %struct.hoge.2, %struct.blam, %struct.wombat.5, i16, i8*, i8*, i16, i8*, i16, i16*, i16, %struct.blam.6**, i16, %struct.foo.7**, i8*, i16, i8**, i8* }
%struct.wibble = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct.foo.1*, %struct.wibble*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct.foo.1 = type { %struct.foo.1*, %struct.wibble*, i32 }
%struct.hoge.2 = type { i16, i16 }
%struct.blam = type { i16, %struct.barney.3* }
%struct.barney.3 = type { i8, %struct.foo.4 }
%struct.foo.4 = type { i64 }
%struct.wombat.5 = type { i16 }
%struct.blam.6 = type <{ %struct.wombat.5, [6 x i8], i8*, i8*, i32, i16, [2 x i8] }>
%struct.foo.7 = type { %struct.wombat.5, i8*, i8*, i8, i8, i32, i8*, i16, %struct.bar*, i16, %struct.barney.9*, i16, %struct.hoge.10*, i8**, i8**, i32*, i32*, i8*, i32, i32, i32* }
%struct.bar = type { i32, i16, i16, %struct.wibble.8, i16, %struct.hoge* }
%struct.wibble.8 = type { i32 }
%struct.barney.9 = type { i16, i16 }
%struct.hoge.10 = type { i16, i16, i16, i16, i16 }
%struct.bar.11 = type { i64, i64 }

@global = external global i32, align 4
@global1 = external global i32, align 4
@global2 = external global i8*, align 8
@global3 = external global %struct.hoge**, align 8
@global4 = external global %struct.hoge**, align 8

; Function Attrs: uwtable
define i32 @foo(%struct.foo* %arg) #0 personality i8* bitcast (i32 (...)* @blam to i8*) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb
  %tmp = load i32, i32* @global, align 4, !tbaa !1
  %tmp4 = add i32 %tmp, -1
  %tmp5 = icmp eq i32 0, 0
  br i1 %tmp5, label %bb12, label %bb6

bb6:                                              ; preds = %bb3
  br label %bb7

bb7:                                              ; preds = %bb7, %bb6
  %tmp8 = load i32, i32* @global, align 4, !tbaa !1
  %tmp9 = and i32 %tmp8, 3
  %tmp10 = icmp eq i32 %tmp9, 0
  br i1 %tmp10, label %bb11, label %bb7

bb11:                                             ; preds = %bb7
  br label %bb12

bb12:                                             ; preds = %bb11, %bb3
  invoke void @zot(%struct.hoge* nonnull undef, i32 %tmp4, i32 undef, i32 9, i32 0, i32 39, %struct.hoge* undef, i32 undef, i32 undef, %struct.bar.11* nonnull undef)
          to label %bb13 unwind label %bb17

bb13:                                             ; preds = %bb12
  br i1 undef, label %bb16, label %bb14

bb14:                                             ; preds = %bb13
  br label %bb19

bb15:                                             ; preds = %bb19
  br label %bb16

bb16:                                             ; preds = %bb15, %bb13
  ret i32 0

bb17:                                             ; preds = %bb12
  %tmp18 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %tmp18

bb19:                                             ; preds = %bb19, %bb14
  br i1 undef, label %bb15, label %bb19
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) #2

; Function Attrs: nobuiltin
declare noalias i8* @eggs(i64) #3

; Function Attrs: nobuiltin
declare noalias i8* @bar(i64) #3

; Function Attrs: uwtable
declare void @zot(%struct.hoge*, i32, i32, i32, i32, i32, %struct.hoge*, i32, i32, %struct.bar.11*) unnamed_addr #0 align 2

declare i32 @blam(...)

; Function Attrs: nobuiltin nounwind
declare void @zot5(i8*) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

; Function Attrs: uwtable
declare i32 @eggs6(%struct.foo*) #0

; Function Attrs: nounwind uwtable
declare void @eggs7(%struct.widget*, i32, i32, i32) unnamed_addr #5 align 2

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nobuiltin "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nobuiltin nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 252700) (llvm/trunk 252705)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
