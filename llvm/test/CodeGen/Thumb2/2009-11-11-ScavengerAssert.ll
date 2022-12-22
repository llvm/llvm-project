; RUN: llc < %s -mtriple=thumbv7-apple-darwin10

%struct.OP = type { ptr, ptr, ptr, i32, i16, i16, i8, i8 }
%struct.SV = type { ptr, i32, i32 }

declare void @Perl_mg_set(ptr) nounwind

define ptr @Perl_pp_complement() nounwind {
entry:
  %0 = load ptr, ptr null, align 4            ; <ptr> [#uses=2]
  br i1 undef, label %bb21, label %bb5

bb5:                                              ; preds = %entry
  br i1 undef, label %bb13, label %bb6

bb6:                                              ; preds = %bb5
  br i1 undef, label %bb8, label %bb7

bb7:                                              ; preds = %bb6
  %1 = getelementptr inbounds %struct.SV, ptr %0, i32 0, i32 0 ; <ptr> [#uses=1]
  %2 = load ptr, ptr %1, align 4                      ; <ptr> [#uses=1]
  %3 = getelementptr inbounds i8, ptr %2, i32 12      ; <ptr> [#uses=1]
  %4 = load i32, ptr %3, align 4                      ; <i32> [#uses=1]
  %storemerge5 = xor i32 %4, -1                   ; <i32> [#uses=1]
  call  void @Perl_sv_setiv(ptr undef, i32 %storemerge5) nounwind
  %5 = getelementptr inbounds %struct.SV, ptr undef, i32 0, i32 2 ; <ptr> [#uses=1]
  %6 = load i32, ptr %5, align 4                      ; <i32> [#uses=1]
  %7 = and i32 %6, 16384                          ; <i32> [#uses=1]
  %8 = icmp eq i32 %7, 0                          ; <i1> [#uses=1]
  br i1 %8, label %bb12, label %bb11

bb8:                                              ; preds = %bb6
  unreachable

bb11:                                             ; preds = %bb7
  call  void @Perl_mg_set(ptr undef) nounwind
  br label %bb12

bb12:                                             ; preds = %bb11, %bb7
  store ptr undef, ptr null, align 4
  br label %bb44

bb13:                                             ; preds = %bb5
  %9 = call  i32 @Perl_sv_2uv(ptr %0) nounwind ; <i32> [#uses=0]
  br i1 undef, label %bb.i, label %bb1.i

bb.i:                                             ; preds = %bb13
  call  void @Perl_sv_setiv(ptr undef, i32 undef) nounwind
  br label %Perl_sv_setuv.exit

bb1.i:                                            ; preds = %bb13
  br label %Perl_sv_setuv.exit

Perl_sv_setuv.exit:                               ; preds = %bb1.i, %bb.i
  %10 = getelementptr inbounds %struct.SV, ptr undef, i32 0, i32 2 ; <ptr> [#uses=1]
  %11 = load i32, ptr %10, align 4                    ; <i32> [#uses=1]
  %12 = and i32 %11, 16384                        ; <i32> [#uses=1]
  %13 = icmp eq i32 %12, 0                        ; <i1> [#uses=1]
  br i1 %13, label %bb20, label %bb19

bb19:                                             ; preds = %Perl_sv_setuv.exit
  call  void @Perl_mg_set(ptr undef) nounwind
  br label %bb20

bb20:                                             ; preds = %bb19, %Perl_sv_setuv.exit
  store ptr undef, ptr null, align 4
  br label %bb44

bb21:                                             ; preds = %entry
  br i1 undef, label %bb23, label %bb22

bb22:                                             ; preds = %bb21
  unreachable

bb23:                                             ; preds = %bb21
  unreachable

bb44:                                             ; preds = %bb20, %bb12
  ret ptr undef
}

declare void @Perl_sv_setiv(ptr, i32) nounwind

declare i32 @Perl_sv_2uv(ptr) nounwind
