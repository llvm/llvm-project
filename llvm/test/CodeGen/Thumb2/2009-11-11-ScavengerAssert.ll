; RUN: llc < %s -mtriple=thumbv7-apple-darwin10

%struct.OP = type { %struct.OP*, %struct.OP*, %struct.OP* ()*, i32, i16, i16, i8, i8 }
%struct.SV = type { i8*, i32, i32 }

declare void @Perl_mg_set(%struct.SV*) nounwind

define %struct.OP* @Perl_pp_complement() nounwind {
entry:
  %0 = load %struct.SV** null, align 4            ; <%struct.SV*> [#uses=2]
  br i1 undef, label %bb21, label %bb5

bb5:                                              ; preds = %entry
  br i1 undef, label %bb13, label %bb6

bb6:                                              ; preds = %bb5
  br i1 undef, label %bb8, label %bb7

bb7:                                              ; preds = %bb6
  %1 = getelementptr inbounds %struct.SV* %0, i32 0, i32 0 ; <i8**> [#uses=1]
  %2 = load i8** %1, align 4                      ; <i8*> [#uses=1]
  %3 = getelementptr inbounds i8* %2, i32 12      ; <i8*> [#uses=1]
  %4 = bitcast i8* %3 to i32*                     ; <i32*> [#uses=1]
  %5 = load i32* %4, align 4                      ; <i32> [#uses=1]
  %storemerge5 = xor i32 %5, -1                   ; <i32> [#uses=1]
  call  void @Perl_sv_setiv(%struct.SV* undef, i32 %storemerge5) nounwind
  %6 = getelementptr inbounds %struct.SV* undef, i32 0, i32 2 ; <i32*> [#uses=1]
  %7 = load i32* %6, align 4                      ; <i32> [#uses=1]
  %8 = and i32 %7, 16384                          ; <i32> [#uses=1]
  %9 = icmp eq i32 %8, 0                          ; <i1> [#uses=1]
  br i1 %9, label %bb12, label %bb11

bb8:                                              ; preds = %bb6
  unreachable

bb11:                                             ; preds = %bb7
  call  void @Perl_mg_set(%struct.SV* undef) nounwind
  br label %bb12

bb12:                                             ; preds = %bb11, %bb7
  store %struct.SV* undef, %struct.SV** null, align 4
  br label %bb44

bb13:                                             ; preds = %bb5
  %10 = call  i32 @Perl_sv_2uv(%struct.SV* %0) nounwind ; <i32> [#uses=0]
  br i1 undef, label %bb.i, label %bb1.i

bb.i:                                             ; preds = %bb13
  call  void @Perl_sv_setiv(%struct.SV* undef, i32 undef) nounwind
  br label %Perl_sv_setuv.exit

bb1.i:                                            ; preds = %bb13
  br label %Perl_sv_setuv.exit

Perl_sv_setuv.exit:                               ; preds = %bb1.i, %bb.i
  %11 = getelementptr inbounds %struct.SV* undef, i32 0, i32 2 ; <i32*> [#uses=1]
  %12 = load i32* %11, align 4                    ; <i32> [#uses=1]
  %13 = and i32 %12, 16384                        ; <i32> [#uses=1]
  %14 = icmp eq i32 %13, 0                        ; <i1> [#uses=1]
  br i1 %14, label %bb20, label %bb19

bb19:                                             ; preds = %Perl_sv_setuv.exit
  call  void @Perl_mg_set(%struct.SV* undef) nounwind
  br label %bb20

bb20:                                             ; preds = %bb19, %Perl_sv_setuv.exit
  store %struct.SV* undef, %struct.SV** null, align 4
  br label %bb44

bb21:                                             ; preds = %entry
  br i1 undef, label %bb23, label %bb22

bb22:                                             ; preds = %bb21
  unreachable

bb23:                                             ; preds = %bb21
  unreachable

bb44:                                             ; preds = %bb20, %bb12
  ret %struct.OP* undef
}

declare void @Perl_sv_setiv(%struct.SV*, i32) nounwind

declare i32 @Perl_sv_2uv(%struct.SV*) nounwind
