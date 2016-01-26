; RUN: llc < %s -mtriple=i386-apple-darwin -relocation-model=pic -disable-fp-elim | FileCheck %s -check-prefix=X86-32
; RUN: llc < %s -mtriple=x86_64-apple-darwin -relocation-model=pic -disable-fp-elim | FileCheck %s -check-prefix=X86-64

; MachineLICM should be able to hoist loop invariant reload out of the loop.
; Only linear scan needs this, -regalloc=greedy sinks the spill instead.
; rdar://7233099

%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { i8*, i32 }
%struct.epoch_t = type { %struct.trans_t*, %struct.trans_t*, i32, i32, i32, i32, i32 }
%struct.trans_t = type { i32, i32, i32, i8* }

@.str12 = external constant [2 x i8], align 1     ; <[2 x i8]*> [#uses=1]
@.str19 = external constant [7 x i8], align 1     ; <[7 x i8]*> [#uses=1]
@.str24 = external constant [4 x i8], align 1     ; <[4 x i8]*> [#uses=1]

define i32 @t1(i32 %c, i8** nocapture %v) nounwind ssp {
; X86-32-LABEL: t1:
entry:
  br i1 undef, label %bb, label %bb3

bb:                                               ; preds = %entry
  unreachable

bb3:                                              ; preds = %entry
  br i1 undef, label %bb.i, label %bb.nph41

bb.i:                                             ; preds = %bb3
  unreachable

bb.nph41:                                         ; preds = %bb3
  %0 = call %struct.FILE* @"\01_fopen$UNIX2003"(i8* undef, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str12, i32 0, i32 0)) nounwind ; <%struct.FILE*> [#uses=3]
  br i1 undef, label %bb4, label %bb5.preheader

bb5.preheader:                                    ; preds = %bb.nph41
  br label %bb5

bb4:                                              ; preds = %bb.nph41
  unreachable

bb5:                                              ; preds = %bb5, %bb5.preheader
  br i1 undef, label %bb7, label %bb5

bb7:                                              ; preds = %bb5
  br i1 undef, label %bb9, label %bb12

bb9:                                              ; preds = %bb7
  unreachable

bb12:                                             ; preds = %bb7
  br i1 undef, label %bb16, label %bb22

bb16:                                             ; preds = %bb12
  unreachable

bb22:                                             ; preds = %bb12
  br label %bb.i1

bb.i1:                                            ; preds = %bb.i1, %bb22
  %1 = icmp eq i8 undef, 69                       ; <i1> [#uses=1]
  br i1 %1, label %imix_test.exit, label %bb.i1

imix_test.exit:                                   ; preds = %bb.i1
  br i1 undef, label %bb23, label %bb26.preheader

bb26.preheader:                                   ; preds = %imix_test.exit
  br i1 undef, label %bb28, label %bb30

bb23:                                             ; preds = %imix_test.exit
  unreachable
; Verify that there are no loads inside the loop.
; X86-32: .p2align 4
; X86-32: %bb28
; X86-32-NOT: (%esp),
; X86-32-NOT: (%ebp),
; X86-32: jmp

bb28:                                             ; preds = %bb28, %bb26.preheader
  %counter.035 = phi i32 [ %3, %bb28 ], [ 0, %bb26.preheader ] ; <i32> [#uses=2]
  %tmp56 = shl i32 %counter.035, 2                ; <i32> [#uses=0]
  %2 = call i8* @fgets(i8* undef, i32 50, %struct.FILE* %0) nounwind ; <i8*> [#uses=0]
  %3 = add nsw i32 %counter.035, 1                ; <i32> [#uses=1]
  %4 = call i32 @feof(%struct.FILE* %0) nounwind  ; <i32> [#uses=0]
  br label %bb28

bb30:                                             ; preds = %bb26.preheader
  %5 = call i32 @strcmp(i8* undef, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str19, i32 0, i32 0)) nounwind readonly ; <i32> [#uses=0]
  br i1 undef, label %bb34, label %bb70

bb32.loopexit:                                    ; preds = %bb45
  %6 = icmp eq i32 undef, 0                       ; <i1> [#uses=1]
  %indvar.next55 = add i32 %indvar54, 1           ; <i32> [#uses=1]
  br i1 %6, label %bb34, label %bb70

bb34:                                             ; preds = %bb32.loopexit, %bb30
  %indvar54 = phi i32 [ %indvar.next55, %bb32.loopexit ], [ 0, %bb30 ] ; <i32> [#uses=3]
  br i1 false, label %bb35, label %bb39.preheader

bb35:                                             ; preds = %bb34
  unreachable

bb39.preheader:                                   ; preds = %bb34
  %7 = getelementptr inbounds %struct.epoch_t, %struct.epoch_t* undef, i32 %indvar54, i32 3 ; <i32*> [#uses=1]
  %8 = getelementptr inbounds %struct.epoch_t, %struct.epoch_t* undef, i32 %indvar54, i32 2 ; <i32*> [#uses=0]
  br i1 false, label %bb42, label %bb45

bb42:                                             ; preds = %bb39.preheader
  unreachable

bb45:                                             ; preds = %bb39.preheader
  %9 = call i32 @strcmp(i8* undef, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str24, i32 0, i32 0)) nounwind readonly ; <i32> [#uses=0]
  br i1 false, label %bb47, label %bb32.loopexit

bb47:                                             ; preds = %bb45
  %10 = load i32, i32* %7, align 4                     ; <i32> [#uses=0]
  unreachable

bb70:                                             ; preds = %bb32.loopexit, %bb30
  br i1 undef, label %bb78, label %bb76

bb76:                                             ; preds = %bb70
  unreachable

bb78:                                             ; preds = %bb70
  br i1 undef, label %bb83, label %bb79

bb79:                                             ; preds = %bb78
  unreachable

bb83:                                             ; preds = %bb78
  call void @rewind(%struct.FILE* %0) nounwind
  unreachable
}

declare %struct.FILE* @"\01_fopen$UNIX2003"(i8*, i8*)

declare i8* @fgets(i8*, i32, %struct.FILE* nocapture) nounwind

declare void @rewind(%struct.FILE* nocapture) nounwind

declare i32 @feof(%struct.FILE* nocapture) nounwind

declare i32 @strcmp(i8* nocapture, i8* nocapture) nounwind readonly

@map_4_to_16 = external constant [16 x i16], align 32 ; <[16 x i16]*> [#uses=2]

define void @t2(i8* nocapture %bufp, i8* nocapture %data, i32 %dsize) nounwind ssp {
; X86-64-LABEL: t2:
entry:
  br i1 undef, label %return, label %bb.nph

bb.nph:                                           ; preds = %entry
; X86-64: movq _map_4_to_16@GOTPCREL(%rip)
; X86-64: .p2align 4
  %tmp5 = zext i32 undef to i64                   ; <i64> [#uses=1]
  %tmp6 = add i64 %tmp5, 1                        ; <i64> [#uses=1]
  %tmp11 = shl i64 undef, 1                       ; <i64> [#uses=1]
  %tmp14 = mul i64 undef, 3                       ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %tmp9 = mul i64 undef, undef                    ; <i64> [#uses=2]
  %tmp12 = add i64 %tmp11, %tmp9                  ; <i64> [#uses=1]
  %scevgep13 = getelementptr i8, i8* %bufp, i64 %tmp12 ; <i8*> [#uses=1]
  %tmp15 = add i64 %tmp14, %tmp9                  ; <i64> [#uses=1]
  %scevgep16 = getelementptr i8, i8* %bufp, i64 %tmp15 ; <i8*> [#uses=1]
  %0 = load i8, i8* undef, align 1                    ; <i8> [#uses=1]
  %1 = zext i8 %0 to i32                          ; <i32> [#uses=1]
  %2 = getelementptr inbounds [16 x i16], [16 x i16]* @map_4_to_16, i64 0, i64 0 ; <i16*> [#uses=1]
  %3 = load i16, i16* %2, align 2                      ; <i16> [#uses=1]
  %4 = trunc i16 %3 to i8                         ; <i8> [#uses=1]
  store i8 %4, i8* undef, align 1
  %5 = and i32 %1, 15                             ; <i32> [#uses=1]
  %6 = zext i32 %5 to i64                         ; <i64> [#uses=1]
  %7 = getelementptr inbounds [16 x i16], [16 x i16]* @map_4_to_16, i64 0, i64 %6 ; <i16*> [#uses=1]
  %8 = load i16, i16* %7, align 2                      ; <i16> [#uses=2]
  %9 = lshr i16 %8, 8                             ; <i16> [#uses=1]
  %10 = trunc i16 %9 to i8                        ; <i8> [#uses=1]
  store i8 %10, i8* %scevgep13, align 1
  %11 = trunc i16 %8 to i8                        ; <i8> [#uses=1]
  store i8 %11, i8* %scevgep16, align 1
  %exitcond = icmp eq i64 undef, %tmp6            ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}
