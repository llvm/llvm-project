; RUN: llc < %s -mtriple=i386-apple-darwin11
; rdar://7604000

%struct.a_t = type { i8*, i64*, i8*, i32, i32, i64*, i64*, i64* }
%struct.b_t = type { i32, i32, i32, i32, i64, i64, i64, i64 }

define void @t(i32 %cNum, i64 %max) nounwind optsize ssp noimplicitfloat {
entry:
  %0 = load %struct.b_t** null, align 4 ; <%struct.b_t*> [#uses=1]
  %1 = getelementptr inbounds %struct.b_t* %0, i32 %cNum, i32 5 ; <i64*> [#uses=1]
  %2 = load i64* %1, align 4                      ; <i64> [#uses=1]
  %3 = icmp ult i64 %2, %max            ; <i1> [#uses=1]
  %4 = getelementptr inbounds %struct.a_t* null, i32 0, i32 7 ; <i64**> [#uses=1]
  %5 = load i64** %4, align 4                     ; <i64*> [#uses=0]
  %6 = load i64* null, align 4                    ; <i64> [#uses=1]
  br i1 %3, label %bb2, label %bb

bb:                                               ; preds = %entry
  br label %bb3

bb2:                                              ; preds = %entry
  %7 = or i64 %6, undef                           ; <i64> [#uses=1]
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %misc_enables.0 = phi i64 [ undef, %bb ], [ %7, %bb2 ] ; <i64> [#uses=0]
  ret void
}
