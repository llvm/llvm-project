; RUN: llc -O1 -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu < %s
; RUN: llc -O1 -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff < %s

; Function Attrs: uwtable
define fastcc void @_Crash_Fn() unnamed_addr #0 {
entry-block:
  br label %_Label_0

_Label_0:                                         ; preds = %_Label_0, %entry-block
  %result.0138 = phi i128 [ %5, %_Label_0 ], [ 0, %entry-block ]
  %iter.sroa.0.0137 = phi i8* [ %0, %_Label_0 ], [ undef, %entry-block ]
  %0 = getelementptr inbounds i8, i8* %iter.sroa.0.0137, i64 1
  %1 = tail call { i128, i1 } @llvm.smul.with.overflow.i128(i128 %result.0138, i128 undef) #2
  %2 = extractvalue { i128, i1 } %1, 0
  %3 = tail call { i128, i1 } @llvm.sadd.with.overflow.i128(i128 %2, i128 0) #2
  %4 = extractvalue { i128, i1 } %3, 1
  %5 = extractvalue { i128, i1 } %3, 0
  %6 = icmp eq i8* %0, null
  br i1 %6, label %bb66.loopexit, label %_Label_0

bb66.loopexit:                                    ; preds = %_Label_0
  unreachable
}

; Function Attrs: nounwind readnone
declare { i128, i1 } @llvm.sadd.with.overflow.i128(i128, i128) #1

; Function Attrs: nounwind readnone
declare { i128, i1 } @llvm.smul.with.overflow.i128(i128, i128) #1

attributes #0 = { uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
