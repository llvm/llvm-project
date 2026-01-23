; REQUIRES: asserts
;
; RUN: llc -mtriple=hexagon -enable-pipeliner=true -debug-only=pipeliner < %s \
; RUN: 2>&1 -pipeliner-experimental-cg=true | FileCheck %s

; Test that the artificial dependence is created as a result of
; CopyToPhi DAG mutation.
; CHECK: Ord  Latency=0 Artificial
target triple = "hexagon"

; Function Attrs: nounwind
define void @foo(ptr nocapture readonly %r64, i16 zeroext %n, i16 zeroext %s, ptr nocapture %p64) #0 {
entry:
  %conv = zext i16 %n to i32
  %cmp = icmp eq i16 %n, 0
  br i1 %cmp, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %tmp = load i64, ptr %r64, align 8
  %v.sroa.0.0.extract.trunc = trunc i64 %tmp to i16
  %v.sroa.4.0.extract.shift = lshr i64 %tmp, 16
  %v.sroa.4.0.extract.trunc = trunc i64 %v.sroa.4.0.extract.shift to i16
  %v.sroa.5.0.extract.shift = lshr i64 %tmp, 32
  %v.sroa.5.0.extract.trunc = trunc i64 %v.sroa.5.0.extract.shift to i16
  %v.sroa.6.0.extract.shift = lshr i64 %tmp, 48
  %v.sroa.6.0.extract.trunc = trunc i64 %v.sroa.6.0.extract.shift to i16
  %conv2 = zext i16 %s to i32
  %add.ptr = getelementptr inbounds i16, ptr %p64, i32 %conv2
  %add.ptr.sum = add nuw nsw i32 %conv2, 1
  %add.ptr3 = getelementptr inbounds i16, ptr %p64, i32 %add.ptr.sum
  %add.ptr.sum50 = add nuw nsw i32 %conv2, 2
  %add.ptr4 = getelementptr inbounds i16, ptr %p64, i32 %add.ptr.sum50
  %add.ptr.sum51 = add nuw nsw i32 %conv2, 3
  %add.ptr5 = getelementptr inbounds i16, ptr %p64, i32 %add.ptr.sum51
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %add.ptr11.phi = phi ptr [ %add.ptr11.inc, %for.body ], [ %add.ptr, %for.body.preheader ]
  %add.ptr16.phi = phi ptr [ %add.ptr16.inc, %for.body ], [ %add.ptr3, %for.body.preheader ]
  %add.ptr21.phi = phi ptr [ %add.ptr21.inc, %for.body ], [ %add.ptr4, %for.body.preheader ]
  %add.ptr26.phi = phi ptr [ %add.ptr26.inc, %for.body ], [ %add.ptr5, %for.body.preheader ]
  %i.058.pmt = phi i32 [ %inc.pmt, %for.body ], [ 0, %for.body.preheader ]
  %v.sroa.0.157 = phi i16 [ %v.sroa.0.0.extract.trunc34, %for.body ], [ %v.sroa.0.0.extract.trunc, %for.body.preheader ]
  %v.sroa.4.156 = phi i16 [ %v.sroa.4.0.extract.trunc36, %for.body ], [ %v.sroa.4.0.extract.trunc, %for.body.preheader ]
  %v.sroa.5.155 = phi i16 [ %v.sroa.5.0.extract.trunc38, %for.body ], [ %v.sroa.5.0.extract.trunc, %for.body.preheader ]
  %v.sroa.6.154 = phi i16 [ %v.sroa.6.0.extract.trunc40, %for.body ], [ %v.sroa.6.0.extract.trunc, %for.body.preheader ]
  %q64.153.pn = phi ptr [ %q64.153, %for.body ], [ %r64, %for.body.preheader ]
  %q64.153 = getelementptr inbounds i64, ptr %q64.153.pn, i32 1
  store i16 %v.sroa.0.157, ptr %add.ptr11.phi, align 2
  store i16 %v.sroa.4.156, ptr %add.ptr16.phi, align 2
  store i16 %v.sroa.5.155, ptr %add.ptr21.phi, align 2
  store i16 %v.sroa.6.154, ptr %add.ptr26.phi, align 2
  %tmp2 = load i64, ptr %q64.153, align 8
  %v.sroa.0.0.extract.trunc34 = trunc i64 %tmp2 to i16
  %v.sroa.4.0.extract.shift35 = lshr i64 %tmp2, 16
  %v.sroa.4.0.extract.trunc36 = trunc i64 %v.sroa.4.0.extract.shift35 to i16
  %v.sroa.5.0.extract.shift37 = lshr i64 %tmp2, 32
  %v.sroa.5.0.extract.trunc38 = trunc i64 %v.sroa.5.0.extract.shift37 to i16
  %v.sroa.6.0.extract.shift39 = lshr i64 %tmp2, 48
  %v.sroa.6.0.extract.trunc40 = trunc i64 %v.sroa.6.0.extract.shift39 to i16
  %inc.pmt = add i32 %i.058.pmt, 1
  %cmp8 = icmp slt i32 %inc.pmt, %conv
  %add.ptr11.inc = getelementptr i16, ptr %add.ptr11.phi, i32 4
  %add.ptr16.inc = getelementptr i16, ptr %add.ptr16.phi, i32 4
  %add.ptr21.inc = getelementptr i16, ptr %add.ptr21.phi, i32 4
  %add.ptr26.inc = getelementptr i16, ptr %add.ptr26.phi, i32 4
  br i1 %cmp8, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv65" }
