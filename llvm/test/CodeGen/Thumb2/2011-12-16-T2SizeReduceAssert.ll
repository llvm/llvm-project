; RUN: llc < %s -mtriple=thumbv7-apple-ios -relocation-model=pic -frame-pointer=all -mcpu=cortex-a8
; RUN: llc < %s -mtriple=thumbv8-none-linux-gnueabi

%struct.LIST_NODE.0.16 = type { %struct.LIST_NODE.0.16*, i8* }

define %struct.LIST_NODE.0.16* @list_AssocListPair(%struct.LIST_NODE.0.16* %List, i8* %Key) nounwind readonly {
entry:
  br label %bb3

bb:                                               ; preds = %bb3
  %Scan.0.idx7.val = load i8*, i8** undef, align 4
  %.idx = getelementptr i8, i8* %Scan.0.idx7.val, i32 4
  %0 = bitcast i8* %.idx to i8**
  %.idx.val = load i8*, i8** %0, align 4
  %1 = icmp eq i8* %.idx.val, %Key
  br i1 %1, label %bb5, label %bb2

bb2:                                              ; preds = %bb
  %Scan.0.idx8.val = load %struct.LIST_NODE.0.16*, %struct.LIST_NODE.0.16** undef, align 4
  br label %bb3

bb3:                                              ; preds = %bb2, %entry
  %Scan.0 = phi %struct.LIST_NODE.0.16* [ %List, %entry ], [ %Scan.0.idx8.val, %bb2 ]
  %2 = icmp eq %struct.LIST_NODE.0.16* %Scan.0, null
  br i1 %2, label %bb5, label %bb

bb5:                                              ; preds = %bb3, %bb
  ret %struct.LIST_NODE.0.16* null
}

declare void @use(i32)
define double @find_max_double(i32 %n, double* nocapture readonly %aa) {
entry:
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  %0 = load double, double* null, align 8
  %cmp2.6 = fcmp ogt double %0, 0.000000e+00
  %idx.1.6 = select i1 %cmp2.6, i32 undef, i32 0
  %idx.1.7 = select i1 undef, i32 undef, i32 %idx.1.6
  %max.1.7 = select i1 undef, double 0.000000e+00, double undef
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %max.0.lcssa = phi double [ undef, %entry ], [ %max.1.7, %for.body ]
  %idx.0.lcssa = phi i32 [ 0, %entry ], [ %idx.1.7, %for.body ]
  tail call void @use(i32 %idx.0.lcssa)
  ret double %max.0.lcssa
}
