; RUN: llc < %s -mtriple=thumbv7-apple-ios -relocation-model=pic -disable-fp-elim -mcpu=cortex-a8 

%struct.LIST_NODE.0.16 = type { %struct.LIST_NODE.0.16*, i8* }

define %struct.LIST_NODE.0.16* @list_AssocListPair(%struct.LIST_NODE.0.16* %List, i8* %Key) nounwind readonly {
entry:
  br label %bb3

bb:                                               ; preds = %bb3
  %Scan.0.idx7.val = load i8** undef, align 4
  %.idx = getelementptr i8* %Scan.0.idx7.val, i32 4
  %0 = bitcast i8* %.idx to i8**
  %.idx.val = load i8** %0, align 4
  %1 = icmp eq i8* %.idx.val, %Key
  br i1 %1, label %bb5, label %bb2

bb2:                                              ; preds = %bb
  %Scan.0.idx8.val = load %struct.LIST_NODE.0.16** undef, align 4
  br label %bb3

bb3:                                              ; preds = %bb2, %entry
  %Scan.0 = phi %struct.LIST_NODE.0.16* [ %List, %entry ], [ %Scan.0.idx8.val, %bb2 ]
  %2 = icmp eq %struct.LIST_NODE.0.16* %Scan.0, null
  br i1 %2, label %bb5, label %bb

bb5:                                              ; preds = %bb3, %bb
  ret %struct.LIST_NODE.0.16* null
}
