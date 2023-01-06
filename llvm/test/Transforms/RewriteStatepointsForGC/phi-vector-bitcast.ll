; REQUIRES: asserts
; RUN: opt < %s -disable-output -passes=rewrite-statepoints-for-gc

; We shouldn't crash when we encounter a vector phi with more than one input
; from the same predecessor.
define void @foo(<2 x ptr addrspace(1)> %arg1, i32 %arg2, i1 %arg3, <2 x ptr addrspace(1)> %arg4) gc "statepoint-example" personality ptr null {
bb:
  %tmp = bitcast <2 x ptr addrspace(1)> %arg1 to <2 x ptr addrspace(1)>
  switch i32 %arg2, label %bb2 [
    i32 1, label %bb4
    i32 2, label %bb4
  ]

bb2:                                              ; preds = %bb
  br i1 %arg3, label %bb8, label %bb4

bb4:                                              ; preds = %bb2, %bb, %bb
  %tmp5 = phi <2 x ptr addrspace(1)> [ %tmp, %bb ], [ %tmp, %bb ], [ %arg4, %bb2 ]
  call void @bar()
  %tmp6 = extractelement <2 x ptr addrspace(1)> %tmp5, i32 1
  ret void

bb8:                                              ; preds = %bb2
  ret void
}

declare void @bar()
