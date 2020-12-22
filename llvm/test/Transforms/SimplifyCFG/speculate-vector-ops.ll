; RUN: opt -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

define i32 @speculate_vector_extract(i32 %d, <4 x i32> %v) #0 {
; CHECK-LABEL: @speculate_vector_extract(
; CHECK-NOT: br
entry:
  %conv = insertelement <4 x i32> undef, i32 %d, i32 0
  %conv2 = insertelement <4 x i32> %conv, i32 %d, i32 1
  %conv3 = insertelement <4 x i32> %conv2, i32 %d, i32 2
  %conv4 = insertelement <4 x i32> %conv3, i32 %d, i32 3
  %tmp6 = add nsw <4 x i32> %conv4, <i32 0, i32 -1, i32 -2, i32 -3>
  %cmp = icmp eq <4 x i32> %tmp6, zeroinitializer
  %cmp.ext = sext <4 x i1> %cmp to <4 x i32>
  %tmp8 = extractelement <4 x i32> %cmp.ext, i32 0
  %tobool = icmp eq i32 %tmp8, 0
  br i1 %tobool, label %cond.else, label %cond.then

return:                                           ; preds = %cond.end28
  ret i32 %cond32

cond.then:                                        ; preds = %entry
  %tmp10 = extractelement <4 x i32> %v, i32 0
  br label %cond.end

cond.else:                                        ; preds = %entry
  %tmp12 = extractelement <4 x i32> %v, i32 3
  br label %cond.end

cond.end:                                         ; preds = %cond.else, %cond.then
  %cond = phi i32 [ %tmp10, %cond.then ], [ %tmp12, %cond.else ]
  %tmp14 = extractelement <4 x i32> %cmp.ext, i32 1
  %tobool15 = icmp eq i32 %tmp14, 0
  br i1 %tobool15, label %cond.else17, label %cond.then16

cond.then16:                                      ; preds = %cond.end
  %tmp20 = extractelement <4 x i32> %v, i32 1
  br label %cond.end18

cond.else17:                                      ; preds = %cond.end
  br label %cond.end18

cond.end18:                                       ; preds = %cond.else17, %cond.then16
  %cond22 = phi i32 [ %tmp20, %cond.then16 ], [ %cond, %cond.else17 ]
  %tmp24 = extractelement <4 x i32> %cmp.ext, i32 2
  %tobool25 = icmp eq i32 %tmp24, 0
  br i1 %tobool25, label %cond.else27, label %cond.then26

cond.then26:                                      ; preds = %cond.end18
  %tmp30 = extractelement <4 x i32> %v, i32 2
  br label %cond.end28

cond.else27:                                      ; preds = %cond.end18
  br label %cond.end28

cond.end28:                                       ; preds = %cond.else27, %cond.then26
  %cond32 = phi i32 [ %tmp30, %cond.then26 ], [ %cond22, %cond.else27 ]
  br label %return
}

attributes #0 = { nounwind }
