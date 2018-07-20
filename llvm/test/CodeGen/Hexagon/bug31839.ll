; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Check for successful compilation.

define i8* @f0(i32 %a0, i32 %a1) {
b0:
  %v0 = call noalias i8* @f1(i32 undef, i32 undef)
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  %v1 = ptrtoint i8* %v0 to i32
  %v2 = bitcast i8* %v0 to i32*
  store volatile i32 %v1, i32* %v2, align 4
  %v3 = getelementptr inbounds i8, i8* %v0, i32 4
  %v4 = bitcast i8* %v3 to i8**
  store i8* %v0, i8** %v4, align 4
  %v5 = getelementptr inbounds i8, i8* %v0, i32 16
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v6 = phi i8* [ %v5, %b1 ], [ null, %b0 ]
  ret i8* %v6
}

declare noalias i8* @f1(i32, i32) local_unnamed_addr
