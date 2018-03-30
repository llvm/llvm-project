; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv62 -enable-pipeliner < %s | FileCheck --check-prefix=CHECK-V62 %s
; RUN: llc -march=hexagon -mcpu=hexagonv65 -enable-pipeliner < %s | FileCheck --check-prefix=CHECK-V65 %s

;
; Make sure we pipeline the loop and that we generate the correct
; base+offset values for the loads.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: r{{[0-9]+}} = memw([[REG1:(r[0-9]+)]]+#{{[0,4]}})
; CHECK: r{{[0-9]+}} = memw([[REG1]]++#4)
; CHECK: }{{[ \t]*}}:endloop
; CHECK-V62-NOT: }{{[ \t]*}}:mem_noshuf
; CHECK-V65: }{{[ \t]*}}:mem_noshuf

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b4

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v7, %b1 ], [ 0, %b0 ]
  %v1 = getelementptr inbounds i8*, i8** undef, i32 %v0
  %v2 = load i8*, i8** %v1, align 4
  %v3 = bitcast i8* %v2 to i32*
  store i32 0, i32* %v3, align 4
  %v4 = load i8*, i8** %v1, align 4
  %v5 = getelementptr inbounds i8, i8* %v4, i32 8
  %v6 = bitcast i8* %v5 to i32*
  store i32 0, i32* %v6, align 4
  %v7 = add nsw i32 %v0, 1
  %v8 = icmp eq i32 %v7, 2
  br i1 %v8, label %b2, label %b1

b2:                                               ; preds = %b1
  br i1 undef, label %b3, label %b4

b3:                                               ; preds = %b2
  unreachable

b4:                                               ; preds = %b2, %b0
  unreachable
}

attributes #0 = { nounwind }
