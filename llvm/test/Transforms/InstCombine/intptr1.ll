; RUN: opt < %s  -instcombine  -S | FileCheck %s

define void @test1(float* %a, float* readnone %a_end, i64* %b.i64) {
; CHECK-LABEL: @test1
entry:
  %cmp1 = icmp ult float* %a, %a_end
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %b = load i64, i64* %b.i64, align 8
; CHECK: load float*, float**
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %a.addr.03 = phi float* [ %incdec.ptr, %for.body ], [ %a, %for.body.preheader ]
  %b.addr.02 = phi i64 [ %add.int, %for.body ], [ %b, %for.body.preheader ]

; CHECK: %a.addr.03 = phi float* [ %incdec.ptr, %for.body ], [ %a, %for.body.preheader ]
; CHECK: %b.addr.02.ptr = phi float* [ %add, %for.body ],
; CHECK-NOT: %b.addr.02 = phi i64

  %tmp = inttoptr i64 %b.addr.02 to float*
; CHECK-NOT: inttoptr i64
  %tmp1 = load float, float* %tmp, align 4
; CHECK: = load
  %mul.i = fmul float %tmp1, 4.200000e+01
  store float %mul.i, float* %a.addr.03, align 4
  %add = getelementptr inbounds float, float* %tmp, i64 1
  %add.int = ptrtoint float* %add to i64
; CHECK %add = getelementptr
; CHECK-NOT: ptrtoint float*
  %incdec.ptr = getelementptr inbounds float, float* %a.addr.03, i64 1
; CHECK: %incdec.ptr = 
  %cmp = icmp ult float* %incdec.ptr, %a_end
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

define void @test1_neg(float* %a, float* readnone %a_end, i64* %b.i64) {
; CHECK-LABEL: @test1_neg
entry:
  %cmp1 = icmp ult float* %a, %a_end
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %b = load i64, i64* %b.i64, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %a.addr.03 = phi float* [ %incdec.ptr, %bb ], [ %a, %for.body.preheader ]
  %b.addr.02 = phi i64 [ %add.int, %bb ], [ %b, %for.body.preheader ]

; CHECK: %a.addr.03 = phi float* [ %incdec.ptr, %bb ], [ %a, %for.body.preheader ]
; CHECK: %b.addr.02 = phi i64

  %tmp = inttoptr i64 %b.addr.02 to float*
; CHECK: inttoptr i64
  %ptrcmp = icmp ult float* %tmp, %a_end
  br i1 %ptrcmp, label %for.end, label %bb

bb:
  %tmp1 = load float, float* %a, align 4
  %mul.i = fmul float %tmp1, 4.200000e+01
  store float %mul.i, float* %a.addr.03, align 4
  %add = getelementptr inbounds float, float* %a, i64 1
  %add.int = ptrtoint float* %add to i64
; CHECK: ptrtoint float*
  %incdec.ptr = getelementptr inbounds float, float* %a.addr.03, i64 1
  %cmp = icmp ult float* %incdec.ptr, %a_end
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @test2(float* %a, float* readnone %a_end, float** %b.float) {
; CHECK-LABEL: @test2
entry:
  %cmp1 = icmp ult float* %a, %a_end
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %b.i64 = bitcast float** %b.float to i64*
  %b = load i64, i64* %b.i64, align 8
; CHECK: load float*, float**
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %a.addr.03 = phi float* [ %incdec.ptr, %for.body ], [ %a, %for.body.preheader ]
  %b.addr.02 = phi i64 [ %add.int, %for.body ], [ %b, %for.body.preheader ]

; CHECK: %a.addr.03 = phi float* [ %incdec.ptr, %for.body ], [ %a, %for.body.preheader ]
; CHECK: %b.addr.02.ptr = phi float* [ %add, %for.body ],
; CHECK-NOT: %b.addr.02 = phi i64

  %tmp = inttoptr i64 %b.addr.02 to float*
; CHECK-NOT: inttoptr i64
  %tmp1 = load float, float* %tmp, align 4
; CHECK: = load
  %mul.i = fmul float %tmp1, 4.200000e+01
  store float %mul.i, float* %a.addr.03, align 4
  %add = getelementptr inbounds float, float* %tmp, i64 1
; CHECK: %add = 
  %add.int = ptrtoint float* %add to i64
; CHECK-NOT: ptrtoint float*
  %incdec.ptr = getelementptr inbounds float, float* %a.addr.03, i64 1
; CHECK: %incdec.ptr = 
  %cmp = icmp ult float* %incdec.ptr, %a_end
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @test3(float* %a, float* readnone %a_end, i8** %b.i8p) {
; CHECK-LABEL: @test3
entry:
  %cmp1 = icmp ult float* %a, %a_end
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %b.i64 = bitcast i8** %b.i8p to i64*
  %b = load i64, i64* %b.i64, align 8
; CHECK: load float*, float**
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %a.addr.03 = phi float* [ %incdec.ptr, %for.body ], [ %a, %for.body.preheader ]
  %b.addr.02 = phi i64 [ %add.int, %for.body ], [ %b, %for.body.preheader ]

; CHECK: %a.addr.03 = phi float* [ %incdec.ptr, %for.body ], [ %a, %for.body.preheader ]
; CHECK: %b.addr.02.ptr = phi float* [ %add, %for.body ],
; CHECK-NOT: %b.addr.02 = phi i64

  %tmp = inttoptr i64 %b.addr.02 to float*
; CHECK-NOT: inttoptr i64
  %tmp1 = load float, float* %tmp, align 4
; CHECK: = load
  %mul.i = fmul float %tmp1, 4.200000e+01
  store float %mul.i, float* %a.addr.03, align 4
  %add = getelementptr inbounds float, float* %tmp, i64 1
; CHECK: %add = getelementptr
  %add.int = ptrtoint float* %add to i64
; CHECK-NOT: ptrtoint float*
  %incdec.ptr = getelementptr inbounds float, float* %a.addr.03, i64 1
; CHECK: %incdec.ptr = 
  %cmp = icmp ult float* %incdec.ptr, %a_end
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @test4(float* %a, float* readnone %a_end, float** %b.float) {
entry:
; CHECK-LABEL: @test4
  %cmp1 = icmp ult float* %a, %a_end
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %b.f = load float*, float** %b.float, align 8
  %b = ptrtoint float* %b.f to i64
; CHECK: load float*, float**
; CHECK-NOT: ptrtoint float*
  br label %for.body
; CHECK: br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %a.addr.03 = phi float* [ %incdec.ptr, %for.body ], [ %a, %for.body.preheader ]
  %b.addr.02 = phi i64 [ %add.int, %for.body ], [ %b, %for.body.preheader ]
  %tmp = inttoptr i64 %b.addr.02 to float*
; CHECK-NOT: inttoptr i64
  %tmp1 = load float, float* %tmp, align 4
; CHECK: = load
  %mul.i = fmul float %tmp1, 4.200000e+01
  store float %mul.i, float* %a.addr.03, align 4
  %add = getelementptr inbounds float, float* %tmp, i64 1
; CHECK: %add = 
  %add.int = ptrtoint float* %add to i64
; CHECK-NOT: ptrtoint float*
  %incdec.ptr = getelementptr inbounds float, float* %a.addr.03, i64 1
; CHECK: %incdec.ptr =
  %cmp = icmp ult float* %incdec.ptr, %a_end
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}
