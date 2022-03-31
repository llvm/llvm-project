; RUN: opt < %s  -passes=instcombine -S | FileCheck %s

define void @test1(float* %a, float* readnone %a_end, i32* %b.i) {
; CHECK-LABEL: @test1
entry:
  %cmp1 = icmp ult float* %a, %a_end
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %b = ptrtoint i32 * %b.i to i64
; CHECK: bitcast
; CHECK-NOT: ptrtoint
  br label %for.body
; CHECK: br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %a.addr.03 = phi float* [ %incdec.ptr, %for.body ], [ %a, %for.body.preheader ]
  %b.addr.02 = phi i64 [ %add.int, %for.body ], [ %b, %for.body.preheader ]
; CHECK:  %a.addr.03 = phi float* [ %incdec.ptr, %for.body ], [ %a, %for.body.preheader ]
; CHECK-NOT: phi i64 
  %tmp = inttoptr i64 %b.addr.02 to float*
; CHECK-NOT: inttoptr
  %tmp1 = load float, float* %tmp, align 4
; CHECK: = load
  %mul.i = fmul float %tmp1, 4.200000e+01
  store float %mul.i, float* %a.addr.03, align 4
  %add = getelementptr inbounds float, float* %tmp, i64 1
; CHECK: %add = 
  %add.int = ptrtoint float* %add to i64
; CHECK-NOT: ptrtoint
  %incdec.ptr = getelementptr inbounds float, float* %a.addr.03, i64 1
; CHECK: %incdec.ptr = 
  %cmp = icmp ult float* %incdec.ptr, %a_end
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

