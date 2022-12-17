; RUN: opt -mtriple=nvptx64-nvidia-cuda -passes=load-store-vectorizer -S -o - %s | FileCheck %s

define void @ldg_f16(half* nocapture align 16 %rd0) {
  %in1b = bitcast half* %rd0 to <2 x half>*
  %load1 = load <2 x half>, <2 x half>* %in1b, align 4
  %p1 = fcmp ogt <2 x half> %load1, zeroinitializer
  %s1 = select <2 x i1> %p1, <2 x half> %load1, <2 x half> zeroinitializer
  store <2 x half> %s1, <2 x half>* %in1b, align 4
  %in2 = getelementptr half, half* %rd0, i64 2
  %in2b = bitcast half* %in2 to <2 x half>*
  %load2 = load <2 x half>, <2 x half>* %in2b, align 4
  %p2 = fcmp ogt <2 x half> %load2, zeroinitializer
  %s2 = select <2 x i1> %p2, <2 x half> %load2, <2 x half> zeroinitializer
  store <2 x half> %s2, <2 x half>* %in2b, align 4
  %in3 = getelementptr half, half* %rd0, i64 4
  %in3b = bitcast half* %in3 to <2 x half>*
  %load3 = load <2 x half>, <2 x half>* %in3b, align 4
  %p3 = fcmp ogt <2 x half> %load3, zeroinitializer
  %s3 = select <2 x i1> %p3, <2 x half> %load3, <2 x half> zeroinitializer
  store <2 x half> %s3, <2 x half>* %in3b, align 4
  %in4 = getelementptr half, half* %rd0, i64 6
  %in4b = bitcast half* %in4 to <2 x half>*
  %load4 = load <2 x half>, <2 x half>* %in4b, align 4
  %p4 = fcmp ogt <2 x half> %load4, zeroinitializer
  %s4 = select <2 x i1> %p4, <2 x half> %load4, <2 x half> zeroinitializer
  store <2 x half> %s4, <2 x half>* %in4b, align 4
  ret void

; CHECK-LABEL: @ldg_f16
; CHECK: %[[LD:.*]] = load <8 x half>, <8 x half>*
; CHECK: shufflevector <8 x half> %[[LD]], <8 x half> poison, <2 x i32> <i32 0, i32 1>
; CHECK: shufflevector <8 x half> %[[LD]], <8 x half> poison, <2 x i32> <i32 2, i32 3>
; CHECK: shufflevector <8 x half> %[[LD]], <8 x half> poison, <2 x i32> <i32 4, i32 5>
; CHECK: shufflevector <8 x half> %[[LD]], <8 x half> poison, <2 x i32> <i32 6, i32 7>
; CHECK: store <8 x half>
}

define void @no_nonpow2_vector(half* nocapture align 16 %rd0) {
  %in1b = bitcast half* %rd0 to <3 x half>*
  %load1 = load <3 x half>, <3 x half>* %in1b, align 4
  %p1 = fcmp ogt <3 x half> %load1, zeroinitializer
  %s1 = select <3 x i1> %p1, <3 x half> %load1, <3 x half> zeroinitializer
  store <3 x half> %s1, <3 x half>* %in1b, align 4
  %in2 = getelementptr half, half* %rd0, i64 3
  %in2b = bitcast half* %in2 to <3 x half>*
  %load2 = load <3 x half>, <3 x half>* %in2b, align 4
  %p2 = fcmp ogt <3 x half> %load2, zeroinitializer
  %s2 = select <3 x i1> %p2, <3 x half> %load2, <3 x half> zeroinitializer
  store <3 x half> %s2, <3 x half>* %in2b, align 4
  %in3 = getelementptr half, half* %rd0, i64 6
  %in3b = bitcast half* %in3 to <3 x half>*
  %load3 = load <3 x half>, <3 x half>* %in3b, align 4
  %p3 = fcmp ogt <3 x half> %load3, zeroinitializer
  %s3 = select <3 x i1> %p3, <3 x half> %load3, <3 x half> zeroinitializer
  store <3 x half> %s3, <3 x half>* %in3b, align 4
  %in4 = getelementptr half, half* %rd0, i64 9
  %in4b = bitcast half* %in4 to <3 x half>*
  %load4 = load <3 x half>, <3 x half>* %in4b, align 4
  %p4 = fcmp ogt <3 x half> %load4, zeroinitializer
  %s4 = select <3 x i1> %p4, <3 x half> %load4, <3 x half> zeroinitializer
  store <3 x half> %s4, <3 x half>* %in4b, align 4
  ret void

; CHECK-LABEL: @no_nonpow2_vector
; CHECK-NOT: shufflevector
}

define void @no_pointer_vector(half** nocapture align 16 %rd0) {
  %in1b = bitcast half** %rd0 to <2 x half*>*
  %load1 = load <2 x half*>, <2 x half*>* %in1b, align 4
  %p1 = icmp ne <2 x half*> %load1, zeroinitializer
  %s1 = select <2 x i1> %p1, <2 x half*> %load1, <2 x half*> zeroinitializer
  store <2 x half*> %s1, <2 x half*>* %in1b, align 4
  %in2 = getelementptr half*, half** %rd0, i64 2
  %in2b = bitcast half** %in2 to <2 x half*>*
  %load2 = load <2 x half*>, <2 x half*>* %in2b, align 4
  %p2 = icmp ne <2 x half*> %load2, zeroinitializer
  %s2 = select <2 x i1> %p2, <2 x half*> %load2, <2 x half*> zeroinitializer
  store <2 x half*> %s2, <2 x half*>* %in2b, align 4
  %in3 = getelementptr half*, half** %rd0, i64 4
  %in3b = bitcast half** %in3 to <2 x half*>*
  %load3 = load <2 x half*>, <2 x half*>* %in3b, align 4
  %p3 = icmp ne <2 x half*> %load3, zeroinitializer
  %s3 = select <2 x i1> %p3, <2 x half*> %load3, <2 x half*> zeroinitializer
  store <2 x half*> %s3, <2 x half*>* %in3b, align 4
  %in4 = getelementptr half*, half** %rd0, i64 6
  %in4b = bitcast half** %in4 to <2 x half*>*
  %load4 = load <2 x half*>, <2 x half*>* %in4b, align 4
  %p4 = icmp ne <2 x half*> %load4, zeroinitializer
  %s4 = select <2 x i1> %p4, <2 x half*> %load4, <2 x half*> zeroinitializer
  store <2 x half*> %s4, <2 x half*>* %in4b, align 4
  ret void

; CHECK-LABEL: @no_pointer_vector
; CHECK-NOT: shufflevector
}
