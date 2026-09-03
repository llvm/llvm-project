; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avx10.2-512 | FileCheck %s --check-prefix=AVX10
; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avx512vnni,+avx512vl | FileCheck %s --check-prefix=VNNI
; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avx512vnni | FileCheck %s --check-prefix=VNNI-NOVLX
; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avxvnni | FileCheck %s --check-prefix=AVXVNNI
; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avxvnniint8 | FileCheck %s --check-prefix=INT8
; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avxvnniint16 | FileCheck %s --check-prefix=INT16
; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avx10.2-512 -vectorizer-maximize-bandwidth | FileCheck %s --check-prefix=AVX10-BW
; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avx512vnni,+avx512vl -vectorizer-maximize-bandwidth | FileCheck %s --check-prefix=VNNI-BW
; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avxvnni -vectorizer-maximize-bandwidth | FileCheck %s --check-prefix=AVXVNNI-BW
; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avxvnniint8 -vectorizer-maximize-bandwidth | FileCheck %s --check-prefix=INT8-BW
; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -mattr=+avxvnniint16 -vectorizer-maximize-bandwidth | FileCheck %s --check-prefix=INT16-BW

target triple = "x86_64-unknown-linux-gnu"

; Signed x Signed byte dot product (i8 -> i32)
; Only AVX10.2 and AVXVNNIINT8 have VPDPBSSD
define i32 @sdot_i8_to_i32(ptr %a, ptr %b) #0 {
; AVX10-LABEL: @sdot_i8_to_i32(
; AVX10:         @llvm.vector.partial.reduce.add
;
; VNNI-LABEL: @sdot_i8_to_i32(
; VNNI-NOT:     @llvm.vector.partial.reduce.add
; VNNI:         ret i32
;
; VNNI-NOVLX-LABEL: @sdot_i8_to_i32(
; VNNI-NOVLX-NOT:   @llvm.vector.partial.reduce.add
; VNNI-NOVLX:       ret i32
;
; AVXVNNI-LABEL: @sdot_i8_to_i32(
; AVXVNNI-NOT:   @llvm.vector.partial.reduce.add
; AVXVNNI:       ret i32
;
; INT8-LABEL: @sdot_i8_to_i32(
; INT8-NOT:     @llvm.vector.partial.reduce.add
; INT8:         ret i32
;
; INT16-LABEL: @sdot_i8_to_i32(
; INT16-NOT:    @llvm.vector.partial.reduce.add
; INT16:        ret i32
;
; AVX10-BW-LABEL: @sdot_i8_to_i32(
; AVX10-BW:       @llvm.vector.partial.reduce.add
;
; VNNI-BW-LABEL: @sdot_i8_to_i32(
; VNNI-BW-NOT:   @llvm.vector.partial.reduce.add
; VNNI-BW:       ret i32
;
; AVXVNNI-BW-LABEL: @sdot_i8_to_i32(
; AVXVNNI-BW-NOT:   @llvm.vector.partial.reduce.add
; AVXVNNI-BW:       ret i32
;
; INT8-BW-LABEL: @sdot_i8_to_i32(
; INT8-BW:       @llvm.vector.partial.reduce.add
;
; INT16-BW-LABEL: @sdot_i8_to_i32(
; INT16-BW-NOT:   @llvm.vector.partial.reduce.add
; INT16-BW:       ret i32
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %add, %loop ]
  %gep.a = getelementptr inbounds i8, ptr %a, i64 %iv
  %load.a = load i8, ptr %gep.a
  %ext.a = sext i8 %load.a to i32
  %gep.b = getelementptr inbounds i8, ptr %b, i64 %iv
  %load.b = load i8, ptr %gep.b
  %ext.b = sext i8 %load.b to i32
  %mul = mul nsw i32 %ext.a, %ext.b
  %add = add nsw i32 %acc, %mul
  %iv.next = add i64 %iv, 1
  %exit = icmp eq i64 %iv.next, 1024
  br i1 %exit, label %end, label %loop
end:
  ret i32 %add
}

; Unsigned x Unsigned byte dot product (i8 -> i32)
; Only AVX10.2 and AVXVNNIINT8 have VPDPBUUD
define i32 @udot_i8_to_i32(ptr %a, ptr %b) #0 {
; AVX10-LABEL: @udot_i8_to_i32(
; AVX10:         @llvm.vector.partial.reduce.add
;
; VNNI-LABEL: @udot_i8_to_i32(
; VNNI-NOT:     @llvm.vector.partial.reduce.add
; VNNI:         ret i32
;
; VNNI-NOVLX-LABEL: @udot_i8_to_i32(
; VNNI-NOVLX-NOT:   @llvm.vector.partial.reduce.add
; VNNI-NOVLX:       ret i32
;
; AVXVNNI-LABEL: @udot_i8_to_i32(
; AVXVNNI-NOT:   @llvm.vector.partial.reduce.add
; AVXVNNI:       ret i32
;
; INT8-LABEL: @udot_i8_to_i32(
; INT8-NOT:     @llvm.vector.partial.reduce.add
; INT8:         ret i32
;
; INT16-LABEL: @udot_i8_to_i32(
; INT16-NOT:    @llvm.vector.partial.reduce.add
; INT16:        ret i32
;
; AVX10-BW-LABEL: @udot_i8_to_i32(
; AVX10-BW:       @llvm.vector.partial.reduce.add
;
; VNNI-BW-LABEL: @udot_i8_to_i32(
; VNNI-BW-NOT:   @llvm.vector.partial.reduce.add
; VNNI-BW:       ret i32
;
; AVXVNNI-BW-LABEL: @udot_i8_to_i32(
; AVXVNNI-BW-NOT:   @llvm.vector.partial.reduce.add
; AVXVNNI-BW:       ret i32
;
; INT8-BW-LABEL: @udot_i8_to_i32(
; INT8-BW:       @llvm.vector.partial.reduce.add
;
; INT16-BW-LABEL: @udot_i8_to_i32(
; INT16-BW-NOT:   @llvm.vector.partial.reduce.add
; INT16-BW:       ret i32
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %add, %loop ]
  %gep.a = getelementptr inbounds i8, ptr %a, i64 %iv
  %load.a = load i8, ptr %gep.a
  %ext.a = zext i8 %load.a to i32
  %gep.b = getelementptr inbounds i8, ptr %b, i64 %iv
  %load.b = load i8, ptr %gep.b
  %ext.b = zext i8 %load.b to i32
  %mul = mul nuw i32 %ext.a, %ext.b
  %add = add i32 %acc, %mul
  %iv.next = add i64 %iv, 1
  %exit = icmp eq i64 %iv.next, 1024
  br i1 %exit, label %end, label %loop
end:
  ret i32 %add
}

; Unsigned x Signed byte dot product (i8 -> i32)
; AVX10.2 has VPDPBSUD, VNNI/AVXVNNI have VPDPBUSD
define i32 @usdot_i8_to_i32(ptr %a, ptr %b) #0 {
; AVX10-LABEL: @usdot_i8_to_i32(
; AVX10:         @llvm.vector.partial.reduce.add
;
; VNNI-LABEL: @usdot_i8_to_i32(
; VNNI:         @llvm.vector.partial.reduce.add
;
; VNNI-NOVLX-LABEL: @usdot_i8_to_i32(
; VNNI-NOVLX:       @llvm.vector.partial.reduce.add
;
; AVXVNNI-LABEL: @usdot_i8_to_i32(
; AVXVNNI-NOT:   @llvm.vector.partial.reduce.add
; AVXVNNI:       ret i32
;
; INT8-LABEL: @usdot_i8_to_i32(
; INT8-NOT:     @llvm.vector.partial.reduce.add
; INT8:         ret i32
;
; INT16-LABEL: @usdot_i8_to_i32(
; INT16-NOT:    @llvm.vector.partial.reduce.add
; INT16:        ret i32
;
; AVX10-BW-LABEL: @usdot_i8_to_i32(
; AVX10-BW:       @llvm.vector.partial.reduce.add
;
; VNNI-BW-LABEL: @usdot_i8_to_i32(
; VNNI-BW:       @llvm.vector.partial.reduce.add
;
; AVXVNNI-BW-LABEL: @usdot_i8_to_i32(
; AVXVNNI-BW:       @llvm.vector.partial.reduce.add
;
; INT8-BW-LABEL: @usdot_i8_to_i32(
; INT8-BW:       @llvm.vector.partial.reduce.add
;
; INT16-BW-LABEL: @usdot_i8_to_i32(
; INT16-BW-NOT:   @llvm.vector.partial.reduce.add
; INT16-BW:       ret i32
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %add, %loop ]
  %gep.a = getelementptr inbounds i8, ptr %a, i64 %iv
  %load.a = load i8, ptr %gep.a
  %ext.a = zext i8 %load.a to i32
  %gep.b = getelementptr inbounds i8, ptr %b, i64 %iv
  %load.b = load i8, ptr %gep.b
  %ext.b = sext i8 %load.b to i32
  %mul = mul nsw i32 %ext.a, %ext.b
  %add = add nsw i32 %acc, %mul
  %iv.next = add i64 %iv, 1
  %exit = icmp eq i64 %iv.next, 1024
  br i1 %exit, label %end, label %loop
end:
  ret i32 %add
}

; Signed x Signed word dot product (i16 -> i32)
; AVX10.2 has VPDPWSSD, VNNI/AVXVNNI also have VPDPWSSD
define i32 @sdot_i16_to_i32(ptr %a, ptr %b) #0 {
; AVX10-LABEL: @sdot_i16_to_i32(
; AVX10:         @llvm.vector.partial.reduce.add
;
; VNNI-LABEL: @sdot_i16_to_i32(
; VNNI:         @llvm.vector.partial.reduce.add
;
; VNNI-NOVLX-LABEL: @sdot_i16_to_i32(
; VNNI-NOVLX:       @llvm.vector.partial.reduce.add
;
; AVXVNNI-LABEL: @sdot_i16_to_i32(
; AVXVNNI:       @llvm.vector.partial.reduce.add
;
; INT8-LABEL: @sdot_i16_to_i32(
; INT8-NOT:     @llvm.vector.partial.reduce.add
; INT8:         ret i32
;
; INT16-LABEL: @sdot_i16_to_i32(
; INT16-NOT:    @llvm.vector.partial.reduce.add
; INT16:        ret i32
;
; AVX10-BW-LABEL: @sdot_i16_to_i32(
; AVX10-BW:       @llvm.vector.partial.reduce.add
;
; VNNI-BW-LABEL: @sdot_i16_to_i32(
; VNNI-BW:       @llvm.vector.partial.reduce.add
;
; AVXVNNI-BW-LABEL: @sdot_i16_to_i32(
; AVXVNNI-BW:       @llvm.vector.partial.reduce.add
;
; INT8-BW-LABEL: @sdot_i16_to_i32(
; INT8-BW-NOT:   @llvm.vector.partial.reduce.add
; INT8-BW:       ret i32
;
; INT16-BW-LABEL: @sdot_i16_to_i32(
; INT16-BW-NOT:   @llvm.vector.partial.reduce.add
; INT16-BW:       ret i32
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %add, %loop ]
  %gep.a = getelementptr inbounds i16, ptr %a, i64 %iv
  %load.a = load i16, ptr %gep.a
  %ext.a = sext i16 %load.a to i32
  %gep.b = getelementptr inbounds i16, ptr %b, i64 %iv
  %load.b = load i16, ptr %gep.b
  %ext.b = sext i16 %load.b to i32
  %mul = mul nsw i32 %ext.a, %ext.b
  %add = add nsw i32 %acc, %mul
  %iv.next = add i64 %iv, 1
  %exit = icmp eq i64 %iv.next, 1024
  br i1 %exit, label %end, label %loop
end:
  ret i32 %add
}

; Unsigned x Unsigned word dot product (i16 -> i32)
; Only AVX10.2 and AVXVNNIINT16 have VPDPWUUD
define i32 @udot_i16_to_i32(ptr %a, ptr %b) #0 {
; AVX10-LABEL: @udot_i16_to_i32(
; AVX10:         @llvm.vector.partial.reduce.add
;
; VNNI-LABEL: @udot_i16_to_i32(
; VNNI-NOT:     @llvm.vector.partial.reduce.add
; VNNI:         ret i32
;
; VNNI-NOVLX-LABEL: @udot_i16_to_i32(
; VNNI-NOVLX-NOT:   @llvm.vector.partial.reduce.add
; VNNI-NOVLX:       ret i32
;
; AVXVNNI-LABEL: @udot_i16_to_i32(
; AVXVNNI-NOT:   @llvm.vector.partial.reduce.add
; AVXVNNI:       ret i32
;
; INT8-LABEL: @udot_i16_to_i32(
; INT8-NOT:     @llvm.vector.partial.reduce.add
; INT8:         ret i32
;
; INT16-LABEL: @udot_i16_to_i32(
; INT16:        @llvm.vector.partial.reduce.add
;
; AVX10-BW-LABEL: @udot_i16_to_i32(
; AVX10-BW:       @llvm.vector.partial.reduce.add
;
; VNNI-BW-LABEL: @udot_i16_to_i32(
; VNNI-BW-NOT:   @llvm.vector.partial.reduce.add
; VNNI-BW:       ret i32
;
; AVXVNNI-BW-LABEL: @udot_i16_to_i32(
; AVXVNNI-BW-NOT:   @llvm.vector.partial.reduce.add
; AVXVNNI-BW:       ret i32
;
; INT8-BW-LABEL: @udot_i16_to_i32(
; INT8-BW-NOT:   @llvm.vector.partial.reduce.add
; INT8-BW:       ret i32
;
; INT16-BW-LABEL: @udot_i16_to_i32(
; INT16-BW:       @llvm.vector.partial.reduce.add
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %add, %loop ]
  %gep.a = getelementptr inbounds i16, ptr %a, i64 %iv
  %load.a = load i16, ptr %gep.a
  %ext.a = zext i16 %load.a to i32
  %gep.b = getelementptr inbounds i16, ptr %b, i64 %iv
  %load.b = load i16, ptr %gep.b
  %ext.b = zext i16 %load.b to i32
  %mul = mul nuw i32 %ext.a, %ext.b
  %add = add i32 %acc, %mul
  %iv.next = add i64 %iv, 1
  %exit = icmp eq i64 %iv.next, 1024
  br i1 %exit, label %end, label %loop
end:
  ret i32 %add
}

; Unsigned x Signed word dot product (i16 -> i32)
; AVX10.2 has VPDPWSUD, AVXVNNIINT16 also has it
define i32 @usdot_i16_to_i32(ptr %a, ptr %b) #0 {
; AVX10-LABEL: @usdot_i16_to_i32(
; AVX10:         @llvm.vector.partial.reduce.add
;
; VNNI-LABEL: @usdot_i16_to_i32(
; VNNI-NOT:     @llvm.vector.partial.reduce.add
; VNNI:         ret i32
;
; VNNI-NOVLX-LABEL: @usdot_i16_to_i32(
; VNNI-NOVLX-NOT:   @llvm.vector.partial.reduce.add
; VNNI-NOVLX:       ret i32
;
; AVXVNNI-LABEL: @usdot_i16_to_i32(
; AVXVNNI-NOT:   @llvm.vector.partial.reduce.add
; AVXVNNI:       ret i32
;
; INT8-LABEL: @usdot_i16_to_i32(
; INT8-NOT:     @llvm.vector.partial.reduce.add
; INT8:         ret i32
;
; INT16-LABEL: @usdot_i16_to_i32(
; INT16:        @llvm.vector.partial.reduce.add
;
; AVX10-BW-LABEL: @usdot_i16_to_i32(
; AVX10-BW:       @llvm.vector.partial.reduce.add
;
; VNNI-BW-LABEL: @usdot_i16_to_i32(
; VNNI-BW-NOT:   @llvm.vector.partial.reduce.add
; VNNI-BW:       ret i32
;
; AVXVNNI-BW-LABEL: @usdot_i16_to_i32(
; AVXVNNI-BW-NOT:   @llvm.vector.partial.reduce.add
; AVXVNNI-BW:       ret i32
;
; INT8-BW-LABEL: @usdot_i16_to_i32(
; INT8-BW-NOT:   @llvm.vector.partial.reduce.add
; INT8-BW:       ret i32
;
; INT16-BW-LABEL: @usdot_i16_to_i32(
; INT16-BW:       @llvm.vector.partial.reduce.add
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %add, %loop ]
  %gep.a = getelementptr inbounds i16, ptr %a, i64 %iv
  %load.a = load i16, ptr %gep.a
  %ext.a = zext i16 %load.a to i32
  %gep.b = getelementptr inbounds i16, ptr %b, i64 %iv
  %load.b = load i16, ptr %gep.b
  %ext.b = sext i16 %load.b to i32
  %mul = mul nsw i32 %ext.a, %ext.b
  %add = add nsw i32 %acc, %mul
  %iv.next = add i64 %iv, 1
  %exit = icmp eq i64 %iv.next, 1024
  br i1 %exit, label %end, label %loop
end:
  ret i32 %add
}

attributes #0 = { "target-cpu"="x86-64" }
