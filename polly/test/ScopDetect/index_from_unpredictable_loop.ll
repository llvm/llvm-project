; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s --check-prefix=AFFINE
; RUN: opt %loadPolly -polly-scops -polly-allow-nonaffine -analyze < %s | FileCheck %s --check-prefix=NONAFFINE

; The loop for.body => for.inc has an unpredictable iteration count could due to
; the undef start value that it is compared to. Therefore the array element
; %arrayidx101 that depends on that exit value cannot be affine.
; Derived from test-suite/MultiSource/Benchmarks/BitBench/uuencode/uuencode.c

define void @encode_line(i8* nocapture readonly %input, i32 %octets) {
entry:
  br i1 undef, label %for.body, label %for.end

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %octets.addr.02 = phi i32 [ undef, %for.inc ], [ %octets, %entry ]
  br i1 false, label %for.inc, label %if.else

if.else:
  %cond = icmp eq i32 %octets.addr.02, 2
  br i1 %cond, label %if.then84, label %for.end

if.then84:
  %0 = add nsw i64 %indvars.iv, 1
  %arrayidx101 = getelementptr inbounds i8, i8* %input, i64 %0
  %1 = load i8, i8* %arrayidx101, align 1
  br label %for.end

for.inc:
  %cmp = icmp sgt i32 %octets.addr.02, 3
  %indvars.iv.next = add nsw i64 %indvars.iv, 3
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}


; AFFINE-NOT: Function: encode_line

; NONAFFINE:      Function: encode_line
; NONAFFINE-NEXT: Region: %for.body---%for.end
