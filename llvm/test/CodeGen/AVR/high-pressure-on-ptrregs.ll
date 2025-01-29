; RUN: llc < %s -mtriple=avr | FileCheck %s

; This tests how LLVM handles IR which puts very high
; presure on the PTRREGS class for the register allocator.
;
; This causes a problem because we only have one small register
; class for loading and storing from pointers - 'PTRREGS'.
; One of these registers is also used for the frame pointer, meaning
; that we only ever have two registers available for these operations.
;
; There is an existing bug filed for this issue - PR14879.
;
; The specific failure:
; LLVM ERROR: ran out of registers during register allocation
;
; It has been assembled from the following c code:
;
; struct ss
; {
;   int a;
;   int b;
;   int c;
; };
;
; void loop(struct ss *x, struct ss **y, int z)
; {
;   int i;
;   for (i=0; i<z; ++i)
;   {
;     x->c += y[i]->b;
;   }
; }

%struct.ss = type { i16, i16, i16 }

; CHECK-LABEL: loop
define void @loop(ptr %x, ptr %y, i16 %z) #0 {
entry:
  %x.addr = alloca ptr, align 2
  %y.addr = alloca ptr, align 2
  %z.addr = alloca i16, align 2
  %i = alloca i16, align 2
  store ptr %x, ptr %x.addr, align 2
  store ptr %y, ptr %y.addr, align 2
  store i16 %z, ptr %z.addr, align 2
  store i16 0, ptr %i, align 2
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %tmp = load i16, ptr %i, align 2
  %tmp1 = load i16, ptr %z.addr, align 2
  %cmp = icmp slt i16 %tmp, %tmp1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp2 = load ptr, ptr %y.addr, align 2
  %tmp3 = load i16, ptr %i, align 2
  %arrayidx = getelementptr inbounds ptr, ptr %tmp2, i16 %tmp3
  %tmp4 = load ptr, ptr %arrayidx, align 2
  %b = getelementptr inbounds %struct.ss, ptr %tmp4, i32 0, i32 1
  %tmp5 = load i16, ptr %b, align 2
  %tmp6 = load ptr, ptr %x.addr, align 2
  %c = getelementptr inbounds %struct.ss, ptr %tmp6, i32 0, i32 2
  %tmp7 = load i16, ptr %c, align 2
  %add = add nsw i16 %tmp7, %tmp5
  store i16 %add, ptr %c, align 2
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %tmp8 = load i16, ptr %i, align 2
  %inc = add nsw i16 %tmp8, 1
  store i16 %inc, ptr %i, align 2
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

