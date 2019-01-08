; RUN: opt < %s -disable-output -da-delinearize=false "-passes=print<da>"      \
; RUN: -aa-pipeline=basic-aa 2>&1 | FileCheck %s
; RUN: opt < %s -analyze -basicaa -da -da-delinearize=false | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s -check-prefix=DELIN
; RUN: opt < %s -analyze -basicaa -da | FileCheck %s -check-prefix=DELIN

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"


;;  for (long int i = 1; i <= 10; i++)
;;    for (long int j = 1; j <= 10; j++) {
;;      A[10*i + j] = 0;
;;      *B++ = A[10*i + j - 1];

define void @banerjee0(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader
; CHECK: 'Dependence Analysis' for function 'banerjee0':
; CHECK: da analyze - none!
; CHECK: da analyze - flow [<= <>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee0':
; DELIN: da analyze - none!
; DELIN: da analyze - flow [<= <>]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc7
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc7 ]
  %i.03 = phi i64 [ 1, %entry ], [ %inc8, %for.inc7 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 1, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 10
  %add = add nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.03, 10
  %add5 = add nsw i64 %mul4, %j.02
  %sub = add nsw i64 %add5, -1
  %arrayidx6 = getelementptr inbounds i64, i64* %A, i64 %sub
  %0 = load i64, i64* %arrayidx6, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %0, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 11
  br i1 %exitcond, label %for.body3, label %for.inc7

for.inc7:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 10
  %inc8 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc8, 11
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end9

for.end9:                                         ; preds = %for.inc7
  ret void
}


;;  for (long int i = 1; i <= n; i++)
;;    for (long int j = 1; j <= m; j++) {
;;      A[10*i + j] = 0;
;;      *B++ = A[10*i + j - 1];

define void @banerjee1(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  %cmp4 = icmp sgt i64 %n, 0
  br i1 %cmp4, label %for.cond1.preheader.preheader, label %for.end9

; CHECK: 'Dependence Analysis' for function 'banerjee1':
; CHECK: da analyze - output [* *]!
; CHECK: da analyze - flow [* <>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - input [* *]!
; CHECK: da analyze - confused!
; CHECK: da analyze - output [* *]!

; DELIN: 'Dependence Analysis' for function 'banerjee1':
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - flow [* <>]!
; DELIN: da analyze - confused!
; DELIN: da analyze - input [* *]!
; DELIN: da analyze - confused!
; DELIN: da analyze - output [* *]!

for.cond1.preheader.preheader:                    ; preds = %entry
  %0 = add i64 %n, 1
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc7
  %B.addr.06 = phi i64* [ %B.addr.1.lcssa, %for.inc7 ], [ %B, %for.cond1.preheader.preheader ]
  %i.05 = phi i64 [ %inc8, %for.inc7 ], [ 1, %for.cond1.preheader.preheader ]
  %1 = add i64 %m, 1
  %cmp21 = icmp sgt i64 %m, 0
  br i1 %cmp21, label %for.body3.preheader, label %for.inc7

for.body3.preheader:                              ; preds = %for.cond1.preheader
  br label %for.body3

for.body3:                                        ; preds = %for.body3.preheader, %for.body3
  %j.03 = phi i64 [ %inc, %for.body3 ], [ 1, %for.body3.preheader ]
  %B.addr.12 = phi i64* [ %incdec.ptr, %for.body3 ], [ %B.addr.06, %for.body3.preheader ]
  %mul = mul nsw i64 %i.05, 10
  %add = add nsw i64 %mul, %j.03
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.05, 10
  %add5 = add nsw i64 %mul4, %j.03
  %sub = add nsw i64 %add5, -1
  %arrayidx6 = getelementptr inbounds i64, i64* %A, i64 %sub
  %2 = load i64, i64* %arrayidx6, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.12, i64 1
  store i64 %2, i64* %B.addr.12, align 8
  %inc = add nsw i64 %j.03, 1
  %exitcond = icmp eq i64 %inc, %1
  br i1 %exitcond, label %for.inc7.loopexit, label %for.body3

for.inc7.loopexit:                                ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.06, i64 %m
  br label %for.inc7

for.inc7:                                         ; preds = %for.inc7.loopexit, %for.cond1.preheader
  %B.addr.1.lcssa = phi i64* [ %B.addr.06, %for.cond1.preheader ], [ %scevgep, %for.inc7.loopexit ]
  %inc8 = add nsw i64 %i.05, 1
  %exitcond7 = icmp eq i64 %inc8, %0
  br i1 %exitcond7, label %for.end9.loopexit, label %for.cond1.preheader

for.end9.loopexit:                                ; preds = %for.inc7
  br label %for.end9

for.end9:                                         ; preds = %for.end9.loopexit, %entry
  ret void
}


;;  for (long int i = 0; i < 10; i++)
;;    for (long int j = 0; j < 10; j++) {
;;      A[10*i + j] = 0;
;;      *B++ = A[10*i + j + 100];

define void @banerjee2(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee2':
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee2':
; DELIN: da analyze - none!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc8
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc8 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc9, %for.inc8 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 10
  %add = add nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.03, 10
  %add5 = add nsw i64 %mul4, %j.02
  %add6 = add nsw i64 %add5, 100
  %arrayidx7 = getelementptr inbounds i64, i64* %A, i64 %add6
  %0 = load i64, i64* %arrayidx7, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %0, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body3, label %for.inc8

for.inc8:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 10
  %inc9 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc9, 10
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end10

for.end10:                                        ; preds = %for.inc8
  ret void
}


;;  for (long int i = 0; i < 10; i++)
;;    for (long int j = 0; j < 10; j++) {
;;      A[10*i + j] = 0;
;;      *B++ = A[10*i + j + 99];

define void @banerjee3(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee3':
; CHECK: da analyze - none!
; CHECK: da analyze - flow [> >]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee3':
; DELIN: da analyze - none!
; DELIN: da analyze - flow [> >]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc8
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc8 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc9, %for.inc8 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 10
  %add = add nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.03, 10
  %add5 = add nsw i64 %mul4, %j.02
  %add6 = add nsw i64 %add5, 99
  %arrayidx7 = getelementptr inbounds i64, i64* %A, i64 %add6
  %0 = load i64, i64* %arrayidx7, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %0, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body3, label %for.inc8

for.inc8:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 10
  %inc9 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc9, 10
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end10

for.end10:                                        ; preds = %for.inc8
  ret void
}


;;  for (long int i = 0; i < 10; i++)
;;    for (long int j = 0; j < 10; j++) {
;;      A[10*i + j] = 0;
;;      *B++ = A[10*i + j - 100];

define void @banerjee4(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee4':
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee4':
; DELIN: da analyze - none!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc7
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc7 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc8, %for.inc7 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 10
  %add = add nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.03, 10
  %add5 = add nsw i64 %mul4, %j.02
  %sub = add nsw i64 %add5, -100
  %arrayidx6 = getelementptr inbounds i64, i64* %A, i64 %sub
  %0 = load i64, i64* %arrayidx6, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %0, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body3, label %for.inc7

for.inc7:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 10
  %inc8 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc8, 10
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end9

for.end9:                                         ; preds = %for.inc7
  ret void
}


;;  for (long int i = 0; i < 10; i++)
;;    for (long int j = 0; j < 10; j++) {
;;      A[10*i + j] = 0;
;;      *B++ = A[10*i + j - 99];

define void @banerjee5(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee5':
; CHECK: da analyze - none!
; CHECK: da analyze - flow [< <]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee5':
; DELIN: da analyze - none!
; DELIN: da analyze - flow [< <]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc7
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc7 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc8, %for.inc7 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 10
  %add = add nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.03, 10
  %add5 = add nsw i64 %mul4, %j.02
  %sub = add nsw i64 %add5, -99
  %arrayidx6 = getelementptr inbounds i64, i64* %A, i64 %sub
  %0 = load i64, i64* %arrayidx6, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %0, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body3, label %for.inc7

for.inc7:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 10
  %inc8 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc8, 10
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end9

for.end9:                                         ; preds = %for.inc7
  ret void
}


;;  for (long int i = 0; i < 10; i++)
;;    for (long int j = 0; j < 10; j++) {
;;      A[10*i + j] = 0;
;;      *B++ = A[10*i + j + 9];

define void @banerjee6(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee6':
; CHECK: da analyze - none!
; CHECK: da analyze - flow [=> <>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee6':
; DELIN: da analyze - none!
; DELIN: da analyze - flow [=> <>]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc8
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc8 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc9, %for.inc8 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 10
  %add = add nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.03, 10
  %add5 = add nsw i64 %mul4, %j.02
  %add6 = add nsw i64 %add5, 9
  %arrayidx7 = getelementptr inbounds i64, i64* %A, i64 %add6
  %0 = load i64, i64* %arrayidx7, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %0, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body3, label %for.inc8

for.inc8:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 10
  %inc9 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc9, 10
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end10

for.end10:                                        ; preds = %for.inc8
  ret void
}


;;  for (long int i = 0; i < 10; i++)
;;    for (long int j = 0; j < 10; j++) {
;;      A[10*i + j] = 0;
;;      *B++ = A[10*i + j + 10];

define void @banerjee7(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee7':
; CHECK: da analyze - none!
; CHECK: da analyze - flow [> <=]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee7':
; DELIN: da analyze - none!
; DELIN: da analyze - flow [> <=]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc8
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc8 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc9, %for.inc8 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 10
  %add = add nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.03, 10
  %add5 = add nsw i64 %mul4, %j.02
  %add6 = add nsw i64 %add5, 10
  %arrayidx7 = getelementptr inbounds i64, i64* %A, i64 %add6
  %0 = load i64, i64* %arrayidx7, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %0, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body3, label %for.inc8

for.inc8:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 10
  %inc9 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc9, 10
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end10

for.end10:                                        ; preds = %for.inc8
  ret void
}


;;  for (long int i = 0; i < 10; i++)
;;    for (long int j = 0; j < 10; j++) {
;;      A[10*i + j] = 0;
;;      *B++ = A[10*i + j + 11];

define void @banerjee8(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee8':
; CHECK: da analyze - none!
; CHECK: da analyze - flow [> <>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee8':
; DELIN: da analyze - none!
; DELIN: da analyze - flow [> <>]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc8
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc8 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc9, %for.inc8 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 10
  %add = add nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.03, 10
  %add5 = add nsw i64 %mul4, %j.02
  %add6 = add nsw i64 %add5, 11
  %arrayidx7 = getelementptr inbounds i64, i64* %A, i64 %add6
  %0 = load i64, i64* %arrayidx7, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %0, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 10
  br i1 %exitcond, label %for.body3, label %for.inc8

for.inc8:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 10
  %inc9 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc9, 10
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end10

for.end10:                                        ; preds = %for.inc8
  ret void
}


;;  for (long int i = 0; i < 20; i++)
;;    for (long int j = 0; j < 20; j++) {
;;      A[30*i + 500*j] = 0;
;;      *B++ = A[i - 500*j + 11];

define void @banerjee9(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee9':
; CHECK: da analyze - output [* *]!
; CHECK: da analyze - flow [<= =|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee9':
; DELIN: da analyze - output [* *]!
; DELIN: da analyze - flow [<= =|<]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc8
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc8 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc9, %for.inc8 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 30
  %mul4 = mul nsw i64 %j.02, 500
  %add = add nsw i64 %mul, %mul4
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %0 = mul i64 %j.02, -500
  %sub = add i64 %i.03, %0
  %add6 = add nsw i64 %sub, 11
  %arrayidx7 = getelementptr inbounds i64, i64* %A, i64 %add6
  %1 = load i64, i64* %arrayidx7, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %1, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 20
  br i1 %exitcond, label %for.body3, label %for.inc8

for.inc8:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 20
  %inc9 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc9, 20
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end10

for.end10:                                        ; preds = %for.inc8
  ret void
}


;;  for (long int i = 0; i < 20; i++)
;;    for (long int j = 0; j < 20; j++) {
;;      A[i + 500*j] = 0;
;;      *B++ = A[i - 500*j + 11];

define void @banerjee10(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee10':
; CHECK: da analyze - none!
; CHECK: da analyze - flow [<> =]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee10':
; DELIN: da analyze - none!
; DELIN: da analyze - flow [<> =]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc7
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc7 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc8, %for.inc7 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %j.02, 500
  %add = add nsw i64 %i.03, %mul
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %0 = mul i64 %j.02, -500
  %sub = add i64 %i.03, %0
  %add5 = add nsw i64 %sub, 11
  %arrayidx6 = getelementptr inbounds i64, i64* %A, i64 %add5
  %1 = load i64, i64* %arrayidx6, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %1, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 20
  br i1 %exitcond, label %for.body3, label %for.inc7

for.inc7:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 20
  %inc8 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc8, 20
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end9

for.end9:                                         ; preds = %for.inc7
  ret void
}


;;  for (long int i = 0; i < 20; i++)
;;    for (long int j = 0; j < 20; j++) {
;;      A[300*i + j] = 0;
;;      *B++ = A[250*i - j + 11];

define void @banerjee11(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee11':
; CHECK: da analyze - none!
; CHECK: da analyze - flow [<= <>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee11':
; DELIN: da analyze - none!
; DELIN: da analyze - flow [<= <>]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc7
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc7 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc8, %for.inc7 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 300
  %add = add nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.03, 250
  %sub = sub nsw i64 %mul4, %j.02
  %add5 = add nsw i64 %sub, 11
  %arrayidx6 = getelementptr inbounds i64, i64* %A, i64 %add5
  %0 = load i64, i64* %arrayidx6, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %0, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 20
  br i1 %exitcond, label %for.body3, label %for.inc7

for.inc7:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 20
  %inc8 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc8, 20
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end9

for.end9:                                         ; preds = %for.inc7
  ret void
}


;;  for (long int i = 0; i < 20; i++)
;;    for (long int j = 0; j < 20; j++) {
;;      A[100*i + j] = 0;
;;      *B++ = A[100*i - j + 11];

define void @banerjee12(i64* %A, i64* %B, i64 %m, i64 %n) nounwind uwtable ssp {
entry:
  br label %for.cond1.preheader

; CHECK: 'Dependence Analysis' for function 'banerjee12':
; CHECK: da analyze - none!
; CHECK: da analyze - flow [= <>]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

; DELIN: 'Dependence Analysis' for function 'banerjee12':
; DELIN: da analyze - none!
; DELIN: da analyze - flow [= <>]!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!
; DELIN: da analyze - confused!
; DELIN: da analyze - none!

for.cond1.preheader:                              ; preds = %entry, %for.inc7
  %B.addr.04 = phi i64* [ %B, %entry ], [ %scevgep, %for.inc7 ]
  %i.03 = phi i64 [ 0, %entry ], [ %inc8, %for.inc7 ]
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.02 = phi i64 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %B.addr.11 = phi i64* [ %B.addr.04, %for.cond1.preheader ], [ %incdec.ptr, %for.body3 ]
  %mul = mul nsw i64 %i.03, 100
  %add = add nsw i64 %mul, %j.02
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %add
  store i64 0, i64* %arrayidx, align 8
  %mul4 = mul nsw i64 %i.03, 100
  %sub = sub nsw i64 %mul4, %j.02
  %add5 = add nsw i64 %sub, 11
  %arrayidx6 = getelementptr inbounds i64, i64* %A, i64 %add5
  %0 = load i64, i64* %arrayidx6, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %B.addr.11, i64 1
  store i64 %0, i64* %B.addr.11, align 8
  %inc = add nsw i64 %j.02, 1
  %exitcond = icmp ne i64 %inc, 20
  br i1 %exitcond, label %for.body3, label %for.inc7

for.inc7:                                         ; preds = %for.body3
  %scevgep = getelementptr i64, i64* %B.addr.04, i64 20
  %inc8 = add nsw i64 %i.03, 1
  %exitcond5 = icmp ne i64 %inc8, 20
  br i1 %exitcond5, label %for.cond1.preheader, label %for.end9

for.end9:                                         ; preds = %for.inc7
  ret void
}
