; RUN: opt -S -loop-vectorize -mtriple=x86_64-apple-darwin %s | FileCheck %s

; Two mostly identical functions. The only difference is the presence of
; fast-math flags on the second. The loop is a pretty simple reduction:

; for (int i = 0; i < 32; ++i)
;   if (arr[i] != 42)
;     tot += arr[i];

define double @sumIfScalar(double* nocapture readonly %arr) {
; CHECK-LABEL: define double @sumIfScalar
; CHECK-NOT: <2 x double>

entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %next.iter]
  %tot = phi double [0.0, %entry], [%tot.next, %next.iter]

  %addr = getelementptr double, double* %arr, i32 %i
  %nextval = load double, double* %addr

  %tst = fcmp une double %nextval, 42.0
  br i1 %tst, label %do.add, label %no.add

do.add:
  %tot.new = fadd double %tot, %nextval
  br label %next.iter

no.add:
  br label %next.iter

next.iter:
  %tot.next = phi double [%tot, %no.add], [%tot.new, %do.add]
  %i.next = add i32 %i, 1
  %again = icmp ult i32 %i.next, 32
  br i1 %again, label %loop, label %done

done:
  ret double %tot.next
}

define double @sumIfVector(double* nocapture readonly %arr) {
; CHECK-LABEL: define double @sumIfVector
; CHECK: <2 x double>
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.next, %next.iter]
  %tot = phi double [0.0, %entry], [%tot.next, %next.iter]

  %addr = getelementptr double, double* %arr, i32 %i
  %nextval = load double, double* %addr

  %tst = fcmp fast une double %nextval, 42.0
  br i1 %tst, label %do.add, label %no.add

do.add:
  %tot.new = fadd fast double %tot, %nextval
  br label %next.iter

no.add:
  br label %next.iter

next.iter:
  %tot.next = phi double [%tot, %no.add], [%tot.new, %do.add]
  %i.next = add i32 %i, 1
  %again = icmp ult i32 %i.next, 32
  br i1 %again, label %loop, label %done

done:
  ret double %tot.next
}
