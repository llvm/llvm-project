; RUN: %lli -O0 -disable-lazy-compilation=false %s

; The intention of this test is to verify that symbols mapped to COMMON in ELF
; work as expected.
;
; Compiled from this C code:
;
; int zero_int;
; double zero_double;
; int zero_arr[10];
; 
; int main()
; {
;     zero_arr[zero_int + 5] = 40;
; 
;     if (zero_double < 1.0)
;         zero_arr[zero_int + 2] = 70;
; 
;     for (int i = 1; i < 10; ++i) {
;         zero_arr[i] = zero_arr[i - 1] + zero_arr[i];
;     }
;     return zero_arr[9] == 110 ? 0 : -1;
; }

@zero_int = common global i32 0, align 4
@zero_arr = common global [10 x i32] zeroinitializer, align 16
@zero_double = common global double 0.000000e+00, align 8

define i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32, i32* @zero_int, align 4
  %add = add nsw i32 %0, 5
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* @zero_arr, i32 0, i64 %idxprom
  store i32 40, i32* %arrayidx, align 4
  %1 = load double, double* @zero_double, align 8
  %cmp = fcmp olt double %1, 1.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %2 = load i32, i32* @zero_int, align 4
  %add1 = add nsw i32 %2, 2
  %idxprom2 = sext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds [10 x i32], [10 x i32]* @zero_arr, i32 0, i64 %idxprom2
  store i32 70, i32* %arrayidx3, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  store i32 1, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.end
  %3 = load i32, i32* %i, align 4
  %cmp4 = icmp slt i32 %3, 10
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %4 = load i32, i32* %i, align 4
  %sub = sub nsw i32 %4, 1
  %idxprom5 = sext i32 %sub to i64
  %arrayidx6 = getelementptr inbounds [10 x i32], [10 x i32]* @zero_arr, i32 0, i64 %idxprom5
  %5 = load i32, i32* %arrayidx6, align 4
  %6 = load i32, i32* %i, align 4
  %idxprom7 = sext i32 %6 to i64
  %arrayidx8 = getelementptr inbounds [10 x i32], [10 x i32]* @zero_arr, i32 0, i64 %idxprom7
  %7 = load i32, i32* %arrayidx8, align 4
  %add9 = add nsw i32 %5, %7
  %8 = load i32, i32* %i, align 4
  %idxprom10 = sext i32 %8 to i64
  %arrayidx11 = getelementptr inbounds [10 x i32], [10 x i32]* @zero_arr, i32 0, i64 %idxprom10
  store i32 %add9, i32* %arrayidx11, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %9 = load i32, i32* %i, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %10 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @zero_arr, i32 0, i64 9), align 4
  %cmp12 = icmp eq i32 %10, 110
  %cond = select i1 %cmp12, i32 0, i32 -1
  ret i32 %cond
}
