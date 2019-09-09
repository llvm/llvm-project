; RUN: opt -inline -inline-threshold=225 -inlinehint-threshold=360 -S < %s | FileCheck %s

@data = common global i32* null, align 8

define i32 @fct1(i32 %a) nounwind uwtable ssp {
entry:
  %a.addr = alloca i32, align 4
  %res = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %tmp = load i32, i32* %a.addr, align 4
  %idxprom = sext i32 %tmp to i64
  %tmp1 = load i32*, i32** @data, align 8
  %arrayidx = getelementptr inbounds i32, i32* %tmp1, i64 %idxprom
  %tmp2 = load i32, i32* %arrayidx, align 4
  %tmp3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %tmp3, 1
  %idxprom1 = sext i32 %add to i64
  %tmp4 = load i32*, i32** @data, align 8
  %arrayidx2 = getelementptr inbounds i32, i32* %tmp4, i64 %idxprom1
  %tmp5 = load i32, i32* %arrayidx2, align 4
  %mul = mul nsw i32 %tmp2, %tmp5
  store i32 %mul, i32* %res, align 4
  store i32 0, i32* %i, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %tmp6 = load i32, i32* %i, align 4
  %tmp7 = load i32, i32* %res, align 4
  %cmp = icmp slt i32 %tmp6, %tmp7
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp8 = load i32, i32* %i, align 4
  %idxprom3 = sext i32 %tmp8 to i64
  %tmp9 = load i32*, i32** @data, align 8
  %arrayidx4 = getelementptr inbounds i32, i32* %tmp9, i64 %idxprom3
  call void @fct0(i32* %arrayidx4)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %tmp10 = load i32, i32* %i, align 4
  %inc = add nsw i32 %tmp10, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  store i32 0, i32* %i, align 4
  br label %for.cond5

for.cond5:                                        ; preds = %for.inc10, %for.end
  %tmp11 = load i32, i32* %i, align 4
  %tmp12 = load i32, i32* %res, align 4
  %cmp6 = icmp slt i32 %tmp11, %tmp12
  br i1 %cmp6, label %for.body7, label %for.end12

for.body7:                                        ; preds = %for.cond5
  %tmp13 = load i32, i32* %i, align 4
  %idxprom8 = sext i32 %tmp13 to i64
  %tmp14 = load i32*, i32** @data, align 8
  %arrayidx9 = getelementptr inbounds i32, i32* %tmp14, i64 %idxprom8
  call void @fct0(i32* %arrayidx9)
  br label %for.inc10

for.inc10:                                        ; preds = %for.body7
  %tmp15 = load i32, i32* %i, align 4
  %inc11 = add nsw i32 %tmp15, 1
  store i32 %inc11, i32* %i, align 4
  br label %for.cond5

for.end12:                                        ; preds = %for.cond5
  store i32 0, i32* %i, align 4
  br label %for.cond13

for.cond13:                                       ; preds = %for.inc18, %for.end12
  %tmp16 = load i32, i32* %i, align 4
  %tmp17 = load i32, i32* %res, align 4
  %cmp14 = icmp slt i32 %tmp16, %tmp17
  br i1 %cmp14, label %for.body15, label %for.end20

for.body15:                                       ; preds = %for.cond13
  %tmp18 = load i32, i32* %i, align 4
  %idxprom16 = sext i32 %tmp18 to i64
  %tmp19 = load i32*, i32** @data, align 8
  %arrayidx17 = getelementptr inbounds i32, i32* %tmp19, i64 %idxprom16
  call void @fct0(i32* %arrayidx17)
  br label %for.inc18

for.inc18:                                        ; preds = %for.body15
  %tmp20 = load i32, i32* %i, align 4
  %inc19 = add nsw i32 %tmp20, 1
  store i32 %inc19, i32* %i, align 4
  br label %for.cond13

for.end20:                                        ; preds = %for.cond13
  %tmp21 = load i32, i32* %res, align 4
  ret i32 %tmp21
}

declare void @fct0(i32*)

define i32 @fct2(i32 %a) nounwind uwtable inlinehint ssp {
entry:
  %a.addr = alloca i32, align 4
  %res = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %tmp = load i32, i32* %a.addr, align 4
  %shl = shl i32 %tmp, 1
  %idxprom = sext i32 %shl to i64
  %tmp1 = load i32*, i32** @data, align 8
  %arrayidx = getelementptr inbounds i32, i32* %tmp1, i64 %idxprom
  %tmp2 = load i32, i32* %arrayidx, align 4
  %tmp3 = load i32, i32* %a.addr, align 4
  %shl1 = shl i32 %tmp3, 1
  %add = add nsw i32 %shl1, 13
  %idxprom2 = sext i32 %add to i64
  %tmp4 = load i32*, i32** @data, align 8
  %arrayidx3 = getelementptr inbounds i32, i32* %tmp4, i64 %idxprom2
  %tmp5 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %tmp2, %tmp5
  store i32 %mul, i32* %res, align 4
  store i32 0, i32* %i, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %tmp6 = load i32, i32* %i, align 4
  %tmp7 = load i32, i32* %res, align 4
  %cmp = icmp slt i32 %tmp6, %tmp7
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp8 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %tmp8 to i64
  %tmp9 = load i32*, i32** @data, align 8
  %arrayidx5 = getelementptr inbounds i32, i32* %tmp9, i64 %idxprom4
  call void @fct0(i32* %arrayidx5)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %tmp10 = load i32, i32* %i, align 4
  %inc = add nsw i32 %tmp10, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  store i32 0, i32* %i, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc11, %for.end
  %tmp11 = load i32, i32* %i, align 4
  %tmp12 = load i32, i32* %res, align 4
  %cmp7 = icmp slt i32 %tmp11, %tmp12
  br i1 %cmp7, label %for.body8, label %for.end13

for.body8:                                        ; preds = %for.cond6
  %tmp13 = load i32, i32* %i, align 4
  %idxprom9 = sext i32 %tmp13 to i64
  %tmp14 = load i32*, i32** @data, align 8
  %arrayidx10 = getelementptr inbounds i32, i32* %tmp14, i64 %idxprom9
  call void @fct0(i32* %arrayidx10)
  br label %for.inc11

for.inc11:                                        ; preds = %for.body8
  %tmp15 = load i32, i32* %i, align 4
  %inc12 = add nsw i32 %tmp15, 1
  store i32 %inc12, i32* %i, align 4
  br label %for.cond6

for.end13:                                        ; preds = %for.cond6
  store i32 0, i32* %i, align 4
  br label %for.cond14

for.cond14:                                       ; preds = %for.inc19, %for.end13
  %tmp16 = load i32, i32* %i, align 4
  %tmp17 = load i32, i32* %res, align 4
  %cmp15 = icmp slt i32 %tmp16, %tmp17
  br i1 %cmp15, label %for.body16, label %for.end21

for.body16:                                       ; preds = %for.cond14
  %tmp18 = load i32, i32* %i, align 4
  %idxprom17 = sext i32 %tmp18 to i64
  %tmp19 = load i32*, i32** @data, align 8
  %arrayidx18 = getelementptr inbounds i32, i32* %tmp19, i64 %idxprom17
  call void @fct0(i32* %arrayidx18)
  br label %for.inc19

for.inc19:                                        ; preds = %for.body16
  %tmp20 = load i32, i32* %i, align 4
  %inc20 = add nsw i32 %tmp20, 1
  store i32 %inc20, i32* %i, align 4
  br label %for.cond14

for.end21:                                        ; preds = %for.cond14
  %tmp21 = load i32, i32* %res, align 4
  ret i32 %tmp21
}

define i32 @fct3(i32 %c) nounwind uwtable ssp {
entry:
  ;CHECK-LABEL: @fct3(
  ;CHECK: call i32 @fct1
  ; The inline keyword gives a sufficient benefits to inline fct2
  ;CHECK-NOT: call i32 @fct2
  %c.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  %tmp = load i32, i32* %c.addr, align 4
  %call = call i32 @fct1(i32 %tmp)
  %tmp1 = load i32, i32* %c.addr, align 4
  %call1 = call i32 @fct2(i32 %tmp1)
  %add = add nsw i32 %call, %call1
  ret i32 %add
}

define i32 @fct4(i32 %c) minsize nounwind uwtable ssp {
entry:
  ;CHECK-LABEL: @fct4(
  ;CHECK: call i32 @fct1
  ; With Oz (minsize attribute), the benefit of inlining fct2
  ; is the same as fct1, thus no inlining for fct2
  ;CHECK: call i32 @fct2
  %c.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  %tmp = load i32, i32* %c.addr, align 4
  %call = call i32 @fct1(i32 %tmp)
  %tmp1 = load i32, i32* %c.addr, align 4
  %call1 = call i32 @fct2(i32 %tmp1)
  %add = add nsw i32 %call, %call1
  ret i32 %add
}
