; RUN: opt -instcombine < %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.S0.0.1.2.3.4.13.22.31.44.48.53.54.55.56.58.59.60.66.68.70.74.77.106.107.108.109.110.113.117.118.128.129 = type <{ i64 }>

; Function Attrs: nounwind
define void @main() #0 {
entry:
  %l_819.i.i = alloca %struct.S0.0.1.2.3.4.13.22.31.44.48.53.54.55.56.58.59.60.66.68.70.74.77.106.107.108.109.110.113.117.118.128.129, align 8
  br i1 undef, label %land.lhs.true, label %for.cond.i

land.lhs.true:                                    ; preds = %entry
  br label %for.cond.i

for.cond.i:                                       ; preds = %land.lhs.true, %entry
  %0 = getelementptr inbounds %struct.S0.0.1.2.3.4.13.22.31.44.48.53.54.55.56.58.59.60.66.68.70.74.77.106.107.108.109.110.113.117.118.128.129, %struct.S0.0.1.2.3.4.13.22.31.44.48.53.54.55.56.58.59.60.66.68.70.74.77.106.107.108.109.110.113.117.118.128.129* %l_819.i.i, i64 0, i32 0
  br label %for.cond.i6.i.i

for.cond.i6.i.i:                                  ; preds = %for.body.i8.i.i, %for.cond.i
  br i1 undef, label %for.body.i8.i.i, label %lbl_707.i.i.i

for.body.i8.i.i:                                  ; preds = %for.cond.i6.i.i
  br label %for.cond.i6.i.i

lbl_707.i.i.i:                                    ; preds = %for.cond.i6.i.i
  br i1 undef, label %lor.rhs.i.i.i, label %lor.end.i.i.i

lor.rhs.i.i.i:                                    ; preds = %lbl_707.i.i.i
  br label %lor.end.i.i.i

lor.end.i.i.i:                                    ; preds = %lor.rhs.i.i.i, %lbl_707.i.i.i
  br label %for.cond1.i.i.i.i

for.cond1.i.i.i.i:                                ; preds = %for.body4.i.i.i.i, %lor.end.i.i.i
  br i1 undef, label %for.body4.i.i.i.i, label %func_39.exit.i.i

for.body4.i.i.i.i:                                ; preds = %for.cond1.i.i.i.i
  br label %for.cond1.i.i.i.i

func_39.exit.i.i:                                 ; preds = %for.cond1.i.i.i.i
  %l_8191.sroa.0.0.copyload.i.i = load i64, i64* %0, align 1
  br label %for.cond1.i.i.i

for.cond1.i.i.i:                                  ; preds = %safe_div_func_uint32_t_u_u.exit.i.i.i, %func_39.exit.i.i
  br i1 undef, label %for.cond7.i.i.i, label %func_11.exit.i

for.cond7.i.i.i:                                  ; preds = %for.end30.i.i.i, %for.cond1.i.i.i
  %storemerge.i.i.i = phi i32 [ %sub.i.i.i, %for.end30.i.i.i ], [ 4, %for.cond1.i.i.i ]
  br i1 undef, label %for.cond22.i.i.i, label %for.end32.i.i.i

for.cond22.i.i.i:                                 ; preds = %for.body25.i.i.i, %for.cond7.i.i.i
  br i1 undef, label %for.body25.i.i.i, label %for.end30.i.i.i

for.body25.i.i.i:                                 ; preds = %for.cond22.i.i.i
  br label %for.cond22.i.i.i

for.end30.i.i.i:                                  ; preds = %for.cond22.i.i.i
  %sub.i.i.i = add nsw i32 0, -1
  br label %for.cond7.i.i.i

for.end32.i.i.i:                                  ; preds = %for.cond7.i.i.i
  %conv33.i.i.i = trunc i64 %l_8191.sroa.0.0.copyload.i.i to i32
  %xor.i.i.i.i = xor i32 %storemerge.i.i.i, -701565022
  %sub.i.i.i.i = sub nsw i32 0, %storemerge.i.i.i
  %xor3.i.i.i.i = xor i32 %sub.i.i.i.i, %storemerge.i.i.i
  %and4.i.i.i.i = and i32 %xor.i.i.i.i, %xor3.i.i.i.i
  %cmp.i.i.i.i = icmp slt i32 %and4.i.i.i.i, 0
  %sub5.i.i.i.i = sub nsw i32 -701565022, %storemerge.i.i.i
  %.sub5.i.i.i.i = select i1 %cmp.i.i.i.i, i32 -701565022, i32 %sub5.i.i.i.i
  br i1 undef, label %safe_div_func_uint32_t_u_u.exit.i.i.i, label %cond.false.i.i.i.i

cond.false.i.i.i.i:                               ; preds = %for.end32.i.i.i
  %div.i.i.i.i = udiv i32 %conv33.i.i.i, %.sub5.i.i.i.i
  br label %safe_div_func_uint32_t_u_u.exit.i.i.i

safe_div_func_uint32_t_u_u.exit.i.i.i:            ; preds = %cond.false.i.i.i.i, %for.end32.i.i.i
  %cond.i.i.i.i = phi i32 [ %div.i.i.i.i, %cond.false.i.i.i.i ], [ %conv33.i.i.i, %for.end32.i.i.i ]
  %cmp35.i.i.i = icmp ne i32 %cond.i.i.i.i, -7
  br label %for.cond1.i.i.i

func_11.exit.i:                                   ; preds = %for.cond1.i.i.i
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %func_11.exit.i
  unreachable

for.end:                                          ; preds = %func_11.exit.i
  br label %for.cond15

for.cond15:                                       ; preds = %for.cond19, %for.end
  br i1 undef, label %for.cond19, label %for.end45

for.cond19:                                       ; preds = %for.cond15
  br label %for.cond15

for.end45:                                        ; preds = %for.cond15
  unreachable
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
