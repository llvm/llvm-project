; RUN: llc -march=x86-64 %s -o -

target triple = "x86_64-unknown-linux-gnu"

define void @autogen_SD1794() {
BB:
  %Cmp45 = icmp slt <4 x i32> undef, undef
  br label %CF243

CF243:                                            ; preds = %CF243, %BB
  br i1 undef, label %CF243, label %CF257

CF257:                                            ; preds = %CF243
  %Shuff144 = shufflevector <4 x i1> undef, <4 x i1> %Cmp45, <4 x i32> <i32 undef, i32 undef, i32 5, i32 undef>
  br label %CF244

CF244:                                            ; preds = %CF244, %CF257
  %Shuff182 = shufflevector <4 x i1> %Shuff144, <4 x i1> zeroinitializer, <4 x i32> <i32 3, i32 5, i32 7, i32 undef>
  br label %CF244
}
