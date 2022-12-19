; RUN: llc -O0 -verify-machineinstrs -mtriple=armv7-apple-darwin < %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=armv7-linux-gnueabi < %s

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  %X = alloca <4 x i32>, align 16
  %Y = alloca <4 x float>, align 16
  store i32 0, ptr %retval
  %tmp = load <4 x i32>, ptr %X, align 16
  call void @__aa(<4 x i32> %tmp, ptr null, i32 3, ptr %Y)
  %0 = load i32, ptr %retval
  ret i32 %0
}

define internal void @__aa(<4 x i32> %v, ptr %p, i32 %offset, ptr %constants) nounwind inlinehint ssp {
entry:
  %__a.addr.i = alloca <4 x i32>, align 16
  %v.addr = alloca <4 x i32>, align 16
  %p.addr = alloca ptr, align 4
  %offset.addr = alloca i32, align 4
  %constants.addr = alloca ptr, align 4
  store <4 x i32> %v, ptr %v.addr, align 16
  store ptr %p, ptr %p.addr, align 4
  store i32 %offset, ptr %offset.addr, align 4
  store ptr %constants, ptr %constants.addr, align 4
  %tmp = load <4 x i32>, ptr %v.addr, align 16
  store <4 x i32> %tmp, ptr %__a.addr.i, align 16
  %tmp.i = load <4 x i32>, ptr %__a.addr.i, align 16
  %0 = bitcast <4 x i32> %tmp.i to <16 x i8>
  %1 = bitcast <16 x i8> %0 to <4 x i32>
  %vcvt.i = sitofp <4 x i32> %1 to <4 x float>
  %tmp1 = load ptr, ptr %p.addr, align 4
  %tmp2 = load i32, ptr %offset.addr, align 4
  %tmp3 = load ptr, ptr %constants.addr, align 4
  call void @__bb(<4 x float> %vcvt.i, ptr %tmp1, i32 %tmp2, ptr %tmp3)
  ret void
}

define internal void @__bb(<4 x float> %v, ptr %p, i32 %offset, ptr %constants) nounwind inlinehint ssp {
entry:
  %v.addr = alloca <4 x float>, align 16
  %p.addr = alloca ptr, align 4
  %offset.addr = alloca i32, align 4
  %constants.addr = alloca ptr, align 4
  %data = alloca i64, align 4
  store <4 x float> %v, ptr %v.addr, align 16
  store ptr %p, ptr %p.addr, align 4
  store i32 %offset, ptr %offset.addr, align 4
  store ptr %constants, ptr %constants.addr, align 4
  %tmp = load i64, ptr %data, align 4
  %tmp1 = load ptr, ptr %p.addr, align 4
  %tmp2 = load i32, ptr %offset.addr, align 4
  %add.ptr = getelementptr i8, ptr %tmp1, i32 %tmp2
  store i64 %tmp, ptr %add.ptr
  ret void
}
