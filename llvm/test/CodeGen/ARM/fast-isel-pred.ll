; RUN: llc -O0 -mtriple=armv7-apple-darwin < %s

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  %X = alloca <4 x i32>, align 16
  %Y = alloca <4 x float>, align 16
  store i32 0, i32* %retval
  %tmp = load <4 x i32>* %X, align 16
  call void @__aa(<4 x i32> %tmp, i8* null, i32 3, <4 x float>* %Y)
  %0 = load i32* %retval
  ret i32 %0
}

define internal void @__aa(<4 x i32> %v, i8* %p, i32 %offset, <4 x float>* %constants) nounwind inlinehint ssp {
entry:
  %__a.addr.i = alloca <4 x i32>, align 16
  %v.addr = alloca <4 x i32>, align 16
  %p.addr = alloca i8*, align 4
  %offset.addr = alloca i32, align 4
  %constants.addr = alloca <4 x float>*, align 4
  store <4 x i32> %v, <4 x i32>* %v.addr, align 16
  store i8* %p, i8** %p.addr, align 4
  store i32 %offset, i32* %offset.addr, align 4
  store <4 x float>* %constants, <4 x float>** %constants.addr, align 4
  %tmp = load <4 x i32>* %v.addr, align 16
  store <4 x i32> %tmp, <4 x i32>* %__a.addr.i, align 16
  %tmp.i = load <4 x i32>* %__a.addr.i, align 16
  %0 = bitcast <4 x i32> %tmp.i to <16 x i8>
  %1 = bitcast <16 x i8> %0 to <4 x i32>
  %vcvt.i = sitofp <4 x i32> %1 to <4 x float>
  %tmp1 = load i8** %p.addr, align 4
  %tmp2 = load i32* %offset.addr, align 4
  %tmp3 = load <4 x float>** %constants.addr, align 4
  call void @__bb(<4 x float> %vcvt.i, i8* %tmp1, i32 %tmp2, <4 x float>* %tmp3)
  ret void
}

define internal void @__bb(<4 x float> %v, i8* %p, i32 %offset, <4 x float>* %constants) nounwind inlinehint ssp {
entry:
  %v.addr = alloca <4 x float>, align 16
  %p.addr = alloca i8*, align 4
  %offset.addr = alloca i32, align 4
  %constants.addr = alloca <4 x float>*, align 4
  %data = alloca i64, align 4
  store <4 x float> %v, <4 x float>* %v.addr, align 16
  store i8* %p, i8** %p.addr, align 4
  store i32 %offset, i32* %offset.addr, align 4
  store <4 x float>* %constants, <4 x float>** %constants.addr, align 4
  %tmp = load i64* %data, align 4
  %tmp1 = load i8** %p.addr, align 4
  %tmp2 = load i32* %offset.addr, align 4
  %add.ptr = getelementptr i8* %tmp1, i32 %tmp2
  %0 = bitcast i8* %add.ptr to i64*
  %arrayidx = getelementptr inbounds i64* %0, i32 0
  store i64 %tmp, i64* %arrayidx
  ret void
}
