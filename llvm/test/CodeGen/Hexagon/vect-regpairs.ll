;RUN: llc -march=hexagon -mcpu=hexagonv66 -mhvx -filetype=obj < %s -o - | llvm-objdump --mcpu=hexagonv66 --mattr=+hvx -d - | FileCheck --check-prefix=CHECK-V66 %s
;RUN: llc -march=hexagon -mcpu=hexagonv67 -mhvx -filetype=obj < %s -o - | llvm-objdump --mcpu=hexagonv67 --mattr=+hvx -d - | FileCheck --check-prefix=CHECK-V67 %s

; Should not attempt to use v<even>:<odd> 'reverse' vector regpairs
; on old or new arches (should not crash).

; CHECK-V66: vcombine
; CHECK-V67: vcombine
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>)
declare <16 x i32> @llvm.hexagon.V6.vd0()
declare <32 x i32> @llvm.hexagon.V6.vmpybus(<16 x i32>, i32)
declare <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32>, <32 x i32>, i32)
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>)
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32 )
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>)
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32 )
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>)
declare <16 x i32> @llvm.hexagon.V6.vmpyihb.acc(<16 x i32>, <16 x i32>, i32)
declare <16 x i32> @llvm.hexagon.V6.vasrhubrndsat(<16 x i32>, <16 x i32>, i32)

declare <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32>, <16 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32>, <16 x i32>, i32)
declare <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32>, <16 x i32>, i32)
declare <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32>, <16 x i32>)


define void @Gaussian7x7u8PerRow(i8* %src, i32 %stride, i32 %width, i8* %dst) #0 {
entry:
  %mul = mul i32 %stride, 3
  %idx.neg = sub i32 0, %mul
  %add.ptr = getelementptr i8, i8* %src, i32 %idx.neg
  bitcast i8* %add.ptr to <16 x i32>*
  %mul1 = shl i32 %stride, 1
  %idx.neg2 = sub i32 0, %mul1
  %add.ptr3 = getelementptr i8, i8* %src, i32 %idx.neg2
  bitcast i8* %add.ptr3 to <16 x i32>*
  %idx.neg5 = sub i32 0, %stride
  %add.ptr6 = getelementptr i8, i8* %src, i32 %idx.neg5
  bitcast i8* %add.ptr6 to <16 x i32>*
  bitcast i8* %src to <16 x i32>*
  %add.ptr10 = getelementptr i8, i8* %src, i32 %stride
  bitcast i8* %add.ptr10 to <16 x i32>*
  %add.ptr12 = getelementptr i8, i8* %src, i32 %mul1
  bitcast i8* %add.ptr12 to <16 x i32>*
  %add.ptr14 = getelementptr i8, i8* %src, i32 %mul
  bitcast i8* %add.ptr14 to <16 x i32>*
  bitcast i8* %dst to <16 x i32>*
  load <16 x i32>, <16 x i32>* %0load <16 x i32>, <16 x i32>* %1load <16 x i32>, <16 x i32>* %2load <16 x i32>, <16 x i32>* %3load <16 x i32>, <16 x i32>* %4load <16 x i32>, <16 x i32>* %5load <16 x i32>, <16 x i32>* %6call <16 x i32> @llvm.hexagon.V6.vd0()
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %15, <16 x i32> %15)
  call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %14, <16 x i32> %8)
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %13, <16 x i32> %9)
  call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %17, <32 x i32> %18, i32 101058054)
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %12, <16 x i32> %10)
  call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %19, <32 x i32> %20, i32 252645135)
  call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %21, <16 x i32> %11, i32 336860180)
  %cmp155 = icmp sgt i32 %width, 64
  br i1 %cmp155, label %for.body.preheader, label %for.end
for.body.preheader:                               %incdec.ptr20 = getelementptr i8, i8* %add.ptr14%23 = bitcast i8* %incdec.ptr20 to <16 x i32>*
  %incdec.ptr19 = getelementptr i8, i8* %add.ptr12%24 = bitcast i8* %incdec.ptr19 to <16 x i32>*
  %incdec.ptr18 = getelementptr i8, i8* %add.ptr10%25 = bitcast i8* %incdec.ptr18 to <16 x i32>*
  %incdec.ptr17 = getelementptr i8, i8* %src%26 = bitcast i8* %incdec.ptr17 to <16 x i32>*
  %incdec.ptr16 = getelementptr i8, i8* %add.ptr6%27 = bitcast i8* %incdec.ptr16 to <16 x i32>*
  %incdec.ptr15 = getelementptr i8, i8* %add.ptr3%28 = bitcast i8* %incdec.ptr15 to <16 x i32>*
  %incdec.ptr = getelementptr i8, i8* %add.ptr%29 = bitcast i8* %incdec.ptr to <16 x i32>*
  br label %for.body
for.body:                                         %optr.0166 = phi <16 x i32>* [ %incdec.ptr28, %for.body ], [ %7, %for.body.preheader ]
  %iptr6.0165 = phi <16 x i32>* [ %incdec.ptr27, %for.body ], [ %23, %for.body.preheader ]
  %iptr5.0164 = phi <16 x i32>* [ %incdec.ptr26, %for.body ], [ %24, %for.body.preheader ]
  %iptr4.0163 = phi <16 x i32>* [ %incdec.ptr25, %for.body ], [ %25, %for.body.preheader ]
  %iptr3.0162 = phi <16 x i32>* [ %incdec.ptr24, %for.body ], [ %26, %for.body.preheader ]
  %iptr2.0161 = phi <16 x i32>* [ %incdec.ptr23, %for.body ], [ %27, %for.body.preheader ]
  %iptr1.0160 = phi <16 x i32>* [ %incdec.ptr22, %for.body ], [ %28, %for.body.preheader ]
  %iptr0.0159 = phi <16 x i32>* [ %incdec.ptr21, %for.body ], [ %29, %for.body.preheader ]
  %dXV1.0158 = phi <32 x i32> [ %49, %for.body ], [ %22, %for.body.preheader ]
  %dXV0.0157 = phi <32 x i32> [ %dXV1.0158, %for.body ], [ %16, %for.body.preheader ]
  %i.0156 = phi i32 [ %sub, %for.body ], [ %width, %for.body.preheader ]
  %incdec.ptr21 = getelementptr <16 x i32>, <16 x i32>* %iptr0.0159%30 = load <16 x i32>, <16 x i32>* %iptr0.0159%incdec.ptr22 = getelementptr <16 x i32>, <16 x i32>* %iptr1.0160%31 = load <16 x i32>, <16 x i32>* %iptr1.0160%incdec.ptr23 = getelementptr <16 x i32>, <16 x i32>* %iptr2.0161%32 = load <16 x i32>, <16 x i32>* %iptr2.0161%incdec.ptr24 = getelementptr <16 x i32>, <16 x i32>* %iptr3.0162%33 = load <16 x i32>, <16 x i32>* %iptr3.0162%incdec.ptr25 = getelementptr <16 x i32>, <16 x i32>* %iptr4.0163%34 = load <16 x i32>, <16 x i32>* %iptr4.0163%incdec.ptr26 = getelementptr <16 x i32>, <16 x i32>* %iptr5.0164%35 = load <16 x i32>, <16 x i32>* %iptr5.0164%incdec.ptr27 = getelementptr <16 x i32>, <16 x i32>* %iptr6.0165%36 = load <16 x i32>, <16 x i32>* %iptr6.0165, !tbaa !8
  call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %dXV1.0158)
  call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %dXV0.0157)
  call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %37, <16 x i32> %38, i32 2)
  call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %dXV1.0158)
  call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %dXV0.0157)
  call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %40, <16 x i32> %41, i32 2)
  call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %37, <16 x i32> %38, i32 4)
  call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %36, <16 x i32> %30)
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %35, <16 x i32> %31)
  call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %44, <32 x i32> %45, i32 101058054)
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %34, <16 x i32> %32)
  call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %46, <32 x i32> %47, i32 252645135)
  call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %48, <16 x i32> %33, i32 336860180)
  call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %49)
  call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %50, <16 x i32> %40, i32 2)
  call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %49)
  call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %52, <16 x i32> %37, i32 2)
  call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %50, <16 x i32> %40, i32 4)
  call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %37, <16 x i32> %39)
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %55, <16 x i32> %40)
  call <32 x i32> @llvm.hexagon.V6.vmpahb(<32 x i32> %56, i32 252972820)
  call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %51, <16 x i32> %40)
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %58, <16 x i32> %37)
  call <32 x i32> @llvm.hexagon.V6.vmpahb(<32 x i32> %59, i32 252972820)
  call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %53, <16 x i32> %43)
  call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %51, <16 x i32> %42)
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %61, <16 x i32> %62)
  call <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32> %57, <32 x i32> %63, i32 17170694)
  call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %54, <16 x i32> %42)
  call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %53, <16 x i32> %39)
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %65, <16 x i32> %66)
  call <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32> %60, <32 x i32> %67, i32 17170694)
  call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %64)
  call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %64)
  call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %69, <16 x i32> %70, i32 12)
  call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %68)
  call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %68)
  call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %72, <16 x i32> %73, i32 12)
  call <16 x i32> @llvm.hexagon.V6.vshuffeb(<16 x i32> %74, <16 x i32> %71)
  %incdec.ptr28 = getelementptr <16 x i32>, <16 x i32>* %1
  store <16 x i32> %75, <16 x i32>* %optr.0166%sub = add i32 %i.0156, -64
  %cmp = icmp sgt i32 %sub, 64
  br i1 %cmp, label %for.body, label %for.end
for.end:                                          ret void
}
declare <32 x i32> @llvm.hexagon.V6.vmpahb(<32 x i32>, i32)
declare <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32>, <32 x i32>, i32)
declare <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32>, <16 x i32>, i32)
declare <16 x i32> @llvm.hexagon.V6.vshuffeb(<16 x i32>, <16 x i32>)

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math""target-cpu"="hexagonv65" "target-features"="+hvx-length64b,+hvxv65,+v65,-long-calls" "unsafe-fp-math"}
!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10}
!10 = !{}
!14 = !{}
!19 = !{}
!24 = !{}
