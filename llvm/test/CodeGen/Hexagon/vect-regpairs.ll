;RUN: llc -mtriple=hexagon -mcpu=hexagonv66 -mhvx -filetype=obj < %s -o - | llvm-objdump --mcpu=hexagonv66 --mattr=+hvx -d - | FileCheck --check-prefix=CHECK-V66 %s
;RUN: llc -mtriple=hexagon -mcpu=hexagonv67 -mhvx -filetype=obj < %s -o - | llvm-objdump --mcpu=hexagonv67 --mattr=+hvx -d - | FileCheck --check-prefix=CHECK-V67 %s

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


define void @Gaussian7x7u8PerRow(ptr %src, i32 %stride, i32 %width, ptr %dst) #0 {
entry:
  %mul = mul i32 %stride, 3
  %idx.neg = sub i32 0, %mul
  %add.ptr = getelementptr i8, ptr %src, i32 %idx.neg
  bitcast ptr %add.ptr to ptr
  %mul1 = shl i32 %stride, 1
  %idx.neg2 = sub i32 0, %mul1
  %add.ptr3 = getelementptr i8, ptr %src, i32 %idx.neg2
  bitcast ptr %add.ptr3 to ptr
  %idx.neg5 = sub i32 0, %stride
  %add.ptr6 = getelementptr i8, ptr %src, i32 %idx.neg5
  bitcast ptr %add.ptr6 to ptr
  bitcast ptr %src to ptr
  %add.ptr10 = getelementptr i8, ptr %src, i32 %stride
  bitcast ptr %add.ptr10 to ptr
  %add.ptr12 = getelementptr i8, ptr %src, i32 %mul1
  bitcast ptr %add.ptr12 to ptr
  %add.ptr14 = getelementptr i8, ptr %src, i32 %mul
  bitcast ptr %add.ptr14 to ptr
  bitcast ptr %dst to ptr
  load <16 x i32>, ptr %0load <16 x i32>, ptr %1load <16 x i32>, ptr %2load <16 x i32>, ptr %3load <16 x i32>, ptr %4load <16 x i32>, ptr %5load <16 x i32>, ptr %6call <16 x i32> @llvm.hexagon.V6.vd0()
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %15, <16 x i32> %15)
  call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %14, <16 x i32> %8)
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %13, <16 x i32> %9)
  call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %17, <32 x i32> %18, i32 101058054)
  call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %12, <16 x i32> %10)
  call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %19, <32 x i32> %20, i32 252645135)
  call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %21, <16 x i32> %11, i32 336860180)
  %cmp155 = icmp sgt i32 %width, 64
  br i1 %cmp155, label %for.body.preheader, label %for.end
for.body.preheader:                               %incdec.ptr20 = getelementptr i8, ptr %add.ptr14%23 = bitcast ptr %incdec.ptr20 to ptr
  %incdec.ptr19 = getelementptr i8, ptr %add.ptr12%24 = bitcast ptr %incdec.ptr19 to ptr
  %incdec.ptr18 = getelementptr i8, ptr %add.ptr10%25 = bitcast ptr %incdec.ptr18 to ptr
  %incdec.ptr17 = getelementptr i8, ptr %src%26 = bitcast ptr %incdec.ptr17 to ptr
  %incdec.ptr16 = getelementptr i8, ptr %add.ptr6%27 = bitcast ptr %incdec.ptr16 to ptr
  %incdec.ptr15 = getelementptr i8, ptr %add.ptr3%28 = bitcast ptr %incdec.ptr15 to ptr
  %incdec.ptr = getelementptr i8, ptr %add.ptr%29 = bitcast ptr %incdec.ptr to ptr
  br label %for.body
for.body:                                         %optr.0166 = phi ptr [ %incdec.ptr28, %for.body ], [ %7, %for.body.preheader ]
  %iptr6.0165 = phi ptr [ %incdec.ptr27, %for.body ], [ %23, %for.body.preheader ]
  %iptr5.0164 = phi ptr [ %incdec.ptr26, %for.body ], [ %24, %for.body.preheader ]
  %iptr4.0163 = phi ptr [ %incdec.ptr25, %for.body ], [ %25, %for.body.preheader ]
  %iptr3.0162 = phi ptr [ %incdec.ptr24, %for.body ], [ %26, %for.body.preheader ]
  %iptr2.0161 = phi ptr [ %incdec.ptr23, %for.body ], [ %27, %for.body.preheader ]
  %iptr1.0160 = phi ptr [ %incdec.ptr22, %for.body ], [ %28, %for.body.preheader ]
  %iptr0.0159 = phi ptr [ %incdec.ptr21, %for.body ], [ %29, %for.body.preheader ]
  %dXV1.0158 = phi <32 x i32> [ %49, %for.body ], [ %22, %for.body.preheader ]
  %dXV0.0157 = phi <32 x i32> [ %dXV1.0158, %for.body ], [ %16, %for.body.preheader ]
  %i.0156 = phi i32 [ %sub, %for.body ], [ %width, %for.body.preheader ]
  %incdec.ptr21 = getelementptr <16 x i32>, ptr %iptr0.0159%30 = load <16 x i32>, ptr %iptr0.0159%incdec.ptr22 = getelementptr <16 x i32>, ptr %iptr1.0160%31 = load <16 x i32>, ptr %iptr1.0160%incdec.ptr23 = getelementptr <16 x i32>, ptr %iptr2.0161%32 = load <16 x i32>, ptr %iptr2.0161%incdec.ptr24 = getelementptr <16 x i32>, ptr %iptr3.0162%33 = load <16 x i32>, ptr %iptr3.0162%incdec.ptr25 = getelementptr <16 x i32>, ptr %iptr4.0163%34 = load <16 x i32>, ptr %iptr4.0163%incdec.ptr26 = getelementptr <16 x i32>, ptr %iptr5.0164%35 = load <16 x i32>, ptr %iptr5.0164%incdec.ptr27 = getelementptr <16 x i32>, ptr %iptr6.0165%36 = load <16 x i32>, ptr %iptr6.0165, !tbaa !8
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
  %incdec.ptr28 = getelementptr <16 x i32>, ptr %1
  store <16 x i32> %75, ptr %optr.0166%sub = add i32 %i.0156, -64
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
