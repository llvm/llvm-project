; RUN: llc < %s -mtriple=x86_64-- -mcpu=corei7 -verify-machineinstrs | FileCheck %s

define void @test(i8** %a, i64* %b, i64 %c, i64 %d) nounwind {
entry:
  %ptrtoarg4 = load i8*, i8** %a, align 8
  %brglist1 = getelementptr i8*, i8** %a, i64 1
  %ptrtoarg25 = load i8*, i8** %brglist1, align 8
  %0 = load i64, i64* %b, align 8
  %1 = mul i64 %0, 4
  %scevgep = getelementptr i8, i8* %ptrtoarg25, i64 %1
  %2 = mul i64 %d, 4
  br label %loop.cond

loop.cond:                                        ; preds = %test.exit, %entry
  %asr.iv6 = phi i8* [ %29, %test.exit ], [ %scevgep, %entry ]
  %iv = phi i64 [ %0, %entry ], [ %28, %test.exit ]
  %3 = icmp eq i64 %iv, %c
  br i1 %3, label %return, label %loop

loop:                                             ; preds = %loop.cond
  %4 = load i64*, i64* addrspace(256)* inttoptr (i64 264 to i64* addrspace(256)*), align 8
  %5 = load i64, i64* %4, align 8
  %vector.size.i = ashr i64 %5, 3
  %num.vector.wi.i = shl i64 %vector.size.i, 3
  %6 = icmp eq i64 %vector.size.i, 0
  br i1 %6, label %scalarIf.i, label %dim_0_vector_pre_head.i

dim_0_vector_pre_head.i:                          ; preds = %loop
  %7 = trunc i64 %5 to i32
  %tempvector_func.i = insertelement <8 x i32> undef, i32 %7, i32 0
  %vectorvector_func.i = shufflevector <8 x i32> %tempvector_func.i, <8 x i32> undef, <8 x i32> zeroinitializer
  br label %vector_kernel_entry.i

vector_kernel_entry.i:                            ; preds = %vector_kernel_entry.i, %dim_0_vector_pre_head.i
  %asr.iv9 = phi i8* [ %scevgep10, %vector_kernel_entry.i ], [ %asr.iv6, %dim_0_vector_pre_head.i ]
  %asr.iv = phi i64 [ %asr.iv.next, %vector_kernel_entry.i ], [ %vector.size.i, %dim_0_vector_pre_head.i ]
  %8 = addrspacecast i8* %ptrtoarg4 to i32 addrspace(1)*
  %asr.iv911 = addrspacecast i8* %asr.iv9 to <8 x i32> addrspace(1)*
  %9 = load <8 x i32>, <8 x i32> addrspace(1)* %asr.iv911, align 4
  %extract8vector_func.i = extractelement <8 x i32> %9, i32 0
  %extract9vector_func.i = extractelement <8 x i32> %9, i32 1
  %extract10vector_func.i = extractelement <8 x i32> %9, i32 2
  %extract11vector_func.i = extractelement <8 x i32> %9, i32 3
  %extract12vector_func.i = extractelement <8 x i32> %9, i32 4
  %extract13vector_func.i = extractelement <8 x i32> %9, i32 5
  %extract14vector_func.i = extractelement <8 x i32> %9, i32 6
  %extract15vector_func.i = extractelement <8 x i32> %9, i32 7
  %10 = atomicrmw min i32 addrspace(1)* %8, i32 %extract8vector_func.i seq_cst
  %11 = atomicrmw min i32 addrspace(1)* %8, i32 %extract9vector_func.i seq_cst
  %12 = atomicrmw min i32 addrspace(1)* %8, i32 %extract10vector_func.i seq_cst
  %13 = atomicrmw min i32 addrspace(1)* %8, i32 %extract11vector_func.i seq_cst
  %14 = atomicrmw min i32 addrspace(1)* %8, i32 %extract12vector_func.i seq_cst
  %15 = atomicrmw min i32 addrspace(1)* %8, i32 %extract13vector_func.i seq_cst
  %16 = atomicrmw min i32 addrspace(1)* %8, i32 %extract14vector_func.i seq_cst
  %17 = atomicrmw min i32 addrspace(1)* %8, i32 %extract15vector_func.i seq_cst
  store <8 x i32> %vectorvector_func.i, <8 x i32> addrspace(1)* %asr.iv911, align 4
  %asr.iv.next = add i64 %asr.iv, -1
  %scevgep10 = getelementptr i8, i8* %asr.iv9, i64 32
  %dim_0_vector_cmp.to.max.i = icmp eq i64 %asr.iv.next, 0
  br i1 %dim_0_vector_cmp.to.max.i, label %scalarIf.i, label %vector_kernel_entry.i

scalarIf.i:                                       ; preds = %vector_kernel_entry.i, %loop
  %exec_wi.i = phi i64 [ 0, %loop ], [ %num.vector.wi.i, %vector_kernel_entry.i ]
  %18 = icmp eq i64 %exec_wi.i, %5
  br i1 %18, label %test.exit, label %dim_0_pre_head.i

dim_0_pre_head.i:                                 ; preds = %scalarIf.i
  %19 = load i64*, i64* addrspace(256)* inttoptr (i64 264 to i64* addrspace(256)*), align 8
  %20 = load i64, i64* %19, align 8
  %21 = trunc i64 %20 to i32
  %22 = mul i64 %vector.size.i, 8
  br label %scalar_kernel_entry.i

scalar_kernel_entry.i:                            ; preds = %scalar_kernel_entry.i, %dim_0_pre_head.i
  %asr.iv12 = phi i64 [ %asr.iv.next13, %scalar_kernel_entry.i ], [ %22, %dim_0_pre_head.i ]
  %23 = addrspacecast i8* %asr.iv6 to i32 addrspace(1)*
  %24 = addrspacecast i8* %ptrtoarg4 to i32 addrspace(1)*
  %scevgep16 = getelementptr i32, i32 addrspace(1)* %23, i64 %asr.iv12
  %25 = load i32, i32 addrspace(1)* %scevgep16, align 4
  %26 = atomicrmw min i32 addrspace(1)* %24, i32 %25 seq_cst
  %scevgep15 = getelementptr i32, i32 addrspace(1)* %23, i64 %asr.iv12
  store i32 %21, i32 addrspace(1)* %scevgep15, align 4
  %asr.iv.next13 = add i64 %asr.iv12, 1
  %dim_0_cmp.to.max.i = icmp eq i64 %5, %asr.iv.next13
  br i1 %dim_0_cmp.to.max.i, label %test.exit, label %scalar_kernel_entry.i

test.exit:                     ; preds = %scalar_kernel_entry.i, %scalarIf.i
  %27 = bitcast i8* %asr.iv6 to i1*
  %28 = add i64 %iv, %d
  store i64 %28, i64* %b, align 8
  %scevgep8 = getelementptr i1, i1* %27, i64 %2
  %29 = bitcast i1* %scevgep8 to i8*
  br label %loop.cond

return:                                           ; preds = %loop.cond
  store i64 %0, i64* %b, align 8
  ret void
}

; CHECK: test
; CHECK: decq
; CHECK-NOT: cmpxchgl
; CHECK: jne
; CHECK: ret
