; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:    < %s | FileCheck %s --check-prefix=POWERPC_64LE

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64-ibm-aix \
; RUN:  < %s | FileCheck %s --check-prefix=POWERPC_64

; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc-ibm-aix \
; RUN:   < %s | FileCheck %s --check-prefix=POWERPC_32

define i32 @test_Greater_than(ptr %colauths, i32 signext %ncols) {
; This testcase is manually reduced to isolate the critical code blocks.
; It is designed to check for vector comparison specifically for zero vectors.
; In the vector.body section, we are expecting a comparison instruction (vcmpequh), 
; merge instructions (vmrghh and vmrglh) which use exactly 2 vectors. 
; The output of the merge instruction is being used by xxland and finally 
; accumulated by vadduwm instruction.

; POWERPC_64LE-LABEL: test_Greater_than:
; POWERPC_64LE:  .LBB0_6: # %vector.body
; POWERPC_64LE-NEXT:    #
; POWERPC_64LE-NEXT:    lxv [[R1:[0-9]+]], -64(4)
; POWERPC_64LE-NEXT:    vcmpequh [[R2:[0-9]+]], [[R2]], [[R3:[0-9]+]]
; POWERPC_64LE-NEXT:    xxlnor [[R1]], [[R1]], [[R1]]
; POWERPC_64LE-NEXT:    vmrghh [[R4:[0-9]+]], [[R2]], [[R2]]
; POWERPC_64LE-NEXT:    vmrglh [[R2]], [[R2]], [[R2]]
; POWERPC_64LE-NEXT:    xxland [[R5:[0-9]+]], [[R5]], [[R6:[0-9]+]]
; POWERPC_64LE-NEXT:    xxland [[R1]], [[R1]], [[R6]]
; POWERPC_64LE-NEXT:    vadduwm [[R7:[0-9]+]], [[R7]], [[R4]]
; POWERPC_64LE:  .LBB0_10: # %vec.epilog.vector.body
; POWERPC_64LE-NEXT:    #
; POWERPC_64LE-NEXT:    lxv [[R8:[0-9]+]], 0(4)
; POWERPC_64LE-NEXT:    addi 4, 4, 16
; POWERPC_64LE-NEXT:    vcmpequh [[R9:[0-9]+]], [[R9]], [[R10:[0-9]+]]
; POWERPC_64LE-NEXT:    xxlnor [[R8]], [[R8]], [[R8]]
; POWERPC_64LE-NEXT:    vmrglh [[R11:[0-9]+]], [[R9]], [[R9]]
; POWERPC_64LE-NEXT:    vmrghh [[R9]], [[R9]], [[R9]]
; POWERPC_64LE-NEXT:    xxland [[R12:[0-9]+]], [[R12]], [[R6]]
; POWERPC_64LE-NEXT:    xxland [[R8]], [[R8]], [[R6]]
; POWERPC_64LE-NEXT:    vadduwm [[R7]], [[R7]], [[R9]]
; POWERPC_64LE-NEXT:    vadduwm [[R3]], [[R3]], [[R11]]
; POWERPC_64LE-NEXT:    bdnz .LBB0_10
; POWERPC_64LE:    blr
;
; POWERPC_64-LABEL: test_Greater_than:
; POWERPC_64:  L..BB0_6: # %vector.body
; POWERPC_64-NEXT:    #
; POWERPC_64-NEXT:    lxv [[R1:[0-9]+]], -64(4)
; POWERPC_64-NEXT:    vcmpequh [[R2:[0-9]+]], [[R2]], [[R3:[0-9]+]]
; POWERPC_64-NEXT:    xxlnor [[R1]], [[R1]], [[R1]]
; POWERPC_64-NEXT:    vmrglh [[R4:[0-9]+]], [[R2]], [[R2]]
; POWERPC_64-NEXT:    vmrghh [[R2]], [[R2]], [[R2]]
; POWERPC_64-NEXT:    xxland [[R5:[0-9]+]], [[R5]], [[R6:[0-9]+]]
; POWERPC_64-NEXT:    xxland [[R1]], [[R1]], [[R6]]
; POWERPC_64-NEXT:    vadduwm [[R7:[0-9]+]], [[R7]], [[R4]]
; POWERPC_64:  L..BB0_10: # %vec.epilog.vector.body
; POWERPC_64-NEXT:    #
; POWERPC_64-NEXT:    lxv [[R8:[0-9]+]], 0(4)
; POWERPC_64-NEXT:    addi 4, 4, 16
; POWERPC_64-NEXT:    vcmpequh [[R9:[0-9]+]], [[R9]], [[R10:[0-9]+]]
; POWERPC_64-NEXT:    xxlnor [[R8]], [[R8]], [[R8]]
; POWERPC_64-NEXT:    vmrghh [[R11:[0-9]+]], [[R9]], [[R9]]
; POWERPC_64-NEXT:    vmrglh [[R9]], [[R9]], [[R9]]
; POWERPC_64-NEXT:    xxland [[R12:[0-9]+]], [[R12]], [[R6]]
; POWERPC_64-NEXT:    xxland [[R8]], [[R8]], [[R6]]
; POWERPC_64-NEXT:    vadduwm [[R7]], [[R7]], [[R9]]
; POWERPC_64-NEXT:    vadduwm [[R3]], [[R3]], [[R11]]
; POWERPC_64-NEXT:    bdnz L..BB0_10
; POWERPC_64:    blr
;
; POWERPC_32-LABEL: test_Greater_than:
; POWERPC_32:  L..BB0_7: # %vector.body
; POWERPC_32-NEXT:    #
; POWERPC_32-NEXT:    lxv [[R1:[0-9]+]], 0(10)
; POWERPC_32-NEXT:    addic [[R13:[0-9]+]], [[R13]], 64
; POWERPC_32-NEXT:    addze [[R14:[0-9]+]], [[R14]]
; POWERPC_32-NEXT:    xor [[R15:[0-9]+]], [[R13]], [[R16:[0-9]+]]
; POWERPC_32-NEXT:    or. [[R15]], [[R15]], [[R14]]
; POWERPC_32-NEXT:    vcmpequh [[R2:[0-9]+]], [[R2]], [[R3:[0-9]+]]
; POWERPC_32-NEXT:    xxlnor [[R1]], [[R1]], [[R1]]
; POWERPC_32-NEXT:    vmrglh [[R4:[0-9]+]], [[R2]], [[R2]]
; POWERPC_32-NEXT:    vmrghh [[R2]], [[R2]], [[R2]]
; POWERPC_32-NEXT:    xxland [[R5:[0-9]+]], [[R5]], [[R6:[0-9]+]]
; POWERPC_32-NEXT:    xxland [[R1]], [[R1]], [[R6]]
; POWERPC_32-NEXT:    vadduwm [[R7:[0-9]+]], [[R7]], [[R4]]
; POWERPC_32:  L..BB0_11: # %vec.epilog.vector.body
; POWERPC_32-NEXT:    #
; POWERPC_32-NEXT:    slwi [[R14]], [[R13]], 1
; POWERPC_32-NEXT:    addic [[R13]], [[R13]], 8
; POWERPC_32-NEXT:    addze [[R17:[0-9]+]], [[R17]]
; POWERPC_32-NEXT:    lxvx [[R8:[0-9]+]], [[R18:[0-9]+]], [[R14]]
; POWERPC_32-NEXT:    xor [[R14]], [[R13]], [[R16]]
; POWERPC_32-NEXT:    or. [[R14]], [[R14]], [[R17]]
; POWERPC_32-NEXT:    vcmpequh [[R9:[0-9]+]], [[R9]], [[R3]]
; POWERPC_32-NEXT:    xxlnor [[R8]], [[R8]], [[R8]]
; POWERPC_32-NEXT:    vmrghh [[R11:[0-9]+]], [[R9]], [[R9]]
; POWERPC_32-NEXT:    vmrglh [[R9]], [[R9]], [[R9]]
; POWERPC_32-NEXT:    xxland [[R12:[0-9]+]], [[R12]], [[R6]]
; POWERPC_32-NEXT:    xxland [[R8]], [[R8]], [[R6]]
; POWERPC_32-NEXT:    vadduwm [[R7]], [[R7]], [[R9]]
; POWERPC_32-NEXT:    vadduwm [[R19:[0-9]+]], [[R19]], [[R11]]
; POWERPC_32-NEXT:    bne 0, L..BB0_11
; POWERPC_32:    blr
    entry:
  %cmp5 = icmp sgt i32 %ncols, 0
  br i1 %cmp5, label %iter.check, label %for.cond.cleanup

iter.check:                                       ; preds = %entry
  %wide.trip.count = zext nneg i32 %ncols to i64
  %min.iters.check = icmp ult i32 %ncols, 8
  br i1 %min.iters.check, label %for.body.preheader, label %vector.main.loop.iter.check

for.body.preheader:                               ; preds = %vec.epilog.iter.check, %vec.epilog.middle.block, %iter.check
  %indvars.iv.ph = phi i64 [ 0, %iter.check ], [ %n.vec, %vec.epilog.iter.check ], [ %n.vec31, %vec.epilog.middle.block ]
  %num_cols_needed.06.ph = phi i32 [ 0, %iter.check ], [ %33, %vec.epilog.iter.check ], [ %40, %vec.epilog.middle.block ]
  br label %for.body

vector.main.loop.iter.check:                      ; preds = %iter.check
  %min.iters.check9 = icmp ult i32 %ncols, 64
  br i1 %min.iters.check9, label %vec.epilog.ph, label %vector.ph

vector.ph:                                        ; preds = %vector.main.loop.iter.check
  %n.vec = and i64 %wide.trip.count, 2147483584
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ %24, %vector.body ]
  %vec.phi10 = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ %25, %vector.body ]
  %vec.phi11 = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ %26, %vector.body ]
  %vec.phi12 = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ %27, %vector.body ]
  %vec.phi13 = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ %28, %vector.body ]
  %vec.phi14 = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ %29, %vector.body ]
  %vec.phi15 = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ %30, %vector.body ]
  %vec.phi16 = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ %31, %vector.body ]
  %0 = getelementptr inbounds nuw i16, ptr %colauths, i64 %index
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 112
  %wide.load = load <8 x i16>, ptr %0, align 2, !tbaa !5
  %wide.load17 = load <8 x i16>, ptr %1, align 2, !tbaa !5
  %wide.load18 = load <8 x i16>, ptr %2, align 2, !tbaa !5
  %wide.load19 = load <8 x i16>, ptr %3, align 2, !tbaa !5
  %wide.load20 = load <8 x i16>, ptr %4, align 2, !tbaa !5
  %wide.load21 = load <8 x i16>, ptr %5, align 2, !tbaa !5
  %wide.load22 = load <8 x i16>, ptr %6, align 2, !tbaa !5
  %wide.load23 = load <8 x i16>, ptr %7, align 2, !tbaa !5
  %8 = icmp ne <8 x i16> %wide.load, zeroinitializer
  %9 = icmp ne <8 x i16> %wide.load17, zeroinitializer
  %10 = icmp ne <8 x i16> %wide.load18, zeroinitializer
  %11 = icmp ne <8 x i16> %wide.load19, zeroinitializer
  %12 = icmp ne <8 x i16> %wide.load20, zeroinitializer
  %13 = icmp ne <8 x i16> %wide.load21, zeroinitializer
  %14 = icmp ne <8 x i16> %wide.load22, zeroinitializer
  %15 = icmp ne <8 x i16> %wide.load23, zeroinitializer
  %16 = zext <8 x i1> %8 to <8 x i32>
  %17 = zext <8 x i1> %9 to <8 x i32>
  %18 = zext <8 x i1> %10 to <8 x i32>
  %19 = zext <8 x i1> %11 to <8 x i32>
  %20 = zext <8 x i1> %12 to <8 x i32>
  %21 = zext <8 x i1> %13 to <8 x i32>
  %22 = zext <8 x i1> %14 to <8 x i32>
  %23 = zext <8 x i1> %15 to <8 x i32>
  %24 = add <8 x i32> %vec.phi, %16
  %25 = add <8 x i32> %vec.phi10, %17
  %26 = add <8 x i32> %vec.phi11, %18
  %27 = add <8 x i32> %vec.phi12, %19
  %28 = add <8 x i32> %vec.phi13, %20
  %29 = add <8 x i32> %vec.phi14, %21
  %30 = add <8 x i32> %vec.phi15, %22
  %31 = add <8 x i32> %vec.phi16, %23
  %index.next = add nuw i64 %index, 64
  %32 = icmp eq i64 %index.next, %n.vec
  br i1 %32, label %middle.block, label %vector.body, !llvm.loop !9

middle.block:                                     ; preds = %vector.body
  %bin.rdx = add <8 x i32> %25, %24
  %bin.rdx24 = add <8 x i32> %26, %bin.rdx
  %bin.rdx25 = add <8 x i32> %27, %bin.rdx24
  %bin.rdx26 = add <8 x i32> %28, %bin.rdx25
  %bin.rdx27 = add <8 x i32> %29, %bin.rdx26
  %bin.rdx28 = add <8 x i32> %30, %bin.rdx27
  %bin.rdx29 = add <8 x i32> %31, %bin.rdx28
  %33 = tail call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %bin.rdx29)
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count
  br i1 %cmp.n, label %for.cond.cleanup, label %vec.epilog.iter.check

vec.epilog.iter.check:                            ; preds = %middle.block
  %n.vec.remaining = and i64 %wide.trip.count, 56
  %min.epilog.iters.check = icmp eq i64 %n.vec.remaining, 0
  br i1 %min.epilog.iters.check, label %for.body.preheader, label %vec.epilog.ph

vec.epilog.ph:                                    ; preds = %vec.epilog.iter.check, %vector.main.loop.iter.check
  %vec.epilog.resume.val = phi i64 [ %n.vec, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %bc.merge.rdx = phi i32 [ %33, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %n.vec31 = and i64 %wide.trip.count, 2147483640
  %34 = insertelement <8 x i32> <i32 poison, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, i32 %bc.merge.rdx, i64 0
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %index32 = phi i64 [ %vec.epilog.resume.val, %vec.epilog.ph ], [ %index.next35, %vec.epilog.vector.body ]
  %vec.phi33 = phi <8 x i32> [ %34, %vec.epilog.ph ], [ %38, %vec.epilog.vector.body ]
  %35 = getelementptr inbounds nuw i16, ptr %colauths, i64 %index32
  %wide.load34 = load <8 x i16>, ptr %35, align 2, !tbaa !5
  %36 = icmp ne <8 x i16> %wide.load34, zeroinitializer
  %37 = zext <8 x i1> %36 to <8 x i32>
  %38 = add <8 x i32> %vec.phi33, %37
  %index.next35 = add nuw i64 %index32, 8
  %39 = icmp eq i64 %index.next35, %n.vec31
  br i1 %39, label %vec.epilog.middle.block, label %vec.epilog.vector.body, !llvm.loop !13

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  %40 = tail call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %38)
  %cmp.n36 = icmp eq i64 %n.vec31, %wide.trip.count
  br i1 %cmp.n36, label %for.cond.cleanup, label %for.body.preheader

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %vec.epilog.middle.block, %entry
  %num_cols_needed.0.lcssa = phi i32 [ 0, %entry ], [ %33, %middle.block ], [ %40, %vec.epilog.middle.block ], [ %spec.select, %for.body ]
  ret i32 %num_cols_needed.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader ]
  %num_cols_needed.06 = phi i32 [ %spec.select, %for.body ], [ %num_cols_needed.06.ph, %for.body.preheader ]
  %arrayidx = getelementptr inbounds nuw i16, ptr %colauths, i64 %indvars.iv
  %41 = load i16, ptr %arrayidx, align 2, !tbaa !5
  %tobool.not = icmp ne i16 %41, 0
  %inc = zext i1 %tobool.not to i32
  %spec.select = add nuw nsw i32 %num_cols_needed.06, %inc
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !14
}

!5 = !{!6, !6, i64 0}
!6 = !{!"short", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11, !12}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.isvectorized", i32 1}
!12 = !{!"llvm.loop.unroll.runtime.disable"}
!13 = distinct !{!13, !10, !11, !12}
!14 = distinct !{!14, !10, !12, !11}
