; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -mattr=+hmx < %s | FileCheck %s

; Check that a register is not overwritten so the last operand of  the third
; activation instruction has the correct value.

; CHECK: .LBB0_6
; CHECK: sub
; CHECK: r[[R19:[0-9]+]] = sub
; CHECK: r[[R18:[0-9]+]] = or(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = or(r{{[0-9]+}},r[[R19]])
; CHECK-NOT: r[[R19]] = r[[R18]]
; CHECK: activation.hf = mxmem(r[[R18]]
; CHECK: r[[R20:[0-9]+]] = or(r{{[0-9]+}},r[[R19]])
; CHECK: activation.hf = mxmem(r{{[0-9]+}},r[[R20]]):single

; RUN: llc -O0 -mtriple=hexagon -mcpu=hexagonv68 -mattr=+hmx < %s | FileCheck -check-prefix=PACKETO0 %s
; Check that weight and activation instructions are packetized together at O0.

; PACKETO0: {
; PACKETO0: activation.hf = mxmem
; PACKETO0-NEXT: weight.hf = mxmem
; PACKETO0-NEXT: }

; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -mattr=+hmx < %s | FileCheck -check-prefix=PACKET %s
; Check that weight and activation instructions are packetized together.
; Also check that none of the operands of activate and weight instructions is
; packetized with them.

; PACKET: .LBB0_8
; PACKET-DAG: r[[R1:[0-9]+]] = or(r{{[0-9]+}},r{{[0-9]+}})
; PACKET-DAG: r[[R2:[0-9]+]] = or(r{{[0-9]+}},r{{[0-9]+}})
; PACKET: }
; PACKET: {
; PACKET: activation.hf = mxmem(r[[R1]],r[[R2]]):single
; PACKET-NEXT: weight.hf = mxmem(r{{[0-9]+}},r{{[0-9]+}}):deep

; PACKET-DAG: r{{[0-9]+}} = add
; PACKET-DAG: r[[R3:[0-9]+]] = or(r{{[0-9]+}},r{{[0-9]+}})
; PACKET-DAG: r[[R4:[0-9]+]] = or(r{{[0-9]+}},r{{[0-9]+}})
; PACKET: }
; PACKET: {
; PACKET: activation.hf = mxmem(r[[R3]],r[[R4]]){{$}}
; PACKET-NEXT: weight.hf = mxmem(r{{[0-9]+}},r{{[0-9]+}}):deep

; PACKET-DAG: r{{[0-9]+}} = add
; PACKET-DAG: r[[R5:[0-9]+]] = or(r{{[0-9]+}},r{{[0-9]+}})
; PACKET-DAG: r[[R6:[0-9]+]] = or(r{{[0-9]+}},r{{[0-9]+}})
; PACKET: }
; PACKET: {
; PACKET: activation.hf = mxmem(r[[R5]],r[[R6]]):single
; PACKET-NEXT: weight.hf = mxmem(r{{[0-9]+}},r{{[0-9]+}}):deep

; PACKET-DAG: r[[R7:[0-9]+]] = or(r{{[0-9]+}},r{{[0-9]+}})
; PACKET-DAG: r[[R8:[0-9]+]] = or(r{{[0-9]+}},r{{[0-9]+}})
; PACKET-DAG: r{{[0-9]+}} = add
; PACKET: }
; PACKET: {
; PACKET: activation.hf = mxmem(r[[R7]],r[[R8]]){{$}}
; PACKET-NEXT: weight.hf = mxmem(r{{[0-9]+}},r{{[0-9]+}}):deep

target triple = "hexagon"

%struct.tile_data = type { ptr, i32, i32, i32, i32, i32 }
%struct.hmx_params = type { %struct.registerPair, [2 x %struct.registerPair], [4 x i32], [2 x i32], i32, [4 x i8] }
%struct.registerPair = type { i32, i32 }

define dso_local void @function(ptr nocapture readonly %outblock, ptr nocapture readonly %inblock, ptr %weight_in, ptr %bias, ptr nocapture readonly %mxparam, ptr %weight_init) local_unnamed_addr #0 {
entry:
  %offset_t_row = getelementptr inbounds %struct.tile_data, ptr %inblock, i32 0, i32 2
  %0 = load i32, ptr %offset_t_row, align 4
  %height = getelementptr inbounds %struct.tile_data, ptr %outblock, i32 0, i32 4
  %1 = load i32, ptr %height, align 4
  %depth = getelementptr inbounds %struct.tile_data, ptr %outblock, i32 0, i32 5
  %2 = load i32, ptr %depth, align 4
  %height2 = getelementptr inbounds %struct.tile_data, ptr %inblock, i32 0, i32 4
  %3 = load i32, ptr %height2, align 4
  %depth4 = getelementptr inbounds %struct.tile_data, ptr %inblock, i32 0, i32 5
  %4 = load i32, ptr %depth4, align 4
  %cmp6197 = icmp eq i32 %2, 0
  br i1 %cmp6197, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %cmp8194 = icmp eq i32 %1, 0
  %addr = getelementptr inbounds %struct.tile_data, ptr %inblock, i32 0, i32 0
  %start = getelementptr inbounds %struct.hmx_params, ptr %mxparam, i32 0, i32 1, i32 0, i32 0
  %start24 = getelementptr inbounds %struct.hmx_params, ptr %mxparam, i32 0, i32 1, i32 1, i32 0
  %cmp36190 = icmp eq i32 %4, 0
  %range = getelementptr inbounds %struct.hmx_params, ptr %mxparam, i32 0, i32 1, i32 1, i32 1
  %arrayidx47 = getelementptr inbounds %struct.hmx_params, ptr %mxparam, i32 0, i32 2, i32 1
  %range61 = getelementptr inbounds %struct.hmx_params, ptr %mxparam, i32 0, i32 1, i32 0, i32 1
  %arrayidx64 = getelementptr inbounds %struct.hmx_params, ptr %mxparam, i32 0, i32 2, i32 0
  %arrayidx70 = getelementptr inbounds %struct.hmx_params, ptr %mxparam, i32 0, i32 2, i32 3
  %arrayidx71 = getelementptr inbounds %struct.hmx_params, ptr %mxparam, i32 0, i32 3, i32 1
  %arrayidx85 = getelementptr inbounds %struct.hmx_params, ptr %mxparam, i32 0, i32 3, i32 0
  %arrayidx95 = getelementptr inbounds %struct.hmx_params, ptr %mxparam, i32 0, i32 2, i32 2
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup9, %entry
  ret void

for.body:                                         ; preds = %for.cond.cleanup9, %for.body.lr.ph
  %weight_in.addr.0200 = phi ptr [ %weight_in, %for.body.lr.ph ], [ %weight.1.lcssa, %for.cond.cleanup9 ]
  %d.0199 = phi i32 [ 0, %for.body.lr.ph ], [ %add105, %for.cond.cleanup9 ]
  %weight.0198 = phi ptr [ %weight_init, %for.body.lr.ph ], [ %weight.1.lcssa, %for.cond.cleanup9 ]
  %mul = shl i32 %d.0199, 1
  %add.ptr = getelementptr inbounds i32, ptr %bias, i32 %mul
  %5 = bitcast ptr %add.ptr to ptr
  tail call void @llvm.hexagon.M8.mxmem2.st.bias(ptr %5)
  br i1 %cmp8194, label %for.cond.cleanup9, label %for.body10

for.cond7.loopexit:                               ; preds = %for.body38, %for.body10
  %weight.2.lcssa = phi ptr [ %weight_in.addr.0200, %for.body10 ], [ %add.ptr99, %for.body38 ]
  %cmp8 = icmp ult i32 %add, %1
  br i1 %cmp8, label %for.body10, label %for.cond.cleanup9

for.cond.cleanup9:                                ; preds = %for.cond7.loopexit, %for.body
  %weight.1.lcssa = phi ptr [ %weight.0198, %for.body ], [ %weight.2.lcssa, %for.cond7.loopexit ]
  %add105 = add i32 %d.0199, 32
  %cmp6 = icmp ult i32 %add105, %2
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.body10:                                       ; preds = %for.cond7.loopexit, %for.body
  %h.0195 = phi i32 [ %add, %for.cond7.loopexit ], [ 0, %for.body ]
  %6 = load ptr, ptr %addr, align 4
  %7 = load i32, ptr %start, align 8
  %and.i = and i32 %7, 1792
  %cmp19 = icmp eq i32 %and.i, 0
  %add = add i32 %h.0195, 8
  %cmp20 = icmp ult i32 %add, %3
  %spec.select = select i1 %cmp20, i32 %0, i32 0
  %cond21 = select i1 %cmp19, i32 0, i32 %spec.select
  %8 = load i32, ptr %start24, align 8
  %and.i189 = and i32 %8, 1792
  %cmp26 = icmp ne i32 %and.i189, 0
  %add28 = add i32 %h.0195, 16
  %cmp29 = icmp ult i32 %add28, %3
  %not.or.cond = and i1 %cmp29, %cmp26
  %mul32 = zext i1 %not.or.cond to i32
  %cond34 = shl i32 %0, %mul32
  tail call void @llvm.hexagon.M8.mxclracc.hf()
  br i1 %cmp36190, label %for.cond7.loopexit, label %for.body38.preheader

for.body38.preheader:                             ; preds = %for.body10
  %shr = lshr exact i32 %h.0195, 3
  %mul11 = mul i32 %shr, %0
  %add.ptr12 = getelementptr inbounds ptr, ptr %6, i32 %mul11
  br label %for.body38

for.body38:                                       ; preds = %for.body38, %for.body38.preheader
  %dd.0193 = phi i32 [ %add100, %for.body38 ], [ 0, %for.body38.preheader ]
  %weight.2192 = phi ptr [ %add.ptr99, %for.body38 ], [ %weight_in.addr.0200, %for.body38.preheader ]
  %ppIn.0191 = phi ptr [ %incdec.ptr, %for.body38 ], [ %add.ptr12, %for.body38.preheader ]
  %add.ptr39 = getelementptr inbounds ptr, ptr %ppIn.0191, i32 %cond34
  %9 = bitcast ptr %add.ptr39 to ptr
  %10 = load i32, ptr %9, align 4
  %add.ptr40 = getelementptr inbounds ptr, ptr %ppIn.0191, i32 %cond21
  %11 = bitcast ptr %add.ptr40 to ptr
  %12 = load i32, ptr %11, align 4
  %incdec.ptr = getelementptr inbounds ptr, ptr %ppIn.0191, i32 1
  %13 = bitcast ptr %ppIn.0191 to ptr
  %14 = load i32, ptr %13, align 4
  %15 = load i32, ptr %start24, align 8
  %or = or i32 %15, %14
  %sub.ptr.sub = sub i32 %12, %14
  %16 = load i32, ptr %range, align 4
  %or46 = or i32 %16, %sub.ptr.sub
  tail call void @llvm.hexagon.M8.mxmems.blk.sm.act.hf(i32 %or, i32 %or46)
  %17 = ptrtoint ptr %weight.2192 to i32
  %18 = load i32, ptr %arrayidx47, align 4
  tail call void @llvm.hexagon.M8.mxmemdp.wei.hf(i32 %17, i32 %18)
  %19 = load i32, ptr %arrayidx47, align 4
  %add50 = add i32 %19, 1
  %add.ptr51 = getelementptr inbounds i8, ptr %weight.2192, i32 %add50
  %20 = load i32, ptr %start, align 8
  %or55 = or i32 %20, %12
  %sub.ptr.sub58 = sub i32 %10, %12
  %21 = load i32, ptr %range61, align 4
  %or62 = or i32 %21, %sub.ptr.sub58
  tail call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %or55, i32 %or62)
  %22 = ptrtoint ptr %add.ptr51 to i32
  %23 = load i32, ptr %arrayidx64, align 8
  tail call void @llvm.hexagon.M8.mxmemdp.wei.hf(i32 %22, i32 %23)
  %24 = load i32, ptr %arrayidx64, align 8
  %add67 = add i32 %24, 1
  %add.ptr68 = getelementptr inbounds i8, ptr %add.ptr51, i32 %add67
  tail call void @llvm.hexagon.M8.mxclracc.hf()
  %25 = ptrtoint ptr %add.ptr68 to i32
  %26 = load i32, ptr %arrayidx70, align 4
  tail call void @llvm.hexagon.M8.mxmemdp.wei.hf(i32 %25, i32 %26)
  %27 = load i32, ptr %arrayidx71, align 4
  %or72 = or i32 %27, %14
  %28 = load i32, ptr %range, align 4
  %or79 = or i32 %28, %sub.ptr.sub
  tail call void @llvm.hexagon.M8.mxmems.blk.sm.act.hf(i32 %or72, i32 %or79)
  %29 = load i32, ptr %arrayidx70, align 4
  %add82 = add i32 %29, 1
  %add.ptr83 = getelementptr inbounds i8, ptr %add.ptr68, i32 %add82
  %30 = load i32, ptr %arrayidx85, align 8
  %or86 = or i32 %30, %12
  %31 = load i32, ptr %range61, align 4
  %or93 = or i32 %31, %sub.ptr.sub58
  tail call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %or86, i32 %or93)
  %32 = ptrtoint ptr %add.ptr83 to i32
  %33 = load i32, ptr %arrayidx95, align 8
  tail call void @llvm.hexagon.M8.mxmemdp.wei.hf(i32 %32, i32 %33)
  %34 = load i32, ptr %arrayidx95, align 8
  %add98 = add i32 %34, 1
  %add.ptr99 = getelementptr inbounds i8, ptr %add.ptr83, i32 %add98
  tail call void @llvm.hexagon.M8.mxclracc.hf()
  %add100 = add i32 %dd.0193, 32
  %cmp36 = icmp ult i32 %add100, %4
  br i1 %cmp36, label %for.body38, label %for.cond7.loopexit
}

; Function Attrs: nounwind writeonly
declare void @llvm.hexagon.M8.mxmem2.st.bias(ptr) #1

; Function Attrs: nounwind
declare void @llvm.hexagon.M8.mxclracc.hf() #2

; Function Attrs: convergent noduplicate nomerge nounwind writeonly
declare void @llvm.hexagon.M8.mxmems.blk.sm.act.hf(i32, i32) #3

; Function Attrs: convergent noduplicate nomerge nounwind writeonly
declare void @llvm.hexagon.M8.mxmemdp.wei.hf(i32, i32) #3

; Function Attrs: convergent noduplicate nomerge nounwind writeonly
declare void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32, i32) #3

attributes #0 = { "target-cpu"="hexagonv68" "target-features"="+hmx" }
attributes #1 = { nounwind writeonly "target-cpu"="hexagonv68" "target-features"="+hmx" }
attributes #2 = { nounwind "target-cpu"="hexagonv68" "target-features"="+hmx" }
attributes #3 = { convergent noduplicate nomerge nounwind writeonly "target-cpu"="hexagonv68" "target-features"="+hmx" }
