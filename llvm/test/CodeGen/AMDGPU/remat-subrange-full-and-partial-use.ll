; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa < %s | FileCheck %s

; Regression test for an assertion in the AMDGPU pipeline:
;   Assertion `SubRegIdx != 0 && LI.hasSubRanges()' failed.
;
; The machine scheduler's trivial-remat sinking uses the Rematerializer, whose
; incremental live-interval update only created subranges when a *new* user
; read a partial lane mask. A 64-bit VGPR is written subreg-by-subreg and then
; read both as the full register and via .sub0. When the full-register user is
; processed first, subranges were never created, so VirtRegRewriter later asserted
; while inspecting the .sub0 use. The kernel must now compile to valid code without
; crashing.

; CHECK-LABEL: remat_subrange_full_and_partial_use:
; CHECK: s_endpgm

define amdgpu_kernel void @remat_subrange_full_and_partial_use(i32 %arg, ptr addrspace(4) %arg1, ptr addrspace(4) %arg2, ptr addrspace(4) %arg3, ptr addrspace(4) %arg4, ptr addrspace(4) %arg5, ptr addrspace(4) %arg6, ptr addrspace(4) %arg7, ptr addrspace(4) %arg8, ptr addrspace(4) %arg9, ptr addrspace(4) %arg10, ptr addrspace(4) %arg11, ptr addrspace(4) %arg12, ptr addrspace(4) %arg13, ptr addrspace(4) %arg14, ptr addrspace(4) %arg15, ptr addrspace(4) %arg16, ptr addrspace(4) %arg17, ptr addrspace(4) %arg18, ptr addrspace(4) %arg19, <2 x i32> %arg20, ptr addrspace(4) %arg21, i32 %arg22, i32 %arg23, i32 %arg24, i32 %arg25, i32 %arg26, i32 %arg27, i32 %arg28, i32 %arg29, i32 %arg30, i32 %arg31, i32 %arg32, i32 %arg33, i32 %arg34, i32 %arg35, i1 %arg36, i32 %arg37, i32 %arg38, i32 %arg39, i32 %arg40, i32 %arg41, ptr addrspace(1) %arg42, i32 %arg43, i1 %arg44, ptr addrspace(8) %arg45, ptr addrspace(8) %arg46, i32 %arg47, i32 %arg48, i1 %arg49, i1 %arg50, i1 %arg51, i32 %arg52, i1 %arg53) {
bbl:
  %load = load i32, ptr addrspace(4) %arg19, align 8
  %load54 = load i32, ptr addrspace(4) %arg11, align 4
  %load55 = load i32, ptr addrspace(4) %arg1, align 8
  %or = or i32 %load55, 1
  %icmp = icmp slt i32 0, %or
  %icmp56 = icmp slt i32 0, %arg
  br i1 %arg36, label %bbl59, label %bbl57

bbl57:                                            ; preds = %bbl
  %shl = shl i32 %arg, 1
  %or58 = or i32 %shl, 1
  br label %bbl59

bbl59:                                            ; preds = %bbl57, %bbl
  %or60 = or i32 %arg32, %arg22
  %or61 = or i32 %or60, %arg25
  %or62 = or i32 %or61, %arg
  %call = tail call <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) null, i32 %or62, i32 0, i32 0)
  %and = and i1 %arg49, %icmp56
  %and63 = and i1 %arg36, %and
  %bitcast = bitcast <2 x i32> %call to <4 x half>
  %and64 = and i1 %arg50, %and63
  %shufflevector = shufflevector <4 x half> %bitcast, <4 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %select = select i1 %and64, <8 x half> %shufflevector, <8 x half> zeroinitializer
  store <8 x half> %select, ptr addrspace(3) null, align 16
  br label %bbl65

bbl65:                                            ; preds = %bbl65, %bbl59
  %icmp66 = icmp slt i32 %arg37, 0
  %and67 = and i1 %icmp66, %arg44
  %call68 = tail call <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) %arg45, i32 0, i32 0, i32 0)
  %bitcast69 = bitcast <2 x i32> %call68 to <4 x half>
  %select70 = select i1 %and67, <4 x half> %bitcast69, <4 x half> zeroinitializer
  %shufflevector71 = shufflevector <4 x half> %select70, <4 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x half> %shufflevector71, ptr addrspace(3) null, align 16
  br i1 %arg51, label %bbl72, label %bbl65

bbl72:                                            ; preds = %bbl65
  %load73 = load float, ptr addrspace(4) %arg5, align 4
  %load74 = load i32, ptr addrspace(4) %arg18, align 4
  %load75 = load i32, ptr addrspace(4) %arg14, align 4
  %mul = mul i32 %load74, %load75
  %load76 = load <2 x i32>, ptr addrspace(4) %arg3, align 8
  %extractelement = extractelement <2 x i32> %load76, i64 0
  %mul77 = mul i32 %mul, %extractelement
  %call78 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %mul77, i32 0, i32 0)
  %bitcast79 = bitcast i16 %call78 to half
  %load80 = load <2 x i32>, ptr addrspace(4) %arg8, align 4
  %extractelement81 = extractelement <2 x i32> %load80, i64 0
  %load82 = load <2 x i32>, ptr addrspace(4) %arg10, align 4
  %extractelement83 = extractelement <2 x i32> %load82, i64 0
  %load84 = load <2 x i32>, ptr addrspace(4) %arg13, align 8
  %load85 = load i32, ptr addrspace(4) %arg7, align 8
  %load86 = load i32, ptr addrspace(4) %arg15, align 8
  %mul87 = mul i32 %load85, %load86
  %load88 = load <2 x i32>, ptr addrspace(4) %arg1, align 8
  %extractelement89 = extractelement <2 x i32> %load88, i64 0
  %mul90 = mul i32 %mul87, %extractelement89
  %mul91 = mul i32 %extractelement81, %extractelement83
  %mul92 = mul i32 %mul91, %arg23
  %or93 = or i32 %mul90, %mul92
  %lshr = lshr i32 %arg27, %arg26
  %extractelement94 = extractelement <2 x i32> %load84, i64 0
  %mul95 = mul i32 %lshr, %extractelement94
  %or96 = or i32 %or93, %mul95
  %icmp97 = icmp slt i32 0, %arg39
  %or98 = or i32 %arg40, %arg41
  %icmp99 = icmp slt i32 0, %or98
  %and100 = and i1 %icmp97, %icmp99
  %fcmp = fcmp ogt half %bitcast79, 0.000000e+00
  %fptrunc = fptrunc float %load73 to half
  %select101 = select i1 %fcmp, half 0.000000e+00, half %fptrunc
  %select102 = select i1 %and100, half %select101, half 0.000000e+00
  %insertelement = insertelement <4 x half> zeroinitializer, half %select102, i64 0
  %bitcast103 = bitcast <4 x half> %insertelement to <2 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %bitcast103, ptr addrspace(8) %arg46, i32 0, i32 0, i32 0)
  %load104 = load <2 x i32>, ptr addrspace(4) %arg6, align 8
  %load105 = load <2 x i32>, ptr addrspace(4) %arg9, align 4
  %load106 = load i32, ptr addrspace(4) %arg12, align 4
  %load107 = load i32, ptr addrspace(4) %arg17, align 4
  %lshr108 = lshr i32 %load106, %load107
  %load109 = load i32, ptr addrspace(4) %arg16, align 8
  %mul110 = mul i32 %load, %load109
  %mul111 = mul i32 %mul110, %arg52
  %mul112 = mul i32 %lshr108, %load54
  %mul113 = mul i32 %mul112, %arg24
  %or114 = or i32 %mul111, %mul113
  %or115 = or i32 %or114, %arg33
  %or116 = or i32 %or115, %arg28
  %extractelement117 = extractelement <2 x i32> %load104, i64 0
  %extractelement118 = extractelement <2 x i32> %load105, i64 0
  %lshr119 = lshr i32 %extractelement117, %extractelement118
  %load120 = load <2 x i32>, ptr addrspace(4) %arg2, align 4
  %extractelement121 = extractelement <2 x i32> %load120, i64 0
  %mul122 = mul i32 %lshr119, %extractelement121
  %insertelement123 = insertelement <2 x i32> zeroinitializer, i32 %mul122, i64 1
  %mul124 = mul <2 x i32> %insertelement123, %arg20
  %extractelement125 = extractelement <2 x i32> %mul124, i64 1
  %icmp126 = icmp slt i32 0, %arg34
  %load127 = load i32, ptr addrspace(4) %arg4, align 8
  %icmp128 = icmp sge i32 0, %load127
  %select129 = select i1 %icmp128, i1 %arg53, i1 false
  %and130 = and i1 %icmp126, %select129
  %icmp131 = icmp slt i32 0, %arg38
  %or132 = or i32 %or96, %arg29
  %call133 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %or132, i32 0, i32 0)
  %bitcast134 = bitcast i16 %call133 to half
  %select135 = select i1 %icmp131, half %bitcast134, half 0.000000e+00
  %insertelement136 = insertelement <4 x half> zeroinitializer, half %select135, i64 0
  %bitcast137 = bitcast <4 x half> %insertelement136 to <2 x i32>
  %or138 = or i32 %or116, %extractelement125
  %select139 = select i1 %and130, i32 0, i32 1
  %or140 = or i32 %or138, %select139
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %bitcast137, ptr addrspace(8) null, i32 %or140, i32 0, i32 0)
  ret void
}

declare <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #0
declare i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32>, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
