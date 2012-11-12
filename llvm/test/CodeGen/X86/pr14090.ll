; RUN: llc < %s -march=x86-64 -print-before=stack-coloring -print-after=stack-coloring >%t 2>&1 && FileCheck <%t %s

define void @foo(i64* %retval.i, i32 %call, i32* %.ph.i80, i32 %fourteen, i32* %out.lo, i32* %out.hi) nounwind align 2 {
entry:
  %_Tmp.i39 = alloca i64, align 8
  %retval.i33 = alloca i64, align 8
  %_Tmp.i = alloca i64, align 8
  %retval.i.i = alloca i64, align 8
  %_First.i = alloca i64, align 8

  %0 = load i64* %retval.i, align 8

  %1 = load i64* %retval.i, align 8

  %_Tmp.i39.0.cast73 = bitcast i64* %_Tmp.i39 to i8*
  call void @llvm.lifetime.start(i64 8, i8* %_Tmp.i39.0.cast73)
  store i64 %1, i64* %_Tmp.i39, align 8
  %cmp.i.i.i40 = icmp slt i32 %call, 0
  %2 = lshr i64 %1, 32
  %3 = trunc i64 %2 to i32
  %sub.i.i.i44 = sub i32 0, %call
  %cmp2.i.i.i45 = icmp ult i32 %3, %sub.i.i.i44
  %or.cond.i.i.i46 = and i1 %cmp.i.i.i40, %cmp2.i.i.i45
  %add.i.i.i47 = add i32 %3, %call
  %sub5.i.i.i48 = lshr i32 %add.i.i.i47, 5
  %trunc.i50 = trunc i64 %1 to i32
  %inttoptr.i51 = inttoptr i32 %trunc.i50 to i32*
  %add61617.i.i.i52 = or i32 %sub5.i.i.i48, -134217728
  %add61617.i.sub5.i.i.i53 = select i1 %or.cond.i.i.i46, i32 %add61617.i.i.i52, i32 %sub5.i.i.i48
  %storemerge2.i.i54 = getelementptr inbounds i32* %inttoptr.i51, i32 %add61617.i.sub5.i.i.i53
  %_Tmp.i39.0.cast74 = bitcast i64* %_Tmp.i39 to i32**
  store i32* %storemerge2.i.i54, i32** %_Tmp.i39.0.cast74, align 8
  %storemerge.i.i55 = and i32 %add.i.i.i47, 31
  %_Tmp.i39.4.raw_idx = getelementptr inbounds i8* %_Tmp.i39.0.cast73, i32 4
  %_Tmp.i39.4.cast = bitcast i8* %_Tmp.i39.4.raw_idx to i32*
  store i32 %storemerge.i.i55, i32* %_Tmp.i39.4.cast, align 4
  %srcval.i56 = load i64* %_Tmp.i39, align 8
  call void @llvm.lifetime.end(i64 8, i8* %_Tmp.i39.0.cast73)

; CHECK: Before Merge disjoint stack slots
; CHECK: [[PREFIX15:MOV64mr.*<fi#]]{{[0-9]}}[[SUFFIX15:.*;]] mem:ST8[%fifteen]
; CHECK: [[PREFIX87:MOV32mr.*;]] mem:ST4[%sunkaddr87]

; CHECK: After Merge disjoint stack slots
; CHECK: [[PREFIX15]]{{[0-9]}}[[SUFFIX15]] mem:ST8[%_Tmp.i39]
; CHECK: [[PREFIX87]] mem:ST4[<unknown>]

  %fifteen = bitcast i64* %retval.i.i to i32**
  %sixteen = bitcast i64* %retval.i.i to i8*
  call void @llvm.lifetime.start(i64 8, i8* %sixteen)
  store i32* %.ph.i80, i32** %fifteen, align 8, !tbaa !0
  %sunkaddr = ptrtoint i64* %retval.i.i to i32
  %sunkaddr86 = add i32 %sunkaddr, 4
  %sunkaddr87 = inttoptr i32 %sunkaddr86 to i32*
  store i32 %fourteen, i32* %sunkaddr87, align 4, !tbaa !3
  %seventeen = load i64* %retval.i.i, align 8
  call void @llvm.lifetime.end(i64 8, i8* %sixteen)
  %eighteen = lshr i64 %seventeen, 32
  %nineteen = trunc i64 %eighteen to i32
  %shl.i.i.i = shl i32 1, %nineteen

  store i32 %shl.i.i.i, i32* %out.lo, align 8
  store i32 %nineteen, i32* %out.hi, align 8

  ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
!3 = metadata !{metadata !"any pointer", metadata !1}
!4 = metadata !{metadata !"vtable pointer", metadata !2}
