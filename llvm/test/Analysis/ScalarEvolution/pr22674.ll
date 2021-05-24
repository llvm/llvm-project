; RUN: opt -loop-reduce -S %s


target datalayout = "e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnux32"

%"class.llvm::AttributeSetNode.230.2029.3828.6141.6912.7683.8454.9482.9996.10253.18506" = type { %"class.llvm::FoldingSetImpl::Node.1.1801.3600.5913.6684.7455.8226.9254.9768.10025.18505", i32 }
%"class.llvm::FoldingSetImpl::Node.1.1801.3600.5913.6684.7455.8226.9254.9768.10025.18505" = type { i8* }
%"struct.std::pair.241.2040.3839.6152.6923.7694.8465.9493.10007.10264.18507" = type { i32, %"class.llvm::AttributeSetNode.230.2029.3828.6141.6912.7683.8454.9482.9996.10253.18506"* }
%"class.llvm::Attribute.222.2021.3820.6133.6904.7675.8446.9474.9988.10245.18509" = type { %"class.llvm::AttributeImpl.2.1802.3601.5914.6685.7456.8227.9255.9769.10026.18508"* }
%"class.llvm::AttributeImpl.2.1802.3601.5914.6685.7456.8227.9255.9769.10026.18508" = type <{ i32 (...)**, %"class.llvm::FoldingSetImpl::Node.1.1801.3600.5913.6684.7455.8226.9254.9768.10025.18505", i8, [3 x i8] }>

; Function Attrs: nounwind uwtable
define void @_ZNK4llvm11AttrBuilder13hasAttributesENS_12AttributeSetEy() #0 align 2 {
entry:
  br i1 undef, label %cond.false, label %_ZNK4llvm12AttributeSet11getNumSlotsEv.exit

_ZNK4llvm12AttributeSet11getNumSlotsEv.exit:      ; preds = %entry
  br i1 undef, label %cond.false, label %for.body.lr.ph.for.body.lr.ph.split_crit_edge

for.body.lr.ph.for.body.lr.ph.split_crit_edge:    ; preds = %_ZNK4llvm12AttributeSet11getNumSlotsEv.exit
  br label %land.lhs.true.i

land.lhs.true.i:                                  ; preds = %for.inc, %for.body.lr.ph.for.body.lr.ph.split_crit_edge
  %I.099 = phi i32 [ 0, %for.body.lr.ph.for.body.lr.ph.split_crit_edge ], [ %inc, %for.inc ]
  %cmp.i = icmp ugt i32 undef, %I.099
  br i1 %cmp.i, label %_ZNK4llvm12AttributeSet12getSlotIndexEj.exit, label %cond.false.i.split

cond.false.i.split:                               ; preds = %land.lhs.true.i
  unreachable

_ZNK4llvm12AttributeSet12getSlotIndexEj.exit:     ; preds = %land.lhs.true.i
  br i1 undef, label %for.end, label %for.inc

for.inc:                                          ; preds = %_ZNK4llvm12AttributeSet12getSlotIndexEj.exit
  %inc = add i32 %I.099, 1
  br i1 undef, label %cond.false, label %land.lhs.true.i

for.end:                                          ; preds = %_ZNK4llvm12AttributeSet12getSlotIndexEj.exit
  %I.099.lcssa129 = phi i32 [ %I.099, %_ZNK4llvm12AttributeSet12getSlotIndexEj.exit ]
  br i1 undef, label %cond.false, label %_ZNK4llvm12AttributeSet3endEj.exit

cond.false:                                       ; preds = %for.end, %for.inc, %_ZNK4llvm12AttributeSet11getNumSlotsEv.exit, %entry
  unreachable

_ZNK4llvm12AttributeSet3endEj.exit:               ; preds = %for.end
  %second.i.i.i = getelementptr inbounds %"struct.std::pair.241.2040.3839.6152.6923.7694.8465.9493.10007.10264.18507", %"struct.std::pair.241.2040.3839.6152.6923.7694.8465.9493.10007.10264.18507"* undef, i32 %I.099.lcssa129, i32 1
  %0 = load %"class.llvm::AttributeSetNode.230.2029.3828.6141.6912.7683.8454.9482.9996.10253.18506"*, %"class.llvm::AttributeSetNode.230.2029.3828.6141.6912.7683.8454.9482.9996.10253.18506"** %second.i.i.i, align 4, !tbaa !2
  %NumAttrs.i.i.i = getelementptr inbounds %"class.llvm::AttributeSetNode.230.2029.3828.6141.6912.7683.8454.9482.9996.10253.18506", %"class.llvm::AttributeSetNode.230.2029.3828.6141.6912.7683.8454.9482.9996.10253.18506"* %0, i32 0, i32 1
  %1 = load i32, i32* %NumAttrs.i.i.i, align 4, !tbaa !8
  %add.ptr.i.i.i55 = getelementptr inbounds %"class.llvm::Attribute.222.2021.3820.6133.6904.7675.8446.9474.9988.10245.18509", %"class.llvm::Attribute.222.2021.3820.6133.6904.7675.8446.9474.9988.10245.18509"* undef, i32 %1
  br i1 undef, label %return, label %for.body11

for.cond9:                                        ; preds = %_ZNK4llvm9Attribute13getKindAsEnumEv.exit
  %cmp10 = icmp eq %"class.llvm::Attribute.222.2021.3820.6133.6904.7675.8446.9474.9988.10245.18509"* %incdec.ptr, %add.ptr.i.i.i55
  br i1 %cmp10, label %return, label %for.body11

for.body11:                                       ; preds = %for.cond9, %_ZNK4llvm12AttributeSet3endEj.exit
  %I5.096 = phi %"class.llvm::Attribute.222.2021.3820.6133.6904.7675.8446.9474.9988.10245.18509"* [ %incdec.ptr, %for.cond9 ], [ undef, %_ZNK4llvm12AttributeSet3endEj.exit ]
  %2 = bitcast %"class.llvm::Attribute.222.2021.3820.6133.6904.7675.8446.9474.9988.10245.18509"* %I5.096 to i32*
  %3 = load i32, i32* %2, align 4, !tbaa !10
  %tobool.i59 = icmp eq i32 %3, 0
  br i1 %tobool.i59, label %cond.false21, label %_ZNK4llvm9Attribute15isEnumAttributeEv.exit

_ZNK4llvm9Attribute15isEnumAttributeEv.exit:      ; preds = %for.body11
  switch i8 undef, label %cond.false21 [
    i8 0, label %_ZNK4llvm9Attribute13getKindAsEnumEv.exit
    i8 1, label %_ZNK4llvm9Attribute13getKindAsEnumEv.exit
    i8 2, label %_ZNK4llvm9Attribute15getKindAsStringEv.exit
  ]

_ZNK4llvm9Attribute13getKindAsEnumEv.exit:        ; preds = %_ZNK4llvm9Attribute15isEnumAttributeEv.exit, %_ZNK4llvm9Attribute15isEnumAttributeEv.exit
  %incdec.ptr = getelementptr inbounds %"class.llvm::Attribute.222.2021.3820.6133.6904.7675.8446.9474.9988.10245.18509", %"class.llvm::Attribute.222.2021.3820.6133.6904.7675.8446.9474.9988.10245.18509"* %I5.096, i32 1
  br i1 undef, label %for.cond9, label %return

cond.false21:                                     ; preds = %_ZNK4llvm9Attribute15isEnumAttributeEv.exit, %for.body11
  unreachable

_ZNK4llvm9Attribute15getKindAsStringEv.exit:      ; preds = %_ZNK4llvm9Attribute15isEnumAttributeEv.exit
  unreachable

return:                                           ; preds = %_ZNK4llvm9Attribute13getKindAsEnumEv.exit, %for.cond9, %_ZNK4llvm12AttributeSet3endEj.exit
  ret void
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (ssh://llvm@gnu-4.sc.intel.com/export/server/git/llvm/clang 4c31740d4f81614b6d278c7825cfdae5a1c78799) (llvm/llvm.git b693958bd09144aed90312709a7e2ccf7124eb53)"}
!2 = !{!3, !7, i64 4}
!3 = !{!"_ZTSSt4pairIjPN4llvm16AttributeSetNodeEE", !4, i64 0, !7, i64 4}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!"any pointer", !5, i64 0}
!8 = !{!9, !4, i64 4}
!9 = !{!"_ZTSN4llvm16AttributeSetNodeE", !4, i64 4}
!10 = !{!7, !7, i64 0}
