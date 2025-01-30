;; Test the unused argument are converted to ISD::UNDEF SDNode.

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff  -debug-only=isel \
; RUN:   %s -o - 2>&1 >/dev/null | FileCheck --check-prefix=CHECK32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff  -debug-only=isel \
; RUN:   %s -o - 2>&1 >/dev/null | FileCheck --check-prefix=CHECK64 %s

define void @bar32(i32 zeroext %var1, i32 noundef zeroext %var2) local_unnamed_addr  {
entry:
  tail call void @foo32(i32 noundef zeroext %var2)
  ret void
}

declare void @foo32(i32 noundef zeroext) local_unnamed_addr

define void @test32() local_unnamed_addr {
entry:
  tail call void @bar32(i32 zeroext poison, i32 noundef zeroext 255)
  ret void
}

; CHECK32:     Initial selection DAG: %bb.0 'test32:entry'
; CHECK32-DAG: SelectionDAG has 18 nodes:
; CHECK32-DAG:   t1: i32 = GlobalAddress<ptr @bar32> 0
; CHECK32-DAG:   t7: i32 = Register $r1
; CHECK32-DAG:       t0: ch,glue = EntryToken
; CHECK32-DAG:     t6: ch,glue = callseq_start t0, TargetConstant:i32<56>, TargetConstant:i32<0>
; CHECK32-DAG:   t9: ch,glue = CopyToReg t6, Register:i32 $r3, undef:i32
; CHECK32-DAG:   t11: ch,glue = CopyToReg t9, Register:i32 $r4, Constant:i32<255>, t9:1
; CHECK32-DAG:   t15: ch,glue = PPCISD::CALL_NOP t11, MCSymbol:i32, Register:i32 $r3, Register:i32 $r4, Register:i32 $r2, RegisterMask:Untyped, t11:1
; CHECK32-DAG:     t16: ch,glue = callseq_end t15, TargetConstant:i32<56>, TargetConstant:i32<0>, t15:1
; CHECK32-DAG:   t17: ch = PPCISD::RET_GLUE t16

; CHECK64:     Initial selection DAG: %bb.0 'test32:entry'
; CHECK64-DAG: SelectionDAG has 20 nodes:
; CHECK64-DAG:   t1: i64 = GlobalAddress<ptr @bar32> 0
; CHECK64-DAG:   t2: i32 = undef
; CHECK64-DAG:   t3: i32 = Constant<255>
; CHECK64-DAG:   t7: i64 = Register $x1
; CHECK64-DAG:       t0: ch,glue = EntryToken
; CHECK64-DAG:     t6: ch,glue = callseq_start t0, TargetConstant:i64<112>, TargetConstant:i64<0>
; CHECK64-DAG:   t11: ch,glue = CopyToReg t6, Register:i64 $x3, undef:i64
; CHECK64-DAG:   t13: ch,glue = CopyToReg t11, Register:i64 $x4, Constant:i64<255>, t11:1
; CHECK64-DAG:   t17: ch,glue = PPCISD::CALL_NOP t13, MCSymbol:i64, Register:i64 $x3, Register:i64 $x4, Register:i64 $x2, RegisterMask:Untyped, t13:1
; CHECK64-DAG:     t18: ch,glue = callseq_end t17, TargetConstant:i64<112>, TargetConstant:i64<0>, t17:1
; CHECK64-DAG:   t19: ch = PPCISD::RET_GLUE t18

define void @bar8(i8 zeroext %var1, i8 noundef zeroext %var2) local_unnamed_addr  {
entry:
  tail call void @foo8(i8 noundef zeroext %var2)
  ret void
}

declare void @foo8(i8 noundef zeroext) local_unnamed_addr

define void @test8() local_unnamed_addr {
entry:
  tail call void @bar8(i8 zeroext poison, i8 noundef zeroext 255)
  ret void
}

; CHECK32:     Initial selection DAG: %bb.0 'test8:entry'
; CHECK32-DAG: SelectionDAG has 20 nodes:
; CHECK32-DAG:   t1: i32 = GlobalAddress<ptr @bar8> 0
; CHECK32-DAG:   t2: i8 = undef
; CHECK32-DAG:   t3: i8 = Constant<-1>
; CHECK32-DAG:   t9: i32 = Register $r1
; CHECK32-DAG:       t0: ch,glue = EntryToken
; CHECK32-DAG:     t8: ch,glue = callseq_start t0, TargetConstant:i32<56>, TargetConstant:i32<0>
; CHECK32-DAG:   t11: ch,glue = CopyToReg t8, Register:i32 $r3, undef:i32
; CHECK32-DAG:   t13: ch,glue = CopyToReg t11, Register:i32 $r4, Constant:i32<255>, t11:1
; CHECK32-DAG:   t17: ch,glue = PPCISD::CALL_NOP t13, MCSymbol:i32, Register:i32 $r3, Register:i32 $r4, Register:i32 $r2, RegisterMask:Untyped, t13:1
; CHECK32-DAG:     t18: ch,glue = callseq_end t17, TargetConstant:i32<56>, TargetConstant:i32<0>, t17:1
; CHECK32-DAG:   t19: ch = PPCISD::RET_GLUE t18


; CHECK64:      Initial selection DAG: %bb.0 'test8:entry'
; CHECK64-DAG: SelectionDAG has 22 nodes:
; CHECK64-DAG:   t1: i64 = GlobalAddress<ptr @bar8> 0
; CHECK64-DAG:   t2: i8 = undef
; CHECK64-DAG:   t3: i8 = Constant<-1>
; CHECK64-DAG:   t4: i32 = undef
; CHECK64-DAG:   t5: i32 = Constant<255>
; CHECK64-DAG:   t9: i64 = Register $x1
; CHECK64-DAG:       t0: ch,glue = EntryToken
; CHECK64-DAG:     t8: ch,glue = callseq_start t0, TargetConstant:i64<112>, TargetConstant:i64<0>
; CHECK64-DAG:   t13: ch,glue = CopyToReg t8, Register:i64 $x3, undef:i64
; CHECK64-DAG:   t15: ch,glue = CopyToReg t13, Register:i64 $x4, Constant:i64<255>, t13:1
; CHECK64-DAG:   t19: ch,glue = PPCISD::CALL_NOP t15, MCSymbol:i64, Register:i64 $x3, Register:i64 $x4, Register:i64 $x2, RegisterMask:Untyped, t15:1
; CHECK64-DAG:     t20: ch,glue = callseq_end t19, TargetConstant:i64<112>, TargetConstant:i64<0>, t19:1
; CHECK64-DAG:   t21: ch = PPCISD::RET_GLUE t20


define void @bar64(i64 zeroext %var1, i64 noundef zeroext %var2) local_unnamed_addr  {
entry:
  tail call void @foo64(i64 noundef zeroext %var2)
  ret void
}

declare void @foo64(i64 noundef zeroext) local_unnamed_addr

; Function Attrs: noinline nounwind
define void @test64() local_unnamed_addr {
entry:
  tail call void @bar64(i64 zeroext poison, i64 noundef zeroext 255)
  ret void
}

; CHECK32:     Initial selection DAG: %bb.0 'test64:entry'
; CHECK32-DAG: SelectionDAG has 26 nodes:
; CHECK32-DAG:   t1: i32 = GlobalAddress<ptr @bar64> 0
; CHECK32-DAG:   t2: i64 = undef
; CHECK32-DAG:   t3: i64 = Constant<255>
; CHECK32-DAG:   t5: i32 = Constant<1>
; CHECK32-DAG:   t11: i32 = Register $r1
; CHECK32-DAG:       t0: ch,glue = EntryToken
; CHECK32-DAG:     t10: ch,glue = callseq_start t0, TargetConstant:i32<56>, TargetConstant:i32<0>
; CHECK32-DAG:   t13: ch,glue = CopyToReg t10, Register:i32 $r3, undef:i32
; CHECK32-DAG:   t15: ch,glue = CopyToReg t13, Register:i32 $r4, undef:i32, t13:1
; CHECK32-DAG:   t17: ch,glue = CopyToReg t15, Register:i32 $r5, Constant:i32<0>, t15:1
; CHECK32-DAG:   t19: ch,glue = CopyToReg t17, Register:i32 $r6, Constant:i32<255>, t17:1
; CHECK32-DAG:   t23: ch,glue = PPCISD::CALL_NOP t19, MCSymbol:i32, Register:i32 $r3, Register:i32 $r4, Register:i32 $r5, Register:i32 $r6, Register:i32 $r2, RegisterMask:Untyped, t19:1
; CHECK32-DAG:     t24: ch,glue = callseq_end t23, TargetConstant:i32<56>, TargetConstant:i32<0>, t23:1
; CHECK32-DAG:   t25: ch = PPCISD::RET_GLUE t24

; CHECK64:     Initial selection DAG: %bb.0 'test64:entry'
; CHECK64-DAG: SelectionDAG has 18 nodes:
; CHECK64-DAG:   t1: i64 = GlobalAddress<ptr @bar64> 0
; CHECK64-DAG:   t7: i64 = Register $x1
; CHECK64-DAG:       t0: ch,glue = EntryToken
; CHECK64-DAG:     t6: ch,glue = callseq_start t0, TargetConstant:i64<112>, TargetConstant:i64<0>
; CHECK64-DAG:   t9: ch,glue = CopyToReg t6, Register:i64 $x3, undef:i64
; CHECK64-DAG:   t11: ch,glue = CopyToReg t9, Register:i64 $x4, Constant:i64<255>, t9:1
; CHECK64-DAG:   t15: ch,glue = PPCISD::CALL_NOP t11, MCSymbol:i64, Register:i64 $x3, Register:i64 $x4, Register:i64 $x2, RegisterMask:Untyped, t11:1
; CHECK64-DAG:     t16: ch,glue = callseq_end t15, TargetConstant:i64<112>, TargetConstant:i64<0>, t15:1
; CHECK64-DAG:   t17: ch = PPCISD::RET_GLUE t16
