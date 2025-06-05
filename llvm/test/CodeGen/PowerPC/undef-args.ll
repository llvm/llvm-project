;; Tests that extending poison results in poison.
;; Also tests that there are no redundant instructions loading 0 into argument registers for unused arguments.

; REQUIRES: asserts

; REQUIRES: asserts

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -debug-only=isel \
; RUN:   %s -o - 2>&1 | FileCheck --check-prefix=CHECKISEL32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -debug-only=isel \
; RUN:   %s -o - 2>&1 | FileCheck --check-prefix=CHECKISEL64 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -filetype=asm \
; RUN:   %s -o - 2>&1 | FileCheck --check-prefix=CHECKASM32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -filetype=asm \
; RUN:   %s -o - 2>&1 | FileCheck --check-prefix=CHECKASM64 %s

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

; CHECKISEL32:     Initial selection DAG: %bb.0 'test32:entry'
; CHECKISEL32-NEXT: SelectionDAG has 18 nodes:
; CHECKISEL32-NEXT:   t1: i32 = GlobalAddress<ptr @bar32> 0
; CHECKISEL32-NEXT:   t7: i32 = Register $r1
; CHECKISEL32-NEXT:       t0: ch,glue = EntryToken
; CHECKISEL32-NEXT:     t6: ch,glue = callseq_start t0, TargetConstant:i32<56>, TargetConstant:i32<0>
; CHECKISEL32-NEXT:   t9: ch,glue = CopyToReg t6, Register:i32 $r3, poison:i32
; CHECKISEL32-NEXT:   t11: ch,glue = CopyToReg t9, Register:i32 $r4, Constant:i32<255>, t9:1
; CHECKISEL32-NEXT:   t15: ch,glue = PPCISD::CALL_NOP t11, MCSymbol:i32, Register:i32 $r3, Register:i32 $r4, Register:i32 $r2, RegisterMask:Untyped, t11:1
; CHECKISEL32-NEXT:     t16: ch,glue = callseq_end t15, TargetConstant:i32<56>, TargetConstant:i32<0>, t15:1
; CHECKISEL32-NEXT:   t17: ch = PPCISD::RET_GLUE t16

; CHECKASM32:      .test32:
; CHECKASM32-NEXT: # %bb.0:                                # %entry
; CHECKASM32-NEXT:         mflr 0
; CHECKASM32-NEXT:         stwu 1, -64(1)
; CHECKASM32-NEXT:         li 4, 255
; CHECKASM32-NEXT:         stw 0, 72(1)
; CHECKASM32-NEXT:         bl .bar32
; CHECKASM32-NEXT:         nop
; CHECKASM32-NEXT:         addi 1, 1, 64
; CHECKASM32-NEXT:         lwz 0, 8(1)
; CHECKASM32-NEXT:         mtlr 0
; CHECKASM32-NEXT:         blr

; CHECKISEL64:     Initial selection DAG: %bb.0 'test32:entry'
; CHECKISEL64-NEXT: SelectionDAG has 20 nodes:
; CHECKISEL64-NEXT:   t1: i64 = GlobalAddress<ptr @bar32> 0
; CHECKISEL64-NEXT:   t2: i32 = poison
; CHECKISEL64-NEXT:   t3: i32 = Constant<255>
; CHECKISEL64-NEXT:   t7: i64 = Register $x1
; CHECKISEL64-NEXT:       t0: ch,glue = EntryToken
; CHECKISEL64-NEXT:     t6: ch,glue = callseq_start t0, TargetConstant:i64<112>, TargetConstant:i64<0>
; CHECKISEL64-NEXT:   t11: ch,glue = CopyToReg t6, Register:i64 $x3, poison:i64
; CHECKISEL64-NEXT:   t13: ch,glue = CopyToReg t11, Register:i64 $x4, Constant:i64<255>, t11:1
; CHECKISEL64-NEXT:   t17: ch,glue = PPCISD::CALL_NOP t13, MCSymbol:i64, Register:i64 $x3, Register:i64 $x4, Register:i64 $x2, RegisterMask:Untyped, t13:1
; CHECKISEL64-NEXT:     t18: ch,glue = callseq_end t17, TargetConstant:i64<112>, TargetConstant:i64<0>, t17:1
; CHECKISEL64-NEXT:   t19: ch = PPCISD::RET_GLUE t18

; CHECKASM64:      .test32:
; CHECKASM64-NEXT: # %bb.0:                                # %entry
; CHECKASM64-NEXT:         mflr 0
; CHECKASM64-NEXT:         stdu 1, -112(1)
; CHECKASM64-NEXT:         li 4, 255
; CHECKASM64-NEXT:         std 0, 128(1)
; CHECKASM64-NEXT:         bl .bar32
; CHECKASM64-NEXT:         nop
; CHECKASM64-NEXT:         addi 1, 1, 112
; CHECKASM64-NEXT:         ld 0, 16(1)
; CHECKASM64-NEXT:         mtlr 0
; CHECKASM64-NEXT:         blr

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

; CHECKISEL32:     Initial selection DAG: %bb.0 'test8:entry'
; CHECKISEL32-NEXT: SelectionDAG has 20 nodes:
; CHECKISEL32-NEXT:   t1: i32 = GlobalAddress<ptr @bar8> 0
; CHECKISEL32-NEXT:   t2: i8 = poison
; CHECKISEL32-NEXT:   t3: i8 = Constant<-1>
; CHECKISEL32-NEXT:   t9: i32 = Register $r1
; CHECKISEL32-NEXT:       t0: ch,glue = EntryToken
; CHECKISEL32-NEXT:     t8: ch,glue = callseq_start t0, TargetConstant:i32<56>, TargetConstant:i32<0>
; CHECKISEL32-NEXT:   t11: ch,glue = CopyToReg t8, Register:i32 $r3, poison:i32
; CHECKISEL32-NEXT:   t13: ch,glue = CopyToReg t11, Register:i32 $r4, Constant:i32<255>, t11:1
; CHECKISEL32-NEXT:   t17: ch,glue = PPCISD::CALL_NOP t13, MCSymbol:i32, Register:i32 $r3, Register:i32 $r4, Register:i32 $r2, RegisterMask:Untyped, t13:1
; CHECKISEL32-NEXT:     t18: ch,glue = callseq_end t17, TargetConstant:i32<56>, TargetConstant:i32<0>, t17:1
; CHECKISEL32-NEXT:   t19: ch = PPCISD::RET_GLUE t18

; CHECKASM32:      .test8:
; CHECKASM32-NEXT: # %bb.0:                                # %entry
; CHECKASM32-NEXT:         mflr 0
; CHECKASM32-NEXT:         stwu 1, -64(1)
; CHECKASM32-NEXT:         li 4, 255
; CHECKASM32-NEXT:         stw 0, 72(1)
; CHECKASM32-NEXT:         bl .bar8
; CHECKASM32-NEXT:         nop
; CHECKASM32-NEXT:         addi 1, 1, 64
; CHECKASM32-NEXT:         lwz 0, 8(1)
; CHECKASM32-NEXT:         mtlr 0
; CHECKASM32-NEXT:         blr

; CHECKASM64:      .test8:
; CHECKASM64-NEXT: # %bb.0:                                # %entry
; CHECKASM64-NEXT:         mflr 0
; CHECKASM64-NEXT:         stdu 1, -112(1)
; CHECKASM64-NEXT:         li 4, 255
; CHECKASM64-NEXT:         std 0, 128(1)
; CHECKASM64-NEXT:         bl .bar8
; CHECKASM64-NEXT:         nop
; CHECKASM64-NEXT:         addi 1, 1, 112
; CHECKASM64-NEXT:         ld 0, 16(1)
; CHECKASM64-NEXT:         mtlr 0
; CHECKASM64-NEXT:         blr

; CHECKISEL64:      Initial selection DAG: %bb.0 'test8:entry'
; CHECKISEL64-NEXT: SelectionDAG has 22 nodes:
; CHECKISEL64-NEXT:   t1: i64 = GlobalAddress<ptr @bar8> 0
; CHECKISEL64-NEXT:   t2: i8 = poison
; CHECKISEL64-NEXT:   t3: i8 = Constant<-1>
; CHECKISEL64-NEXT:   t4: i32 = poison
; CHECKISEL64-NEXT:   t5: i32 = Constant<255>
; CHECKISEL64-NEXT:   t9: i64 = Register $x1
; CHECKISEL64-NEXT:       t0: ch,glue = EntryToken
; CHECKISEL64-NEXT:     t8: ch,glue = callseq_start t0, TargetConstant:i64<112>, TargetConstant:i64<0>
; CHECKISEL64-NEXT:   t13: ch,glue = CopyToReg t8, Register:i64 $x3, poison:i64
; CHECKISEL64-NEXT:   t15: ch,glue = CopyToReg t13, Register:i64 $x4, Constant:i64<255>, t13:1
; CHECKISEL64-NEXT:   t19: ch,glue = PPCISD::CALL_NOP t15, MCSymbol:i64, Register:i64 $x3, Register:i64 $x4, Register:i64 $x2, RegisterMask:Untyped, t15:1
; CHECKISEL64-NEXT:     t20: ch,glue = callseq_end t19, TargetConstant:i64<112>, TargetConstant:i64<0>, t19:1
; CHECKISEL64-NEXT:   t21: ch = PPCISD::RET_GLUE t20


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

; CHECKISEL32:     Initial selection DAG: %bb.0 'test64:entry'
; CHECKISEL32-NEXT: SelectionDAG has 27 nodes:
; CHECKISEL32-NEXT:   t1: i32 = GlobalAddress<ptr @bar64> 0
; CHECKISEL32-NEXT:   t3: i64 = Constant<255>
; CHECKISEL32-NEXT:   t12: i32 = Register $r1
; CHECKISEL32-NEXT:       t0: ch,glue = EntryToken
; CHECKISEL32-NEXT:     t11: ch,glue = callseq_start t0, TargetConstant:i32<56>, TargetConstant:i32<0>
; CHECKISEL32-NEXT:     t5: i32 = extract_element poison:i64, Constant:i32<1>
; CHECKISEL32-NEXT:   t14: ch,glue = CopyToReg t11, Register:i32 $r3, t5
; CHECKISEL32-NEXT:     t7: i32 = extract_element poison:i64, Constant:i32<0>
; CHECKISEL32-NEXT:   t16: ch,glue = CopyToReg t14, Register:i32 $r4, t7, t14:1
; CHECKISEL32-NEXT:   t18: ch,glue = CopyToReg t16, Register:i32 $r5, Constant:i32<0>, t16:1
; CHECKISEL32-NEXT:   t20: ch,glue = CopyToReg t18, Register:i32 $r6, Constant:i32<255>, t18:1
; CHECKISEL32-NEXT:   t24: ch,glue = PPCISD::CALL_NOP t20, MCSymbol:i32, Register:i32 $r3, Register:i32 $r4, Register:i32 $r5, Register:i32 $r6, Register:i32 $r2, RegisterMask:Untyped, t20:1
; CHECKISEL32-NEXT:     t25: ch,glue = callseq_end t24, TargetConstant:i32<56>, TargetConstant:i32<0>, t24:1
; CHECKISEL32-NEXT:   t26: ch = PPCISD::RET_GLUE t25

; CHECKASM32:      .test64:
; CHECKASM32-NEXT: # %bb.0:                                # %entry
; CHECKASM32-NEXT:         mflr 0
; CHECKASM32-NEXT:         stwu 1, -64(1)
; CHECKASM32-NEXT:         li 5, 0
; CHECKASM32-NEXT:         li 6, 255
; CHECKASM32-NEXT:         stw 0, 72(1)
; CHECKASM32-NEXT:         bl .bar64
; CHECKASM32-NEXT:         nop
; CHECKASM32-NEXT:         addi 1, 1, 64
; CHECKASM32-NEXT:         lwz 0, 8(1)
; CHECKASM32-NEXT:         mtlr 0
; CHECKASM32-NEXT:         blr

; CHECKISEL64:     Initial selection DAG: %bb.0 'test64:entry'
; CHECKISEL64-NEXT: SelectionDAG has 18 nodes:
; CHECKISEL64-NEXT:   t1: i64 = GlobalAddress<ptr @bar64> 0
; CHECKISEL64-NEXT:   t7: i64 = Register $x1
; CHECKISEL64-NEXT:       t0: ch,glue = EntryToken
; CHECKISEL64-NEXT:     t6: ch,glue = callseq_start t0, TargetConstant:i64<112>, TargetConstant:i64<0>
; CHECKISEL64-NEXT:   t9: ch,glue = CopyToReg t6, Register:i64 $x3, poison:i64
; CHECKISEL64-NEXT:   t11: ch,glue = CopyToReg t9, Register:i64 $x4, Constant:i64<255>, t9:1
; CHECKISEL64-NEXT:   t15: ch,glue = PPCISD::CALL_NOP t11, MCSymbol:i64, Register:i64 $x3, Register:i64 $x4, Register:i64 $x2, RegisterMask:Untyped, t11:1
; CHECKISEL64-NEXT:     t16: ch,glue = callseq_end t15, TargetConstant:i64<112>, TargetConstant:i64<0>, t15:1
; CHECKISEL64-NEXT:   t17: ch = PPCISD::RET_GLUE t16

; CHECKASM64:      .test64:
; CHECKASM64-NEXT: # %bb.0:                                # %entry
; CHECKASM64-NEXT:         mflr 0
; CHECKASM64-NEXT:         stdu 1, -112(1)
; CHECKASM64-NEXT:         li 4, 255
; CHECKASM64-NEXT:         std 0, 128(1)
; CHECKASM64-NEXT:         bl .bar64
; CHECKASM64-NEXT:         nop
; CHECKASM64-NEXT:         addi 1, 1, 112
; CHECKASM64-NEXT:         ld 0, 16(1)
; CHECKASM64-NEXT:         mtlr 0
; CHECKASM64-NEXT:         blr
