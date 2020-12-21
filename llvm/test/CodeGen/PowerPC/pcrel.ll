; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -target-abi=elfv2 -mtriple=powerpc64-- \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   --filetype=obj < %s | \
; RUN:   llvm-objdump --mcpu=pwr10 -dr - | FileCheck %s --check-prefix=CHECK-O

; Constant Pool Index.
; CHECK-S-LABEL: ConstPool
; CHECK-S:       plfd f1, .LCPI0_0@PCREL(0), 1
; CHECK-S:       blr

; CHECK-O-LABEL: ConstPool
; CHECK-O:       plfd 1, 0(0), 1
; CHECK-O-NEXT:  R_PPC64_PCREL34  .rodata.cst8
; CHECK-O:       blr
define dso_local double @ConstPool() local_unnamed_addr {
  entry:
    ret double 0x406ECAB439581062
}

@valIntLoc = common dso_local local_unnamed_addr global i32 0, align 4
define dso_local signext i32 @ReadLocalVarInt() local_unnamed_addr  {
; CHECK-S-LABEL: ReadLocalVarInt
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    plwa r3, valIntLoc@PCREL(0), 1
; CHECK-S-NEXT:    blr

; CHECK-O-LABEL: ReadLocalVarInt
; CHECK-O:         plwa 3, 0(0), 1
; CHECK-O-NEXT:    R_PPC64_PCREL34 valIntLoc
; CHECK-O-NEXT:    blr
entry:
  %0 = load i32, i32* @valIntLoc, align 4
  ret i32 %0
}

@valIntGlob = external global i32, align 4
define dso_local signext i32 @ReadGlobalVarInt() local_unnamed_addr  {
; CHECK-S-LABEL: ReadGlobalVarInt
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    pld r3, valIntGlob@got@pcrel(0), 1
; CHECK-S-NEXT: .Lpcrel0:
; CHECK-S-NEXT:    .reloc .Lpcrel0-8,R_PPC64_PCREL_OPT,.-(.Lpcrel0-8)
; CHECK-S-NEXT:    lwa r3, 0(r3)
; CHECK-S-NEXT:    blr

; CHECK-O-LABEL: ReadGlobalVarInt
; CHECK-O:         pld 3, 0(0), 1
; CHECK-O-NEXT:    R_PPC64_GOT_PCREL34 valIntGlob
; CHECK-O-NEXT:    R_PPC64_PCREL_OPT *ABS*+0x8
; CHECK-O-NEXT:    lwa 3, 0(3)
; CHECK-O-NEXT:    blr
entry:
  %0 = load i32, i32* @valIntGlob, align 4
  ret i32 %0
}
