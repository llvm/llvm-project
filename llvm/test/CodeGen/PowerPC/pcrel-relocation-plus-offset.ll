; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   --filetype=obj < %s | \
; RUN:   llvm-objdump --mcpu=pwr10 -dr - | FileCheck %s --check-prefix=CHECK-O
; RUN: llc -verify-machineinstrs -target-abi=elfv2 -mtriple=powerpc64-- \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -target-abi=elfv2 -mtriple=powerpc64-- \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   --filetype=obj < %s | \
; RUN:   llvm-objdump --mcpu=pwr10 -dr - | FileCheck %s --check-prefix=CHECK-O



@array1 = external local_unnamed_addr global [10 x i32], align 4
@array2 = common dso_local local_unnamed_addr global [10 x i32] zeroinitializer, align 4

define dso_local signext i32 @getElementLocal7() local_unnamed_addr {
; CHECK-S-LABEL: getElementLocal7:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    plwa r3, array2@PCREL+28(0), 1
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <getElementLocal7>:
; CHECK-O:         plwa 3, 0(0), 1
; CHECK-O-NEXT:      R_PPC64_PCREL34      array2+0x1c
; CHECK-O-NEXT:    blr
entry:
  %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @array2, i64 0, i64 7), align 4
  ret i32 %0
}

define dso_local signext i32 @getElementLocalNegative() local_unnamed_addr {
; CHECK-S-LABEL: getElementLocalNegative:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    plwa r3, array2@PCREL-8(0), 1
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <getElementLocalNegative>:
; CHECK-O:         plwa 3, 0(0), 1
; CHECK-O-NEXT:      R_PPC64_PCREL34      array2-0x8
; CHECK-O-NEXT:    blr
entry:
  %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @array2, i64 0, i64 -2), align 4
  ret i32 %0
}

define dso_local signext i32 @getElementExtern4() local_unnamed_addr {
; CHECK-S-LABEL: getElementExtern4:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    pld r3, array1@got@pcrel(0), 1
; CHECK-S-NEXT:  .Lpcrel0:
; CHECK-S-NEXT:    .reloc .Lpcrel0-8,R_PPC64_PCREL_OPT,.-(.Lpcrel0-8)
; CHECK-S-NEXT:    lwa r3, 16(r3)
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <getElementExtern4>:
; CHECK-O:         pld 3, 0(0), 1
; CHECK-O-NEXT:      R_PPC64_GOT_PCREL34  array1
; CHECK-O-NEXT:      R_PPC64_PCREL_OPT *ABS*+0x8
; CHECK-O:         lwa 3, 16(3)
; CHECK-O-NEXT:    blr
entry:
  %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @array1, i64 0, i64 4), align 4
  ret i32 %0
}

define dso_local signext i32 @getElementExternNegative() local_unnamed_addr {
; CHECK-S-LABEL: getElementExternNegative:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    pld r3, array1@got@pcrel(0), 1
; CHECK-S-NEXT:  .Lpcrel1:
; CHECK-S-NEXT:    .reloc .Lpcrel1-8,R_PPC64_PCREL_OPT,.-(.Lpcrel1-8)
; CHECK-S-NEXT:    lwa r3, -4(r3)
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <getElementExternNegative>:
; CHECK-O:         pld 3, 0(0), 1
; CHECK-O-NEXT:      R_PPC64_GOT_PCREL34  array1
; CHECK-O-NEXT:      R_PPC64_PCREL_OPT *ABS*+0x8
; CHECK-O:         lwa 3, -4(3)
; CHECK-O-NEXT:    blr
entry:
  %0 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @array1, i64 0, i64 -1), align 4
  ret i32 %0
}


