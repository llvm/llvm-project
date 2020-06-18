; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   --filetype=obj < %s | \
; RUN:   llvm-objdump --mcpu=future -dr - | FileCheck %s --check-prefix=CHECK-O


@array1 = external local_unnamed_addr global [10 x i32], align 4
@array2 = common dso_local local_unnamed_addr global [10 x i32] zeroinitializer, align 4

define dso_local signext i32 @getElementLocal7() local_unnamed_addr {
; CHECK-S-LABEL: getElementLocal7:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    plwa r3, array2@PCREL+28(0), 1
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <getElementLocal7>:
; CHECK-O:         00 00 10 04 00 00 60 a4       plwa 3, 0(0), 1
; CHECK-O-NEXT:    0000000000000000:  R_PPC64_PCREL34      array2+0x1c
; CHECK-O-NEXT:    20 00 80 4e                   blr
entry:
  %0 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @array2, i64 0, i64 7), align 4
  ret i32 %0
}

define dso_local signext i32 @getElementLocalNegative() local_unnamed_addr {
; CHECK-S-LABEL: getElementLocalNegative:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    plwa r3, array2@PCREL-8(0), 1
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <getElementLocalNegative>:
; CHECK-O:         00 00 10 04 00 00 60 a4       plwa 3, 0(0), 1
; CHECK-O-NEXT:    0000000000000020:  R_PPC64_PCREL34      array2-0x8
; CHECK-O-NEXT:    20 00 80 4e                   blr
entry:
  %0 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @array2, i64 0, i64 -2), align 4
  ret i32 %0
}

define dso_local signext i32 @getElementExtern4() local_unnamed_addr {
; CHECK-S-LABEL: getElementExtern4:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    pld r3, array1@got@pcrel(0), 1
; CHECK-S-NEXT:    lwa r3, 16(r3)
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <getElementExtern4>:
; CHECK-O:         00 00 10 04 00 00 60 e4       pld 3, 0(0), 1
; CHECK-O-NEXT:    0000000000000040:  R_PPC64_GOT_PCREL34  array1
; CHECK-O-NEXT:    12 00 63 e8                   lwa 3, 16(3)
; CHECK-O-NEXT:    20 00 80 4e                   blr
entry:
  %0 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @array1, i64 0, i64 4), align 4
  ret i32 %0
}

define dso_local signext i32 @getElementExternNegative() local_unnamed_addr {
; CHECK-S-LABEL: getElementExternNegative:
; CHECK-S:       # %bb.0: # %entry
; CHECK-S-NEXT:    pld r3, array1@got@pcrel(0), 1
; CHECK-S-NEXT:    lwa r3, -4(r3)
; CHECK-S-NEXT:    blr
; CHECK-O-LABEL: <getElementExternNegative>:
; CHECK-O:         00 00 10 04 00 00 60 e4       pld 3, 0(0), 1
; CHECK-O-NEXT:    0000000000000060:  R_PPC64_GOT_PCREL34  array1
; CHECK-O-NEXT:    fe ff 63 e8                   lwa 3, -4(3)
; CHECK-O-NEXT:    20 00 80 4e                   blr
entry:
  %0 = load i32, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @array1, i64 0, i64 -1), align 4
  ret i32 %0
}


