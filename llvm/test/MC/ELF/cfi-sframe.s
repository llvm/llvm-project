// TODO: Add other architectures as they gain sframe support
// REQUIRES: x86-registered-target
// RUN: llvm-mc --assemble --filetype=obj --gsframe -triple x86_64 %s -o %t.o
// RUN: llvm-readelf --sframe %t.o | FileCheck %s

	.cfi_sections .sframe
f1:
	.cfi_startproc  // FRE 0
	nop
	.cfi_def_cfa_offset 16  // FRE 1
	.cfi_def_cfa_offset 8   // location didn't change. No new FRE, but new offset.
	nop
	.cfi_def_cfa_offset 8   // offset didn't change. No new FRE.
	nop
	.cfi_def_cfa_offset 16  // FRE 2. new location, new offset.
	nop
	.cfi_register 0, 1      // Uninteresting register. No new FRE.
	nop

        .cfi_endproc

f2:
	.cfi_startproc
	nop
	nop
        .cfi_endproc

// CHECK: SFrame section '.sframe' {
// CHECK-NEXT:  Header {
// CHECK-NEXT:    Magic: 0xDEE2
// CHECK-NEXT:    Version: V2 (0x2)
// CHECK-NEXT:    Flags [ (0x4)
// CHECK:    ABI: AMD64EndianLittle (0x3)
// CHECK-NEXT:    CFA fixed FP offset (unused): 0
// CHECK-NEXT:    CFA fixed RA offset: 0
// CHECK-NEXT:    Auxiliary header length: 0
// CHECK-NEXT:    Num FDEs: 2
// CHECK-NEXT:    Num FREs: 0
// CHECK-NEXT:    FRE subsection length: 0
// CHECK-NEXT:    FDE subsection offset: 0
// CHECK-NEXT:    FRE subsection offset: 40
// CHECK:    Function Index [
// CHECK-NEXT:        FuncDescEntry [0] {
// CHECK-NEXT:          PC {
// CHECK-NEXT:            Relocation: {{.*}}PC32{{.*}}
// CHECK-NEXT:            Symbol Name: .text
// CHECK-NEXT:            Start Address: {{.*}}
// CHECK-NEXT:          }
// CHECK-NEXT:          Size: 0x5
// CHECK-NEXT:          Start FRE Offset: 0x0
// CHECK-NEXT:          Num FREs: 0
// CHECK-NEXT:          Info {
// CHECK-NEXT:            FRE Type: Addr1 (0x0)
// CHECK-NEXT:            FDE Type: PCInc (0x0)
// CHECK-NEXT:            Raw: 0x0
// CHECK-NEXT:          }
// CHECK-NEXT:          Repetitive block size (unused): 0x0
// CHECK-NEXT:          Padding2: 0x0
// CHECK-NEXT:          FREs [
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        FuncDescEntry [1] {
// CHECK-NEXT:          PC {
// CHECK-NEXT:            Relocation: {{.*}}PC32{{.*}}
// CHECK-NEXT:            Symbol Name: .text
// CHECK-NEXT:            Start Address: {{.*}}
// CHECK-NEXT:          }
// CHECK-NEXT:          Size: 0x2
// CHECK-NEXT:          Start FRE Offset: 0x0
// CHECK-NEXT:          Num FREs: 0
// CHECK-NEXT:          Info {
// CHECK-NEXT:            FRE Type: Addr1 (0x0)
// CHECK-NEXT:            FDE Type: PCInc (0x0)
// CHECK-NEXT:            Raw: 0x0
// CHECK-NEXT:          }
// CHECK-NEXT:          Repetitive block size (unused): 0x0
// CHECK-NEXT:          Padding2: 0x0
// CHECK-NEXT:          FREs [
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
