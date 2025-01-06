/// Check that file attributes are recorded in a .hexagon.attributes section.

q0&=vcmp.gt(v0.bf,v0.bf)     // hvxv73, hvx-qfloat
r3:2=cround(r1:0,#0x0)       // v67, audio
v3:0.w=vrmpyz(v0.b,r0.b)     // hvxv73, zreg
v1:0.sf=vadd(v0.bf,v0.bf)    // hvxv73, hvx-ieee-fp

// RUN: llvm-mc --mattr=+v67,+hvxv73,+hvx-qfloat,+hvx-ieee-fp,+zreg,+audio %s \
// RUN:   -triple=hexagon -filetype=obj --hexagon-add-build-attributes -o %t.o

// RUN: llvm-readelf -A %t.o | \
// RUN:   FileCheck %s --match-full-lines --implicit-check-not={{.}} --check-prefix=READELF

/// llvm-objudmp should be able to determine subtarget features
/// without manually passing in features when an attribute section is present.
// RUN: llvm-objdump -d %t.o | FileCheck %s --check-prefix=OBJDUMP

// RUN: llvm-mc --mattr=+v67,+hvxv73,+hvx-qfloat,+hvx-ieee-fp,+zreg,+audio %s \
// RUN:   -triple=hexagon -filetype=asm --hexagon-add-build-attributes | \
// RUN:     FileCheck %s --match-full-lines --implicit-check-not={{.}} --check-prefix=ASM

//      READELF: BuildAttributes {
// READELF-NEXT:   FormatVersion: 0x41
// READELF-NEXT:   Section 1 {
// READELF-NEXT:     SectionLength: 31
// READELF-NEXT:     Vendor: hexagon
// READELF-NEXT:     Tag: Tag_File (0x1)
// READELF-NEXT:     Size: 19
// READELF-NEXT:     FileAttributes {
// READELF-NEXT:       Attribute {
// READELF-NEXT:         Tag: 4
// READELF-NEXT:         TagName: arch
// READELF-NEXT:         Value: 67
// READELF-NEXT:       }
// READELF-NEXT:       Attribute {
// READELF-NEXT:         Tag: 5
// READELF-NEXT:         TagName: hvx_arch
// READELF-NEXT:         Value: 73
// READELF-NEXT:       }
// READELF-NEXT:       Attribute {
// READELF-NEXT:         Tag: 6
// READELF-NEXT:         TagName: hvx_ieeefp
// READELF-NEXT:         Value: 1
// READELF-NEXT:       }
// READELF-NEXT:       Attribute {
// READELF-NEXT:         Tag: 7
// READELF-NEXT:         TagName: hvx_qfloat
// READELF-NEXT:         Value: 1
// READELF-NEXT:       }
// READELF-NEXT:       Attribute {
// READELF-NEXT:         Tag: 8
// READELF-NEXT:         TagName: zreg
// READELF-NEXT:         Value: 1
// READELF-NEXT:       }
// READELF-NEXT:       Attribute {
// READELF-NEXT:         Tag: 9
// READELF-NEXT:         TagName: audio
// READELF-NEXT:         Value: 1
// READELF-NEXT:       }
// READELF-NEXT:       Attribute {
// READELF-NEXT:         Tag: 10
// READELF-NEXT:         TagName: cabac
// READELF-NEXT:         Value: 1
// READELF-NEXT:       }
// READELF-NEXT:     }
// READELF-NEXT:   }
// READELF-NEXT: }

//      OBJDUMP: 1c80e0d0 {      q0 &= vcmp.gt(v0.bf,v0.bf) }
// OBJDUMP-NEXT: 8ce0c042 {      r3:2 = cround(r1:0,#0x0) }
// OBJDUMP-NEXT: 19e8c000 {      v3:0.w = vrmpyz(v0.b,r0.b) }
// OBJDUMP-NEXT: 1d40e0c0 {      v1:0.sf = vadd(v0.bf,v0.bf) }

//      ASM: .attribute      4, 67   // Tag_arch
// ASM-NEXT: .attribute      5, 73   // Tag_hvx_arch
// ASM-NEXT: .attribute      6, 1    // Tag_hvx_ieeefp
// ASM-NEXT: .attribute      7, 1    // Tag_hvx_qfloat
// ASM-NEXT: .attribute      8, 1    // Tag_zreg
// ASM-NEXT: .attribute      9, 1    // Tag_audio
// ASM-NEXT: .attribute      10, 1   // Tag_cabac
// ASM-EMPTY:
// ASM-NEXT:        {
// ASM-NEXT:                q0 &= vcmp.gt(v0.bf,v0.bf)
// ASM-NEXT:        }
// ASM-NEXT:        {
// ASM-NEXT:                r3:2 = cround(r1:0,#0)
// ASM-NEXT:        }
// ASM-NEXT:        {
// ASM-NEXT:                v3:0.w = vrmpyz(v0.b,r0.b)
// ASM-NEXT:        }
// ASM-NEXT:        {
// ASM-NEXT:                v1:0.sf = vadd(v0.bf,v0.bf)
// ASM-NEXT:        }
