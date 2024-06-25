/// Check .attribute parsing.

// RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-readelf -A - | \
// RUN:     FileCheck %s --match-full-lines --implicit-check-not={{.}}

.attribute 4, 71 // Tag_arch
.attribute Tag_cabac, 1
.attribute Tag_hvx_arch, 68
.attribute 7, 1 // Tag_hvx_qfloat

//      CHECK: BuildAttributes {
// CHECK-NEXT:   FormatVersion: 0x41
// CHECK-NEXT:   Section 1 {
// CHECK-NEXT:     SectionLength: 25
// CHECK-NEXT:     Vendor: hexagon
// CHECK-NEXT:     Tag: Tag_File (0x1)
// CHECK-NEXT:     Size: 13
// CHECK-NEXT:     FileAttributes {
// CHECK-NEXT:       Attribute {
// CHECK-NEXT:         Tag: 4
// CHECK-NEXT:         TagName: arch
// CHECK-NEXT:         Value: 71
// CHECK-NEXT:       }
// CHECK-NEXT:       Attribute {
// CHECK-NEXT:         Tag: 10
// CHECK-NEXT:         TagName: cabac
// CHECK-NEXT:         Value: 1
// CHECK-NEXT:       }
// CHECK-NEXT:       Attribute {
// CHECK-NEXT:         Tag: 5
// CHECK-NEXT:         TagName: hvx_arch
// CHECK-NEXT:         Value: 68
// CHECK-NEXT:       }
// CHECK-NEXT:       Attribute {
// CHECK-NEXT:         Tag: 7
// CHECK-NEXT:         TagName: hvx_qfloat
// CHECK-NEXT:         Value: 1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
