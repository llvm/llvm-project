// RUN: llvm-mc -filetype=obj -triple i686-pc-windows-msvc < %s | llvm-readobj -S --section-data - | FileCheck %s

.text
.set var, 42
.long var
.set var, 19
.long var

// CHECK:Sections [
// CHECK:  Section {
// CHECK:    Name: .text (2E 74 65 78 74 00 00 00)
// CHECK:    SectionData (
// CHECK:      0000: 2A000000 13000000
// CHECK:    )
