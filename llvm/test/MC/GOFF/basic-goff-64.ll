; RUN: llc %s -mtriple s390x-ibm-zos -filetype=obj -o - | od -v -Ax -tx1 | FileCheck --ignore-case %s
; REQUIRES: systemz-registered-target

@x = global i32 0, align 4
@y = internal global i32 1, align 4
@z = external global i32, align 4

; Function Attrs: noinline nounwind optnone
define hidden void @foo() {
entry:
  store i32 8200, ptr @x, align 4
  %0 = load i32, ptr @x, align 4
  store i32 2, ptr @y, align 4
  store i32 100, ptr @z, align 4
  call void @bar(i32 noundef signext %0)
  ret void
}

declare void @bar(i32 noundef signext)

; Records for basic-goff-64#C
; Requires a continuation record due to the name's length. ESD items
; with names of length greater than 8 require continuation records.
; Byte 1 of the first record for this ESD entry is 0x01 to indicate
; that this record is continued, and byte 2 of the second record
; to indicate that this is the final continuation record.
; Byte 3 of first record for this ESD entry is 0x00 to indicate
; SD (Section Definition).
; This is the "root" SD Node.
; CHECK: 000050 03 01 00 00 00 00 00 01 00 00 00 00 00 00 00 00
; CHECK: 000060 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000070 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000080 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 60
; CHECK: 000090 00 01 00 00 00 00 00 0f 82 81 a2 89 83 60 87 96
; CHECK: 0000a0 03 02 00 86 86 60 f6 f4 7b c3 00 00 00 00 00 00
; CHECK: 0000b0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0000c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0000d0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0000e0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00


; The next two logical records represent the ADA.
; Record for C_WSA64.
; Byte 3 is 0x01 to indicate ED (Element Definition).
; This represents the writable static area.
; CHECK: 0000f0 03 00 00 01 00 00 00 02 00 00 00 01 00 00 00 00
; CHECK: 000100 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000110 00 00 00 00 00 00 00 00 03 80 00 00 00 00 00 00
; CHECK: 000120 00 00 00 00 00 00 00 00 00 00 00 00 04 04 01 01
; CHECK: 000130 00 40 04 00 00 00 00 07 c3 6d e6 e2 c1 f6 f4 00

; Records for basic-goff-64#S
; Requires a continuation record.
; Byte 3 is 0x03 to indicate PR (Part Reference).
; This represents the ADA (associated data area). 
; CHECK: 000140 03 01 00 03 00 00 00 03 00 00 00 02 00 00 00 00
; CHECK: 000150 00 00 00 00 00 00 00 00 00 00 00 28 00 00 00 00
; CHECK: 000160 00 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00
; CHECK: 000170 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 01
; CHECK: 000180 10 01 24 00 00 00 00 0f 82 81 a2 89 83 60 87 96
; CHECK: 000190 03 02 00 86 86 60 f6 f4 7b e2 00 00 00 00 00 00
; CHECK: 0001a0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0001b0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0001c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0001d0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00


; The next two logical records represent the code section. Source
; code and affiliated metadata (such as the PPA1 and PPA2 sections)
; reside here.
; Record for C_CODE64.
; Byte 3 is 0x01 to indicate ED (Element Definition).
; CHECK: 0001e0 03 00 00 01 00 00 00 04 00 00 00 01 00 00 00 00
; CHECK: 0001f0 00 00 00 00 00 00 00 00 00 00 00 ac 00 00 00 00
; CHECK: 000200 00 00 00 00 00 00 00 00 01 80 00 00 00 00 00 00
; CHECK: 000210 00 00 00 00 00 00 00 00 00 00 00 00 04 04 00 0a
; CHECK: 000220 00 00 03 00 00 00 00 08 c3 6d c3 d6 c4 c5 f6 f4

; Records for basic-goff-64#C. Note that names for ESD entries
; need not be unique.
; Byte 3 is 0x02 to indicate LD (Label Definition).
; CHECK: 000230 03 01 00 02 00 00 00 05 00 00 00 04 00 00 00 00
; CHECK: 000240 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000250 00 00 00 00 00 00 00 00 01 00 00 00 00 00 00 03
; CHECK: 000260 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 02
; CHECK: 000270 00 01 20 00 00 00 00 0f 82 81 a2 89 83 60 87 96
; CHECK: 000280 03 02 00 86 86 60 f6 f4 7b c3 00 00 00 00 00 00
; CHECK: 000290 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0002a0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0002b0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0002c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00


; Records for the C_@@QPPA2 section, which contains the offset
; to the PPA2. 
; CHECK: 0002d0 03 01 00 01 00 00 00 06 00 00 00 01 00 00 00 00
; CHECK: 0002e0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 0002f0 00 00 00 00 00 00 00 00 03 80 00 00 00 00 00 00
; CHECK: 000300 00 00 00 00 00 00 00 00 00 00 00 00 04 04 01 09
; CHECK: 000310 00 00 03 00 00 00 00 09 c3 6d 7c 7c d8 d7 d7 c1
; CHECK: 000320 03 02 00 f2 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000330 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000340 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000350 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000360 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 000370 03 00 00 03 00 00 00 07 00 00 00 06 00 00 00 00
; CHECK: 000380 00 00 00 00 00 00 00 00 00 00 00 08 00 00 00 00
; CHECK: 000390 00 00 00 00 00 00 00 00 03 20 00 00 00 00 00 00
; CHECK: 0003a0 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 01
; CHECK: 0003b0 10 01 03 00 00 00 00 06 4b 50 97 97 81 f2 00 00


; Record for function foo().
; Byte 3 is 0x02 to indicate LD.
; This record is owned by ESD entry with ID 4, which is 
; C_CODE64. That is in turn owned by ESD entry with ID 1,
; which is C_WSA64, which is owned by Section Definition
; basic-goff-64#C. All functions in GOFF defined in the
; compilation unit are defined in this manner.
; Byte 63 is 0x02 = 0b00000010. Bits 5-7 indicate that
; this record is executable, since it contains code. Note
; that Bits 5-7 should be 001 if it is not executable and
; 000 if executability is not specified.
; Byte 65 is 0x03 = 0b00000011. Bits 4-7 indicate the
; binding scope, which is library scope. This means 
; that symbol is NOT available fo dynamic binding. However,
; it is still available for static linking. This is due to the
; hidden attribute on the function definition.
; Note that the symbol name is written beginning in byte
; 72. 0x86 0x96 0x96 spell `foo` in EBCDIC encoding.
; CHECK: 03 00 00 02 00 00 00 0a 00 00 00 04 00 00 00 00
; CHECK: 00 00 00 10 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 01 00 00 00 00 00 00 03
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 02
; CHECK: 00 03 20 00 00 00 00 03 86 96 96 00 00 00 00 00

; Record for Section Definition for global variable x.
; Note that all initialized global variables require their
; own section definition. This includes functions defined in
; this module. Note that bytes 4-7 indicate that ESDID is 9.
; Byte 3 is 0x00 to indicate SD.
; CHECK: 03 00 00 00 00 00 00 0b 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 01 a7 00 00 00 00 00 00 00

; Record for C_WSA64 belonging to Section for global variable x.
; Byte 3 is 0x01 to indicate ED.
; Bytes 8-11 indicate that Parent ESDID is 9.
; CHECK: 03 00 00 01 00 00 00 0c 00 00 00 0b 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 03 80 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 04 04 01 01
; CHECK: 00 40 04 00 00 00 00 07 c3 6d e6 e2 c1 f6 f4 00

; Record for PR of global variable x.
; Byte 3 is 0x03 to indicate PR.
; Bytes 8-11 indicate that Parent ESDID is A, the above
; C_WSA64.
; Byte 65 is 0x04 = 0b00000100. Bits 4-7 indicate the
; binding scope, which is Import-Export scope. This means 
; that symbol is available for dynamic binding.
; Byte 66 is 0x20 = 0b00100000. Bits 1-2 indicate the linkage.
; In this case 0b01 indicates XPLINK.
; CHECK: 03 00 00 03 00 00 00 0d 00 00 00 0c 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 04 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 01
; CHECK: 10 04 20 00 00 00 00 01 a7 00 00 00 00 00 00 00

; Global variable y works much like x, but with a few
; differences:
; y is explicitly listed as internal, so its binding scope is
; set to B'0001 (Section Scope). This is true for the SD record
; as well as the PR record.
; CHECK: 03 00 00 00 00 00 00 0e 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 01 00 00 00 00 00 01 a8 00 00 00 00 00 00 00
; CHECK: 03 00 00 01 00 00 00 0f 00 00 00 0e 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 03 80 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 04 04 01 01
; CHECK: 00 40 04 00 00 00 00 07 c3 6d e6 e2 c1 f6 f4 00
; CHECK: 03 00 00 03 00 00 00 10 00 00 00 0f 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 04 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 01
; CHECK: 10 01 20 00 00 00 00 01 a8 00 00 00 00 00 00 00


; Record for C_WSA64. Child of section basic-goff-64#C
; and contains an extern global variable. (z)
; CHECK: 03 00 00 01 00 00 00 11 00 00 00 01 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 03 80 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 04 04 01 01
; CHECK: 00 40 00 00 00 00 00 07 c3 6d e6 e2 c1 f6 f4 00


; Record for external global variable z. This is NOT
; initialized in this compilation unit and so unlike
; global variable x, it lacks its own section definition.
; Byte 3 is 0x03 to indicate PR.
; Bytes 8-11 indicate that parent ESDID is 0d, which is
; above C_WSA64.
; Byte 65 is 0x04 = 0b00000100. Bits 4-7 indicate the
; binding scope, which is Import-Export scope. This is
; required because it is imported (not defined in this
; module.
; CHECK: 03 00 00 03 00 00 00 12 00 00 00 11 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 01
; CHECK: 10 04 20 00 00 00 00 01 a9 00 00 00 00 00 00 00


; Record for externally defined function bar().
; Byte 3 is 0x04 to indicate External Reference (ErWx).
; Bytes 8-11 indicate that parent ESDID is 01, the section
; definition for this module. (basic-goff-64#C). 
; Byte 65 indicates that the binding scope is Import-Export
; Scope, since the definition may be something we dynamically
; link against.
; CHECK: 03 00 00 04 00 00 00 13 00 00 00 01 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
; CHECK: 00 00 00 00 00 00 00 00 00 00 00 00 04 00 00 02
; CHECK: 00 04 20 00 00 00 00 03 82 81 99 00 00 00 00 00


; TXT Records:

; TXT Record corresponding to global variable y.
; Bytes 4-7 indicate that the owning ESD has ID 0x10. This is the
; PR for global variable x.
; Byte 22-23 contain the data length, which is 4, as expected for
; an i32 type.
; CHECK-DAG: 03 10 00 00 00 00 00 10 00 00 00 00 00 00 00 00

; TXT Record corresponding to global variable x.
; CHECK-DAG: 03 10 00 00 00 00 00 0d 00 00 00 00 00 00 00 00

; TXT Record corresponding to the C_CODE64 Section.
; This contains the bodies of the function(s) that make up
; a module.
; CHECK-DAG: 03 11 00 00 00 00 00 04 00 00 00 00 00 00 00 00

; TXT Record containing ADA.
; CHECK-DAG: 03 10 00 00 00 00 00 03 00 00 00 00 00 00 00 00
