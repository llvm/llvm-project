@ RUN: llvm-mc < %s -triple armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd | FileCheck %s

@ CHECK: Name: .ARM.attribute
@ CHECK: SectionData (

@ <format-version>
@ CHECK: 41

@ <section-length>
@ CHECK: 130000 00

@ <vendor-name> "aeabi\0"
@ CHECK: 616561 626900

@ <file-tag>
@ CHECK: 01

@ <size>
@ CHECK: 09000000

	.fpu	neon
@ CHECK: 0A03
@ CHECK: 0C01

@ CHECK: )
