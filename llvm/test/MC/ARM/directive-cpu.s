@ RUN: llvm-mc < %s -triple armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd - | FileCheck %s

@ CHECK: Name: .ARM.attribute
@ CHECK: SectionData (

@ <format-version>
@ CHECK: 41

@ <section-length>
@ CHECK: 1A0000 00

@ <vendor-name> "aeabi\0"
@ CHECK: 616561 626900

@ <file-tag>
@ CHECK: 01

@ <size>
@ CHECK: 10000000

	.cpu	cortex-a8
@ CHECK: 05636F72 7465782D 613800

@ CHECK: )
