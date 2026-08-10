; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s

target triple = "thumbv7m-apple-unknown-macho"

; CHECK: <BITCODE_WRAPPER_HEADER Magic=0x0b17c0de Version=0x{{[0-9a-f]+}} Offset=0x00000014 Size=0x{{[0-9a-f]+}} CPUType=0x0000000c/>
