# RUN: not llvm-mc --triple=loongarch32 %s 2>&1 | FileCheck %s

# CHECK: :[[#@LINE+1]]:1: error: instruction requires the following: LA64 Basic Integer and Privilege Instruction Set
movgr2fr.d $fa0, $a0

# CHECK: :[[#@LINE+1]]:1: error: instruction requires the following: LA64 Basic Integer and Privilege Instruction Set
movfr2gr.d $a0, $fa0
