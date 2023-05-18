# RUN: not llvm-mc -triple mips-unknown-unknown %s 2>%t1
# RUN: FileCheck %s < %t1

    .set mips0
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips1
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips2
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips3
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips4
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips5
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips32
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips32r2
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips32r6
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips64
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips64r2
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips64r6
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set arch=mips32
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set mips16
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set nomips16
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set micromips
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set nomicromips
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set msa
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set nomsa
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set dsp
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set dspr2
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set nodsp
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set push
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set pop
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set reorder
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set noreorder
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set macro
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set nomacro
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set at
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set at=$3
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set noat
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .set fp=32
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .cpload $25
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .llvm_internal_mips_reallow_module_directive
    .module fp=32
# CHECK-NOT: :[[@LINE-1]]:13: error: .module directive must appear before any code

    .cpsetup $25, 8, __cerror
    .module fp=64
# CHECK: :[[@LINE-1]]:13: error: .module directive must appear before any code
